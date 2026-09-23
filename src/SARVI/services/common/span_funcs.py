import torch
import string
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from stop_words import get_stop_words

from ...models.schemas import (
    Any
)
from .utils.span_utils import(
    get_edge_words, has_bad_surrounding, is_number, token_label_candidates, join_sentencepiece_tokens, valid_words_for_weight, lexicon_lemmas, to_list, labels_for_window, safe_tensor_item
)

###
stopwords_es = list(set(get_stop_words('spanish')+list(string.ascii_lowercase)+["ñ","ç","ch"]))
###

def generate_sequences(tokenizer: Any, sequence: list, max_len: int = 10):
    """
    Genera todas las subsecuencias posibles de tokens dentro de una secuencia, limitando la longitud máxima de cada span.

    Parameters
    ----------
        `tokenizer`: Any
            - Tokenizer utilizado para obtener el token de padding cuando el span llega al final de la secuencia

        `sequence`: list
            - Lista de IDs de tokens sobre la que se generan los spans. Se asume que contiene tokens especiales al inicio y al final.

        `max_len`: int
            - Longitud máxima permitida para cada span generado. Por defecto es **10**

    Returns
    -------
        `sequences_span`: list
            - Lista de tuplas con la forma **(seq, span_tokens, next_token)**: **seq** contiene los índices del span, **span_tokens** contiene los IDs del span y **next_token** contiene el token siguiente.
    """
    sequences_span = []
    n = len(sequence)-2

    for start in range(1, n + 1):
        for end in range(start, min(start + max_len, n + 1)):
            seq = list(range(start, end + 1))

            if end + 1 < len(sequence):
                next_token = [sequence[end + 1]]
            else:
                next_token = [tokenizer.pad_token_id]

            span_tokens = sequence[start:end + 1]
            sequences_span.append((seq, span_tokens, next_token))

    return sequences_span

def is_valid_decoder(decoder: list, next_token: list, tokenizer: Any, limit_stopwords_surround: int = 3):
    """
    Valida si un span decodificado representa una secuencia completa y útil para el modelo, descartando spans incompletos, mal segmentados o poco informativos.

    Parameters
    ----------
        `decoder`: list
            - Lista de tokens decodificados que forman el span candidato

        `next_token`: list
            - Lista con el token siguiente al span. Se usa para comprobar si la última palabra del span está completa

        `tokenizer`: Any
            - Tokenizer utilizado para comprobar el token de padding y otras reglas asociadas a la tokenización

        `limit_stopwords_surround`: int
            - Número máximo de palabras de contexto que se revisan alrededor de cada palabra de contenido para detectar spans poco informativos. Por defecto es **3**

    Returns
    -------
        ``: bool
            - **True** si el span es válido según las reglas definidas. En caso contrario, devuelve **False**.
    """
    if not decoder:
        return False

    # 2. First token must start a full word
    if not decoder[0].startswith("▁"):
        return False

    # 3. Last word must be complete
    if next_token[0] != tokenizer.pad_token and not next_token[0].startswith("▁") and next_token[0].strip("▁") not in string.punctuation:
        return False

    # 4. No token should contain newlines
    if any("\n" in token for token in decoder):
        return False

    # 5. Do not allow spans covering multiple sentences
    if any(token.strip("▁") in string.punctuation for token in decoder):
        return False

    # 6. Reject spans made only of punctuation, stopwords or numbers
    if all(token.strip("▁") in string.punctuation or token.strip("▁").lower() in stopwords_es or is_number(token.strip("▁")) for token in decoder):
        return False

    # 7. Remove the span if it starts or ends with a stopword
    if get_edge_words(decoder)[0] in stopwords_es or get_edge_words(decoder)[1] in stopwords_es:
        return False

    # 8. Reject spans where content words are surrounded by too many
    # punctuation / stopword / number words
    if not has_bad_surrounding(decoder, stopwords_es, limit_stopwords_surround=limit_stopwords_surround):
        return False

    return True

def validate_span_in_entity_list(token_span, entity_list, full=True):
    """
    Checks whether a token span is contained in or overlaps a known entity.

    Parameters
    ----------
        `token_span`: Any
            - Token or character positions used for alignment.
        `entity_list`: Any
            - Known entities used for span comparison.
        `full`: Any
            - Argument controlling full.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    span_start, span_end = token_span[0], token_span[-1]+1
    for entity in entity_list:
        ent_start, ent_end = entity[0][0], entity[0][-1]
        if full:
            if span_start == ent_start and span_end == ent_end:
                return True
        else:
            if span_start >= ent_start and span_end <= ent_end:
                return True
    return False

def calculate_weight_span(token_span, entity_list, text, decoded_span):
    """
    Calculates a score for a candidate span using its entity coverage and text.

    Parameters
    ----------
        `token_span`: Any
            - Token or character positions used for alignment.
        `entity_list`: Any
            - Known entities used for span comparison.
        `text`: Any
            - Text containing the entity or span.
        `decoded_span`: Any
            - Decoded text corresponding to the candidate span.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    span_start, span_end = token_span[0], token_span[-1]+1

    for entity in entity_list:
        ent_start, ent_end = entity[0][0], entity[0][-1]
        if span_start == ent_start and span_end == ent_end:
            return 2.0
        else:
            if span_start >= ent_start and span_end <= ent_end:
                entity_real = text[entity[1][0]:entity[1][-1]]
                entity_pred = "".join((" " + t[1:]) if t.startswith("▁") else t for t in decoded_span).strip()

                valid_words_real = set([token for token in entity_real.split(" ") if token not in string.punctuation and token.lower() not in stopwords_es and not is_number(token)])
                valid_words_pred = set([token for token in entity_pred.split(" ") if token not in string.punctuation and token.lower() not in stopwords_es and not is_number(token)])

                if valid_words_real == valid_words_pred:
                    return 1.8

                overlap = len(valid_words_pred & valid_words_real)
                coverage = overlap / len(valid_words_real) if valid_words_real else 0
                return 1.0 + 0.8 * coverage

def build_entities_from_window_labels(labels: list[str]) -> dict[str, dict]:
    """
    Groups token labels from one window into entity spans.

    Parameters
    ----------
        `labels`: list[str]
            - Entity or relation labels used by the model.

    Returns
    -------
        `dict[str, dict]`
            - Mapping containing the processed values.
    """
    entities = {}

    for token_idx, raw_label in enumerate(labels):
        for candidate in token_label_candidates(raw_label):
            identity = candidate["identity"]
            entity = entities.setdefault(identity, {"identity": identity, "label": candidate["entity"], "tokens": [], "priority_by_token": {}})
            entity["tokens"].append(token_idx)
            entity["priority_by_token"][token_idx] = max(int(candidate["pos"]), int(entity["priority_by_token"].get(token_idx, -1)))

    for entity in entities.values():
        entity["tokens"] = sorted(set(entity["tokens"]))
        entity["token_set"] = set(entity["tokens"])
        entity["start"] = min(entity["tokens"]) if entity["tokens"] else None
        entity["end"] = max(entity["tokens"]) + 1 if entity["tokens"] else None

    return entities

def choose_entity_for_span(token_idx: list[int], entities: dict[str, dict]) -> dict | None:
    """
    Selects the best matching entity for a candidate token span.

    Parameters
    ----------
        `token_idx`: list[int]
            - Token positions used to identify the entity.
        `entities`: dict[str, dict]
            - Argument controlling entities.

    Returns
    -------
        `dict | None`
            - Mapping containing the processed values.
    """
    token_set = set(token_idx)
    candidates = []

    for entity in entities.values():
        if not token_set:
            continue

        if token_set.issubset(entity["token_set"]):
            priorities = [entity["priority_by_token"].get(idx, -1) for idx in token_idx]
            candidates.append({
                **entity,
                "span_priority": min(priorities),
                "is_exact": token_idx[0] == entity["start"] and token_idx[-1] + 1 == entity["end"],
                "length": len(entity["tokens"]),
            })

    if not candidates:
        return None

    return sorted(candidates, key=lambda item: (item["span_priority"], item["is_exact"], -item["length"]), reverse=True)[0]

def calculate_weight_span_from_entity(token_idx: list[int], entity: dict, window_input_ids: list[int], tokenizer: Any, decoded_span: list[str], lexicon_lemma_set: set[str]) -> float:
    """
    Scores a candidate span against an entity using token and lexical evidence.

    Parameters
    ----------
        `token_idx`: list[int]
            - Token positions used to identify the entity.
        `entity`: dict
            - Entity record being processed.
        `window_input_ids`: list[int]
            - Encoded IDs for the NER window.
        `tokenizer`: Any
            - Tokenizer used to encode text and obtain offsets.
        `decoded_span`: list[str]
            - Decoded text corresponding to the candidate span.
        `lexicon_lemma_set`: set[str]
            - Set of normalized lexicon lemmas.

    Returns
    -------
        `float`
            - Derived value produced by the operation.
    """
    if token_idx[0] == entity["start"] and token_idx[-1] + 1 == entity["end"]:
        return 2.0

    entity_token_ids = [window_input_ids[idx] for idx in entity["tokens"] if 0 <= idx < len(window_input_ids)]
    entity_tokens = tokenizer.convert_ids_to_tokens(entity_token_ids)
    entity_text = join_sentencepiece_tokens(entity_tokens)
    span_text = join_sentencepiece_tokens(decoded_span)

    valid_words_real = valid_words_for_weight(entity_text, entity["label"], lexicon_lemma_set)
    valid_words_pred = valid_words_for_weight(span_text, entity["label"], lexicon_lemma_set)

    if valid_words_real and valid_words_real == valid_words_pred:
        return 1.8

    overlap = len(valid_words_pred & valid_words_real)
    coverage = overlap / len(valid_words_real) if valid_words_real else 0.0
    return 1.0 + 0.8 * coverage

def construct_span_data(data: dict, id2label: dict, label2id: dict, tokenizer: Any, skip_incomplete_spans: bool = True, lexicon: pd.DataFrame|None = None):
    """
    Creates span-classification examples from tokenized NER windows.

    Parameters
    ----------
        `data`: dict
            - Input dataframe containing the records to process.
        `id2label`: dict
            - Mapping from numeric identifiers to labels.
        `label2id`: dict
            - Mapping from labels to numeric identifiers.
        `tokenizer`: Any
            - Tokenizer used to encode text and obtain offsets.
        `skip_incomplete_spans`: bool
            - Token or character positions used for alignment.
        `lexicon`: pd.DataFrame | None
            - Optional lexicon used to score candidate spans.

    Returns
    -------
        `Any`
            - Constructed data ready for the next pipeline step.
    """
    input_ids = data["input_ids"]
    lexicon_lemma_set = lexicon_lemmas(lexicon)

    span_hidden_states = []
    span_token_idx = []
    span_global_token_idx = []
    span_char_idx = []
    span_window_index = []
    span_text_instance = []
    span_file_names = []
    span_tokens = []
    decoded_spans = []
    targets = []
    y_true = []
    weights = []

    def is_ignore_label(value: Any) -> bool:
        value = safe_tensor_item(value)
        try:
            return int(value) == -100
        except Exception:
            return str(value) == "-100"

    def span_labels_are_ignored(raw_labels: Any, token_span_idx: list[int]) -> bool:
        if raw_labels is None:
            return False
        try:
            return all(is_ignore_label(raw_labels[idx]) for idx in token_span_idx)
        except Exception:
            return False

    # for window_idx in tqdm(range(int(input_ids.shape[0])), desc="Constructing NER dataset: Creating correct spans", unit="window", total=len(range(int(input_ids.shape[0]))), leave=False):
    for window_idx in range(int(input_ids.shape[0])):
        window_input_ids = to_list(input_ids[window_idx])
        window_attention_mask = to_list(data.get("attention_mask")[window_idx]) if data.get("attention_mask") is not None else [1] * len(window_input_ids)
        active_len = int(sum(int(mask) for mask in window_attention_mask))
        active_input_ids = window_input_ids[:active_len]
        token_span_sequences = generate_sequences(tokenizer, active_input_ids, max_len=15)

        window_labels = labels_for_window(data, window_idx, id2label)
        raw_window_labels = data.get("window_labels")[window_idx] if data.get("window_labels") is not None else None
        window_entities = build_entities_from_window_labels(window_labels) if window_labels is not None else {}

        for token_span_idx, token_ids, next_token_ids in token_span_sequences:
            decoded_span = tokenizer.convert_ids_to_tokens(token_ids)
            decoded_next_token = tokenizer.convert_ids_to_tokens(next_token_ids)

            if skip_incomplete_spans and not is_valid_decoder(decoded_span, decoded_next_token, tokenizer):
                continue

            span_hidden_states.append(data["last_hidden_state"][window_idx, token_span_idx[0]:token_span_idx[-1] + 1].mean(dim=0))
            span_token_idx.append(token_span_idx)

            if data.get("global_token_indices") is not None:
                span_global_token_idx.append(to_list(data.get("global_token_indices")[window_idx, token_span_idx[0]:token_span_idx[-1] + 1]))

            if data.get("window_offset_mapping") is not None:
                span_offsets = to_list(data.get("window_offset_mapping")[window_idx, token_span_idx[0]:token_span_idx[-1] + 1])
                valid_offsets = [tuple(offset) for offset in span_offsets if len(offset) == 2 and int(offset[0]) != int(offset[1])]
                if valid_offsets:
                    span_char_idx.append((min(int(start) for start, _ in valid_offsets), max(int(end) for _, end in valid_offsets)))
                else:
                    span_char_idx.append(None)

            span_window_index.append(window_idx)

            if data.get("window_to_text") is not None:
                local_text_idx = int(safe_tensor_item(data.get("window_to_text")[window_idx]))
                if data.get("text_indices") is not None:
                    span_text_instance.append(int(safe_tensor_item(data.get("text_indices")[local_text_idx])))
                else:
                    span_text_instance.append(local_text_idx)

                if data.get("file_names") is not None and local_text_idx < len(data.get("file_names")):
                    span_file_names.append(data.get("file_names")[local_text_idx])

            span_tokens.append(token_ids)
            decoded_spans.append(decoded_span)

            if data.get("window_labels") is not None:
                entity = choose_entity_for_span(token_span_idx, window_entities)

                if entity is None:
                    target = "O"
                    weight = 0.5
                else:
                    target = entity["label"]
                    weight = calculate_weight_span_from_entity(token_idx=token_span_idx, entity=entity, window_input_ids=window_input_ids, tokenizer=tokenizer, decoded_span=decoded_span, lexicon_lemma_set=lexicon_lemma_set)

                targets.append(target)
                if target == "O" and span_labels_are_ignored(raw_window_labels, token_span_idx):
                    y_true.append(-100)
                else:
                    y_true.append(label2id[target])
                weights.append(weight)

    out = {**data, "last_hidden_state": torch.stack(span_hidden_states, dim=0) if span_hidden_states else data["last_hidden_state"].new_empty((0, data["last_hidden_state"].size(-1))),
        "span_token_idx": span_token_idx, "Token idx": span_token_idx, "span_window_index": span_window_index, "Tokens": span_tokens, "Decoded span": decoded_spans}

    if span_global_token_idx:
        out["span_global_token_idx"] = span_global_token_idx

    if span_char_idx:
        out["span_char_idx"] = span_char_idx

    if span_text_instance:
        out["span_text_instance"] = span_text_instance
        out["Text instance"] = span_text_instance

    if span_file_names:
        out["span_file_names"] = span_file_names
        out["File"] = span_file_names

    if data.get("window_labels") is not None:
        out["Target"] = targets
        out["Weight"] = torch.tensor(weights, dtype=data["last_hidden_state"].dtype, device=data["last_hidden_state"].device)
        out["weights"] = out["Weight"]
        out["y_true"] = torch.tensor(y_true, dtype=torch.long, device=data["last_hidden_state"].device)

    return out
