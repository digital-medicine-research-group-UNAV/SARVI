import ast
import torch
import string
import pandas as pd

from stop_words import get_stop_words
from transformers import AutoTokenizer, AutoModel

from .utils.ner_utils import (
    get_edge_words, is_number, has_bad_surrounding, merge_sentencepiece_words, char_span_to_token_span, lit
)

from ...data_io.reader import (
    read_torch_checkpoint
)
from ...models.schemas import (
    Any, PipelineContext
)
from ...models.neural_networks import (
    SpanClassifier
)

###
device = "cuda" if torch.cuda.is_available() else "cpu"
stopwords_es = list(set(get_stop_words('spanish')+list(string.ascii_lowercase)+["ñ","ç","ch"]))

id2label_ner = {0: 'ACTOR', 1: 'CLINENTITY', 2: 'O', 3: 'TIMEX3'}
label2id_ner = {'ACTOR': 0, 'CLINENTITY': 1, 'O': 2, 'TIMEX3': 3}

tokenizer_ner = AutoTokenizer.from_pretrained("IIC/RigoBERTa-Clinical", trim_offsets=False, use_fast=True)
model_ner = AutoModel.from_pretrained("IIC/RigoBERTa-Clinical").to(device)
###

def average_overlapping_hidden_states_checked(windows: list[dict[str, Any]], last_hidden_state_list: list[torch.Tensor], *, window_tokens_no_special: int, stride: int, strict: bool = True) -> tuple[list[torch.Tensor], list[str]]:
    """
    Promedia los estados ocultos de los mismos tokens del texto original cuando aparecen
    en varias ventanas solapadas, comprobando que dimensiones, máscaras e índices sean coherentes.

    Parameters
    ----------
        `windows`: list[dict[str, Any]]
            - Lista de ventanas generadas previamente. Cada ventana debe contener, al menos,
            `attention_mask`, `input_ids` y `absolute_token_indices`. Esta última clave
            identifica qué token del texto original ocupa cada posición de la ventana.

        `last_hidden_state_list`: list[torch.Tensor]
            - Lista de tensores con los estados ocultos generados por el modelo para cada ventana. Cada tensor debe tener forma **(1, seq_len, hidden_dim)**

        `window_tokens_no_special`: int
            - Número máximo de tokens reales por ventana, sin contar tokens especiales como `CLS` y `SEP`

        `stride`: int
            - Desplazamiento entre ventanas consecutivas, medido en tokens del texto completo.
            Se usa junto con `window_tokens_no_special` para calcular el solapamiento esperado.

        `strict`: bool
            - Si es **True**, lanza errores cuando detecta inconsistencias en los datos, dimensiones o solapamientos. Si es **False**, registra advertencias en **logs** y continúa cuando sea posible. Por defecto es **True**.

    Returns
    -------
        out, logs: tuple[list[torch.Tensor], list[str]]
            - Tupla formada por:
            
                - `out`: lista de tensores con los estados ocultos actualizados, donde cada
                  aparición del mismo token original recibe el promedio de todas sus apariciones.
                - `logs`: lista de mensajes informativos o advertencias sobre el proceso.
    """

    logs = []

    if len(windows) != len(last_hidden_state_list):
        raise ValueError("windows y last_hidden_state_list deben tener la misma longitud")

    if window_tokens_no_special <= 0:
        raise ValueError("window_tokens_no_special debe ser > 0")
    if stride <= 0:
        raise ValueError("stride debe ser > 0")

    overlap = window_tokens_no_special - stride
    if overlap < 0:
        msg = f"overlap negativo: window_tokens_no_special({window_tokens_no_special}) - stride({stride}) = {overlap}"
        if strict:
            raise ValueError(msg)
        logs.append("WARNING: " + msg)
        # sin solape útil
        return [h.clone() for h in last_hidden_state_list], logs

    out = [h.clone() for h in last_hidden_state_list]

    if overlap == 0:
        logs.append("INFO: overlap=0, no se promedia nada.")
        return out, logs

    occurrences: dict[int, list[tuple[int, int]]] = {}
    token_id_by_abs_idx: dict[int, int] = {}

    for i, h_i in enumerate(out):
        if h_i.dim() != 3 or h_i.size(0) != 1:
            msg = f"Ventanas deben ser (1, seq_len, hidden_dim). Got {tuple(h_i.shape)} at i={i}"
            if strict:
                raise ValueError(msg)
            logs.append("WARNING: " + msg)
            continue

        Li = h_i.shape[1]
        window = windows[i]
        missing_keys = [key for key in ("attention_mask", "input_ids", "absolute_token_indices") if key not in window]
        if missing_keys:
            msg = f"Ventana {i} no contiene las claves requeridas: {missing_keys}"
            if strict:
                raise ValueError(msg)
            logs.append("WARNING: " + msg)
            continue

        attention_mask = window["attention_mask"]
        input_ids = window["input_ids"]
        absolute_token_indices = window["absolute_token_indices"]

        if len(attention_mask) != Li or len(input_ids) != Li or len(absolute_token_indices) != Li:
            msg = (
                f"Ventana {i} no alinea metadatos con hidden states: "
                f"hidden_len={Li}, attention_mask={len(attention_mask)}, "
                f"input_ids={len(input_ids)}, absolute_token_indices={len(absolute_token_indices)}"
            )
            if strict:
                raise ValueError(msg)
            logs.append("WARNING: " + msg)
            continue

        real_positions = [
            pos
            for pos, abs_idx in enumerate(absolute_token_indices)
            if abs_idx is not None and int(attention_mask[pos]) == 1
        ]
        real_i = len(real_positions)
        if real_i <= 0:
            msg = f"Ventana {i} no tiene tokens reales: seq_len={Li}"
            if strict:
                raise ValueError(msg)
            logs.append("WARNING: " + msg)
            continue

        if strict and i < len(out) - 1 and real_i != window_tokens_no_special:
            raise ValueError(
                f"Ventana {i} tiene {real_i} tokens reales, esperado {window_tokens_no_special}. "
                "Si la ventana puede ser corta antes de la última, usa strict=False."
            )

        for pos in real_positions:
            abs_idx = int(absolute_token_indices[pos])
            token_id = int(input_ids[pos])

            previous_token_id = token_id_by_abs_idx.get(abs_idx)
            if previous_token_id is not None and previous_token_id != token_id:
                msg = (
                    f"Token original {abs_idx} no coincide entre ventanas: "
                    f"input_id previo={previous_token_id}, input_id actual={token_id} en ventana {i}, posición {pos}"
                )
                if strict:
                    raise ValueError(msg)
                logs.append("WARNING: " + msg)
                continue

            token_id_by_abs_idx[abs_idx] = token_id
            occurrences.setdefault(abs_idx, []).append((i, pos))

    averaged_token_count = 0
    averaged_occurrence_count = 0

    for abs_idx, positions in occurrences.items():
        if len(positions) < 2:
            continue

        avg = torch.stack([out[widx][0, pos, :] for widx, pos in positions], dim=0).mean(dim=0)
        for widx, pos in positions:
            out[widx][0, pos, :] = avg

        averaged_token_count += 1
        averaged_occurrence_count += len(positions)

    if averaged_token_count == 0:
        logs.append("INFO: no se encontraron tokens solapados para promediar.")
    else:
        logs.append(
            f"OK: promediados {averaged_token_count} tokens originales "
            f"en {averaged_occurrence_count} apariciones; overlap esperado por par completo={overlap}."
        )

    return out, logs

def series_to_striding_ner_windows(serie, tokenizer: Any, *, window_tokens: int = 512, stride: int = 128, padding: bool = False) -> list[dict[str, Any]]:
    """
    Divide el texto de una serie en ventanas de tokens con solapamiento, usando stride, y tokeniza cada ventana para preparar entradas compatibles con el modelo.

    Parameters
    ----------
        `serie`: pd.Series
            - Fila o serie que contiene el texto a procesar. Debe incluir la clave `Text` con el contenido completo del documento

        `tokenizer`: Any
            - Tokenizer utilizado para convertir el texto en tokens, obtener los `input_ids`, las máscaras de atención y los offsets de caracteres

        `window_tokens`: int
            - Número máximo de tokens por ventana, sin contar tokens especiales. Por defecto es **512**

        `stride`: int
            - Desplazamiento entre ventanas consecutivas, medido en tokens del texto completo.
            Por ejemplo, con `window_tokens=510` y `stride=128`, las ventanas empiezan
            en los tokens 0, 128, 256, etc. Por defecto es **128**

        `padding`: bool
            - Indica si se debe aplicar padding hasta la longitud máxima `window_tokens + 2` para incluir tokens especiales. Por defecto es **False**.

    Returns
    -------
        `windows`: list[dict[str, Any]]
            - Lista de diccionarios, donde cada diccionario representa una ventana del texto. Cada ventana incluye su índice, posiciones de caracteres, texto, **input_ids**, **attention_mask** y **offset_mapping**
    """
    text = serie["Text"]

    # Tokenize full doc no special tokens
    enc_full = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    full_ids = enc_full["input_ids"]
    full_offsets = enc_full["offset_mapping"]
    if not full_ids:
        return []

    if stride <= 0 or stride > window_tokens:
        raise ValueError("Invalid stride/window size")
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    if special_tokens != 2:
        raise ValueError(f"Expected tokenizer to add exactly 2 special tokens, got {special_tokens}")

    windows = []
    token_start = 0
    widx = 0

    while token_start < len(full_ids):
        token_end = min(token_start + window_tokens, len(full_ids))
        win_char_start = full_offsets[token_start][0]
        win_char_end = full_offsets[token_end - 1][1]
        win_text = text[win_char_start:win_char_end]

        # Original token slice
        win_ids_slice = full_ids[token_start:token_end]
        win_offsets_slice = full_offsets[token_start:token_end]

        # Reuse the exact full-document token IDs so overlapped token positions
        # refer to the same real text tokens across consecutive windows.
        input_ids = tokenizer.build_inputs_with_special_tokens(win_ids_slice)
        attention_mask = [1] * len(input_ids)
        offset_mapping = [(0, 0)] + win_offsets_slice + [(0, 0)]
        absolute_token_indices = [None] + list(range(token_start, token_end)) + [None]

        if padding:
            max_length = window_tokens + special_tokens
            pad_length = max_length - len(input_ids)
            if pad_length < 0:
                raise ValueError(f"Window {widx} has length {len(input_ids)} > max_length {max_length}")
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
            offset_mapping = offset_mapping + [(0, 0)] * pad_length
            absolute_token_indices = absolute_token_indices + [None] * pad_length

        windows.append({
            "window_index": widx,
            "token_start": token_start,
            "token_end": token_end,
            "char_start": win_char_start,
            "char_end": win_char_end,
            "text": win_text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
            "absolute_token_indices": absolute_token_indices,
        })

        if token_end == len(full_ids):
            break
        token_start += stride
        widx += 1

    return windows

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
    # 1. Must not be empty
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

def span_collate_fn(batch):
    """
    Agrupa una lista de ejemplos individuales en un batch de tensores para poder usarlo dentro de un DataLoader.

    Parameters
    ----------
        `batch`: list
            - Lista de ejemplos devueltos por el dataset. Cada ejemplo debe contener `span_repr`, `cls_repr`, `span_width`, `weight` y `label`

    Returns
    -------
        `span_reprs`, `cls_reprs`, `span_widths`, `weights`, `labels`: tuple
            - Tupla formada por:
                - `span_reprs`: tensor con las representaciones de los spans.
                - `cls_reprs`: tensor con las representaciones del token **CLS**
                - `span_widths`: tensor con la anchura o longitud de cada span
                - `labels`: tensor con las etiquetas correspondientes
    """
    span_reprs = []
    cls_reprs = []
    span_widths = []

    for span_repr, cls_repr, span_width in batch:
        span_reprs.append(span_repr)
        cls_reprs.append(cls_repr)
        span_widths.append(span_width)

    span_reprs = torch.stack(span_reprs)   # [N, span_dim]
    cls_reprs = torch.stack(cls_reprs)     # [N, cls_dim]
    span_widths = torch.stack(span_widths) # [N]

    return span_reprs, cls_reprs, span_widths

def prepara_data_from_ner_pred_to_icd_pred(df_data, df_final_ner):
    ner_labels = [label for label in dict.fromkeys(list(id2label_ner.values()) + list(label2id_ner.keys())) if label != "O"]
    out_labels = list(dict.fromkeys(["Description" if label == "CLINENTITY" else label for label in ner_labels]))

    tmp = df_final_ner.copy()
    tmp["ner_label"] = tmp["pred_label"].where(tmp["pred_label"].isin(ner_labels), tmp["pred_id"].map({v: k for k, v in label2id_ner.items()}))
    tmp = tmp[tmp["ner_label"].isin(ner_labels)].copy()
    tmp["Decoded span"] = tmp["Decoded span"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    tmp["Token idx"] = tmp["Token idx"].apply(lambda x: tuple(ast.literal_eval(x) if isinstance(x, str) else x))
    tmp["entity"] = tmp["Decoded span"].apply(lambda toks: " ".join(merge_sentencepiece_words(toks)))
    tmp["out_label"] = tmp["ner_label"].replace({"CLINENTITY": "Description"})
    tmp = tmp.drop_duplicates(subset=["File", "out_label", "Token idx", "entity"])

    entities_by_file = tmp.groupby(["File", "out_label"]).agg(entity=("entity", list), token_idx=("Token idx", list)).reset_index()
    entities_by_file = entities_by_file.pivot(index="File", columns="out_label", values=["entity", "token_idx"]).reset_index()
    entities_by_file.columns = ["File"] + [f"All {label}" if kind == "entity" else f"All {label} Token idx" for kind, label in entities_by_file.columns[1:]]

    df_final_ner = df_data[["archivo_origen", "Text"]].merge(entities_by_file, left_on="archivo_origen", right_on="File", how="left")
    for col in [col for label in out_labels for col in (f"All {label}", f"All {label} Token idx")]: df_final_ner[col] = df_final_ner[col].apply(lambda x: x if isinstance(x, list) else [])
    df_final_ner = df_final_ner.rename(columns={"archivo_origen": "Original File"})[["Text"] + [col for label in out_labels for col in (f"All {label}", f"All {label} Token idx")] + ["Original File"]]
    return df_final_ner

def corrected_entities_to_df(answer_global: dict, original_texts: dict[str, str]) -> pd.DataFrame: 
    rows = []

    labels = ["ACTOR", "CLINENTITY", "TIMEX3"]

    for item in answer_global["data"]:
        file_id = item["file_id"]
        text = original_texts[file_id]

        row = {"Text": text, "Original File": file_id}

        grouped = {label: {"texts": [], "token_idxs": []} for label in labels}

        for ent in item["entities"]:
            if ent["decision"] == "REJECT":
                continue

            corrected_text = ent["corrected_text"]
            corrected_start = ent["corrected_start"]
            corrected_end = ent["corrected_end"]
            corrected_label = ent["corrected_label"]

            if corrected_label is None:
                continue

            if corrected_label not in grouped:
                continue

            token_idxs = char_span_to_token_span(text=text, start=corrected_start, end=corrected_end, tokenizer_ner=tokenizer_ner)

            grouped[corrected_label]["texts"].append(corrected_text)
            grouped[corrected_label]["token_idxs"].append(token_idxs)

        row["All ACTOR"] = grouped["ACTOR"]["texts"]
        row["All ACTOR Token idx"] = grouped["ACTOR"]["token_idxs"]

        row["All Description"] = grouped["CLINENTITY"]["texts"]
        row["All Description Token idx"] = grouped["CLINENTITY"]["token_idxs"]

        row["All TIMEX3"] = grouped["TIMEX3"]["texts"]
        row["All TIMEX3 Token idx"] = grouped["TIMEX3"]["token_idxs"]

        rows.append(row)

    columns = ["Text", "All ACTOR", "All ACTOR Token idx", "All Description", "All Description Token idx", "All TIMEX3", "All TIMEX3 Token idx", "Original File"]

    return pd.DataFrame(rows, columns=columns)

def span_from_tokens(text, idxs, base=0):
    offsets = tokenizer_ner(text, return_offsets_mapping=True, add_special_tokens=False, truncation=False)["offset_mapping"]

    idxs = lit(idxs)
    idxs = list(idxs) if isinstance(idxs, (list, tuple)) else []

    spans = [offsets[i - base] for i in idxs if 0 <= i - base < len(offsets)]

    if not spans:
        return None, None

    return min(s[0] for s in spans), max(s[1] for s in spans)


def build_ner_json(df, token_index_base=0):
    data = []

    LABEL_MAP = {"All ACTOR": "ACTOR", "All Description": "CLINENTITY", "All TIMEX3": "TIMEX3"}

    for _, row in df.iterrows():
        text = row["Text"]
        entities_out = []
        ent_id = 0

        for col, label in LABEL_MAP.items():
            ents = lit(row.get(col, []))
            idxs = lit(row.get(f"{col} Token idx", []))

            if not isinstance(ents, list) or not isinstance(idxs, list):
                continue

            for ent_text, token_idxs in zip(ents, idxs):
                start, end = span_from_tokens(text, token_idxs, base=token_index_base)

                if start is None:
                    continue

                entities_out.append({
                    "target_label": label,
                    "entity": {
                        "ent_id": ent_id,
                        "text": ent_text,
                        "start": start,
                        "end": end,
                    }
                })

                ent_id += 1

        data.append({
            "file_id": row["Original File"],
            "text": text,
            "entities": entities_out
        })

    return {"data": data}

def entity_matches_correction(ent: dict, cor: dict, original_text: str) -> bool:
    # return (ent["entity"]["start"] == cor["original_start"] and ent["entity"]["end"] == cor["original_end"] and ent["entity"]["text"] == cor["original_text"] and original_text[cor["original_start"]:cor["original_end"]] == cor["original_text"])
    return (ent["entity"]["start"] == cor["original_start"] and ent["entity"]["end"] == cor["original_end"] and ent["entity"]["text"] == cor["original_text"])


def get_missing_entities(original_entities: list[dict], corrected_entities: list[dict], original_text: str) -> list[dict]:
    missing = []

    for ent in original_entities:
        exists = any(entity_matches_correction(ent, cor, original_text) for cor in corrected_entities)
        if not exists:
            missing.append(ent)

    return missing

def initialize_span_ner_model(ctx: PipelineContext, checkpoint_name: str) -> SpanClassifier:
    checkpoint = read_torch_checkpoint(ctx, checkpoint_name)

    model = SpanClassifier(span_dim=1024, cls_dim=1024, num_classes=len(id2label_ner), max_span_width=15).to(ctx.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model
