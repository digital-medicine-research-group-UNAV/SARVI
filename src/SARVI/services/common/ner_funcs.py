from __future__ import annotations

import ast
import torch
import pandas as pd
from typing import Any, TYPE_CHECKING
from collections import defaultdict

from transformers import AutoTokenizer, AutoModel

from .utils.ner_utils import (
    brat_entities_overlap, char_span_to_token_span, format_brat_t_line, lit, token_overlaps_span,
    entity_exists, entity_text_for_span, text_num_tokens_from_windows, shift_absolute_token_indices, parse_label_piece
)
from .utils.ann_utils import(
    normalize_ann_lines, parse_brat_t_entities, parse_brat_text_boundaries, ann_output_to_df_ann, df_ann_to_ann_list,
    normalize_original_texts
)
from .utils.span_utils import (
    merge_sentencepiece_words
)
from .utils.lemma_utils import (
    load_lemmatizer, run_lemmatizer
)
from .utils.nn_utils import (
    checkpoint_state_dict, load_checkpoint_with_resized_head, build_inputs_with_special_tokens
)

from ...data_io.reader import (
    read_torch_checkpoint, read_dsv_single
)
from ...models.neural_networks import (
    Span_NERClassifier, BIO_NERClassifier
)

if TYPE_CHECKING:
    from ...models.schemas import PipelineContext

###
device = "cuda" if torch.cuda.is_available() else "cpu"

NER_WINDOW_LABELS = {"DISO", "Date", "LIVB", "Neg_cue", "Spec_cue"}

tokenizer_ner = AutoTokenizer.from_pretrained("IIC/RigoBERTa-Clinical", trim_offsets=False, use_fast=True)
model_ner = AutoModel.from_pretrained("IIC/RigoBERTa-Clinical").to(device)

lemmatizer = load_lemmatizer(device)
###

def classification_report_ner(gold_ann: Any, pred_ann: Any, labels: list[str] | None = None, target_labels: list[str] | None = None, digits: int = 4, print_report: bool = True) -> dict[str, Any]:
    """
    Entity-level classification report for BRAT .ann T annotations.

    Error types:
        Type 1: complete false positive
        Type 2: complete false negative
        Type 3: wrong label / correct span
        Type 4: wrong label / overlapping span
        Type 5: correct label / overlapping span
    """
    def parse_t_entities(value: Any) -> list[dict[str, Any]]:
        return parse_brat_t_entities(value)

    def overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
        return brat_entities_overlap(a, b)

    def safe_div(num: int, den: int) -> float:
        return num / den if den else 0.0

    selected_labels = target_labels if target_labels is not None else labels

    gold_entities = parse_t_entities(gold_ann)
    pred_entities = parse_t_entities(pred_ann)
    
    if selected_labels is not None:
        label_list = list(selected_labels)
        label_set = set(label_list)
        gold_entities = [entity for entity in gold_entities if entity["label"] in label_set]
        pred_entities = [entity for entity in pred_entities if entity["label"] in label_set]
    else:
        label_list = sorted({entity["label"] for entity in gold_entities + pred_entities})

    label_set = set(label_list)

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)
    errors = {label: defaultdict(int) for label in label_list}

    for entity in gold_entities:
        if entity["label"] in label_set:
            support[entity["label"]] += 1

    gold_matched = set()
    pred_matched = set()

    # Exact true positives.
    for gi, gold in enumerate(gold_entities):
        for pi, pred in enumerate(pred_entities):
            if pi in pred_matched:
                continue
            if gold["label"] == pred["label"] and gold["start"] == pred["start"] and gold["end"] == pred["end"]:
                gold_matched.add(gi)
                pred_matched.add(pi)
                if gold["label"] in label_set:
                    tp[gold["label"]] += 1
                break

    # Type 3: wrong label / correct span.
    for gi, gold in enumerate(gold_entities):
        if gi in gold_matched:
            continue
        for pi, pred in enumerate(pred_entities):
            if pi in pred_matched:
                continue
            if gold["start"] == pred["start"] and gold["end"] == pred["end"] and gold["label"] != pred["label"]:
                gold_matched.add(gi)
                pred_matched.add(pi)
                if gold["label"] in label_set:
                    errors[gold["label"]]["type_3"] += 1
                    fn[gold["label"]] += 1
                if pred["label"] in label_set:
                    fp[pred["label"]] += 1
                break

    # Type 5: correct label / overlapping span.
    for gi, gold in enumerate(gold_entities):
        if gi in gold_matched:
            continue
        candidates = [(pi, pred) for pi, pred in enumerate(pred_entities) if pi not in pred_matched and gold["label"] == pred["label"] and overlaps(gold, pred)]
        if not candidates:
            continue

        pi, pred = max(candidates, key=lambda item: min(gold["end"], item[1]["end"]) - max(gold["start"], item[1]["start"]))
        gold_matched.add(gi)
        pred_matched.add(pi)
        if gold["label"] in label_set:
            errors[gold["label"]]["type_5"] += 1
            fn[gold["label"]] += 1
            fp[gold["label"]] += 1

    # Type 4: wrong label / overlapping span.
    for gi, gold in enumerate(gold_entities):
        if gi in gold_matched:
            continue
        candidates = [(pi, pred) for pi, pred in enumerate(pred_entities) if pi not in pred_matched and gold["label"] != pred["label"] and overlaps(gold, pred)]
        if not candidates:
            continue

        pi, pred = max(candidates, key=lambda item: min(gold["end"], item[1]["end"]) - max(gold["start"], item[1]["start"]))
        gold_matched.add(gi)
        pred_matched.add(pi)
        if gold["label"] in label_set:
            errors[gold["label"]]["type_4"] += 1
            fn[gold["label"]] += 1
        if pred["label"] in label_set:
            fp[pred["label"]] += 1

    # Type 2: complete false negative.
    for gi, gold in enumerate(gold_entities):
        if gi in gold_matched:
            continue
        if gold["label"] in label_set:
            errors[gold["label"]]["type_2"] += 1
            fn[gold["label"]] += 1

    # Type 1: complete false positive.
    for pi, pred in enumerate(pred_entities):
        if pi in pred_matched:
            continue
        if pred["label"] in label_set:
            errors[pred["label"]]["type_1"] += 1
            fp[pred["label"]] += 1

    rows = {}
    for label in label_list:
        precision = safe_div(tp[label], tp[label] + fp[label])
        recall = safe_div(tp[label], tp[label] + fn[label])
        f1 = safe_div(2 * precision * recall, precision + recall)
        rows[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support[label],
            "type_1": errors[label]["type_1"],
            "type_2": errors[label]["type_2"],
            "type_3": errors[label]["type_3"],
            "type_4": errors[label]["type_4"],
            "type_5": errors[label]["type_5"],
        }

    total_tp = sum(tp[label] for label in label_list)
    total_fp = sum(fp[label] for label in label_list)
    total_fn = sum(fn[label] for label in label_list)
    total_support = sum(support[label] for label in label_list)
    micro_p = safe_div(total_tp, total_tp + total_fp)
    micro_r = safe_div(total_tp, total_tp + total_fn)
    micro_f1 = safe_div(2 * micro_p * micro_r, micro_p + micro_r)

    macro = {
        "precision": sum(rows[label]["precision"] for label in label_list) / len(label_list) if label_list else 0.0,
        "recall": sum(rows[label]["recall"] for label in label_list) / len(label_list) if label_list else 0.0,
        "f1-score": sum(rows[label]["f1-score"] for label in label_list) / len(label_list) if label_list else 0.0,
        "support": total_support,
    }
    weighted = {
        "precision": safe_div(sum(rows[label]["precision"] * support[label] for label in label_list), total_support),
        "recall": safe_div(sum(rows[label]["recall"] * support[label] for label in label_list), total_support),
        "f1-score": safe_div(sum(rows[label]["f1-score"] * support[label] for label in label_list), total_support),
        "support": total_support,
    }

    for avg_name, avg_row in (("micro avg", {"precision": micro_p, "recall": micro_r, "f1-score": micro_f1, "support": total_support}), ("macro avg", macro), ("weighted avg", weighted)):
        rows[avg_name] = {
            **avg_row,
            "type_1": sum(errors[label]["type_1"] for label in label_list),
            "type_2": sum(errors[label]["type_2"] for label in label_list),
            "type_3": sum(errors[label]["type_3"] for label in label_list),
            "type_4": sum(errors[label]["type_4"] for label in label_list),
            "type_5": sum(errors[label]["type_5"] for label in label_list),
        }

    if print_report:
        label_width = max([len(str(label)) for label in rows] + [9])
        header = f"{'':>{label_width}}  {'precision':>9} {'recall':>9} {'f1-score':>9} {'support':>9} {'type_1':>8} {'type_2':>8} {'type_3':>8} {'type_4':>8} {'type_5':>8}"
        print(header)
        print()

        for label in label_list + ["micro avg", "macro avg", "weighted avg"]:
            row = rows[label]
            print(
                f"{label:>{label_width}}  "
                f"{row['precision']:>9.{digits}f} {row['recall']:>9.{digits}f} {row['f1-score']:>9.{digits}f} "
                f"{int(row['support']):>9} {int(row['type_1']):>8} {int(row['type_2']):>8} {int(row['type_3']):>8} {int(row['type_4']):>8} {int(row['type_5']):>8}"
            )
            if label_list and label == label_list[-1]:
                print()

    return {"report": rows, "gold_entities": gold_entities, "pred_entities": pred_entities}

def create_ultimate_df_ann(principal_df_ann: pd.DataFrame | dict, secondary_df_ann: pd.DataFrame | dict, overlapped_entities: bool = False, extend_entities: bool = False, add_extra_entities: bool = False, file_col: str = "archivo_origen", t_col: str = "T", return_type: str = "ann_df", original_texts: dict | pd.DataFrame | None = None) -> pd.DataFrame | dict[str, list[str]]:
    """
    Builds an ultimate df_ann from a principal df_ann and a secondary df_ann.

    The output starts as a literal copy of principal_df_ann.
    """
    if return_type not in {"ann_df", "ann_list"}:
        raise ValueError("return_type must be either 'ann_df' or 'ann_list'.")

    principal_df_ann = ann_output_to_df_ann(principal_df_ann, file_col=file_col, t_col=t_col)
    secondary_df_ann = ann_output_to_df_ann(secondary_df_ann, file_col=file_col, t_col=t_col)
    originals = normalize_original_texts(original_texts)

    def char_spans_without_linebreaks(text: str, start: int, end: int) -> list[tuple[int, int]]:
        spans = []
        span_start = None

        for idx in range(start, end):
            if text[idx] in "\r\n":
                if span_start is not None and span_start < idx:
                    spans.append((span_start, idx))
                span_start = None
            elif span_start is None:
                span_start = idx

        if span_start is not None and span_start < end:
            spans.append((span_start, end))

        return spans or [(start, end)]

    def text_from_char_spans(text: str, char_spans: list[tuple[int, int]]) -> str:
        return " ".join(text[start:end] for start, end in char_spans).replace("\t", " ")

    def refresh_entity_boundaries(entity: dict[str, Any], file_name: Any, force_continuous_without_text: bool = False) -> None:
        text = originals.get(file_name)
        if text is None:
            if force_continuous_without_text or not entity.get("char_spans"):
                entity["char_spans"] = [(int(entity["start"]), int(entity["end"]))]
            return

        entity["char_spans"] = char_spans_without_linebreaks(str(text), int(entity["start"]), int(entity["end"]))

    def final_entity_text(entity: dict[str, Any], candidates: list[dict[str, Any]], file_name: Any) -> str:
        text = originals.get(file_name)
        if text is not None:
            return text_from_char_spans(str(text), entity.get("char_spans") or [(int(entity["start"]), int(entity["end"]))])
        return entity_text_for_span(entity, candidates)

    def entity_dedupe_key(entity: dict[str, Any]) -> tuple:
        return (str(entity["label"]), int(entity["start"]), int(entity["end"]), str(entity.get("text", "")))

    def dedupe_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen = set()
        deduped = []

        for entity in entities:
            key = entity_dedupe_key(entity)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entity)

        return deduped

    if file_col not in principal_df_ann.columns:
        raise ValueError(f"principal_df_ann must have a '{file_col}' column.")
    if file_col not in secondary_df_ann.columns:
        raise ValueError(f"secondary_df_ann must have a '{file_col}' column.")
    if t_col not in principal_df_ann.columns:
        raise ValueError(f"principal_df_ann must have a '{t_col}' column.")
    if t_col not in secondary_df_ann.columns:
        raise ValueError(f"secondary_df_ann must have a '{t_col}' column.")

    ultimate_df_ann = principal_df_ann.copy(deep=True)

    principal_entities_by_file = defaultdict(list)
    for entity in parse_brat_t_entities(principal_df_ann, file_col=file_col, t_col=t_col):
        principal_entities_by_file[entity["file"]].append(entity)

    secondary_entities_by_file = defaultdict(list)
    for entity in parse_brat_t_entities(secondary_df_ann, file_col=file_col, t_col=t_col):
        secondary_entities_by_file[entity["file"]].append(entity)

    for row_idx, row in ultimate_df_ann.iterrows():
        file_name = row[file_col]
        principal_entities = [dict(entity) for entity in principal_entities_by_file.get(file_name, [])]
        secondary_entities = secondary_entities_by_file.get(file_name, [])
        ultimate_entities = [dict(entity) for entity in principal_entities]

        if extend_entities:
            for entity in ultimate_entities:
                same_label_overlaps = [secondary for secondary in secondary_entities if secondary["label"] == entity["label"] and brat_entities_overlap(entity, secondary)]
                if not same_label_overlaps:
                    continue

                candidates = [dict(entity)] + [dict(secondary) for secondary in same_label_overlaps]
                entity["start"] = min(int(candidate["start"]) for candidate in candidates)
                entity["end"] = max(int(candidate["end"]) for candidate in candidates)
                refresh_entity_boundaries(entity, file_name, force_continuous_without_text=True)
                entity["text"] = final_entity_text(entity, candidates, file_name)

        if overlapped_entities:
            for secondary in secondary_entities:
                if any(brat_entities_overlap(secondary, principal) for principal in principal_entities):
                    if not entity_exists(secondary, ultimate_entities):
                        ultimate_entities.append(dict(secondary))

        if add_extra_entities:
            for secondary in secondary_entities:
                if not any(brat_entities_overlap(secondary, principal) for principal in principal_entities):
                    if not entity_exists(secondary, ultimate_entities):
                        ultimate_entities.append(dict(secondary))

        for entity in ultimate_entities:
            refresh_entity_boundaries(entity, file_name)
            if originals.get(file_name) is not None:
                entity["text"] = final_entity_text(entity, [entity], file_name)

        ultimate_entities = dedupe_entities(ultimate_entities)
        ultimate_entities = sorted(ultimate_entities, key=lambda item: (int(item["start"]), int(item["end"]), item["label"]))
        ultimate_df_ann.at[row_idx, t_col] = [format_brat_t_line(i, entity) for i, entity in enumerate(ultimate_entities, start=1)]

    if return_type == "ann_list":
        return df_ann_to_ann_list(ultimate_df_ann)

    return ultimate_df_ann


def parse_target_brat_entities(ann_info: Any, target_labels: set[str] = NER_WINDOW_LABELS) -> list[dict[str, Any]]:
    if ann_info is None:
        return []

    if isinstance(ann_info, pd.Series) or isinstance(ann_info, dict):
        t_lines = normalize_ann_lines(ann_info.get("T"))
    else:
        t_lines = normalize_ann_lines(getattr(ann_info, "T", None))

    entities = []

    for line in t_lines:
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        ann_id = parts[0]
        label_and_boundaries = parts[1].split(maxsplit=1)
        if len(label_and_boundaries) != 2:
            continue

        label, boundary_text = label_and_boundaries
        if label not in target_labels:
            continue

        char_spans = parse_brat_text_boundaries(boundary_text)
        if not char_spans:
            continue

        entities.append({
            "id": ann_id,
            "label": label,
            "char_spans": char_spans,
            "is_discontinuous": len(char_spans) > 1,
            "start": min(start for start, _ in char_spans),
            "end": max(end for _, end in char_spans),
            "length": sum(end - start for start, end in char_spans),
        })

    for entity_order, entity in enumerate(sorted(entities, key=lambda item: (item["start"], item["end"], item["id"])), start=1):
        entity["entity_order"] = entity_order

    return entities

def remap_entities_to_lemmatized_offsets(entities: list[dict[str, Any]], token_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remapped_entities = []

    for entity in entities:
        remapped_char_spans = []

        for span_start, span_end in entity["char_spans"]:
            overlapping_tokens = [token for token in token_data if token.get("original_start") is not None and token.get("original_end") is not None and token_overlaps_span(token["original_start"], token["original_end"], span_start, span_end)]

            if not overlapping_tokens:
                continue

            remapped_char_spans.append((min(token["lemma_start"] for token in overlapping_tokens), max(token["lemma_end"] for token in overlapping_tokens)))

        if not remapped_char_spans:
            continue

        remapped_entity = {
            **entity,
            "char_spans": remapped_char_spans,
            "is_discontinuous": len(remapped_char_spans) > 1,
            "start": min(start for start, _ in remapped_char_spans),
            "end": max(end for _, end in remapped_char_spans),
            "length": sum(end - start for start, end in remapped_char_spans),
        }
        remapped_entities.append(remapped_entity)

    for entity_order, entity in enumerate(sorted(remapped_entities, key=lambda item: (item["start"], item["end"], item["id"])), start=1):
        entity["entity_order"] = entity_order

    return remapped_entities

def entity_token_labels_for_full_text(text: str, full_offsets: list[tuple[int, int]], ann_info: Any, entities: list[dict[str, Any]] | None = None) -> dict[int, list[tuple[dict[str, Any], str]]]:
    if entities is None:
        entities = parse_target_brat_entities(ann_info)

    token_labels: dict[int, list[tuple[dict[str, Any], str]]] = {}

    for entity in entities:
        entity_token_indices = []
        component_token_indices = []

        for component_idx, (span_start, span_end) in enumerate(entity["char_spans"]):
            tokens_in_component = [token_idx for token_idx, (tok_start, tok_end) in enumerate(full_offsets) if tok_start != tok_end and token_overlaps_span(tok_start, tok_end, span_start, span_end)]

            if not tokens_in_component:
                continue

            component_token_indices.append(tokens_in_component)
            entity_token_indices.extend(tokens_in_component)

            previous_prefix = None
            previous_end = None

            for local_idx, token_idx in enumerate(tokens_in_component):
                tok_start, tok_end = full_offsets[token_idx]

                if entity["is_discontinuous"] and component_idx > 0:
                    prefix = "I"
                elif local_idx == 0:
                    prefix = "B"
                else:
                    same_word = previous_end is not None and tok_start == previous_end
                    prefix = previous_prefix if same_word else "I"

                entity_order = entity["entity_order"]
                if entity["is_discontinuous"]:
                    label = f"{{{entity_order}.{component_idx + 1}_{prefix}-{entity['label']}}}"
                else:
                    label = f"{entity_order}_{prefix}-{entity['label']}"

                token_labels.setdefault(token_idx, []).append((entity, label))
                previous_prefix = prefix
                previous_end = tok_end

        entity["token_indices"] = sorted(set(entity_token_indices))

    for token_idx, labels in token_labels.items():
        labels.sort(key=lambda item: (-item[0]["length"], item[0]["start"], item[0]["end"], item[0]["id"]))
        token_labels[token_idx] = labels

    return token_labels

def labels_for_window(input_len: int, absolute_token_indices: list[int | None], token_labels: dict[int, list[tuple[dict[str, Any], str]]]) -> list[str]:
    window_real_token_indices = {int(idx) for idx in absolute_token_indices if idx is not None}
    labels = ["O"] * input_len

    for pos, abs_idx in enumerate(absolute_token_indices):
        if abs_idx is None:
            continue

        token_label_items = []

        for entity, label in token_labels.get(int(abs_idx), []):
            entity_tokens = entity.get("token_indices", [])
            if entity_tokens and set(entity_tokens).issubset(window_real_token_indices):
                token_label_items.append(label)

        if token_label_items:
            labels[pos] = ";".join(token_label_items)

    return labels

def series_to_striding_ner_windows(serie, ann_info: Any | None = None, *, tokenizer: Any, window_tokens: int = 512, stride: int = 128, padding: bool = False, lemma: bool = False) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[list[str]]]:
    """
    Divide el texto de una serie en ventanas de tokens con solapamiento, usando stride, y tokeniza cada ventana para preparar entradas compatibles con el modelo.

    Parameters
    ----------
        `serie`: pd.Series
            - Fila o serie que contiene el texto a procesar. Debe incluir la clave `Text` con el contenido completo del documento

        `ann_info`: Any | None
            - Fila o diccionario con las anotaciones BRAT del mismo documento. Si es `None`,
            la función conserva el comportamiento de inferencia y devuelve solo `windows`.

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
        
        `lemma`: bool
            - Si se necesita hacer una lemmatización del texto o no. Por defecto es **False**.

    Returns
    -------
        `windows`: list[dict[str, Any]]
            - Lista de diccionarios, donde cada diccionario representa una ventana del texto. Cada ventana incluye su índice, posiciones de caracteres, texto, **input_ids**, **attention_mask** y **offset_mapping**

        `window_labels`: list[list[str]]
            - Solo cuando `ann_info` no es `None`. Lista alineada con `windows`; cada sublista
            tiene la misma longitud que los `input_ids` de su ventana y contiene etiquetas BIO.
    """
    text = serie["Text"]
    lemmatized_token_data = None

    if lemma:
        text, lemmatized_token_data = run_lemmatizer(lemmatizer, text)

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
    window_labels = [] if ann_info is not None else None
    if ann_info is not None:
        entities = parse_target_brat_entities(ann_info)
        if lemma:
            entities = remap_entities_to_lemmatized_offsets(entities, lemmatized_token_data or [])
        full_token_labels = entity_token_labels_for_full_text(text, full_offsets, ann_info, entities=entities)
    else:
        full_token_labels = {}

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

        # Reuse the exact full-document token IDs so overlapped token positions refer to the same real text tokens across consecutive windows.
        input_ids = build_inputs_with_special_tokens(win_ids_slice, cls_token_id=tokenizer.cls_token_id, sep_token_id=tokenizer.sep_token_id)
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

        window = {
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
        }
        windows.append(window)

        if window_labels is not None:
            window_labels.append(labels_for_window(len(input_ids), absolute_token_indices, full_token_labels))

        if token_end == len(full_ids):
            break
        token_start += stride
        widx += 1

    if window_labels is not None:
        return windows, window_labels

    return windows

def collated_window_label_ids(labels: list, window_index: int, shifted_indices: list[int], attention_mask: list[int], ignore_o_labels: bool = False, o_label_id: int | None = None) -> list[int | str]:
    label_ids = []

    for label_id, token_idx, token_mask in zip(labels[window_index], shifted_indices, attention_mask):
        if token_idx < 0 or int(token_mask) == 0:
            label_ids.append(-100)
        elif ignore_o_labels and (label_id == "O" or (o_label_id is not None and str(label_id) == str(o_label_id))):
            label_ids.append(-100)
        else:
            try:
                label_ids.append(int(label_id))
            except Exception:
                label_ids.append(str(label_id))

    return label_ids

def span_labels_present_in_window_labels(window_labels: list | None) -> list[str] | None:
    if window_labels is None:
        return None

    present_labels = set()

    for document_labels in window_labels:
        for window in document_labels:
            for raw_label in window:
                for piece in str(raw_label).split(";"):
                    parsed = parse_label_piece(piece, 0)
                    if parsed is not None:
                        present_labels.add(parsed["entity"])
                    elif piece in NER_WINDOW_LABELS:
                        present_labels.add(piece)

    return ["O"] + sorted(present_labels)

def ner_collate_fn(batch, ignore_o_labels: bool = False, o_label_id: int | None = None):
    """
    Agrupa una lista de textos individuales en un batch de tensores para poder usarlo dentro de un DataLoader.

    Cada ejemplo del batch representa un texto completo y contiene todas sus ventanas o subsecuencias asociadas. La función aplana las ventanas de todos los textos del batch, pero mantiene la información necesaria para saber a qué texto pertenece cada ventana y para evitar que los tokens de textos diferentes se mezclen durante el promediado de embeddings solapados.

    Parameters
    ----------
        `batch`: list
            - Lista de ejemplos devueltos por el dataset. Cada ejemplo debe contener `text_index`, `file_name`, `windows` y, opcionalmente, `window_labels`.
            - `windows` debe ser una lista de diccionarios generados por `series_to_striding_ner_windows`. Cada ventana debe contener `input_ids`, `attention_mask` y `absolute_token_indices`.
        `ignore_o_labels`: bool
            - Si es `True`, las etiquetas "O" se sustituyen por `-100` para ignorarlas en la loss y métricas enmascaradas.
        `o_label_id`: int | None
            - ID de la etiqueta "O" cuando las etiquetas ya están codificadas como enteros.

    Returns
    -------
        `batch_out`: dict
            - Diccionario con los tensores y metadatos necesarios para el modelo:
            
                - `input_ids`: tensor con los IDs de entrada de todas las ventanas aplanadas.
                - `attention_mask`: tensor con las máscaras de atención de todas las ventanas aplanadas.
                - `global_token_indices`: tensor con los índices globales de token, usados para promediar correctamente tokens solapados sin mezclar textos diferentes.
                - `window_to_text`: tensor que indica a qué texto del batch pertenece cada ventana.
                - `window_local_index`: tensor que indica el índice local de cada ventana dentro de su texto original.
                - `text_indices`: tensor con los índices originales de los textos dentro del dataset.
                - `file_names`: lista con los nombres de archivo asociados a cada texto.
                - `text_token_offsets`: tensor con el offset global de tokens de cada texto dentro del batch.
                - `text_token_lengths`: tensor con el número de tokens originales de cada texto.
                - `window_labels`: etiquetas asociadas a cada ventana, si existen. En caso contrario, `None`.
    """
    input_ids = []
    attention_mask = []
    global_token_indices = []

    window_to_text = []
    window_local_index = []
    window_offset_mapping = []

    text_indices = []
    file_names = []

    text_token_offsets = []
    text_token_lengths = []

    flat_window_labels = []

    token_offset = 0

    for batch_text_idx, item in enumerate(batch):
        windows = item["windows"]
        labels = item.get("window_labels", None)

        text_indices.append(item["text_index"])
        file_names.append(item["file_name"])

        text_num_tokens = text_num_tokens_from_windows(windows)

        text_token_offsets.append(token_offset)
        text_token_lengths.append(text_num_tokens)

        for widx, w in enumerate(windows):
            input_ids.append(w["input_ids"])
            attention_mask.append(w["attention_mask"])

            shifted_indices = shift_absolute_token_indices(w["absolute_token_indices"], token_offset)

            global_token_indices.append(shifted_indices)
            window_offset_mapping.append(w.get("offset_mapping", [(0, 0)] * len(w["input_ids"])))

            window_to_text.append(batch_text_idx)
            window_local_index.append(widx)

            if labels is not None:
                flat_window_labels.append(collated_window_label_ids(labels, widx, shifted_indices, w["attention_mask"], ignore_o_labels=ignore_o_labels, o_label_id=o_label_id))

        token_offset += text_num_tokens

    batch_out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "global_token_indices": torch.tensor(global_token_indices, dtype=torch.long),

        "window_to_text": torch.tensor(window_to_text, dtype=torch.long),
        "window_local_index": torch.tensor(window_local_index, dtype=torch.long),
        "window_offset_mapping": torch.tensor(window_offset_mapping, dtype=torch.long),

        "text_indices": torch.tensor(text_indices, dtype=torch.long),
        "file_names": file_names,

        "text_token_offsets": torch.tensor(text_token_offsets, dtype=torch.long),
        "text_token_lengths": torch.tensor(text_token_lengths, dtype=torch.long),
    }

    if len(flat_window_labels) > 0:
        try:
            batch_out["window_labels"] = torch.tensor(flat_window_labels, dtype=torch.long)
        except:
            batch_out["window_labels"] = flat_window_labels
    else:
        batch_out["window_labels"] = None

    return batch_out

# def prepara_data_from_ner_pred_to_icd_pred(df_data, df_final_ner):
#     ner_labels = [label for label in dict.fromkeys(list(id2label_ner.values()) + list(label2id_ner.keys())) if label != "O"]
#     out_labels = list(dict.fromkeys(["Description" if label == "CLINENTITY" else label for label in ner_labels]))

#     tmp = df_final_ner.copy()
#     tmp["ner_label"] = tmp["pred_label"].where(tmp["pred_label"].isin(ner_labels), tmp["pred_id"].map({v: k for k, v in label2id_ner.items()}))
#     tmp = tmp[tmp["ner_label"].isin(ner_labels)].copy()
#     tmp["Decoded span"] = tmp["Decoded span"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
#     tmp["Token idx"] = tmp["Token idx"].apply(lambda x: tuple(ast.literal_eval(x) if isinstance(x, str) else x))
#     tmp["entity"] = tmp["Decoded span"].apply(lambda toks: " ".join(merge_sentencepiece_words(toks)))
#     tmp["out_label"] = tmp["ner_label"].replace({"CLINENTITY": "Description"})
#     tmp = tmp.drop_duplicates(subset=["File", "out_label", "Token idx", "entity"])

#     entities_by_file = tmp.groupby(["File", "out_label"]).agg(entity=("entity", list), token_idx=("Token idx", list)).reset_index()
#     entities_by_file = entities_by_file.pivot(index="File", columns="out_label", values=["entity", "token_idx"]).reset_index()
#     entities_by_file.columns = ["File"] + [f"All {label}" if kind == "entity" else f"All {label} Token idx" for kind, label in entities_by_file.columns[1:]]

#     df_final_ner = df_data[["archivo_origen", "Text"]].merge(entities_by_file, left_on="archivo_origen", right_on="File", how="left")
#     for col in [col for label in out_labels for col in (f"All {label}", f"All {label} Token idx")]: df_final_ner[col] = df_final_ner[col].apply(lambda x: x if isinstance(x, list) else [])
#     df_final_ner = df_final_ner.rename(columns={"archivo_origen": "Original File"})[["Text"] + [col for label in out_labels for col in (f"All {label}", f"All {label} Token idx")] + ["Original File"]]
#     return df_final_ner

# def corrected_entities_to_df(answer_global: dict, original_texts: dict[str, str]) -> pd.DataFrame: 
#     rows = []

#     labels = ["ACTOR", "CLINENTITY", "TIMEX3"]

#     for item in answer_global["data"]:
#         file_id = item["file_id"]
#         text = original_texts[file_id]

#         row = {"Text": text, "Original File": file_id}

#         grouped = {label: {"texts": [], "token_idxs": []} for label in labels}

#         for ent in item["entities"]:
#             if ent["decision"] == "REJECT":
#                 continue

#             corrected_text = ent["corrected_text"]
#             corrected_start = ent["corrected_start"]
#             corrected_end = ent["corrected_end"]
#             corrected_label = ent["corrected_label"]

#             if corrected_label is None:
#                 continue

#             if corrected_label not in grouped:
#                 continue

#             token_idxs = char_span_to_token_span(text=text, start=corrected_start, end=corrected_end, tokenizer_ner=tokenizer_ner)

#             grouped[corrected_label]["texts"].append(corrected_text)
#             grouped[corrected_label]["token_idxs"].append(token_idxs)

#         row["All ACTOR"] = grouped["ACTOR"]["texts"]
#         row["All ACTOR Token idx"] = grouped["ACTOR"]["token_idxs"]

#         row["All Description"] = grouped["CLINENTITY"]["texts"]
#         row["All Description Token idx"] = grouped["CLINENTITY"]["token_idxs"]

#         row["All TIMEX3"] = grouped["TIMEX3"]["texts"]
#         row["All TIMEX3 Token idx"] = grouped["TIMEX3"]["token_idxs"]

#         rows.append(row)

#     columns = ["Text", "All ACTOR", "All ACTOR Token idx", "All Description", "All Description Token idx", "All TIMEX3", "All TIMEX3 Token idx", "Original File"]

#     return pd.DataFrame(rows, columns=columns)

# def span_from_tokens(text, idxs, base=0):
#     offsets = tokenizer_ner(text, return_offsets_mapping=True, add_special_tokens=False, truncation=False)["offset_mapping"]

#     idxs = lit(idxs)
#     idxs = list(idxs) if isinstance(idxs, (list, tuple)) else []

#     spans = [offsets[i - base] for i in idxs if 0 <= i - base < len(offsets)]

#     if not spans:
#         return None, None

#     return min(s[0] for s in spans), max(s[1] for s in spans)


# def build_ner_json(df, token_index_base=0):
#     data = []

#     LABEL_MAP = {"All ACTOR": "ACTOR", "All Description": "CLINENTITY", "All TIMEX3": "TIMEX3"}

#     for _, row in df.iterrows():
#         text = row["Text"]
#         entities_out = []
#         ent_id = 0

#         for col, label in LABEL_MAP.items():
#             ents = lit(row.get(col, []))
#             idxs = lit(row.get(f"{col} Token idx", []))

#             if not isinstance(ents, list) or not isinstance(idxs, list):
#                 continue

#             for ent_text, token_idxs in zip(ents, idxs):
#                 start, end = span_from_tokens(text, token_idxs, base=token_index_base)

#                 if start is None:
#                     continue

#                 entities_out.append({
#                     "target_label": label,
#                     "entity": {
#                         "ent_id": ent_id,
#                         "text": ent_text,
#                         "start": start,
#                         "end": end,
#                     }
#                 })

#                 ent_id += 1

#         data.append({
#             "file_id": row["Original File"],
#             "text": text,
#             "entities": entities_out
#         })

#     return {"data": data}

def initialize_span_ner_model(ctx: "PipelineContext", encoder: Any, num_labels: int, id2label: dict, label2id: dict, tokenizer: Any, checkpoint_name: str | None, skip_incomplete_spans: bool, source_label2id: dict[str, int] | None = None, target_label2id: dict[str, int] | None = None, freeze_encoder: bool = False, base_encoder_name: str = None) -> Span_NERClassifier:
    MedLexSp = read_dsv_single(ctx.paths.docs_dir / "MedLexSp_v2/lexicon/MedLexSp_v2.dsv", header=None, names=["umls_cui", "lemma", "variant_forms", "pos", "semantic_types", "semantic_group"])
    if base_encoder_name is None:
        encoder_use = encoder
        tokenizer_use = tokenizer
    else:
        encoder_use = AutoModel.from_pretrained(base_encoder_name).to(device)
        tokenizer_use = AutoTokenizer.from_pretrained(base_encoder_name, trim_offsets=False, use_fast=True)

    model = Span_NERClassifier(encoder=encoder_use, num_labels=num_labels, id2label=id2label, label2id=label2id, tokenizer=tokenizer_use, skip_incomplete_spans=skip_incomplete_spans, lexicon=MedLexSp, freeze_encoder=freeze_encoder).to(ctx.device)
    if checkpoint_name is not None:
        checkpoint = checkpoint_state_dict(read_torch_checkpoint(ctx, checkpoint_name))
        if source_label2id is not None:
            load_checkpoint_with_resized_head(model, checkpoint, source_label2id=source_label2id, target_label2id=target_label2id)
        else:
            model.load_state_dict(checkpoint)

    return model

def initialize_bio_ner_model(ctx: "PipelineContext", encoder: Any, num_labels: int, checkpoint_name: str | None, source_label2id: dict[str, int] | None = None, target_label2id: dict[str, int] | None = None, base_encoder_name: str = None) -> BIO_NERClassifier:
    if base_encoder_name is None:
        encoder_use = encoder
    else:
        encoder_use = AutoModel.from_pretrained(base_encoder_name).to(device)

    model = BIO_NERClassifier(encoder=encoder_use, num_labels=num_labels).to(ctx.device)
    if checkpoint_name is not None:
        checkpoint = checkpoint_state_dict(read_torch_checkpoint(ctx, checkpoint_name))
        if source_label2id is not None:
            load_checkpoint_with_resized_head(model, checkpoint, source_label2id=source_label2id, target_label2id=target_label2id)
        else:
            model.load_state_dict(checkpoint)

    return model
