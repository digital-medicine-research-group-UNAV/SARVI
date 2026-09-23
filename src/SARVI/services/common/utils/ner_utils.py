import ast
import torch
import string
import numpy as np
import pandas as pd

from stop_words import get_stop_words

from ....models.schemas import (
    Any
)

###
stopwords_es = list(set(get_stop_words('spanish')+list(string.ascii_lowercase)+["ñ","ç","ch"]))
###

def average_overlapping_hidden_states(last_hidden_state: torch.Tensor, global_token_indices: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Averages hidden states contributed by overlapping NER windows.

    Parameters
    ----------
        `last_hidden_state`: torch.Tensor
            - Identifier values used to link or index records.
        `global_token_indices`: torch.Tensor
            - Token data used for encoding or alignment.
        `attention_mask`: torch.Tensor | None
            - Mask identifying non-padding input tokens.

    Returns
    -------
        `torch.Tensor`
            - Derived value produced by the operation.
    """
    if last_hidden_state.dim() != 3:
        raise ValueError(f"last_hidden_state debe tener forma (num_windows, seq_len, hidden_dim). Got {tuple(last_hidden_state.shape)}")

    if global_token_indices.shape != last_hidden_state.shape[:2]:
        raise ValueError(f"global_token_indices debe alinear con last_hidden_state en las dos primeras dimensiones: indices={tuple(global_token_indices.shape)}, hidden={tuple(last_hidden_state.shape)}")

    if attention_mask is not None and attention_mask.shape != global_token_indices.shape:
        raise ValueError(f"attention_mask debe tener la misma forma que global_token_indices: attention_mask={tuple(attention_mask.shape)}, indices={tuple(global_token_indices.shape)}")

    valid_mask = global_token_indices.ge(0)
    if attention_mask is not None:
        valid_mask = valid_mask & attention_mask.bool()

    if not valid_mask.any():
        return last_hidden_state

    flat_hidden = last_hidden_state.reshape(-1, last_hidden_state.size(-1))
    flat_indices = global_token_indices.reshape(-1)
    flat_valid = valid_mask.reshape(-1)

    valid_indices = flat_indices[flat_valid]
    valid_hidden = flat_hidden[flat_valid]
    unique_indices, inverse = torch.unique(valid_indices, sorted=False, return_inverse=True)

    sums = valid_hidden.new_zeros((unique_indices.size(0), valid_hidden.size(-1)))
    sums.index_add_(0, inverse, valid_hidden)

    counts = valid_hidden.new_zeros((unique_indices.size(0), 1))
    counts.index_add_(0, inverse, torch.ones((valid_hidden.size(0), 1), dtype=valid_hidden.dtype, device=valid_hidden.device))

    averaged = sums / counts
    out = last_hidden_state.clone().reshape(-1, last_hidden_state.size(-1))
    out[flat_valid] = averaged[inverse]

    return out.reshape_as(last_hidden_state)

def normalize_list(value: Any):
    """
    Normaliza un valor de entrada para devolverlo siempre como una lista.

    Parameters
    ----------
        `value`: Any
            - Valor que se quiere convertir a lista. Puede ser `None`, un `np.ndarray`, una lista, una tupla, una cadena de texto u otro objeto iterable

    Returns
    -------
        ``: list
            - Lista normalizada. Si el valor es **None** devuelve una lista vacía. Si es una cadena con formato de lista, intenta interpretarla como tal. Si no es iterable, devuelve una lista con el propio valor
    """
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, str):
        value = value.strip()

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except Exception:
            pass

        value = value.strip("[]")
        if not value:
            return []

        return [x.strip().strip("'").strip('"') for x in value.split(",")]

    try:
        return list(value)
    except TypeError:
        return [value]

def token_spans_are_adjacent(idx_a: list, idx_b: list):
    """
    Processes token spans are adjacent for use by the pipeline.

    Parameters
    ----------
        `idx_a`: list
            - Identifier values used to link the first span
        `idx_b`: list
            - Identifier values used to link the second span

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    if not idx_a or not idx_b:
        return False

    return max(idx_a) + 1 == min(idx_b) or max(idx_b) + 1 == min(idx_a)

def same_entity_component(row_a: dict, row_b: dict):
    """
    Processes same entity component for use by the pipeline.

    Parameters
    ----------
        `row_a`: dict
            - Dict with the info of the first span. Must contain **token_set** and **token_idx**
        `row_b`: dict
            - Dict with the info of the second span. Must contain **token_set** and **token_idx**

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    if row_a["token_set"] & row_b["token_set"]:
        return True

    if token_spans_are_adjacent(row_a["token_idx"], row_b["token_idx"]):
        return True

    return False

def build_entity_token_idx_from_component(component: list):
    """
    Builds the final list of token indices for an entity from a span component.

    Parameters
    ----------
        `component`: list
            - List of spans that are part of the same entity. Each element must contain the **token_idx** key

    Returns
    -------
        `all_idx`: list
            - Sorted list of all token indices that make up the final entity, with no duplicates
    """
    all_idx = set()

    for row in component:
        all_idx.update(row["token_idx"])

    return sorted(all_idx)

def get_final_entity_components_from_group(group: pd.DataFrame, token_idx_col: str):
    """
    Retrieves the final entity components within a group of spans that share the same file, text instance, and predicted label.

    Parameters
    ----------
        `group`: pd.DataFrame
            - Group of rows in the DataFrame corresponding to the same combination of file, text instance, and predicted label

        `token_idx_col`: str
            - Name of the column containing the token indices for each span

    Returns
    -------
        `components`: list
            - List of components. Each component includes: **source_row_indices** with the indices of the original rows that make up the component, and **final_token_idx** with the joined indices of the entity
    """
    rows = []

    for row_index, row in group.iterrows():
        token_idx = normalize_list(row[token_idx_col])
        token_idx = [int(x) for x in token_idx]

        if len(token_idx) == 0:
            continue

        rows.append({"row_index": row_index, "token_idx": token_idx, "token_set": set(token_idx)})

    visited = [False] * len(rows)
    components = []

    for i in range(len(rows)):
        if visited[i]:
            continue

        stack = [i]
        visited[i] = True
        component = []

        while stack:
            idx = stack.pop()
            component.append(rows[idx])

            for j in range(len(rows)):
                if visited[j]:
                    continue

                if same_entity_component(rows[idx], rows[j]):
                    visited[j] = True
                    stack.append(j)

        final_token_idx = build_entity_token_idx_from_component(component)

        components.append({"source_row_indices": [x["row_index"] for x in component], "final_token_idx": final_token_idx})

    return components

def char_span_to_token_span(text: str, start: int, end: int, tokenizer_ner: Any):
    """
    Maps a character span to the corresponding token indices.

    Parameters
    ----------
        `text`: str
            - Text containing the entity or span.
        `start`: int
            - Start value or position.
        `end`: int
            - End value or position.
        `tokenizer_ner`: Any
            - Tokenizer used to encode the model input.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    encoding = tokenizer_ner(text, return_offsets_mapping=True, add_special_tokens=False)

    token_idxs = []

    for i, (tok_start, tok_end) in enumerate(encoding["offset_mapping"]):
        if tok_start < end and tok_end > start:
            token_idxs.append(i)

    return token_idxs

def lit(x):
    """
    Processes lit for use by the pipeline.

    Parameters
    ----------
        `x`: Any
            - Prediction value or collection to remap.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x

def char_spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """
    Processes char spans overlap for use by the pipeline.

    Parameters
    ----------
        `a_start`: int
            - Argument controlling a start.
        `a_end`: int
            - Argument controlling a end.
        `b_start`: int
            - Argument controlling b start.
        `b_end`: int
            - Argument controlling b end.

    Returns
    -------
        `bool`
            - Boolean indicating whether the condition is satisfied.
    """
    return int(a_start) < int(b_end) and int(a_end) > int(b_start)

def brat_entities_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """
    Processes brat entities overlap for use by the pipeline.

    Parameters
    ----------
        `a`: dict[str, Any]
            - First code or value to compare.
        `b`: dict[str, Any]
            - Second code or value to compare.

    Returns
    -------
        `bool`
            - Boolean indicating whether the condition is satisfied.
    """
    return char_spans_overlap(a["start"], a["end"], b["start"], b["end"])

def format_brat_t_line(entity_id: int, entity: dict[str, Any]) -> str:
    """
    Formats an entity as a BRAT text-bound annotation line.

    Parameters
    ----------
        `entity_id`: int
            - Entity record or collection of entity records.
        `entity`: dict[str, Any]
            - Entity record being processed.

    Returns
    -------
        `str`
            - Derived value produced by the operation.
    """
    return f"T{entity_id}\t{entity['label']} {int(entity['start'])} {int(entity['end'])}\t{entity.get('text', '')}"

def token_overlaps_span(token_start: int, token_end: int, span_start: int, span_end: int) -> bool:
    """
    Processes token overlaps span for use by the pipeline.

    Parameters
    ----------
        `token_start`: int
            - Token data used for encoding or alignment.
        `token_end`: int
            - Token data used for encoding or alignment.
        `span_start`: int
            - Token or character positions used for alignment.
        `span_end`: int
            - Token or character positions used for alignment.

    Returns
    -------
        `bool`
            - Boolean indicating whether the condition is satisfied.
    """
    return token_start < span_end and token_end > span_start


def parse_label_piece(piece: str, pos: int) -> dict | None:
    """
    Parses a token label into its prefix and entity type.

    Examples
    --------
    "B-DISO"         -> B-DISO
    "2_B-DISO"       -> B-DISO with entity id 2
    "{2.1_B-DISO}"   -> B-DISO with discontinuous entity id 2, part 1
    "{2.2_I-DISO}"   -> I-DISO with discontinuous entity id 2, part 2
    "O"              -> None

    Parameters
    ----------
        `piece`: str
            - Argument controlling piece.
        `pos`: int
            - Argument controlling pos.

    Returns
    -------
        `dict | None`
            - Mapping containing the processed values.
    """
    piece = piece.strip()

    if piece == "" or piece == "O":
        return None

    disc_id = None
    entity_id = None
    component_id = None
    is_discontinuous = False

    if piece.startswith("{") and piece.endswith("}"):
        is_discontinuous = True
        inner = piece[1:-1]

        if "_" in inner:
            id_part, piece = inner.split("_", 1)
            if "." in id_part:
                disc_id, component_id = id_part.split(".", 1)
            else:
                disc_id = id_part
        else:
            piece = inner
    elif "_" in piece:
        id_part, label_part = piece.split("_", 1)
        if label_part.startswith("B-") or label_part.startswith("I-"):
            entity_id = id_part
            piece = label_part

    if not (piece.startswith("B-") or piece.startswith("I-")):
        return None

    bio, entity = piece.split("-", 1)
    if entity_id is None:
        entity_id = disc_id

    return {"bio": bio, "entity": entity, "entity_id": entity_id, "disc_id": disc_id, "component_id": component_id, "is_discontinuous": is_discontinuous, "pos": pos}


def parse_multilabel(raw_label: str) -> list[dict]:
    """
    Parses labels separated by ';'.

    Example
    -------
    "B-Date;B-DISO"
    """
    pieces = raw_label.split(";")

    candidates = []

    for pos, piece in enumerate(pieces):
        parsed = parse_label_piece(piece, pos)

        if parsed is not None:
            candidates.append(parsed)

    return candidates


def label_entity(label: str) -> str | None:
    if label == "O":
        return None

    return label.split("-", 1)[1]


def candidate_identity(candidate: dict) -> str | None:
    """
    Processes candidate identity for use by the pipeline.

    Parameters
    ----------
        `candidate`: dict
            - Identifier values used to link or index records.

    Returns
    -------
        `str | None`
            - Derived value produced by the operation.
    """
    if candidate["is_discontinuous"]:
        if candidate["disc_id"] is not None and candidate["component_id"] is not None:
            return f"{candidate['disc_id']}.{candidate['component_id']}"

        return candidate["disc_id"]

    return candidate["entity_id"]


def candidate_block_key(candidate: dict) -> tuple:
    """
    Processes candidate block key for use by the pipeline.

    Parameters
    ----------
        `candidate`: dict
            - Identifier values used to link or index records.

    Returns
    -------
        `tuple`
            - Tuple containing the derived values.
    """
    identity = candidate_identity(candidate)

    if identity is None:
        return ("entity", candidate["entity"])

    return ("identity", candidate["entity"], identity)


def can_continue_previous_entity(candidate: dict, previous_output_label: str, previous_identity: str | None) -> bool:
    """
    Processes can continue previous entity for use by the pipeline.
    B-DISO followed by B-DISO can continue the same entity.

    Parameters
    ----------
        `candidate`: dict
            - Identifier values used to link or index records.
        `previous_output_label`: str
            - Entity or relation labels used by the model.
        `previous_identity`: str | None
            - Entity record or collection of entity records.

    Returns
    -------
        `bool`
            - Boolean indicating whether the condition is satisfied.
    """
    if previous_output_label == "O":
        return False

    previous_entity = label_entity(previous_output_label)

    if candidate["entity"] != previous_entity:
        return False

    identity = candidate_identity(candidate)

    if identity is not None or previous_identity is not None:
        return identity == previous_identity

    return True


def candidate_to_output_label(candidate: dict, previous_output_label: str, previous_identity: str | None) -> str:
    """
    Converts selected candidate into the final normal BIO label.

    Rules
    -----
    - {1.1_B-DISO} -> B-DISO
    - {1.2_I-DISO} after {1.1_B-DISO} -> I-DISO
    - {1.2_I-DISO} after O -> B-DISO
    - B-DISO after B-DISO -> B-DISO
    - I-DISO without previous DISO -> O
    """
    bio = candidate["bio"]
    entity = candidate["entity"]

    if candidate["is_discontinuous"]:
        if bio == "I":
            if can_continue_previous_entity(candidate=candidate, previous_output_label=previous_output_label, previous_identity=previous_identity):
                return f"I-{entity}"
            return f"B-{entity}"
        return f"B-{entity}"

    if bio == "B":
        return f"B-{entity}"

    if bio == "I":
        if can_continue_previous_entity(candidate=candidate, previous_output_label=previous_output_label, previous_identity=previous_identity):
            return f"I-{entity}"
        return "O"
    return "O"


def normalize_bio_sequence(labels: list[str]) -> list[str]:
    """
    Normalizes multilabel/discontinuous BIO labels.

    Main rules
    ----------
    1. Entities after ';' are not selected as new entities.
    2. But if an entity after ';' continues the currently active valid entity, we keep that active entity.
    3. Discontinuous entities are converted into normal entities.
    4. Separated discontinuous parts become separate entities.
    5. Consecutive B-ENT labels can belong to the same entity.
    """
    output_labels = []

    previous_output_label = "O"
    previous_identity = None

    blocked_candidates = set()

    for raw_label in labels:
        candidates = parse_multilabel(raw_label)
        present_block_keys = {candidate_block_key(candidate) for candidate in candidates}

        # Remove blocked entity instances that are no longer present.
        blocked_candidates = {block_key for block_key in blocked_candidates if block_key in present_block_keys}
        selected_candidate = None

        # 1. Prefer a candidate that continues the currently active valid entity, even if it appears after ';'
        for candidate in candidates:
            if can_continue_previous_entity(candidate=candidate, previous_output_label=previous_output_label, previous_identity=previous_identity):
                selected_candidate = candidate
                break

        # 2. Otherwise, only use the first-position candidate. Do not start new entities from labels after ';'.
        if selected_candidate is None:
            first_candidates = [candidate for candidate in candidates if candidate["pos"] == 0]

            if len(first_candidates) > 0:
                first_candidate = first_candidates[0]

                if candidate_block_key(first_candidate) not in blocked_candidates:
                    selected_candidate = first_candidate

        # 3. Produce final output label
        if selected_candidate is None:
            output_label = "O"
            selected_block_key = None
            selected_identity = None
        else:
            output_label = candidate_to_output_label(candidate=selected_candidate, previous_output_label=previous_output_label, previous_identity=previous_identity)

            if output_label == "O":
                selected_block_key = None
                selected_identity = None
            else:
                selected_block_key = candidate_block_key(selected_candidate)
                selected_identity = candidate_identity(selected_candidate)

        output_labels.append(output_label)

        # 4. Block entities that appear after ';' but were not selected. These are overlapping entity instances that started inside another entity.
        for candidate in candidates:
            if candidate["pos"] > 0:
                block_key = candidate_block_key(candidate)
                if selected_block_key != block_key:
                    blocked_candidates.add(block_key)

        previous_output_label = output_label
        previous_identity = selected_identity

    return output_labels


def same_entity(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a["label"] == b["label"] and int(a["start"]) == int(b["start"]) and int(a["end"]) == int(b["end"])

def entity_exists(entity: dict[str, Any], entities: list[dict[str, Any]]) -> bool:
    """
    Processes entity exists for use by the pipeline.

    Parameters
    ----------
        `entity`: dict[str, Any]
            - Entity record being processed.
        `entities`: list[dict[str, Any]]
            - Argument controlling entities.

    Returns
    -------
        `bool`
            - Boolean indicating whether the condition is satisfied.
    """
    return any(same_entity(entity, other) for other in entities)

def entity_text_for_span(entity: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    """
    Processes entity text for span for use by the pipeline.

    Parameters
    ----------
        `entity`: dict[str, Any]
            - Entity record being processed.
        `candidates`: list[dict[str, Any]]
            - Identifier values used to link or index records.

    Returns
    -------
        `str`
            - Derived value produced by the operation.
    """
    for candidate in candidates:
        if int(candidate["start"]) == int(entity["start"]) and int(candidate["end"]) == int(entity["end"]) and candidate["text"]:
            return candidate["text"]
    return entity.get("text", "")

def text_num_tokens_from_windows(windows: list[dict[str, Any]]) -> int:
    """
    Processes text num tokens from windows for use by the pipeline.

    Parameters
    ----------
        `windows`: list[dict[str, Any]]
            - Argument controlling windows.

    Returns
    -------
        `int`
            - Derived value produced by the operation.
    """
    max_abs_idx = -1

    for window in windows:
        for absolute_idx in window["absolute_token_indices"]:
            if absolute_idx is not None:
                max_abs_idx = max(max_abs_idx, int(absolute_idx))

    return max_abs_idx + 1

def shift_absolute_token_indices(absolute_token_indices: list[int | None], token_offset: int) -> list[int]:
    """
    Processes shift absolute token indices for use by the pipeline.

    Parameters
    ----------
        `absolute_token_indices`: list[int | None]
            - Token data used for encoding or alignment.
        `token_offset`: int
            - Token or character positions used for alignment.

    Returns
    -------
        `list[int]`
            - List of parsed, filtered, or generated values.
    """
    return [-1 if absolute_idx is None else token_offset + int(absolute_idx) for absolute_idx in absolute_token_indices]

def entity_matches_correction(ent: dict, cor: dict, original_text: str) -> bool:
    # return (ent["entity"]["start"] == cor["original_start"] and ent["entity"]["end"] == cor["original_end"] and ent["entity"]["text"] == cor["original_text"] and original_text[cor["original_start"]:cor["original_end"]] == cor["original_text"])
    """
    Checks whether a predicted entity matches its proposed correction.

    Parameters
    ----------
        `ent`: dict
            - Entity record or collection of entity records.
        `cor`: dict
            - Argument controlling cor.
        `original_text`: str
            - Source text being processed.

    Returns
    -------
        `bool`
            - Boolean indicating whether the condition is satisfied.
    """
    return (ent["entity"]["start"] == cor["original_start"] and ent["entity"]["end"] == cor["original_end"] and ent["entity"]["text"] == cor["original_text"])


def get_missing_entities(original_entities: list[dict], corrected_entities: list[dict], original_text: str) -> list[dict]:
    """
    Finds entities present in the original annotations but missing after correction.

    Parameters
    ----------
        `original_entities`: list[dict]
            - Argument controlling original entities.
        `corrected_entities`: list[dict]
            - Argument controlling corrected entities.
        `original_text`: str
            - Source text being processed.

    Returns
    -------
        `list[dict]`
            - List of parsed, filtered, or generated values.
    """
    missing = []

    for ent in original_entities:
        exists = any(entity_matches_correction(ent, cor, original_text) for cor in corrected_entities)
        if not exists:
            missing.append(ent)

    return missing
