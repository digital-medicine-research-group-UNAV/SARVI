from __future__ import annotations

import re
from typing import Any

from .nn_utils import (
    make_entity_marker
)

def sentence_spans_until_period(text: str) -> list[tuple[int, int]]:
    """
    Splits text into sentence spans ending at periods.

    Parameters
    ----------
        `text`: str
            - Text containing the entity or span.

    Returns
    -------
        `list[tuple[int, int]]`
            - List of parsed, filtered, or generated values.
    """
    spans = []
    start = 0

    for match in re.finditer(r"\.", text):
        end = match.end()
        if end > start:
            spans.append((start, end))
        start = end

    if start < len(text):
        spans.append((start, len(text)))

    return [(start, end) for start, end in spans if text[start:end].strip()]

def entity_sentence_index(entity: dict[str, Any], sentence_spans: list[tuple[int, int]]) -> int | None:
    """
    Processes entity sentence index for use by the pipeline.

    Parameters
    ----------
        `entity`: dict[str, Any]
            - Entity record being processed.
        `sentence_spans`: list[tuple[int, int]]
            - Token or character positions used for alignment.

    Returns
    -------
        `int | None`
            - Derived value produced by the operation.
    """
    ent_start = int(entity["start"])
    ent_end = int(entity["end"])

    for idx, (sent_start, sent_end) in enumerate(sentence_spans):
        if sent_start <= ent_start and ent_end <= sent_end:
            return idx

    for idx, (sent_start, sent_end) in enumerate(sentence_spans):
        if sent_start < ent_end and sent_end > ent_start:
            return idx

    return None

def char_spans_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """
    Processes char spans overlap for use by the pipeline.

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
    return int(a["start"]) < int(b["end"]) and int(a["end"]) > int(b["start"])

def entities_with_label(entities: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    """
    Processes entities with label for use by the pipeline.

    Parameters
    ----------
        `entities`: list[dict[str, Any]]
            - Argument controlling entities.
        `label`: str
            - Entity or relation label.

    Returns
    -------
        `list[dict[str, Any]]`
            - List of parsed, filtered, or generated values.
    """
    return [entity for entity in entities if entity.get("label") == label]

def make_entity_pairs(entities: list[dict[str, Any]], label_a: str, label_b: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """
    Creates candidate entity pairs for relation extraction.

    Parameters
    ----------
        `entities`: list[dict[str, Any]]
            - Argument controlling entities.
        `label_a`: str
            - Entity or relation labels used by the model.
        `label_b`: str
            - Entity or relation labels used by the model.

    Returns
    -------
        `list[tuple[dict[str, Any], dict[str, Any]]]`
            - List of parsed, filtered, or generated values.
    """
    left_entities = entities_with_label(entities, label_a)
    right_entities = entities_with_label(entities, label_b)
    pairs = []

    for first in left_entities:
        for second in right_entities:
            if first.get("id") == second.get("id"):
                continue
            if label_a == label_b and int(first["start"]) >= int(second["start"]):
                continue
            if first.get("original_id", first.get("id")) == second.get("original_id", second.get("id")) and not (first.get("parent_is_discontinuous", False) and second.get("parent_is_discontinuous", False)):
                continue
            if char_spans_overlap(first, second):
                continue

            ordered = tuple(sorted((first, second), key=lambda item: (int(item["start"]), int(item["end"]), str(item.get("id", "")))))
            if ordered not in pairs:
                pairs.append(ordered)

    return sorted(pairs, key=lambda pair: (int(pair[0]["start"]), int(pair[1]["start"]), str(pair[0].get("id", "")), str(pair[1].get("id", ""))))

def selected_sentence_groups(left_sentence_idx: int, right_sentence_idx: int, num_sentences: int, w: int, v: int) -> tuple[list[int], list[int] | None, int]:
    """
    Selects left and right sentence groups for a relation example.

    Parameters
    ----------
        `left_sentence_idx`: int
            - Identifier values used to link or index records.
        `right_sentence_idx`: int
            - Identifier values used to link or index records.
        `num_sentences`: int
            - Total number of sentences in the document.
        `w`: int
            - Number of neighboring sentences included in the window.
        `v`: int
            - Additional sentence distance used for relation context.

    Returns
    -------
        `tuple[list[int], list[int] | None, int]`
            - List of parsed, filtered, or generated values.
    """
    left_idx = min(left_sentence_idx, right_sentence_idx)
    right_idx = max(left_sentence_idx, right_sentence_idx)
    middle_count = max(0, right_idx - left_idx - 1)

    if left_idx == right_idx:
        before = list(range(max(0, left_idx - v), left_idx))
        after = list(range(left_idx + 1, min(num_sentences, left_idx + 1 + v)))
        return before + [left_idx] + after, None, 0

    before = list(range(max(0, left_idx - v), left_idx))
    after = list(range(right_idx + 1, min(num_sentences, right_idx + 1 + v)))
    between_start = left_idx + 1
    between_end = right_idx

    left_take = min(w, middle_count)
    right_take = min(w, middle_count - left_take)
    if middle_count > w * 2:
        left_take = w
        right_take = w

    left_middle = list(range(between_start, between_start + left_take))
    right_middle = list(range(between_end - right_take, between_end)) if right_take > 0 else []

    left_group = before + [left_idx] + left_middle
    right_group = right_middle + [right_idx] + after
    return left_group, right_group, middle_count

def insert_pair_markers(text: str, left_entity: dict[str, Any], right_entity: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """
    Inserts relation markers around the two entities and tracks their positions.

    Parameters
    ----------
        `text`: str
            - Text containing the entity or span.
        `left_entity`: dict[str, Any]
            - Entity record or collection of entity records.
        `right_entity`: dict[str, Any]
            - Entity record or collection of entity records.

    Returns
    -------
        `tuple[str, list[dict[str, Any]]]`
            - List of parsed, filtered, or generated values.
    """
    insertions = [
        {
            "orig_pos": int(left_entity["start"]),
            "text": make_entity_marker("S", str(left_entity["label"]), closing=False),
            "kind": "start",
            "role": "S",
            "entity_id": left_entity.get("id"),
            "label": left_entity.get("label"),
        },
        {
            "orig_pos": int(left_entity["end"]),
            "text": make_entity_marker("S", str(left_entity["label"]), closing=True),
            "kind": "end",
            "role": "S",
            "entity_id": left_entity.get("id"),
            "label": left_entity.get("label"),
        },
        {
            "orig_pos": int(right_entity["start"]),
            "text": make_entity_marker("O", str(right_entity["label"]), closing=False),
            "kind": "start",
            "role": "O",
            "entity_id": right_entity.get("id"),
            "label": right_entity.get("label"),
        },
        {
            "orig_pos": int(right_entity["end"]),
            "text": make_entity_marker("O", str(right_entity["label"]), closing=True),
            "kind": "end",
            "role": "O",
            "entity_id": right_entity.get("id"),
            "label": right_entity.get("label"),
        },
    ]
    insertions = sorted(insertions, key=lambda item: (item["orig_pos"], 0 if item["kind"] == "end" else 1))

    parts = []
    cursor = 0
    marked_insertions = []
    added = 0

    for insertion in insertions:
        orig_pos = insertion["orig_pos"]
        parts.append(text[cursor:orig_pos])
        marker_start = orig_pos + added
        parts.append(insertion["text"])
        marker_end = marker_start + len(insertion["text"])
        marked_insertions.append({**insertion, "marked_start": marker_start, "marked_end": marker_end})
        added += len(insertion["text"])
        cursor = orig_pos

    parts.append(text[cursor:])
    return "".join(parts), marked_insertions

def original_to_marked_pos(insertions: list[dict[str, Any]], original_pos: int, include_at_pos: bool) -> int:
    """
    Processes original to marked pos for use by the pipeline.

    Parameters
    ----------
        `insertions`: list[dict[str, Any]]
            - Argument controlling insertions.
        `original_pos`: int
            - Character position in the unmarked text.
        `include_at_pos`: bool
            - Whether insertions at the current position are included.

    Returns
    -------
        `int`
            - Derived value produced by the operation.
    """
    total_added = 0
    for insertion in insertions:
        if insertion["orig_pos"] < original_pos or (include_at_pos and insertion["orig_pos"] == original_pos):
            total_added += len(insertion["text"])
    return original_pos + total_added

def marked_sentence_bounds(sentence_indices: list[int], sentence_spans: list[tuple[int, int]], insertions: list[dict[str, Any]]) -> tuple[int, int]:
    """
    Processes marked sentence bounds for use by the pipeline.

    Parameters
    ----------
        `sentence_indices`: list[int]
            - Argument controlling sentence indices.
        `sentence_spans`: list[tuple[int, int]]
            - Token or character positions used for alignment.
        `insertions`: list[dict[str, Any]]
            - Argument controlling insertions.

    Returns
    -------
        `tuple[int, int]`
            - Tuple containing the derived values.
    """
    context_start = sentence_spans[min(sentence_indices)][0]
    context_end = sentence_spans[max(sentence_indices)][1]
    marked_start = original_to_marked_pos(insertions, context_start, include_at_pos=False)
    marked_end = original_to_marked_pos(insertions, context_end, include_at_pos=True)
    return marked_start, marked_end

def token_indices_for_marked_bounds(full_offsets: list[tuple[int, int]], marked_start: int, marked_end: int) -> list[int]:
    """
    Processes token indices for marked bounds for use by the pipeline.

    Parameters
    ----------
        `full_offsets`: list[tuple[int, int]]
            - Token or character positions used for alignment.
        `marked_start`: int
            - Start position in the marked text.
        `marked_end`: int
            - End position in the marked text.

    Returns
    -------
        `list[int]`
            - List of parsed, filtered, or generated values.
    """
    return [idx for idx, (tok_start, tok_end) in enumerate(full_offsets) if int(tok_start) < marked_end and int(tok_end) > marked_start]

def build_position_arrays_for_special_tokens(input_ids: list[int], payload_ids: list[int], payload_positions: list[int]) -> tuple[list[int | None], list[int]]:
    """
    Processes build position arrays for special tokens for use by the pipeline.

    Parameters
    ----------
        `input_ids`: list[int]
            - Encoded input token IDs.
        `payload_ids`: list[int]
            - Encoded IDs for the payload sequence.
        `payload_positions`: list[int]
            - Positions occupied by payload tokens.

    Returns
    -------
        `tuple[list[int | None], list[int]]`
            - List of parsed, filtered, or generated values.
    """
    payload_cursor = 0
    original_position_ids = []

    for input_id in input_ids:
        if payload_cursor < len(payload_ids) and int(input_id) == int(payload_ids[payload_cursor]):
            original_position_ids.append(payload_positions[payload_cursor])
            payload_cursor += 1
        else:
            original_position_ids.append(None)

    real_positions = [pos for pos in original_position_ids if pos is not None]
    min_token_idx = min(real_positions) if real_positions else 0
    position_ids = [-1 if pos is None else int(pos) - min_token_idx for pos in original_position_ids]
    return original_position_ids, position_ids

def build_re_input_ids_with_one_sep(tokenizer: Any, ids_a: list[int], ids_b: list[int] | None = None) -> list[int]:
    """
    Builds relation-model input IDs with a single separator token.

    Parameters
    ----------
        `tokenizer`: Any
            - Tokenizer used to encode text and obtain offsets.
        `ids_a`: list[int]
            - Identifier values used to link or index records.
        `ids_b`: list[int] | None
            - Identifier values used to link or index records.

    Returns
    -------
        `list[int]`
            - List of parsed, filtered, or generated values.
    """
    cls_token_id = getattr(tokenizer, "cls_token_id", None)
    sep_token_id = getattr(tokenizer, "sep_token_id", None)
    if cls_token_id is None or sep_token_id is None:
        raise ValueError("Tokenizer must define cls_token_id and sep_token_id for RE one-SEP inputs.")

    if ids_b is None:
        return [int(cls_token_id)] + list(ids_a) + [int(sep_token_id)]

    return [int(cls_token_id)] + list(ids_a) + [int(sep_token_id)] + list(ids_b)

def pad_re_input(example: dict[str, Any]) -> tuple[list[int], list[int]]:
    """
    Pads a relation example and creates its attention mask.

    Parameters
    ----------
        `example`: dict[str, Any]
            - Argument controlling example.

    Returns
    -------
        `tuple[list[int], list[int]]`
            - List of parsed, filtered, or generated values.
    """
    max_length = int(example.get("max_length", len(example["input_ids"])))
    pad_length = max_length - len(example["input_ids"])
    if pad_length < 0:
        raise ValueError(f"RE example has length {len(example['input_ids'])} > max_length {max_length}")

    pad_token_id = int(example.get("pad_token_id", 0))
    input_ids = example["input_ids"] + [pad_token_id] * pad_length
    attention_mask = example["attention_mask"] + [0] * pad_length
    return input_ids, attention_mask

def re_marker_position_ids(example: dict[str, Any]) -> list[int]:
    """
    Creates marker position IDs for a relation example.

    Parameters
    ----------
        `example`: dict[str, Any]
            - Argument controlling example.

    Returns
    -------
        `list[int]`
            - List of parsed, filtered, or generated values.
    """
    left_entity = example.get("left_entity", {})
    right_entity = example.get("right_entity", {})
    marker_positions = example.get("marker_positions", {})

    return [
        int(marker_positions.get(f"<S:{left_entity.get('label')}>", [-1])[0]),
        int(marker_positions.get(f"</S:{left_entity.get('label')}>", [-1])[0]),
        int(marker_positions.get(f"<O:{right_entity.get('label')}>", [-1])[0]),
        int(marker_positions.get(f"</O:{right_entity.get('label')}>", [-1])[0]),
    ]

def re_sep_position_id(example: dict[str, Any]) -> int:
    """
    Processes re sep position id for use by the pipeline.

    Parameters
    ----------
        `example`: dict[str, Any]
            - Argument controlling example.

    Returns
    -------
        `int`
            - Derived value produced by the operation.
    """
    sep_candidates = [pos for pos, (position_id, token_mask) in enumerate(zip(example["position_ids"], example["attention_mask"])) if int(token_mask) == 1 and int(position_id) < 0 and pos != 0]
    return int(sep_candidates[0]) if sep_candidates else -1

def resolve_max_length(tokenizer: Any, max_length: int | None = None) -> int:
    """
    Resolves the usable sequence length from the tokenizer and override.

    Parameters
    ----------
        `tokenizer`: Any
            - Tokenizer used to encode text and obtain offsets.
        `max_length`: int | None
            - Maximum encoded sequence length.

    Returns
    -------
        `int`
            - Derived value produced by the operation.
    """
    if max_length is not None:
        return int(max_length)

    tokenizer_max_length = int(getattr(tokenizer, "model_max_length", 0) or 0)
    if 0 < tokenizer_max_length < 100_000:
        return tokenizer_max_length

    return 512

def middle_token_count_to_bin(middle_token_count: int, token_distance_bins: dict[int, tuple[int | float, int | float]]) -> int:
    """
    Assigns the number of middle tokens to a configured distance bin.

    Parameters
    ----------
        `middle_token_count`: int
            - Number of tokens between the two entities.
        `token_distance_bins`: dict[int, tuple[int | float, int | float]]
            - Configured bins for relation distances.

    Returns
    -------
        `int`
            - Derived value produced by the operation.
    """
    for bin_id, (lower, upper) in token_distance_bins.items():
        if float(lower) <= float(middle_token_count) <= float(upper):
            return int(bin_id)

    raise ValueError(f"middle_token_count={middle_token_count} does not fit in token_distance_bins.")
