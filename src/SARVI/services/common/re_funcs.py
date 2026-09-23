from __future__ import annotations

import random
import torch
import pandas as pd
from typing import Any, TYPE_CHECKING

from transformers import AutoModel

from ...models.neural_networks import (
    REClassifier
)
from ...data_io.reader import (
    read_torch_checkpoint
)
from .utils.ann_utils import (
    normalize_ann_lines, parse_brat_text_boundaries
)
from .utils.re_utils import (
    pad_re_input,
    re_marker_position_ids, re_sep_position_id,
    entity_sentence_index, token_indices_for_marked_bounds,
    selected_sentence_groups, marked_sentence_bounds, sentence_spans_until_period,
    build_position_arrays_for_special_tokens, make_entity_marker, build_re_input_ids_with_one_sep, insert_pair_markers
)

if TYPE_CHECKING:
    from ...models.schemas import PipelineContext

###
device = "cuda" if torch.cuda.is_available() else "cpu"
###

def parse_re_target_entities(ann_info: Any, target_labels: set[str], t_col: str = "T") -> list[dict[str, Any]]:
    """
    Extracts relation-extraction target entities from annotation data.

    Parameters
    ----------
        `ann_info`: Any
            - Annotation row or object to parse.
        `target_labels`: set[str]
            - Entity or relation labels used by the model.
        `t_col`: str
            - BRAT text-bound column.

    Returns
    -------
        `list[dict[str, Any]]`
            - List of parsed, filtered, or generated values.
    """
    if ann_info is None:
        return []

    if isinstance(ann_info, pd.Series) or isinstance(ann_info, dict):
        t_lines = normalize_ann_lines(ann_info.get(t_col))
    else:
        t_lines = normalize_ann_lines(getattr(ann_info, t_col, None))

    entities = []
    for line in t_lines:
        parts = str(line).split("\t")
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

        is_discontinuous = len(char_spans) > 1
        for component_idx, (start, end) in enumerate(char_spans, start=1):
            entity_id = f"{ann_id}.{component_idx}" if is_discontinuous else ann_id
            entities.append({
                "id": entity_id,
                "original_id": ann_id,
                "component_idx": component_idx,
                "label": label,
                "char_spans": [(start, end)],
                "original_char_spans": char_spans,
                "is_discontinuous": False,
                "parent_is_discontinuous": is_discontinuous,
                "start": start,
                "end": end,
                "length": end - start,
                "text": parts[2] if len(parts) > 2 else "",
                "line": str(line),
            })

    for entity_order, entity in enumerate(sorted(entities, key=lambda item: (item["start"], item["end"], item["id"])), start=1):
        entity["entity_order"] = entity_order

    return entities

def parse_re_relations(ann_info: Any, r_col: str = "R") -> list[dict[str, Any]]:
    """
    Extracts relation annotations and their linked entity identifiers.

    Parameters
    ----------
        `ann_info`: Any
            - Annotation row or object to parse.
        `r_col`: str
            - Argument controlling r col.

    Returns
    -------
        `list[dict[str, Any]]`
            - List of parsed, filtered, or generated values.
    """
    if ann_info is None:
        return []

    if isinstance(ann_info, pd.Series) or isinstance(ann_info, dict):
        r_lines = normalize_ann_lines(ann_info.get(r_col))
    else:
        r_lines = normalize_ann_lines(getattr(ann_info, r_col, None))

    relations = []
    for line in r_lines:
        parts = str(line).split("\t")
        if len(parts) < 2:
            continue

        relation_id = parts[0]
        pieces = parts[1].split()
        if not pieces:
            continue

        relation_label = pieces[0]
        args = {}
        for piece in pieces[1:]:
            if ":" not in piece:
                continue
            arg_name, entity_id = piece.split(":", 1)
            args[arg_name] = entity_id

        if len(args) < 2:
            continue

        relations.append({
            "id": relation_id,
            "label": relation_label,
            "args": args,
            "line": str(line),
        })

    return relations

def relation_label_for_pair(left_entity: dict[str, Any], right_entity: dict[str, Any], relations: list[dict[str, Any]], outside_label: str = "O") -> str:
    """
    Finds the relation label assigned to a pair of entities.

    Parameters
    ----------
        `left_entity`: dict[str, Any]
            - Entity record or collection of entity records.
        `right_entity`: dict[str, Any]
            - Entity record or collection of entity records.
        `relations`: list[dict[str, Any]]
            - Argument controlling relations.
        `outside_label`: str
            - Label assigned to non-entity tokens.

    Returns
    -------
        `str`
            - Derived value produced by the operation.
    """
    left_id = left_entity.get("original_id", left_entity.get("id"))
    right_id = right_entity.get("original_id", right_entity.get("id"))
    pair_ids = {left_id, right_id}

    if left_id == right_id and left_entity.get("parent_is_discontinuous", False) and right_entity.get("parent_is_discontinuous", False):
        return "union"

    for relation in relations:
        relation_entity_ids = set(relation.get("args", {}).values())
        if pair_ids <= relation_entity_ids:
            return str(relation["label"])

    return outside_label

def re_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Pads and collates relation-extraction examples into a model batch.

    Parameters
    ----------
        `batch`: list[dict[str, Any]]
            - Argument controlling batch.

    Returns
    -------
        `dict[str, Any]`
            - Mapping containing the processed values.
    """
    input_ids = []
    attention_mask = []
    marker_position_ids = []
    sep_position_ids = []
    middle_token_counts = []
    middle_token_count_bins = []
    relation_labels = []
    example_to_text = []
    example_local_index = []
    text_indices = []
    file_names = []
    
    ##############################
    ## Position_ids/original_position_ids are useful while constructing the RE window, but the current REDataset, REClassifier and training notebooks do not consume them from the collated batch.
    
    # position_ids = []
    # original_position_ids = []
    # middle_sentence_counts = []
    # middle_token_count_difficulties = []
    # examples_metadata = []
    ##############################

    labels_present = any(item.get("relation_labels") is not None for item in batch)

    for batch_text_idx, item in enumerate(batch):
        examples = item["examples"]
        labels = item.get("relation_labels")

        text_indices.append(item["text_index"])
        file_names.append(item["file_name"])

        for example_idx, example in enumerate(examples):
            input_ids_padded, attention_mask_padded = pad_re_input(example)

            input_ids.append(input_ids_padded)
            attention_mask.append(attention_mask_padded)
            marker_position_ids.append(re_marker_position_ids(example))
            sep_position_ids.append(re_sep_position_id(example))
            
            middle_token_counts.append(int(example.get("middle_token_count", 0)))
            middle_token_count_bins.append(int(example.get("middle_token_count_bin", -1)))
            

            example_to_text.append(batch_text_idx)
            example_local_index.append(example_idx)

            ##############################
            # position_ids_padded = example["position_ids"] + [-1] * pad_length
            # original_position_ids_padded = [-1 if value is None else int(value) for value in example.get("original_position_ids", [])] + [-1] * pad_length

            # position_ids.append(position_ids_padded)
            # original_position_ids.append(original_position_ids_padded)

            # middle_sentence_counts.append(int(example.get("middle_sentence_count", 0)))
            # middle_token_count_difficulties.append(example.get("middle_token_count_difficulty"))

            # examples_metadata.append({
            #     "file_name": example.get("file_name", item["file_name"]),
            #     "left_entity": example.get("left_entity"),
            #     "right_entity": example.get("right_entity"),
            #     "subject_entity": example.get("subject_entity"),
            #     "object_entity": example.get("object_entity"),
            #     "marker_positions": example.get("marker_positions"),
            #     "marked_text": example.get("marked_text"),
            #     "selected_sentence_indices": example.get("selected_sentence_indices"),
            #     "middle_sentence_count": example.get("middle_sentence_count"),
            #     "middle_token_count": example.get("middle_token_count"),
            #     "middle_token_count_bin": example.get("middle_token_count_bin"),
            #     "middle_token_count_difficulty": example.get("middle_token_count_difficulty"),
            #     "selected_full_token_indices": example.get("selected_full_token_indices"),
            # })
            ##############################

            if labels is not None:
                relation_labels.append(int(labels[example_idx]))

    if not input_ids:
        batch_out = {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "attention_mask": torch.empty((0, 0), dtype=torch.long),
            "marker_position_ids": torch.empty((0, 4), dtype=torch.long),
            "sep_position_ids": torch.empty((0,), dtype=torch.long),
            "middle_token_count": torch.empty((0,), dtype=torch.long),
            "middle_token_count_bin": torch.empty((0,), dtype=torch.long),
            "example_to_text": torch.empty((0,), dtype=torch.long),
            "example_local_index": torch.empty((0,), dtype=torch.long),
            "text_indices": torch.tensor(text_indices, dtype=torch.long),
            "file_names": file_names,
            "relation_labels": torch.empty((0,), dtype=torch.long) if labels_present else None,

            # #############################
            # "position_ids": torch.empty((0, 0), dtype=torch.long),
            # "original_position_ids": torch.empty((0, 0), dtype=torch.long),
            # "marker_position_names": ["subject_start", "subject_end", "object_start", "object_end"],
            # "middle_sentence_count": torch.empty((0,), dtype=torch.long),
            # "middle_token_count_difficulty": [],
            # "examples_metadata": examples_metadata,
            # #############################
        }
        return batch_out
    else:
        batch_out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "marker_position_ids": torch.tensor(marker_position_ids, dtype=torch.long),
            "sep_position_ids": torch.tensor(sep_position_ids, dtype=torch.long),
            "middle_token_count": torch.tensor(middle_token_counts, dtype=torch.long),
            "middle_token_count_bin": torch.tensor(middle_token_count_bins, dtype=torch.long),
            "example_to_text": torch.tensor(example_to_text, dtype=torch.long),
            "example_local_index": torch.tensor(example_local_index, dtype=torch.long),
            "text_indices": torch.tensor(text_indices, dtype=torch.long),
            "file_names": file_names,

            ##############################
            # "position_ids": torch.tensor(position_ids, dtype=torch.long),
            # "original_position_ids": torch.tensor(original_position_ids, dtype=torch.long),
            # "marker_position_names": ["subject_start", "subject_end", "object_start", "object_end"],
            # "middle_sentence_count": torch.tensor(middle_sentence_counts, dtype=torch.long),
            # "middle_token_count_difficulty": middle_token_count_difficulties,
            # "examples_metadata": examples_metadata,
            ##############################
        }

        if labels_present:
            batch_out["relation_labels"] = torch.tensor(relation_labels, dtype=torch.long)
        else:
            batch_out["relation_labels"] = None

        return batch_out


def undersample_re_negatives(data_prepared: list[list[dict[str, Any]]], relation_labels: list[list[str | int]], outside_label: str = "O", max_hard_negative_ratio: int | float = 2, max_medium_negative_ratio: int | float = 3, max_easy_negative_ratio: int | float = 1, seed: int = 8) -> tuple[list[list[dict[str, Any]]], list[list[str | int]]]:
    if len(data_prepared) != len(relation_labels):
        raise ValueError("data_prepared and relation_labels must have the same number of texts.")

    positives = []
    negatives_by_difficulty = {"hard": [], "medium": [], "easy": []}
    kept_entries = []

    for text_idx, (examples, labels) in enumerate(zip(data_prepared, relation_labels)):
        if len(examples) != len(labels):
            raise ValueError(f"Text {text_idx} has {len(examples)} examples but {len(labels)} labels.")

        for example_idx, (example, label) in enumerate(zip(examples, labels)):
            entry = (text_idx, example_idx, example, label)
            if label != outside_label:
                positives.append(entry)
                kept_entries.append(entry)
                continue

            difficulty = example.get("middle_token_count_difficulty")
            if difficulty in negatives_by_difficulty:
                negatives_by_difficulty[difficulty].append(entry)

    n_pos = len(positives)
    max_by_difficulty = {"hard": int(max_hard_negative_ratio * n_pos), "medium": int(max_medium_negative_ratio * n_pos), "easy": int(max_easy_negative_ratio * n_pos)}

    rng = random.Random(seed)
    for difficulty, entries in negatives_by_difficulty.items():
        max_negatives = max_by_difficulty[difficulty]
        if len(entries) > max_negatives:
            kept_entries.extend(rng.sample(entries, max_negatives))
        else:
            kept_entries.extend(entries)

    kept_entries.sort(key=lambda item: (item[0], item[1]))

    filtered_data = [[] for _ in data_prepared]
    filtered_labels = [[] for _ in relation_labels]
    for text_idx, _, example, label in kept_entries:
        filtered_data[text_idx].append(example)
        filtered_labels[text_idx].append(label)

    return filtered_data, filtered_labels


def remap_re_relation_labels(relation_labels: list[list[str | int]], label2id: dict[str, int], id2label: dict[int, str], outside_label: str = "O", data_prepared: list[list[dict[str, Any]]] | None = None, source_id2label: dict[int, str] | None = None, return_ids: bool = False) -> list[list[str | int]]:
    """
    Keeps only labels present in `label2id`, every other label becomes `outside_label`.
    Use this after `prepare_data` and before `construct_loader_re` when a checkpoint/classifier was trained with a smaller relation label set.

    Parameters
    ----------
        `relation_labels`: list[list[str | int]]
            - Entity or relation labels used by the model.
        `label2id`: dict[str, int]
            - Mapping from labels to numeric identifiers.
        `id2label`: dict[int, str]
            - Mapping from numeric identifiers to labels.
        `outside_label`: str
            - Label assigned to non-entity tokens.
        `data_prepared`: list[list[dict[str, Any]]] | None
            - Prepared examples grouped by source document.
        `source_id2label`: dict[int, str] | None
            - Entity or relation labels used by the model.
        `return_ids`: bool
            - Identifier values used to link or index records.

    Returns
    -------
        `list[list[str | int]]`
            - List of parsed, filtered, or generated values.
    """
    if outside_label not in label2id:
        raise ValueError(f"outside_label='{outside_label}' must exist in label2id.")

    normalized_id2label = {int(idx): label for idx, label in id2label.items()}
    source_id2label = ({int(idx): label for idx, label in source_id2label.items()} if source_id2label is not None else normalized_id2label)

    expected_label2id = {label: int(idx) for label, idx in label2id.items()}
    expected_id2label = {idx: label for label, idx in expected_label2id.items()}

    if expected_id2label != normalized_id2label:
        raise ValueError("label2id and id2label do not describe the same label mapping.")

    if data_prepared is not None and len(data_prepared) != len(relation_labels):
        raise ValueError("data_prepared and relation_labels must have the same number of texts.")

    allowed_labels = set(label2id)
    remapped_relation_labels = []

    for text_idx, labels_per_text in enumerate(relation_labels):
        if data_prepared is not None and len(data_prepared[text_idx]) != len(labels_per_text):
            raise ValueError(f"Text {text_idx} has {len(data_prepared[text_idx])} examples but {len(labels_per_text)} labels.")

        remapped_labels_per_text = []
        for label_idx, label in enumerate(labels_per_text):
            if isinstance(label, str):
                label_name = label
            else:
                label_name = source_id2label.get(int(label), outside_label)

            remapped_label = label_name if label_name in allowed_labels else outside_label
            if data_prepared is not None:
                data_prepared[text_idx][label_idx]["relation_label"] = remapped_label

            remapped_labels_per_text.append(label2id[remapped_label] if return_ids else remapped_label)

        remapped_relation_labels.append(remapped_labels_per_text)

    return remapped_relation_labels

def build_re_pair_window(text: str, left_entity: dict[str, Any], right_entity: dict[str, Any], tokenizer: Any, max_length: int, w: int, v: int, padding: bool = True) -> tuple[dict[str, Any] | None, str | None]:
    sentence_spans = sentence_spans_until_period(text)
    if not sentence_spans:
        return None, "Text has no sentence spans."

    left_sentence_idx = entity_sentence_index(left_entity, sentence_spans)
    right_sentence_idx = entity_sentence_index(right_entity, sentence_spans)
    if left_sentence_idx is None or right_sentence_idx is None:
        return None, "At least one entity could not be assigned to a sentence."

    marked_text, insertions = insert_pair_markers(text, left_entity, right_entity)

    current_v = max(0, int(v))
    current_w = max(0, int(w))
    final = None
    same_sentence = left_sentence_idx == right_sentence_idx
    middle_sentence_count = abs(right_sentence_idx - left_sentence_idx) - 1 if not same_sentence else 0

    while True:
        left_sent_indices, right_sent_indices, middle_sentence_count = selected_sentence_groups(left_sentence_idx, right_sentence_idx, len(sentence_spans), current_w, current_v)

        enc_full = tokenizer(marked_text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        full_ids = enc_full["input_ids"]
        full_offsets = enc_full["offset_mapping"]

        left_marked_start, left_marked_end = marked_sentence_bounds(left_sent_indices, sentence_spans, insertions)
        left_token_indices = token_indices_for_marked_bounds(full_offsets, left_marked_start, left_marked_end)
        left_ids = [full_ids[idx] for idx in left_token_indices]

        if right_sent_indices is None:
            right_marked_start = None
            right_marked_end = None
            right_token_indices = None
            right_ids = None
            total_length = len(build_re_input_ids_with_one_sep(tokenizer, left_ids))
        else:
            right_marked_start, right_marked_end = marked_sentence_bounds(right_sent_indices, sentence_spans, insertions)
            right_token_indices = token_indices_for_marked_bounds(full_offsets, right_marked_start, right_marked_end)
            right_ids = [full_ids[idx] for idx in right_token_indices]
            total_length = len(build_re_input_ids_with_one_sep(tokenizer, left_ids, right_ids))

        if total_length <= max_length:
            final = (left_sent_indices, right_sent_indices, left_marked_start, left_marked_end, right_marked_start, right_marked_end, full_ids, full_offsets, left_token_indices, right_token_indices, current_w, current_v, middle_sentence_count)
            break

        if current_v > 0:
            current_v -= 1
            continue
        if current_w > 0:
            current_w -= 1
            continue

        return None, "Pair could not be computed with v=0 and w=0 because the entity sentences exceed max_length."

    (left_sent_indices, right_sent_indices, left_marked_start, left_marked_end, right_marked_start, right_marked_end, full_ids, full_offsets, left_token_indices, right_token_indices, final_w, final_v, middle_sentence_count) = final

    left_ids = [full_ids[idx] for idx in left_token_indices]
    if right_token_indices is None:
        right_ids = None
        input_ids = build_re_input_ids_with_one_sep(tokenizer, left_ids)
    else:
        right_ids = [full_ids[idx] for idx in right_token_indices]
        input_ids = build_re_input_ids_with_one_sep(tokenizer, left_ids, right_ids)
    attention_mask = [1] * len(input_ids)
    original_position_ids, position_ids = build_position_arrays_for_special_tokens(input_ids, list(left_ids) + list(right_ids or []), list(left_token_indices) + list(right_token_indices or []))

    if padding:
        pad_length = max_length - len(input_ids)
        if pad_length < 0:
            return None, "Pair could not be computed because the selected tokens exceed max_length."
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_length
        attention_mask = attention_mask + [0] * pad_length
        original_position_ids = original_position_ids + [None] * pad_length
        position_ids = position_ids + [-1] * pad_length

    marker_positions = {}
    marker_token_ids = {insertion["text"]: tokenizer.convert_tokens_to_ids(insertion["text"]) for insertion in insertions}
    for pos, input_id in enumerate(input_ids):
        for marker, marker_id in marker_token_ids.items():
            if marker_id is not None and int(input_id) == int(marker_id):
                marker_positions.setdefault(marker, []).append(pos)

    required_markers = [
        make_entity_marker("S", str(left_entity["label"]), closing=False),
        make_entity_marker("S", str(left_entity["label"]), closing=True),
        make_entity_marker("O", str(right_entity["label"]), closing=False),
        make_entity_marker("O", str(right_entity["label"]), closing=True),
    ]
    missing_markers = [marker for marker in required_markers if not marker_positions.get(marker)]
    if missing_markers:
        return None, f"Pair could not be computed because required entity markers are missing from the selected tokens: {missing_markers}."

    subject_end_marker = make_entity_marker("S", str(left_entity["label"]), closing=True)
    object_start_marker = make_entity_marker("O", str(right_entity["label"]), closing=False)
    subject_end_pos = marker_positions.get(subject_end_marker, [-1])[0]
    object_start_pos = marker_positions.get(object_start_marker, [-1])[0]
    middle_token_count = max(0, int(object_start_pos) - int(subject_end_pos) - 1) if subject_end_pos >= 0 and object_start_pos >= 0 else 0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "max_length": max_length,
        "pad_token_id": tokenizer.pad_token_id,
        "marker_positions": marker_positions,
        "middle_token_count": middle_token_count,
        "w": final_w,
        "v": final_v,
        "left_entity": left_entity,
        "right_entity": right_entity,

        ##############################
        # "text": marked_text[left_marked_start:left_marked_end] if right_marked_start is None else marked_text[left_marked_start:left_marked_end] + tokenizer.sep_token + marked_text[right_marked_start:right_marked_end],
        # "marked_text": marked_text[left_marked_start:left_marked_end] if right_marked_start is None else marked_text[left_marked_start:left_marked_end] + tokenizer.sep_token + marked_text[right_marked_start:right_marked_end],
        # "marked_text_a": marked_text[left_marked_start:left_marked_end],
        # "marked_text_b": None if right_marked_start is None else marked_text[right_marked_start:right_marked_end],
        # "full_marked_text": marked_text,
        # "selected_sentence_indices": left_sent_indices if right_sent_indices is None else left_sent_indices + right_sent_indices,
        # "selected_sentence_indices_a": left_sent_indices,
        # "selected_sentence_indices_b": right_sent_indices,
        # "middle_sentence_count": middle_sentence_count,
        # "subject_entity": left_entity,
        # "object_entity": right_entity,
        # "marked_char_start": left_marked_start,
        # "marked_char_end": left_marked_end if right_marked_end is None else right_marked_end,
        # "marked_char_start_a": left_marked_start,
        # "marked_char_end_a": left_marked_end,
        # "marked_char_start_b": right_marked_start,
        # "marked_char_end_b": right_marked_end,
        # "full_offset_mapping": full_offsets,
        # "selected_full_token_indices": left_token_indices if right_token_indices is None else left_token_indices + right_token_indices,
        # "selected_full_token_indices_a": left_token_indices,
        # "selected_full_token_indices_b": right_token_indices,
        ##############################
    }, None

def initialize_re_model(ctx: "PipelineContext", encoder:Any, num_labels: int, middle_token_count_bin_embeddings: int, checkpoint_name: str|None, base_encoder_name: str = None, tokenizer: Any|None = None) -> REClassifier:
    """
    Initializes the relation-extraction model and its entity markers.

    Parameters
    ----------
        `ctx`: 'PipelineContext'
            - Pipeline context containing runtime configuration and device information.
        `encoder`: Any
            - Encoder component used by the classifier.
        `num_labels`: int
            - Number of output labels in the classifier.
        `middle_token_count_bin_embeddings`: int
            - Identifier values used to link or index records.
        `checkpoint_name`: str | None
            - Checkpoint file or registry name to restore.
        `base_encoder_name`: str
            - Base encoder used when rebuilding the tokenizer or model.
        `tokenizer`: Any | None
            - Tokenizer used to encode text and obtain offsets.

    Returns
    -------
        `REClassifier`
            - Initialized model or processing component.
    """
    if base_encoder_name is None:
        encoder_use = encoder
    else:
        encoder_use = AutoModel.from_pretrained(base_encoder_name).to(device)
        encoder_use.resize_token_embeddings(len(tokenizer))

    model = REClassifier(encoder=encoder_use, num_labels=num_labels, middle_token_count_bin_embeddings=middle_token_count_bin_embeddings).to(ctx.device)
    if checkpoint_name is not None:
        checkpoint = read_torch_checkpoint(ctx, checkpoint_name)
        model.load_state_dict(checkpoint)

    return model
