from __future__ import annotations

import random
import pandas as pd
import torch
from typing import Any, TYPE_CHECKING

from transformers import AutoModel

from ...data_io.reader import (
    read_torch_checkpoint
)
from ...models.neural_networks import (
    ATTClassifier
)
from .utils.ann_utils import (
    normalize_ann_lines, parse_brat_text_boundaries
)
from .utils.re_utils import (
    build_position_arrays_for_special_tokens,
    entity_sentence_index,
    marked_sentence_bounds,
    pad_re_input,
    sentence_spans_until_period,
    token_indices_for_marked_bounds,
)
from .utils.nn_utils import (
    make_entity_marker
)

if TYPE_CHECKING:
    from ...models.schemas import PipelineContext

###
device = "cuda" if torch.cuda.is_available() else "cpu"
###

def parse_att_target_entities(ann_info: Any, entity_label: str, t_col: str = "T") -> list[dict[str, Any]]:
    """
    Extracts target entities for attribute prediction from annotation data.

    Parameters
    ----------
        `ann_info`: Any
            - Annotation row or object to parse.
        `entity_label`: str
            - Label identifying the target entity type.
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
        if label != entity_label:
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


def parse_att_attributes(ann_info: Any, a_col: str = "A") -> list[dict[str, Any]]:
    """
    Extracts attribute annotations and their target entity identifiers.

    Parameters
    ----------
        `ann_info`: Any
            - Annotation row or object to parse.
        `a_col`: str
            - BRAT attribute column.

    Returns
    -------
        `list[dict[str, Any]]`
            - List of parsed, filtered, or generated values.
    """
    if ann_info is None:
        return []

    if isinstance(ann_info, pd.Series) or isinstance(ann_info, dict):
        a_lines = normalize_ann_lines(ann_info.get(a_col))
    else:
        a_lines = normalize_ann_lines(getattr(ann_info, a_col, None))

    attributes = []
    for line in a_lines:
        parts = str(line).split("\t")
        if len(parts) < 2:
            continue

        attr_id = parts[0]
        pieces = parts[1].split()
        if len(pieces) < 2:
            continue

        attributes.append({
            "id": attr_id,
            "label": pieces[0],
            "target_id": pieces[1],
            "value": pieces[2] if len(pieces) > 2 else pieces[0],
            "line": str(line),
        })

    return attributes


def attribute_label_for_entity(entity: dict[str, Any], attributes: list[dict[str, Any]], attribute_label: str | None = None, outside_label: str = "O") -> str:
    """
    Finds the requested attribute label associated with an entity.

    Parameters
    ----------
        `entity`: dict[str, Any]
            - Entity record being processed.
        `attributes`: list[dict[str, Any]]
            - Attribute records associated with the entities.
        `attribute_label`: str | None
            - Attribute label assigned to the entity.
        `outside_label`: str
            - Label assigned to non-entity tokens.

    Returns
    -------
        `str`
            - Derived value produced by the operation.
    """
    entity_ids = {str(entity.get("id")), str(entity.get("original_id", entity.get("id")))}

    for attribute in attributes:
        if str(attribute.get("target_id")) not in entity_ids:
            continue
        if attribute_label is not None and str(attribute.get("label")) != attribute_label:
            continue
        return str(attribute.get("value", attribute.get("label")))

    return outside_label


def insert_att_markers(text: str, entity: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """
    Inserts attribute markers around an entity and records their positions.

    Parameters
    ----------
        `text`: str
            - Text containing the entity or span.
        `entity`: dict[str, Any]
            - Entity record being processed.

    Returns
    -------
        `tuple[str, list[dict[str, Any]]]`
            - List of parsed, filtered, or generated values.
    """
    insertions = [
        {
            "orig_pos": int(entity["start"]),
            "text": make_entity_marker("S", str(entity["label"]), closing=False),
            "kind": "start",
            "role": "S",
            "entity_id": entity.get("id"),
            "label": entity.get("label"),
        },
        {
            "orig_pos": int(entity["end"]),
            "text": make_entity_marker("S", str(entity["label"]), closing=True),
            "kind": "end",
            "role": "S",
            "entity_id": entity.get("id"),
            "label": entity.get("label"),
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


def selected_att_sentence_indices(entity_sentence_idx: int, num_sentences: int, w: int) -> list[int]:
    """
    Selects the sentence window used to classify an attribute.

    Parameters
    ----------
        `entity_sentence_idx`: int
            - Sentence index containing the target entity.
        `num_sentences`: int
            - Total number of sentences in the document.
        `w`: int
            - Number of neighboring sentences included in the window.

    Returns
    -------
        `list[int]`
            - List of parsed, filtered, or generated values.
    """
    current_w = max(0, int(w))
    start = max(0, entity_sentence_idx - current_w)
    end = min(num_sentences, entity_sentence_idx + current_w + 1)
    return list(range(start, end))


def build_att_entity_window(text: str, entity: dict[str, Any], tokenizer: Any, max_length: int, w: int, padding: bool = True) -> tuple[dict[str, Any] | None, str | None]:
    """
    Builds a tokenized sentence window centered on an entity for attribute classification.

    Parameters
    ----------
        `text`: str
            - Text containing the entity or span.
        `entity`: dict[str, Any]
            - Entity record being processed.
        `tokenizer`: Any
            - Tokenizer used to encode text and obtain offsets.
        `max_length`: int
            - Maximum encoded sequence length.
        `w`: int
            - Number of neighboring sentences included in the window.
        `padding`: bool
            - Whether encoded sequences are padded to a common length.

    Returns
    -------
        `tuple[dict[str, Any] | None, str | None]`
            - Mapping containing the processed values.
    """
    sentence_spans = sentence_spans_until_period(text)
    if not sentence_spans:
        return None, "Text has no sentence spans."

    sentence_idx = entity_sentence_index(entity, sentence_spans)
    if sentence_idx is None:
        return None, "Entity could not be assigned to a sentence."

    marked_text, insertions = insert_att_markers(text, entity)
    current_w = max(0, int(w))
    final = None

    while True:
        sentence_indices = selected_att_sentence_indices(sentence_idx, len(sentence_spans), current_w)

        enc_full = tokenizer(marked_text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        full_ids = enc_full["input_ids"]
        full_offsets = enc_full["offset_mapping"]

        marked_start, marked_end = marked_sentence_bounds(sentence_indices, sentence_spans, insertions)
        token_indices = token_indices_for_marked_bounds(full_offsets, marked_start, marked_end)
        selected_ids = [full_ids[idx] for idx in token_indices]

        input_ids = [int(tokenizer.cls_token_id)] + list(selected_ids) + [int(tokenizer.sep_token_id)]
        if len(input_ids) <= max_length:
            final = (sentence_indices, marked_start, marked_end, full_ids, token_indices, current_w)
            break

        if current_w > 0:
            current_w -= 1
            continue

        return None, "Entity window could not be computed with w=0 because the entity sentence exceeds max_length."

    sentence_indices, marked_start, marked_end, full_ids, token_indices, final_w = final
    selected_ids = [full_ids[idx] for idx in token_indices]
    input_ids = [int(tokenizer.cls_token_id)] + list(selected_ids) + [int(tokenizer.sep_token_id)]
    attention_mask = [1] * len(input_ids)
    original_position_ids, position_ids = build_position_arrays_for_special_tokens(input_ids, selected_ids, token_indices)

    if padding:
        pad_length = max_length - len(input_ids)
        if pad_length < 0:
            return None, "Entity window could not be computed because the selected tokens exceed max_length."
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
        make_entity_marker("S", str(entity["label"]), closing=False),
        make_entity_marker("S", str(entity["label"]), closing=True),
    ]
    missing_markers = [marker for marker in required_markers if not marker_positions.get(marker)]
    if missing_markers:
        return None, f"Entity window could not be computed because required markers are missing from the selected tokens: {missing_markers}."

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "max_length": max_length,
        "pad_token_id": tokenizer.pad_token_id,
        "marker_positions": marker_positions,
        "w": final_w,
        "entity": entity,
    }, None


def att_marker_position_ids(example: dict[str, Any]) -> list[int]:
    """
    Creates marker position identifiers for an attribute-classification example.

    Parameters
    ----------
        `example`: dict[str, Any]
            - Argument controlling example.

    Returns
    -------
        `list[int]`
            - List of parsed, filtered, or generated values.
    """
    entity = example.get("entity", {})
    marker_positions = example.get("marker_positions", {})

    return [
        int(marker_positions.get(f"<S:{entity.get('label')}>", [-1])[0]),
        int(marker_positions.get(f"</S:{entity.get('label')}>", [-1])[0]),
    ]


def undersample_att_negatives(data_prepared: list[list[dict[str, Any]]], attribute_labels: list[list[str | int]], outside_label: str | int = "O", max_negative_ratio: int | float = 1, seed: int = 8) -> tuple[list[list[dict[str, Any]]], list[list[str | int]]]:
    """
    Keeps all non-outside examples and samples outside-label examples by ratio.

    `max_negative_ratio` is the maximum number of negative examples to keep per
    non-negative example. For example, ratio=2 keeps at most twice as many `O`
    examples as all non-`O` examples combined.
    """
    if len(data_prepared) != len(attribute_labels):
        raise ValueError("data_prepared and attribute_labels must have the same number of texts.")
    if max_negative_ratio < 0:
        raise ValueError("max_negative_ratio must be >= 0.")

    positives = []
    negatives = []

    for text_idx, (examples, labels) in enumerate(zip(data_prepared, attribute_labels)):
        if len(examples) != len(labels):
            raise ValueError(f"Text {text_idx} has {len(examples)} examples but {len(labels)} labels.")

        for example_idx, (example, label) in enumerate(zip(examples, labels)):
            entry = (text_idx, example_idx, example, label)
            if label == outside_label:
                negatives.append(entry)
            else:
                positives.append(entry)

    max_negatives = int(max_negative_ratio * len(positives))
    rng = random.Random(seed)
    kept_entries = positives.copy()
    if len(negatives) > max_negatives:
        kept_entries.extend(rng.sample(negatives, max_negatives))
    else:
        kept_entries.extend(negatives)

    kept_entries.sort(key=lambda item: (item[0], item[1]))

    filtered_data = [[] for _ in data_prepared]
    filtered_labels = [[] for _ in attribute_labels]
    for text_idx, _, example, label in kept_entries:
        filtered_data[text_idx].append(example)
        filtered_labels[text_idx].append(label)

    return filtered_data, filtered_labels


def remap_att_attribute_labels(attribute_labels: list[list[str | int]], label2id: dict[str, int], id2label: dict[int, str], outside_label: str = "O", data_prepared: list[list[dict[str, Any]]] | None = None, source_id2label: dict[int, str] | None = None, return_ids: bool = False) -> list[list[str | int]]:
    """
    Keeps only labels present in `label2id`, every other label becomes `outside_label`.

    Use this after `prepare_data` and before `construct_loader_att` when a checkpoint/classifier was trained with a smaller attribute label set.
    """
    if outside_label not in label2id:
        raise ValueError(f"outside_label='{outside_label}' must exist in label2id.")

    normalized_id2label = {int(idx): label for idx, label in id2label.items()}
    source_id2label = ({int(idx): label for idx, label in source_id2label.items()} if source_id2label is not None else normalized_id2label)

    expected_label2id = {label: int(idx) for label, idx in label2id.items()}
    expected_id2label = {idx: label for label, idx in expected_label2id.items()}

    if expected_id2label != normalized_id2label:
        raise ValueError("label2id and id2label do not describe the same label mapping.")

    if data_prepared is not None and len(data_prepared) != len(attribute_labels):
        raise ValueError("data_prepared and attribute_labels must have the same number of texts.")

    allowed_labels = set(label2id)
    remapped_attribute_labels = []

    for text_idx, labels_per_text in enumerate(attribute_labels):
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
                data_prepared[text_idx][label_idx]["attribute_label"] = remapped_label

            remapped_labels_per_text.append(label2id[remapped_label] if return_ids else remapped_label)

        remapped_attribute_labels.append(remapped_labels_per_text)

    return remapped_attribute_labels


remap_att_relation_labels = remap_att_attribute_labels

def att_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Reduces negative attribute examples while retaining the requested negative-to-positive ratio.

    Parameters
    ----------
        `data_prepared`: list[list[dict[str, Any]]]
            - Prepared examples grouped by source document.
        `attribute_labels`: list[list[str | int]]
            - Attribute labels aligned with the prepared examples.
        `outside_label`: str | int
            - Label assigned to non-entity tokens.
        `max_negative_ratio`: int | float
            - Maximum number of negative examples per positive example.
        `seed`: int
            - Seed used to make random sampling reproducible.

    Returns
    -------
        `tuple[list[list[dict[str, Any]]], list[list[str | int]]]`
            - List of parsed, filtered, or generated values.
    """
    input_ids = []
    attention_mask = []
    marker_position_ids = []
    attribute_labels = []
    example_to_text = []
    example_local_index = []
    text_indices = []
    file_names = []

    labels_present = any(item.get("attribute_labels") is not None for item in batch)

    for batch_text_idx, item in enumerate(batch):
        examples = item["examples"]
        labels = item.get("attribute_labels")

        text_indices.append(item["text_index"])
        file_names.append(item["file_name"])

        for example_idx, example in enumerate(examples):
            input_ids_padded, attention_mask_padded = pad_re_input(example)

            input_ids.append(input_ids_padded)
            attention_mask.append(attention_mask_padded)
            marker_position_ids.append(att_marker_position_ids(example))
            example_to_text.append(batch_text_idx)
            example_local_index.append(example_idx)

            if labels is not None:
                attribute_labels.append(int(labels[example_idx]))

    if not input_ids:
        return {
            "input_ids": torch.empty((0, 0), dtype=torch.long),
            "attention_mask": torch.empty((0, 0), dtype=torch.long),
            "marker_position_ids": torch.empty((0, 2), dtype=torch.long),
            "example_to_text": torch.empty((0,), dtype=torch.long),
            "example_local_index": torch.empty((0,), dtype=torch.long),
            "text_indices": torch.tensor(text_indices, dtype=torch.long),
            "file_names": file_names,
            "attribute_labels": torch.empty((0,), dtype=torch.long) if labels_present else None,
        }

    batch_out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "marker_position_ids": torch.tensor(marker_position_ids, dtype=torch.long),
        "example_to_text": torch.tensor(example_to_text, dtype=torch.long),
        "example_local_index": torch.tensor(example_local_index, dtype=torch.long),
        "text_indices": torch.tensor(text_indices, dtype=torch.long),
        "file_names": file_names,
    }

    if labels_present:
        batch_out["attribute_labels"] = torch.tensor(attribute_labels, dtype=torch.long)
    else:
        batch_out["attribute_labels"] = None

    return batch_out


def initialize_att_model(ctx: "PipelineContext", encoder: Any, num_labels: int, checkpoint_name: str | None, base_encoder_name: str = None, tokenizer: Any|None = None) -> ATTClassifier:
    """
    Initializes the attribute classifier and optionally loads its encoder and checkpoint.

    Parameters
    ----------
        `ctx`: 'PipelineContext'
            - Pipeline context containing runtime configuration and device information.
        `encoder`: Any
            - Encoder component used by the classifier.
        `num_labels`: int
            - Number of output labels in the classifier.
        `checkpoint_name`: str | None
            - Checkpoint file or registry name to restore.
        `base_encoder_name`: str
            - Base encoder used when rebuilding the tokenizer or model.
        `tokenizer`: Any | None
            - Tokenizer used to encode text and obtain offsets.

    Returns
    -------
        `ATTClassifier`
            - Initialized model or processing component.
    """
    if base_encoder_name is None:
        encoder_use = encoder
    else:
        encoder_use = AutoModel.from_pretrained(base_encoder_name).to(device)
        encoder_use.resize_token_embeddings(len(tokenizer))

    model = ATTClassifier(encoder=encoder_use, num_labels=num_labels).to(ctx.device)
    if checkpoint_name is not None:
        checkpoint = read_torch_checkpoint(ctx, checkpoint_name)
        model.load_state_dict(checkpoint)

    return model
