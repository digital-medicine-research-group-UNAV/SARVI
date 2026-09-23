from __future__ import annotations

import torch
import numpy as np
from typing import Any, Optional, TYPE_CHECKING

from transformers import AutoTokenizer

if TYPE_CHECKING:
    from ....models.neural_networks import Span_NERClassifier, BIO_NERClassifier

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """
    Moves tensor values in a batch to the selected device.
    Non-tensor values, such as file_names, are kept unchanged.

    Parameters
    ----------
        `batch`: dict
            - Argument controlling batch.
        `device`: torch.device
            - Device used for tensor computation.

    Returns
    -------
        `dict`
            - Mapping containing the processed values.
    """
    batch_out = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            batch_out[key] = value.to(device)
        else:
            batch_out[key] = value

    return batch_out

def checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]

    return checkpoint

def copy_matching_classifier_rows(model: Span_NERClassifier | BIO_NERClassifier, state_dict: dict[str, torch.Tensor], source_label2id: dict[str, int] | None = None, target_label2id: dict[str, int] | None = None) -> None:
    """
    Copies classifier weights for labels shared by two label mappings.

    Parameters
    ----------
        `model`: Span_NERClassifier | BIO_NERClassifier
            - Model used to perform the requested inference or initialization.
        `state_dict`: dict[str, torch.Tensor]
            - Argument controlling state dict.
        `source_label2id`: dict[str, int] | None
            - Entity or relation labels used by the model.
        `target_label2id`: dict[str, int] | None
            - Entity or relation labels used by the model.

    Returns
    -------
        `None`
            - No value; the operation updates its input in place.
    """
    weight_key = "classifier.4.weight"
    bias_key = "classifier.4.bias"

    if weight_key not in state_dict:
        return

    source_weight = state_dict[weight_key]
    source_bias = state_dict.get(bias_key)
    target_layer = model.classifier[4]

    if source_weight.shape == target_layer.weight.shape:
        return

    with torch.no_grad():
        if source_label2id is not None and target_label2id is not None:
            shared_labels = set(source_label2id) & set(target_label2id)
            for label in shared_labels:
                source_idx = int(source_label2id[label])
                target_idx = int(target_label2id[label])
                if source_idx >= source_weight.size(0) or target_idx >= target_layer.weight.size(0):
                    continue

                target_layer.weight[target_idx].copy_(source_weight[source_idx].to(target_layer.weight.device))
                if source_bias is not None and source_idx < source_bias.size(0):
                    target_layer.bias[target_idx].copy_(source_bias[source_idx].to(target_layer.bias.device))
            return

        shared_rows = min(source_weight.size(0), target_layer.weight.size(0))
        target_layer.weight[:shared_rows].copy_(source_weight[:shared_rows].to(target_layer.weight.device))
        if source_bias is not None:
            target_layer.bias[:shared_rows].copy_(source_bias[:shared_rows].to(target_layer.bias.device))

def load_checkpoint_with_resized_head(model: Span_NERClassifier | BIO_NERClassifier, state_dict: dict[str, torch.Tensor], source_label2id: dict[str, int] | None = None, target_label2id: dict[str, int] | None = None) -> None:
    """
    Loads checkpoint weights while adapting a classifier head to its labels.

    Parameters
    ----------
        `model`: Span_NERClassifier | BIO_NERClassifier
            - Model used to perform the requested inference or initialization.
        `state_dict`: dict[str, torch.Tensor]
            - Argument controlling state dict.
        `source_label2id`: dict[str, int] | None
            - Entity or relation labels used by the model.
        `target_label2id`: dict[str, int] | None
            - Entity or relation labels used by the model.

    Returns
    -------
        `None`
            - No value; the operation updates its input in place.
    """
    model_state = model.state_dict()
    compatible_state = {key: value for key, value in state_dict.items() if key in model_state and model_state[key].shape == value.shape}

    model.load_state_dict(compatible_state, strict=False)
    copy_matching_classifier_rows(model, state_dict, source_label2id=source_label2id, target_label2id=target_label2id)

def detach_to_cpu(value: Any) -> Any:
    """
    Detaches a tensor or nested value and moves it to CPU.

    Parameters
    ----------
        `value`: Any
            - Value to validate or normalize.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: detach_to_cpu(item) for key, item in value.items() if key != "encoder_outputs"}
    if isinstance(value, list):
        return [detach_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_to_cpu(item) for item in value)
    return value

def detach_output(output) -> dict[Any, torch.Tensor | Any]:
    """
    Processes detach output for use by the pipeline.

    Parameters
    ----------
        `output`: Any
            - Model output or predictions to transform.

    Returns
    -------
        `dict[Any, torch.Tensor | Any]`
            - Mapping containing the processed values.
    """
    return detach_to_cpu(output)

def tensor_to_python(value: Any, squeeze_single: bool = False) -> Any:
    """
    Converts tensors and nested values into Python-native objects.

    Parameters
    ----------
        `value`: Any
            - Value to validate or normalize.
        `squeeze_single`: bool
            - Argument controlling squeeze single.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if squeeze_single and value.numel() == 1:
            return value.reshape(-1)[0].item()
        if value.dim() == 0:
            return value.item()
        return value.tolist()

    if isinstance(value, np.ndarray):
        if squeeze_single and value.size == 1:
            return value.reshape(-1)[0].item()
        if value.ndim == 0:
            return value.item()
        return value.tolist()

    if isinstance(value, list):
        converted = [tensor_to_python(item, squeeze_single=squeeze_single) for item in value]
        return converted[0] if squeeze_single and len(converted) == 1 else converted

    if isinstance(value, tuple):
        converted = tuple(tensor_to_python(item, squeeze_single=squeeze_single) for item in value)
        return converted[0] if squeeze_single and len(converted) == 1 else converted

    if isinstance(value, dict):
        return {key: tensor_to_python(item, squeeze_single=squeeze_single) for key, item in value.items()}

    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass

    return value

def make_entity_marker(role: str, label: str, closing: bool = False) -> str:
    """
    Creates an opening or closing marker for an entity label.

    Parameters
    ----------
        `role`: str
            - Entity marker role, such as opening or closing.
        `label`: str
            - Entity or relation label.
        `closing`: bool
            - Argument controlling closing.

    Returns
    -------
        `str`
            - Derived value produced by the operation.
    """
    prefix = "/" if closing else ""
    return f"<{prefix}{role}:{label}>"

def entity_marker_tokens(labels: list[str] | tuple[str, ...] | set[str], type_marker:str) -> list[str]:
    """
    Returns all marker tokens required by the supplied entity labels.

    Parameters
    ----------
        `labels`: list[str] | tuple[str, ...] | set[str]
            - Entity or relation labels used by the model.
        `type_marker`: str
            - Marker type inserted around entities.

    Returns
    -------
        `list[str]`
            - List of parsed, filtered, or generated values.
    """
    markers = []
    for label in sorted(set(labels)):
        if type_marker == "re":
            for role in ("S", "O"):
                markers.append(make_entity_marker(role, label, closing=False))
                markers.append(make_entity_marker(role, label, closing=True))
        elif type_marker == "att":
            markers.append(make_entity_marker("S", label, closing=False))
            markers.append(make_entity_marker("S", label, closing=True))
    return markers

def ensure_entity_marker_tokens(tokenizer: Any|None, labels: list[str] | tuple[str, ...] | set[str], type_marker: str, base_encoder_name: str|None = None) -> list[str]|Any:
    """
    Adds required entity markers to the tokenizer and returns the updated tokenizer.

    Parameters
    ----------
        `tokenizer`: Any | None
            - Tokenizer used to encode text and obtain offsets.
        `labels`: list[str] | tuple[str, ...] | set[str]
            - Entity or relation labels used by the model.
        `type_marker`: str
            - Marker type inserted around entities.
        `base_encoder_name`: str | None
            - Base encoder used when rebuilding the tokenizer or model.

    Returns
    -------
        `list[str] | Any`
            - List of parsed, filtered, or generated values.
    """
    if base_encoder_name is None:
        tokenizer_use = tokenizer
    else:
        tokenizer_use = AutoTokenizer.from_pretrained(base_encoder_name, trim_offsets=False, use_fast=True)

    markers = entity_marker_tokens(labels, type_marker)
    existing_specials = list(getattr(tokenizer_use, "additional_special_tokens", []) or [])
    merged_specials = existing_specials + [marker for marker in markers if marker not in existing_specials]
    if len(merged_specials) != len(existing_specials):
        tokenizer_use.add_special_tokens({"additional_special_tokens": merged_specials})

    if base_encoder_name is None:
        return markers
    else:
        return tokenizer_use


#####   FUNCTION REMOVED IN TRANSFORMERS 5.X -> COPIED FROM THE LATEST 4.X VERSION FOR SAME USSAGE
def build_inputs_with_special_tokens(token_ids_0: list[int], token_ids_1: Optional[list[int]] = None, cls_token_id: int = 0, sep_token_id: int = 2) -> list[int]:
    """
    Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
    adding special tokens. An XLM-RoBERTa sequence has the following format:

    - single sequence: `<s> X </s>`
    - pair of sequences: `<s> A </s></s> B </s>`

    Args:
        token_ids_0 (`list[int]`):
            List of IDs to which the special tokens will be added.
        token_ids_1 (`list[int]`, *optional*):
            Optional second list of IDs for sequence pairs.

    Returns:
        `list[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

    """

    if token_ids_1 is None:
        return [cls_token_id] + token_ids_0 + [sep_token_id]
    cls = [cls_token_id]
    sep = [sep_token_id]
    return cls + token_ids_0 + sep + sep + token_ids_1 + sep
