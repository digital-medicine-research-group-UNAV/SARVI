from __future__ import annotations

import torch
import numpy as np
import pandas as pd
from typing import Any

from .ann_utils import (
    normalize_ann_lines
)

def remap_ids(x: Any, id_no_hs_to_id_hs: dict):
    """
    Remaps prediction IDs using the supplied ID mapping.

    Parameters
    ----------
        `x`: Any
            - Prediction value or collection to remap.
        `id_no_hs_to_id_hs`: dict
            - Mapping from non-hierarchical IDs to hierarchical IDs.

    Returns
    -------
        `Any`
            - Normalized or parsed representation of the input.
    """
    if isinstance(x, int):
        return id_no_hs_to_id_hs[x]
    elif isinstance(x, list):
        return [remap_ids(item, id_no_hs_to_id_hs) for item in x]
    elif isinstance(x, tuple):
        return tuple(remap_ids(item, id_no_hs_to_id_hs) for item in x)
    elif isinstance(x, dict):
        return {k: remap_ids(v, id_no_hs_to_id_hs) for k, v in x.items()}
    else:
        return x

def find_file_column(df: pd.DataFrame) -> str | None:
    """
    Finds the dataframe column containing source filenames.

    Parameters
    ----------
        `df`: pd.DataFrame
            - Input dataframe containing the records to process.

    Returns
    -------
        `str | None`
            - Matching value or collection, when available.
    """
    for column in ("archivo_origen", "Original File", "file_name", "File"):
        if column in df.columns:
            return column
    return None

def annotation_row_for_data_row(data_row: pd.Series, data_ann: pd.DataFrame, row_pos: int):
    """
    Finds the annotation row corresponding to one data row.

    Parameters
    ----------
        `data_row`: pd.Series
            - Argument controlling data row.
        `data_ann`: pd.DataFrame
            - Annotation data used to align entities, attributes, or relations.
        `row_pos`: int
            - Position of the data row in the annotation table.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    file_col = find_file_column(data_ann)
    data_file_col = find_file_column(pd.DataFrame([data_row]))

    if file_col is not None and data_file_col is not None:
        matches = data_ann[data_ann[file_col] == data_row[data_file_col]]
        if len(matches) > 0:
            return matches.iloc[0]

    return data_ann.iloc[row_pos]

def has_value(value) -> bool:
    """
    Processes has value for use by the pipeline.

    Parameters
    ----------
        `value`: Any
            - Value to validate or normalize.

    Returns
    -------
        `bool`
            - Boolean indicating whether the condition is satisfied.
    """
    if isinstance(value, (list, tuple)):
        return True
    return not pd.isna(value)

def extract_brat_note_codes(value) -> dict[str, str]:
    """
    Extracts ICD codes from BRAT note annotations.

    Parameters
    ----------
        `value`: Any
            - Value to validate or normalize.

    Returns
    -------
        `dict[str, str]`
            - Mapping containing the processed values.
    """
    note_codes = {}

    for line in normalize_ann_lines(value):
        parts = str(line).split("\t")
        if len(parts) < 3:
            continue

        note_info = parts[1].split()
        if len(note_info) < 2 or note_info[0] != "AnnotatorNotes":
            continue

        target_id = note_info[1]
        code = parts[2].strip().upper()
        confidence = ""
        judger = ""
        if len(parts) >= 6:
            confidence = parts[3].strip().upper()
            judger = parts[5].strip().split(" ")[-1]
        elif len(parts) >= 4:
            confidence = parts[3].strip().upper()
        if code:
            note_codes[target_id] = (code, confidence, judger)

    return note_codes

def value_to_int(value):
    """
    Processes value to int for use by the pipeline.

    Parameters
    ----------
        `value`: Any
            - Value to validate or normalize.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    if hasattr(value, "item") and callable(value.item):
        return int(value.item())
    return int(value)

def normalize_icd_code_label(label):
    """
    Normalizes the representation of an ICD code label.

    Parameters
    ----------
        `label`: Any
            - Entity or relation label.

    Returns
    -------
        `Any`
            - Normalized or parsed representation of the input.
    """
    if isinstance(label, str):
        return label.upper()
    return label

def flatten_optional_nested(values):
    """
    Flattens an optional one-level nested collection.

    Parameters
    ----------
        `values`: Any
            - Values to flatten or normalize.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    if values is None:
        return None
    if len(values) == 0:
        return []
    if all(isinstance(item, (list, tuple)) for item in values):
        return [value for item in values for value in item]
    return list(values)

def as_prediction_id_list(predictions):
    """
    Converts prediction output into a list of integer IDs.

    Parameters
    ----------
        `predictions`: Any
            - Prediction IDs or model outputs.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    if predictions is None:
        return []
    if torch.is_tensor(predictions):
        return [int(value) for value in predictions.detach().cpu().reshape(-1).tolist()]
    if isinstance(predictions, np.ndarray):
        return [int(value) for value in predictions.reshape(-1).tolist()]
    return [int(value.item() if hasattr(value, "item") else value) for value in predictions]

def remap_icd_prediction_ids(value, id_map: dict):
    """
    Remaps ICD prediction IDs in tensors or nested collections.

    Parameters
    ----------
        `value`: Any
            - Value to validate or normalize.
        `id_map`: dict
            - Mapping from old prediction IDs to new IDs.

    Returns
    -------
        `Any`
            - Normalized or parsed representation of the input.
    """
    if torch.is_tensor(value):
        if value.numel() == 1:
            scalar = value.detach().cpu().item()
            if isinstance(scalar, (int, np.integer)):
                return id_map.get(int(scalar), int(scalar))
        return value
    if isinstance(value, np.ndarray):
        return [remap_icd_prediction_ids(item, id_map) for item in value.tolist()]
    if isinstance(value, (int, np.integer)):
        return id_map.get(int(value), int(value))
    if isinstance(value, list):
        return [remap_icd_prediction_ids(item, id_map) for item in value]
    if isinstance(value, tuple):
        return tuple(remap_icd_prediction_ids(item, id_map) for item in value)
    if isinstance(value, dict):
        return {key: remap_icd_prediction_ids(item, id_map) for key, item in value.items()}
    return value
