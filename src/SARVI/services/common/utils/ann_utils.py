import ast
import re
import pandas as pd

from torch.utils.data import DataLoader, SequentialSampler
from collections import defaultdict

from ....models.schemas import (
    Any
)
from .nn_utils import tensor_to_python

def normalize_ann_lines(value: Any) -> list[str]:
    """
    Converts annotation input into a clean list of annotation lines.

    Parameters
    ----------
        `value`: Any
            - Value to validate or normalize.

    Returns
    -------
        `list[str]`
            - List of parsed, filtered, or generated values.
    """
    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed]
        except Exception:
            pass

        return [value]

    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    return [str(value)]

def parse_brat_text_boundaries(boundary_text: str) -> list[tuple[int, int]]:
    """
    Parses BRAT text-boundary offsets into character spans.

    Parameters
    ----------
        `boundary_text`: str
            - Source text being processed.

    Returns
    -------
        `list[tuple[int, int]]`
            - List of parsed, filtered, or generated values.
    """
    spans = []

    for part in boundary_text.split(";"):
        pieces = part.strip().split()
        if len(pieces) < 2:
            continue
        spans.append((int(pieces[0]), int(pieces[1])))

    return spans

def ann_df_to_t_lines(df: pd.DataFrame, column: str = "T") -> list[str]:
    """
    Extracts BRAT text-bound annotation lines from an annotation dataframe.

    Parameters
    ----------
        `df`: pd.DataFrame
            - Input dataframe containing the records to process.
        `column`: str
            - Argument controlling column.

    Returns
    -------
        `list[str]`
            - List of parsed, filtered, or generated values.
    """
    if column not in df.columns:
        raise ValueError(f"Expected DataFrame with a '{column}' column.")

    lines = []

    for value in df[column]:
        lines.extend(normalize_ann_lines(value))

    return [str(line) for line in lines if str(line).startswith("T")]

def parse_brat_t_entities(df: pd.DataFrame, file_col: str = "archivo_origen", t_col: str = "T") -> list[dict[str, Any]]:
    """
    Parses BRAT text-bound entities from annotation rows.

    Parameters
    ----------
        `df`: pd.DataFrame
            - Input dataframe containing the records to process.
        `file_col`: str
            - Name of the source-file column.
        `t_col`: str
            - BRAT text-bound column.

    Returns
    -------
        `list[dict[str, Any]]`
            - List of parsed, filtered, or generated values.
    """
    if t_col not in df.columns:
        raise ValueError(f"Expected DataFrame with a '{t_col}' column.")

    entities = []

    for row_idx, row in df.iterrows():
        file_name = row.get(file_col, row_idx)

        for line in normalize_ann_lines(row.get(t_col)):
            if not str(line).startswith("T"):
                continue

            parts = str(line).split("\t")
            if len(parts) < 2:
                continue

            ann_id = parts[0]
            label_and_boundaries = parts[1].split(maxsplit=1)
            if len(label_and_boundaries) != 2:
                continue

            label, boundary_text = label_and_boundaries
            boundaries = parse_brat_text_boundaries(boundary_text)
            if not boundaries:
                continue

            entities.append({
                "id": ann_id,
                "file": file_name,
                "row_idx": row_idx,
                "label": label,
                "start": min(start for start, _ in boundaries),
                "end": max(end for _, end in boundaries),
                "char_spans": boundaries,
                "text": parts[2] if len(parts) > 2 else "",
                "line": str(line),
            })

    return entities

def normalize_original_texts(value: dict | pd.DataFrame | None) -> dict:
    """
    Converts original text input into a filename-to-text mapping.

    Parameters
    ----------
        `value`: dict | pd.DataFrame | None
            - Value to validate or normalize.

    Returns
    -------
        `dict`
            - Mapping containing the processed values.
    """
    if value is None:
        return {}
    if isinstance(value, pd.DataFrame):
        file_col = "archivo_origen" if "archivo_origen" in value.columns else "File"
        text_col = "Text" if "Text" in value.columns else "text"
        return dict(zip(value[file_col], value[text_col]))
    return dict(value)

def find_file_column(df: pd.DataFrame) -> str:
    """
    Finds the dataframe column containing source filenames.

    Parameters
    ----------
        `df`: pd.DataFrame
            - Input dataframe containing the records to process.

    Returns
    -------
        `str`
            - Matching value or collection, when available.
    """
    for column in ("archivo_origen", "Original File", "file_name", "File"):
        if column in df.columns:
            return column
    raise ValueError("Expected original_ann to contain a file column such as 'archivo_origen', 'file_name' or 'File'.")

def ann_id_prefix(line: str) -> str | None:
    """
    Processes ann id prefix for use by the pipeline.

    Parameters
    ----------
        `line`: str
            - Annotation line to add or parse.

    Returns
    -------
        `str | None`
            - Derived value produced by the operation.
    """
    if not line:
        return None
    if line[0] == "#":
        return "#"
    return line[0] if line[0].isalpha() else None

def ann_id_number(line: str, prefix: str) -> int | None:
    """
    Processes ann id number for use by the pipeline.

    Parameters
    ----------
        `line`: str
            - Annotation line to add or parse.
        `prefix`: str
            - Argument controlling prefix.

    Returns
    -------
        `int | None`
            - Derived value produced by the operation.
    """
    if prefix == "#":
        match = re.match(r"#(\d+)\b", line)
    else:
        match = re.match(rf"{re.escape(prefix)}(\d+)\b", line)
    return int(match.group(1)) if match else None

def add_ann_line(ann_by_file: dict[str, list[str]], counters: dict[Any, defaultdict[str, int]], file_name: Any, line: str) -> None:
    """
    Processes add ann line for use by the pipeline.

    Parameters
    ----------
        `ann_by_file`: dict[str, list[str]]
            - Annotation data used to align entities, attributes, or relations.
        `counters`: dict[Any, defaultdict[str, int]]
            - Argument controlling counters.
        `file_name`: Any
            - Source filename used to locate the text.
        `line`: str
            - Annotation line to add or parse.

    Returns
    -------
        `None`
            - No value; the operation updates its input in place.
    """
    file_key = str(file_name)
    ann_by_file.setdefault(file_key, []).append(str(line))

    prefix = ann_id_prefix(str(line))
    if prefix is None:
        return

    line_id = ann_id_number(str(line), prefix)
    if line_id is not None:
        counters[file_key][prefix] = max(counters[file_key][prefix], line_id)

def base_ann_from_original(original_ann: pd.DataFrame | dict | None, ann_columns: tuple[str, ...] = ("T", "A", "R", "#")) -> tuple[dict[str, list[str]], dict[Any, defaultdict[str, int]]]:
    """
    Builds annotation lines and ID counters from original annotations.

    Parameters
    ----------
        `original_ann`: pd.DataFrame | dict | None
            - Original BRAT annotations.
        `ann_columns`: tuple[str, ...]
            - Annotation columns to preserve.

    Returns
    -------
        `tuple[dict[str, list[str]], dict[Any, defaultdict[str, int]]]`
            - List of parsed, filtered, or generated values.
    """
    ann_by_file = {}
    counters = defaultdict(lambda: defaultdict(int))

    if original_ann is None:
        return ann_by_file, counters

    if isinstance(original_ann, pd.DataFrame):
        file_col = find_file_column(original_ann)

        for row_idx, row in original_ann.iterrows():
            file_name = row.get(file_col, row_idx)
            for column in ann_columns:
                if column not in original_ann.columns:
                    continue
                for line in normalize_ann_lines(row.get(column)):
                    add_ann_line(ann_by_file, counters, file_name, line)

        return ann_by_file, counters

    for file_name, value in dict(original_ann).items():
        for line in normalize_ann_lines(value):
            add_ann_line(ann_by_file, counters, file_name, line)

    return ann_by_file, counters

def next_ann_id(counters: dict[Any, defaultdict[str, int]], file_name: Any, prefix: str) -> str:
    """
    Generates the next available annotation ID for a file and type.

    Parameters
    ----------
        `counters`: dict[Any, defaultdict[str, int]]
            - Argument controlling counters.
        `file_name`: Any
            - Source filename used to locate the text.
        `prefix`: str
            - Argument controlling prefix.

    Returns
    -------
        `str`
            - Derived value produced by the operation.
    """
    file_key = str(file_name)
    counters[file_key][prefix] += 1
    return f"{prefix}{counters[file_key][prefix]}" if prefix != "#" else f"#{counters[file_key][prefix]}"

def label_from_id(id2label: dict, label_id: Any) -> str:
    """
    Processes label from id for use by the pipeline.

    Parameters
    ----------
        `id2label`: dict
            - Mapping from numeric identifiers to labels.
        `label_id`: Any
            - Numeric label identifier.

    Returns
    -------
        `str`
            - Derived value produced by the operation.
    """
    label_id = int(label_id)
    return str(id2label[label_id] if label_id in id2label else id2label[str(label_id)])

def iter_dataset_examples(dataset: Any, example_key: str):
    """
    Iterates over dataset examples while preserving their source files.

    Parameters
    ----------
        `dataset`: Any
            - Argument controlling dataset.
        `example_key`: str
            - Argument controlling example key.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    if hasattr(dataset, "flat_indices") and getattr(dataset, "flatten_examples", False):
        for text_idx, example_idx in dataset.flat_indices:
            examples = dataset.data_prepared[text_idx]
            if example_idx >= len(examples):
                continue
            file_name = dataset.file_names[text_idx] if getattr(dataset, "file_names", None) is not None else examples[example_idx].get("file_name")
            yield file_name, examples[example_idx]
        return

    if hasattr(dataset, "data_prepared"):
        for text_idx, examples in enumerate(dataset.data_prepared):
            file_name = dataset.file_names[text_idx] if getattr(dataset, "file_names", None) is not None else None
            for example in examples:
                yield file_name if file_name is not None else example.get("file_name"), example
        return

    for item in dataset:
        file_name = item.get("file_name")
        for example in item.get(example_key, []):
            yield file_name if file_name is not None else example.get("file_name"), example

def ensure_sequential_predictions(data_loader: DataLoader, converter_name: str) -> None:
    """
    Checks that predictions will be read in dataset order.

    Parameters
    ----------
        `data_loader`: DataLoader
            - DataLoader providing model examples.
        `converter_name`: str
            - Argument controlling converter name.

    Returns
    -------
        `None`
            - No value; the operation updates its input in place.
    """
    if not isinstance(getattr(data_loader, "sampler", None), SequentialSampler):
        raise ValueError(
            f"{converter_name} needs predictions in dataset order, but this data_loader is not sequential. "
            "Build the inference/evaluation loader with shuffle=False before running the classifier."
        )

def token_offsets_for_file(file_name: Any, text_cache: dict, originals: dict, tokenizer: Any | None):
    """
    Gets and caches token offsets for a source file.

    Parameters
    ----------
        `file_name`: Any
            - Source filename used to locate the text.
        `text_cache`: dict
            - Cache of token offsets by source file.
        `originals`: dict
            - Mapping from source filenames to original text.
        `tokenizer`: Any | None
            - Tokenizer used to encode text and obtain offsets.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    if file_name in text_cache:
        return text_cache[file_name]
    text = originals.get(file_name)
    if text is None or tokenizer is None:
        return None
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)["offset_mapping"]
    text_cache[file_name] = offsets
    return offsets

def token_span_to_char_span(file_name: Any, token_indices: list[int], text_cache: dict, originals: dict, tokenizer: Any | None):
    """
    Converts token indices to character offsets for a source file.

    Parameters
    ----------
        `file_name`: Any
            - Source filename used to locate the text.
        `token_indices`: list[int]
            - Token positions belonging to the span.
        `text_cache`: dict
            - Cache of token offsets by source file.
        `originals`: dict
            - Mapping from source filenames to original text.
        `tokenizer`: Any | None
            - Tokenizer used to encode text and obtain offsets.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    offsets = token_offsets_for_file(file_name, text_cache, originals, tokenizer)
    if offsets is None:
        return None
    valid_offsets = [offsets[idx] for idx in token_indices if 0 <= idx < len(offsets) and offsets[idx][0] != offsets[idx][1]]
    if not valid_offsets:
        return None
    return min(start for start, _ in valid_offsets), max(end for _, end in valid_offsets)

def clean_overlapping_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Removes overlapping entities using their span priority.

    Parameters
    ----------
        `entities`: list[dict[str, Any]]
            - Argument controlling entities.

    Returns
    -------
        `list[dict[str, Any]]`
            - List of parsed, filtered, or generated values.
    """
    cleaned = []

    grouped = defaultdict(list)
    for entity in entities:
        if int(entity["end"]) <= int(entity["start"]):
            continue
        grouped[(entity["file"], entity["label"])].append({**entity, "start": int(entity["start"]), "end": int(entity["end"])})

    for (file_name, label), group in grouped.items():
        group = sorted(group, key=lambda item: (item["start"], item["end"]))
        current = None

        for entity in group:
            if current is None:
                current = {"file": file_name, "label": label, "start": entity["start"], "end": entity["end"]}
                continue

            if entity["start"] <= current["end"]:
                current["start"] = min(current["start"], entity["start"])
                current["end"] = max(current["end"], entity["end"])
            else:
                cleaned.append(current)
                current = {"file": file_name, "label": label, "start": entity["start"], "end": entity["end"]}

        if current is not None:
            cleaned.append(current)

    return cleaned

def ann_lines_from_entities(entities: list[dict[str, Any]], originals: dict, original_ann: pd.DataFrame | dict | None = None) -> dict[str, list[str]]:
    """
    Converts predicted entities into BRAT annotation lines.

    Parameters
    ----------
        `entities`: list[dict[str, Any]]
            - Argument controlling entities.
        `originals`: dict
            - Mapping from source filenames to original text.
        `original_ann`: pd.DataFrame | dict | None
            - Original BRAT annotations.

    Returns
    -------
        `dict[str, list[str]]`
            - List of parsed, filtered, or generated values.
    """
    ann_by_file, counters = base_ann_from_original(original_ann)

    for entity in sorted(clean_overlapping_entities(entities), key=lambda item: (item["file"], item["start"], item["end"], item["label"])):
        file_name = entity["file"]
        text = originals.get(file_name, "")
        entity_text = text[entity["start"]:entity["end"]] if text else entity.get("text", "")
        entity_text = str(entity_text).replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
        ann_id = next_ann_id(counters, file_name, "T")
        ann_by_file.setdefault(str(file_name), []).append(f"{ann_id}\t{entity['label']} {entity['start']} {entity['end']}\t{entity_text}")

    return ann_by_file

def append_token_entity(entities: list[dict[str, Any]], file_name: Any, label: str | None, token_indices: list[int], text_cache: dict, originals: dict, tokenizer: Any | None) -> None:
    """
    Adds a token-level prediction as a character-based entity.

    Parameters
    ----------
        `entities`: list[dict[str, Any]]
            - Argument controlling entities.
        `file_name`: Any
            - Source filename used to locate the text.
        `label`: str | None
            - Entity or relation label.
        `token_indices`: list[int]
            - Token positions belonging to the span.
        `text_cache`: dict
            - Cache of token offsets by source file.
        `originals`: dict
            - Mapping from source filenames to original text.
        `tokenizer`: Any | None
            - Tokenizer used to encode text and obtain offsets.

    Returns
    -------
        `None`
            - No value; the operation updates its input in place.
    """
    if label is None or not token_indices:
        return
    char_span = token_span_to_char_span(file_name, token_indices, text_cache, originals, tokenizer)
    if char_span is None:
        return
    entities.append({"file": file_name, "label": label, "start": int(char_span[0]), "end": int(char_span[1])})

def ner_outputs_to_ann(results: dict, id2label: dict, *, data_loader: DataLoader | None = None, original_texts: dict | pd.DataFrame | None = None, tokenizer: Any | None = None, outside_label: str = "O", original_ann: pd.DataFrame | dict | None = None) -> dict[str, list[str]]:

    """
    Converts NER model outputs into BRAT annotations.

    Parameters
    ----------
        `results`: dict
            - Model outputs to convert or enrich.
        `id2label`: dict
            - Mapping from numeric identifiers to labels.
        `data_loader`: DataLoader | None
            - DataLoader providing model examples.
        `original_texts`: dict | pd.DataFrame | None
            - Mapping from source filenames to original text.
        `tokenizer`: Any | None
            - Tokenizer used to encode text and obtain offsets.
        `outside_label`: str
            - Label assigned to non-entity tokens.
        `original_ann`: pd.DataFrame | dict | None
            - Original BRAT annotations.

    Returns
    -------
        `dict[str, list[str]]`
            - List of parsed, filtered, or generated values.
    """
    originals = normalize_original_texts(original_texts)
    original_offset_cache = {}
    entities = []

    model_outputs = results.get("model_outputs")
    if model_outputs:
        for output in model_outputs:
            pred_labels = tensor_to_python(output.get("pred_label", []))
            pred_ids = tensor_to_python(output.get("pred_id", []))
            files = output.get("File") or output.get("span_file_names") or []
            global_token_spans = output.get("span_global_token_idx", [])
            char_spans = output.get("span_char_idx", [])
            text_offsets = tensor_to_python(output.get("text_token_offsets", []))
            batch_file_names = output.get("file_names", [])

            if not pred_labels and pred_ids:
                pred_labels = [label_from_id(id2label, pred_id) for pred_id in pred_ids]

            for i, label in enumerate(pred_labels):
                if label == outside_label:
                    continue

                file_name = files[i] if i < len(files) else None
                if file_name is None:
                    continue

                char_span = None
                if i < len(global_token_spans):
                    global_indices = [int(idx) for idx in tensor_to_python(global_token_spans[i]) if int(idx) >= 0]
                    if file_name in batch_file_names and text_offsets:
                        local_file_idx = batch_file_names.index(file_name)
                        token_offset = int(text_offsets[local_file_idx])
                        local_indices = [idx - token_offset for idx in global_indices]
                    else:
                        local_indices = global_indices
                    char_span = token_span_to_char_span(file_name, local_indices, original_offset_cache, originals, tokenizer)

                if char_span is None and i < len(char_spans):
                    candidate = tensor_to_python(char_spans[i])
                    if candidate is not None:
                        char_span = (int(candidate[0]), int(candidate[1]))

                if char_span is None:
                    continue

                entities.append({"file": file_name, "label": label, "start": int(char_span[0]), "end": int(char_span[1])})

        return ann_lines_from_entities(entities, originals, original_ann=original_ann)

    if data_loader is None:
        raise ValueError("data_loader is required to convert BIO outputs because BIO results only contain flattened predictions.")

    all_pred = tensor_to_python(results.get("all_pred"))
    if all_pred is None:
        raise ValueError("results must contain either model_outputs or all_pred.")
    ensure_sequential_predictions(data_loader, "ner_outputs_to_ann")

    pred_pos = 0

    for item in data_loader.dataset:
        file_name = item.get("file_name")

        for window in item["windows"]:
            labels = item.get("window_labels")
            active_label = None
            active_tokens = []
            previous_token_idx = None

            def flush_entity():
                if active_label is None or not active_tokens:
                    return
                append_token_entity(entities, file_name, active_label, active_tokens, original_offset_cache, originals, tokenizer)

            for pos, abs_idx in enumerate(window["absolute_token_indices"]):
                if abs_idx is None or int(window["attention_mask"][pos]) == 0:
                    continue
                if labels is not None and labels[window["window_index"]][pos] == -100:
                    continue
                if pred_pos >= len(all_pred):
                    break

                token_idx = int(abs_idx)
                bio_label = label_from_id(id2label, all_pred[pred_pos])
                pred_pos += 1

                if bio_label == outside_label or "-" not in bio_label:
                    flush_entity()
                    active_label = None
                    active_tokens = []
                    previous_token_idx = token_idx
                    continue

                bio, label = bio_label.split("-", 1)
                starts_new = bio == "B" or active_label != label or previous_token_idx is None or token_idx != previous_token_idx + 1

                if starts_new:
                    flush_entity()
                    active_label = label
                    active_tokens = [token_idx]
                else:
                    active_tokens.append(token_idx)

                previous_token_idx = token_idx

            flush_entity()

    return ann_lines_from_entities(entities, originals, original_ann=original_ann)

def att_outputs_to_ann(results: dict, id2label: dict, *, data_loader: DataLoader, original_ann: pd.DataFrame | dict | None = None, attribute_label: str | None = "Status", outside_label: str = "O") -> dict[str, list[str]]:
    """
    Converts attribute model outputs into BRAT annotations.

    Parameters
    ----------
        `results`: dict
            - Model outputs to convert or enrich.
        `id2label`: dict
            - Mapping from numeric identifiers to labels.
        `data_loader`: DataLoader
            - DataLoader providing model examples.
        `original_ann`: pd.DataFrame | dict | None
            - Original BRAT annotations.
        `attribute_label`: str | None
            - Attribute label assigned to the entity.
        `outside_label`: str
            - Label assigned to non-entity tokens.

    Returns
    -------
        `dict[str, list[str]]`
            - List of parsed, filtered, or generated values.
    """
    all_pred = tensor_to_python(results.get("all_pred"))
    if all_pred is None:
        return base_ann_from_original(original_ann)[0]
    ensure_sequential_predictions(data_loader, "att_outputs_to_ann")

    ann_by_file, counters = base_ann_from_original(original_ann)

    for pred_id, (file_name, example) in zip(all_pred, iter_dataset_examples(data_loader.dataset, "examples")):
        pred_label = label_from_id(id2label, pred_id)
        if pred_label == outside_label or file_name is None:
            continue

        entity = example.get("entity", {})
        target_id = entity.get("original_id", entity.get("id"))
        if target_id is None:
            continue

        ann_id = next_ann_id(counters, file_name, "A")
        if attribute_label is None:
            line = f"{ann_id}\t{pred_label} {target_id}"
        else:
            line = f"{ann_id}\t{attribute_label} {target_id} {pred_label}"
        ann_by_file.setdefault(str(file_name), []).append(line)

    return ann_by_file

def re_outputs_to_ann(results: dict, id2label: dict, *, data_loader: DataLoader, original_ann: pd.DataFrame | dict | None = None, outside_label: str = "O", arg1_name: str = "Arg1", arg2_name: str = "Arg2") -> dict[str, list[str]]:
    """
    Converts RE classifier predictions into BRAT R-lines grouped by file name.
    """
    all_pred = tensor_to_python(results.get("all_pred"))
    if all_pred is None:
        return base_ann_from_original(original_ann)[0]
    ensure_sequential_predictions(data_loader, "re_outputs_to_ann")

    ann_by_file, counters = base_ann_from_original(original_ann)

    for pred_id, (file_name, example) in zip(all_pred, iter_dataset_examples(data_loader.dataset, "examples")):
        pred_label = label_from_id(id2label, pred_id)
        if pred_label == outside_label or file_name is None:
            continue

        left_entity = example.get("left_entity", {})
        right_entity = example.get("right_entity", {})
        left_id = left_entity.get("original_id", left_entity.get("id"))
        right_id = right_entity.get("original_id", right_entity.get("id"))
        if left_id is None or right_id is None:
            continue

        ann_id = next_ann_id(counters, file_name, "R")
        ann_by_file.setdefault(str(file_name), []).append(f"{ann_id}\t{pred_label} {arg1_name}:{left_id} {arg2_name}:{right_id}")

    return ann_by_file

def diso_entity_ids_by_file(original_ann: pd.DataFrame | None, entity_label: str = "DISO", file_col: str | None = None, t_col: str = "T") -> dict[str, list[str]]:
    if original_ann is None:
        return {}
    if file_col is None:
        file_col = find_file_column(original_ann)

    ids_by_file = defaultdict(list)
    for row_idx, row in original_ann.iterrows():
        file_name = str(row.get(file_col, row_idx))
        for line in normalize_ann_lines(row.get(t_col)):
            parts = str(line).split("\t")
            if len(parts) < 2:
                continue
            label_parts = parts[1].split(maxsplit=1)
            if len(label_parts) != 2 or label_parts[0] != entity_label:
                continue
            ids_by_file[file_name].append(parts[0])

    return ids_by_file

def icd_outputs_to_ann(results: dict, id2label: dict, data_loader: DataLoader, original_ann: pd.DataFrame | None, node_list: dict, entity_label: str = "DISO", outside_label: str = "O") -> dict[str, list[str]]:
    """
    Converts ICD predictions into BRAT note annotations.

    Parameters
    ----------
        `results`: dict
            - Model outputs to convert or enrich.
        `id2label`: dict
            - Mapping from numeric identifiers to labels.
        `data_loader`: DataLoader
            - DataLoader providing model examples.
        `original_ann`: pd.DataFrame | None
            - Original BRAT annotations.
        `node_list`: dict
            - ICD hierarchy nodes used to enrich predictions.
        `entity_label`: str
            - Label identifying the target entity type.
        `outside_label`: str
            - Label assigned to non-entity tokens.

    Returns
    -------
        `dict[str, list[str]]`
            - List of parsed, filtered, or generated values.
    """
    all_pred = tensor_to_python(results.get("all_pred"))
    all_results_sources = results.get("sources")
    all_results_step_end = [elem[2]["selected_threshold"][1] for elem in results.get("items")]

    if original_ann is None:
        raise ValueError("original_ann is required so ICD predictions can be attached to existing DISO T-lines.")
    if all_pred is None:
        return base_ann_from_original(original_ann)[0]
    ensure_sequential_predictions(data_loader, "icd_outputs_to_ann")

    ann_by_file, counters = base_ann_from_original(original_ann)
    diso_ids_by_file = diso_entity_ids_by_file(original_ann, entity_label=entity_label)
    diso_pos_by_file = defaultdict(int)
    dataset = data_loader.dataset
    file_names = getattr(dataset, "file_names", None)

    if file_names is None:
        raise ValueError("data_loader.dataset must expose file_names for ICD annotation conversion.")

    for pred_id, result_source, result_step_end, file_name in zip(all_pred, all_results_sources, all_results_step_end, file_names):
        file_key = str(file_name)
        entity_ids = diso_ids_by_file.get(file_key, [])
        entity_pos = diso_pos_by_file[file_key]
        if entity_pos >= len(entity_ids):
            raise ValueError(f"Could not map ICD prediction {entity_pos + 1} for file {file_key} to a {entity_label} T-line.")

        target_id = entity_ids[entity_pos]
        diso_pos_by_file[file_key] += 1
        code = label_from_id(id2label, pred_id)
        if code == outside_label:
            continue

        ann_id = next_ann_id(counters, file_key, "#")
        ann_by_file.setdefault(file_key, []).append(f"{ann_id}\tAnnotatorNotes {target_id}\t{code}\t{result_source}\tMax confidence: {node_list[code].path[-1] if result_source == "NO_HS" else node_list[code].path[result_step_end]}")

    return ann_by_file


def ann_output_to_df_ann(ann_output: pd.DataFrame | dict, file_col: str = "archivo_origen", t_col: str = "T") -> pd.DataFrame:
    if isinstance(ann_output, pd.DataFrame):
        return ann_output

    rows = []
    for file_name, ann_lines in dict(ann_output).items():
        rows.append({
            file_col: file_name,
            t_col: [line for line in normalize_ann_lines(ann_lines) if str(line).startswith("T")],
        })

    return pd.DataFrame(rows, columns=[file_col, t_col])

def df_ann_to_ann_list(df_ann: pd.DataFrame | dict, ann_columns: tuple[str, ...] = ("T", "A", "R", "#")) -> dict[str, list[str]]:
    """
    Converts annotation dataframe columns into per-file annotation lists.

    Parameters
    ----------
        `df_ann`: pd.DataFrame | dict
            - Input dataframe containing the records to process.
        `ann_columns`: tuple[str, ...]
            - Annotation columns to preserve.

    Returns
    -------
        `dict[str, list[str]]`
            - List of parsed, filtered, or generated values.
    """
    return base_ann_from_original(df_ann, ann_columns=ann_columns)[0]

def ann_bool_text(value: Any) -> str:
    """
    Normalizes a value to the BRAT boolean text representation.

    Parameters
    ----------
        `value`: Any
            - Value to validate or normalize.

    Returns
    -------
        `str`
            - Derived value produced by the operation.
    """
    if pd.isna(value):
        return "False"
    if isinstance(value, str):
        return "True" if value.strip().lower() == "true" else "False"
    return "True" if bool(value) else "False"

def update_diso_comment_notes_from_df_final(df_ann: pd.DataFrame | dict, df_final: pd.DataFrame, file_col_ann: str | None = None, file_col_final: str = "nombre_archivo", entity_id_col: str = "ann_ent_id", cie10_col: str = "CIE10_selected_V2", judge_col: str = "tree_5_V2", entity_label: str = "DISO", ann_columns: tuple[str, ...] = ("T", "A", "R", "#")) -> dict[str, list[str]]:
    """
    Updates BRAT DISO notes with ICD and judgement values from the final dataframe.

    Parameters
    ----------
        `df_ann`: pd.DataFrame | dict
            - Input dataframe containing the records to process.
        `df_final`: pd.DataFrame
            - Final dataframe containing model decisions.
        `file_col_ann`: str | None
            - Filename column in the annotation data.
        `file_col_final`: str
            - Filename column in the final dataframe.
        `entity_id_col`: str
            - Column containing the entity identifier.
        `cie10_col`: str
            - Column containing the selected ICD-10 code.
        `judge_col`: str
            - Column containing the tree judgement.
        `entity_label`: str
            - Label identifying the target entity type.
        `ann_columns`: tuple[str, ...]
            - Annotation columns to preserve.

    Returns
    -------
        `dict[str, list[str]]`
            - List of parsed, filtered, or generated values.
    """
    if not isinstance(df_ann, pd.DataFrame):
        df_ann = pd.DataFrame([{"archivo_origen": file_name, **ann_data} for file_name, ann_data in dict(df_ann).items()])

    if file_col_ann is None:
        file_col_ann = find_file_column(df_ann)

    required_final_cols = {file_col_final, entity_id_col, cie10_col, judge_col}
    missing_final_cols = required_final_cols.difference(df_final.columns)
    if missing_final_cols:
        raise ValueError(f"df_final is missing required columns: {sorted(missing_final_cols)}")

    if "#" not in df_ann.columns:
        df_ann = df_ann.copy()
        df_ann["#"] = [[] for _ in range(len(df_ann))]
    else:
        df_ann = df_ann.copy(deep=True)

    diso_ids_by_file = set()
    for _, row in df_ann.iterrows():
        file_name = str(row[file_col_ann])
        for line in normalize_ann_lines(row.get("T")):
            parts = str(line).split("\t")
            if len(parts) < 2:
                continue
            label_parts = parts[1].split(maxsplit=1)
            if label_parts and label_parts[0] == entity_label:
                diso_ids_by_file.add((file_name, parts[0]))

    updates = {}
    for _, row in df_final.iterrows():
        file_name = str(row[file_col_final])
        entity_id = str(row[entity_id_col])
        if (file_name, entity_id) not in diso_ids_by_file:
            continue
        updates[(file_name, entity_id)] = {"cie10": "" if pd.isna(row[cie10_col]) else str(row[cie10_col]), "judge": ann_bool_text(row[judge_col]),}

    for row_idx, row in df_ann.iterrows():
        file_name = str(row[file_col_ann])
        notes = normalize_ann_lines(row.get("#"))
        updated_targets = set()
        updated_notes = []

        for line in notes:
            parts = str(line).split("\t")
            if len(parts) < 3:
                updated_notes.append(str(line))
                continue

            target_id = parts[1].split()[-1]
            update = updates.get((file_name, target_id))
            if update is None:
                updated_notes.append(str(line))
                continue

            parts[2] = update["cie10"]
            parts = [part for part in parts if not str(part).startswith("Juzgador:")]
            parts.append(f"Juzgador: {update['judge']}")
            updated_notes.append("\t".join(parts))
            updated_targets.add(target_id)

        next_note_number = max((ann_id_number(note, "#") or 0 for note in updated_notes), default=0)
        for (update_file, target_id), update in updates.items():
            if update_file != file_name or target_id in updated_targets:
                continue
            next_note_number += 1
            updated_notes.append(f"#{next_note_number}\tAnnotatorNotes {target_id}\t{update['cie10']}\tJuzgador: {update['judge']}")

        df_ann.at[row_idx, "#"] = updated_notes

    return df_ann_to_ann_list(df_ann, ann_columns=ann_columns)

def ann_parts(row):
    ts = normalize_ann_lines(row.get("T"))
    rs = normalize_ann_lines(row.get("R"))
    ats = normalize_ann_lines(row.get("A"))
    notes = normalize_ann_lines(row.get("#"))

    text = {}
    label = {}
    code = {}

    for line in ts:
        p = line.split("\t")
        if len(p) >= 3:
            ent_id = p[0]
            label[ent_id] = p[1].split()[0]
            text[ent_id] = p[2]

    for line in notes:
        p = line.split("\t")
        if len(p) >= 3:
            # Example: #1 AnnotatorNotes T3 R35
            target_id = p[1].split()[-1]
            code[target_id] = p[2].split()[0]

    return text, label, code, rs, ats


def related_ent(target_id, wanted_relation, wanted_label, text, label, rs):
    """
    Finds an entity linked by a requested BRAT relation.

    Parameters
    ----------
        `target_id`: Any
            - Identifier of the source entity.
        `wanted_relation`: Any
            - Relation type to search for.
        `wanted_label`: Any
            - Entity label to search for.
        `text`: Any
            - Text containing the entity or span.
        `label`: Any
            - Entity or relation label.
        `rs`: Any
            - BRAT relation lines.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    for line in rs:
        p = line.split("\t")
        if len(p) < 2:
            continue

        bits = p[1].split()
        if not bits or bits[0] != wanted_relation:
            continue

        args = dict(x.split(":") for x in bits[1:] if ":" in x)
        arg1, arg2 = args.get("Arg1"), args.get("Arg2")

        other = None
        if arg1 == target_id:
            other = arg2
        elif arg2 == target_id:
            other = arg1

        if other and label.get(other) == wanted_label:
            return text.get(other)

    return None
