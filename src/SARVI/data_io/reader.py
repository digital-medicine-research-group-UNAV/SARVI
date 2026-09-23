from __future__ import annotations

import json
import torch
import pickle
import textwrap
import pandas as pd
from pathlib import Path
from docx import Document
from tqdm.auto import tqdm
from collections import defaultdict

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.schemas import PipelineContext

def read_parquet_file(ctx: "PipelineContext", name: str) -> Any:
    return pd.read_parquet(ctx.paths.docs_dir / name)

def read_torch_checkpoint(ctx: "PipelineContext", name: str) -> Any:
    if (ctx.llm_config.device != "cpu"):
        return torch.load(ctx.paths.docs_dir / name)
    else:
        return torch.load(ctx.paths.docs_dir / name, map_location=torch.device('cpu'))

def read_docx_single(path: Path) -> str:
    doc = Document(path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip() != ""])
    text = textwrap.dedent(text)
    return text


def read_docx_list(folder_path: Path) -> dict[str, str]:
    total = defaultdict(str)
    docx_list = [f for f in folder_path.glob("*.docx") if f.is_file()]

    for report in docx_list:
        text = read_docx_single(report)
        total[report.stem] = text

    return total

def read_txt_single(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = textwrap.dedent(text)
    return text


def read_txt_list(folder_path: Path) -> dict[str, str]:
    total = defaultdict(str)
    txt_list = [f for f in folder_path.glob("*.txt") if f.is_file()]

    for report in txt_list:
        informe_texto = read_txt_single(report)
        total[report.stem] = informe_texto

    return total

def read_ann_single(path: Path) -> dict[str, list[str]]:
    annotations = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line.strip():
                continue

            annotation_id = line.split("\t", 1)[0]
            annotation_type = annotation_id[0]

            annotations[annotation_type].append(line)

    return dict(annotations)


def read_ann_list(folder_path: Path) -> dict[str, dict[str, list[str]]]:
    total = defaultdict(dict)
    ann_list = [f for f in folder_path.glob("*.ann") if f.is_file()]

    for report in ann_list:
        ann_data = read_ann_single(report)
        total[report.stem] = ann_data

    return dict(total)


def read_excel_single(ctx: "PipelineContext", xlsx_name: str, **kwargs: object) -> Any:
    return pd.read_excel(ctx.paths.docs_dir / xlsx_name, **kwargs)


def read_json_single(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data

def read_jsonl_single(path: Path) -> list:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def read_json_diagnosticos(folders: list[Path]) -> dict[str, dict]:
    data = defaultdict(dict)

    for folder in folders:
        json_list = [f for f in folder.glob("*.json") if f.is_file()]
        for jsonfile in tqdm(json_list, desc="Procesando JSONs...", unit="JSON", total=len(json_list)):
            data[jsonfile.stem] = read_json_single(jsonfile)
    
    return data


def read_schema_info_single(schema_path: Path) -> tuple:
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    root_props = schema.get("properties", {}) or {}
    valid_keys = set(root_props.keys())
    required = set(schema.get("required", []) or [])
    allow_extra = schema.get("additionalProperties", True)

    for key, value in root_props.items():
        if isinstance(value, dict) and value.get("type") == "array" and "items" in value:
            items = value["items"]
            if isinstance(items, dict) and items.get("type") == "object":
                item_props = items.get("properties", {}) or {}
                valid_keys = set(item_props.keys())
                required = set(items.get("required", []) or [])
                allow_extra = items.get("additionalProperties", True)
                break

    return valid_keys, required, allow_extra

def read_pickle_single(path: Path) -> Any:
    with path.open("rb") as f:
        data = pickle.load(f)

    return data


def read_pickle_list(folder_path: Path) -> dict[str, Any]:
    total = defaultdict(object)

    pickle_list = [f for f in folder_path.glob("*") if f.is_file() and f.suffix.lower() in {".pickle", ".pkl"}]

    for pickle_file in pickle_list:
        total[pickle_file.stem] = read_pickle_single(pickle_file)

    return dict(total)

def read_dsv_single(path: Path, sep: str = "|", **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(path, sep=sep, **kwargs)