import json
import torch
import textwrap
import pandas as pd
from pathlib import Path
from docx import Document
from tqdm.auto import tqdm
from collections import defaultdict

from ..models import PipelineContext, Any

def read_checkpoint(ctx: PipelineContext) -> Any:
    return torch.load(ctx.paths.docs_dir / "best_model_checkpoint.pt")

def cargar_docx_single(path: Path) -> str:
    """
    Load the `.docx` archive and transform it into a plain `str`

    Parameters
    ----------
        `path`: Path
            - Path of the single `.docx` archive

    Returns
    -------
        `texto`: str
            - Plain text of the original doc
    """
    doc = Document(path)
    texto = "\n".join([p.text for p in doc.paragraphs if p.text.strip() != ""])
    texto = textwrap.dedent(texto)
    return texto


def cargar_docx_lista(folder_path: Path) -> dict[str, str]:
    """
    Load all the the `.docx` archives inside a directory and creates a dict with all the transformed `.docx` into plain `str`

    Parameters
    ----------
        `folder_path`: Path
            - Path of all the `.docx` archives

    Returns
    -------
        `total`: dict[str, str]
            - Dict where the key is the file name and the value is the plain text
    """
    total = defaultdict(str)
    docx_list = [f for f in folder_path.glob("*.docx") if f.is_file()]

    for report in docx_list:
        informe_texto = cargar_docx_single(report)
        total[report.stem] = informe_texto

    return total


def read_embeddings_pre_created(ctx: PipelineContext, pt_name: str):
    if (ctx.llm_config.device != "cpu"):
        return torch.load(ctx.paths.docs_dir / pt_name)
    else:
        return torch.load(ctx.paths.docs_dir / pt_name, map_location=torch.device('cpu'))


def read_excel(ctx: PipelineContext, xlsx_name: str, **kwargs: object):
    return pd.read_excel(ctx.paths.docs_dir / xlsx_name, **kwargs)


def read_json_single(path: Path) -> dict:
    """
    Read the a single `.json` and returns a dict with that data

    Parameters
    ----------
        `path`: Path
            - JSON path

    Returns
    -------
        `data`: dict
            - The `.json` data
    """
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
    """
    Read the data from the multiple `.json` and returns a flatten list with all of them

    Allows using multiple directories with multiple `.json` inside them

    Parameters
    ----------
        `folders`: list[Path]
            - List of all the directories to investigate

    Returns
    -------
        `data`: list[dict]
            - List with all the `.json` read
    """
    data = defaultdict(dict)

    for folder in folders:
        json_list = [f for f in folder.glob("*.json") if f.is_file()]
        for jsonfile in tqdm(json_list, desc="Procesando JSONs...", unit="JSON", total=len(json_list)):
            data[jsonfile.stem] = read_json_single(jsonfile)
    
    return data


def load_schema_info(schema_path: Path) -> tuple:
    """
    Load valid and required keys from a JSON schema file.
    Automatically targets the inner item schema if the root defines an array of objects,
    otherwise targets the root object properties.

    Parameters
    ----------
        `schema_path`: Path
            - Path to the JSON schema file (e.g., 'esquema.json').

    Returns
    -------
        `tuple`: (valid_keys, required, allow_extra)
            - `valid_keys`: set with the allowed property names (from `items.properties` if a collection exists, otherwise from root `properties`).
            - `required`: set with the required property names for that level.
            - `allow_extra`: bool indicating if additional properties are allowed at that same level.
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        esquema = json.load(f)

    # Por defecto: usar propiedades de raíz
    root_props = esquema.get("properties", {}) or {}
    valid_keys = set(root_props.keys())
    required = set(esquema.get("required", []) or [])
    allow_extra = esquema.get("additionalProperties", True)

    # Si existe alguna colección (array) con items tipo "object", usamos ese nivel (caso diagnosticos)
    # Ejemplo: esquema_diagnosticos.json → usar items.required/props (los dicts internos). :contentReference[oaicite:2]{index=2}
    for k, v in root_props.items():
        if isinstance(v, dict) and v.get("type") == "array" and "items" in v:
            items = v["items"]
            if isinstance(items, dict) and items.get("type") == "object":
                item_props = items.get("properties", {}) or {}
                valid_keys = set(item_props.keys())
                required = set(items.get("required", []) or [])
                allow_extra = items.get("additionalProperties", True)
                break

    return valid_keys, required, allow_extra