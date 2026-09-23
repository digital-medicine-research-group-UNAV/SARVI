from __future__ import annotations

import json
import torch
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models.schemas import PipelineContext

def _ensure_parent_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def write_json(ctx: "PipelineContext", report: Path, texto: dict) -> None:
    path = _ensure_parent_dir(ctx.paths.data_intermediate / ctx.folder_and_archive_name / "JSON" / f"{report.stem}.json")
    with path.open('w', encoding='utf-8') as f:
        json.dump(texto, f, ensure_ascii=False, indent=4)

def write_json_extra_docs(ctx: "PipelineContext", file_name: str, content: dict) -> None:
    path = _ensure_parent_dir(ctx.paths.docs_dir / f"{file_name}.json")
    with path.open('w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)

def write_excel_log(df: pd.DataFrame, ctx: "PipelineContext", i: int) -> None:
    path = _ensure_parent_dir(ctx.paths.data_output / ctx.folder_and_archive_name / "xlsx_logs" / f"df_{i}.xlsx")
    df.to_excel(path, index=False)

def write_excel_final(df: pd.DataFrame, ctx: "PipelineContext") -> None:
    path = _ensure_parent_dir(ctx.paths.data_output / ctx.folder_and_archive_name / f"df_FINAL.xlsx")
    df.to_excel(path, index=False)

def write_torch_checkpoint(ctx: "PipelineContext", name: str, checkpoint: Any) -> None:
    path = _ensure_parent_dir(ctx.paths.docs_dir / f"{name}.pt")
    torch.save(checkpoint, path)

def write_parquet(ctx: "PipelineContext", name: str, data: pd.DataFrame) -> None:
    path = _ensure_parent_dir(ctx.paths.docs_dir / f"{name}.parquet")
    data.to_parquet(path, index=False)

def write_ann_single(ctx: "PipelineContext", data: list, name: str, folder: Path, extra: str = "") -> None:
    path =  _ensure_parent_dir(ctx.paths.docs_dir / folder / f"{name}{extra}.ann")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(data))

def write_ann_list(ctx: "PipelineContext", data: dict, folder: Path, extra: str = "") -> None:
    path =  _ensure_parent_dir(ctx.paths.docs_dir / folder)
    for key, value in data.items():
        write_ann_single(ctx, value, key, folder, extra)
