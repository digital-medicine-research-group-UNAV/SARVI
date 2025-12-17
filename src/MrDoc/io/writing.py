import json
import pandas as pd
from pathlib import Path
from ..models import PipelineContext

def create_intermediate_folder_name(ctx: PipelineContext):
    intermediate_dir = ctx.paths.data_intermediate / f"Informes_JSON_{ctx.folder_and_archive_name}"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

def create_output_folder_name(ctx: PipelineContext):
    output_dir = ctx.paths.data_output / f"{ctx.folder_and_archive_name}_logs"
    output_dir.mkdir(parents=True, exist_ok=True)

def write_json(ctx: PipelineContext, report: Path, texto: dict):
    with open(ctx.paths.data_intermediate / f"Informes_JSON_{ctx.folder_and_archive_name}" / f"{report.stem}.json", 'w', encoding='utf-8') as f:
        json.dump(texto, f, ensure_ascii=False, indent=4)

def write_excel_log(df: pd.DataFrame, ctx: PipelineContext, i: int):
    df.to_excel(f"{ctx.paths.data_output}/{ctx.folder_and_archive_name}_logs/df_{ctx.folder_and_archive_name}_{i}.xlsx", index=False)

def write_excel_final(df: pd.DataFrame, ctx: PipelineContext):
    df.to_excel(f"{ctx.paths.data_output}/df_{ctx.folder_and_archive_name}_FINAL.xlsx", index=False)