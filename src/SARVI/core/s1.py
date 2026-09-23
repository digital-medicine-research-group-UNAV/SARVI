from __future__ import annotations

import asyncio
import importlib
import multiprocessing as mp
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from natsort import natsorted
from tqdm.auto import tqdm

from ..data_io.reader import (
    read_ann_list,
    read_docx_single,
    textwrap,
)
from ..data_io.writing import write_json
from ..models.schemas import DOCXToJSONSConfig
from ..services.common.utils.ann_utils import find_file_column
from ..services.common.icd_pred_funcs import result_ann_row_to_diagnosticos
from ..services.common.utils.json_utils import validate_json_created
from ..services.sync_funcs.llm_funcs import process_docx as process_docx_SYNC
from ..services.common.llm_funcs import prompts as prompts_total
from ..services.common.llm_loader import load_llm

if TYPE_CHECKING:
    from ..models.schemas import PipelineContext

EXTRACTOR_MODULES = (
    ("NER", f"{f"{__package__}.extractors"}.ner"),
    ("RE", f"{f"{__package__}.extractors"}.re"),
    ("ATT", f"{f"{__package__}.extractors"}.att"),
    ("ICD10", f"{f"{__package__}.extractors"}.icd"),
)


def initialize_variables(ctx: "PipelineContext"):
    """
    Initialize the variables neeeded for the S1 procedure

    Parameters
    ----------
        `ctx`: PipelineContext
            - Context with the types of variables to be used

    Returns
    -------
        ``: DOCXToJSONSConfig
            - Object with all the variables needed. Most of them are `None` so they can be initialized in their respective process
    """
    print("\n\033[1m0. Initializing variables\033[0m\n\n")

    report_list = natsorted(p for p in (ctx.paths.data_input / ctx.folder_and_archive_name).iterdir() if p.is_file())

    if ctx.ussage == "generative":
        print("\n\033[1m0.1. Initializing LLM\033[0m\n\n")
        llm = load_llm(ctx.llm_config)
        prompt = textwrap.dedent(prompts_total["report_to_data"])
    else:
        llm = None
        prompt = None

    return DOCXToJSONSConfig(
        report_list=report_list,
        prompt=prompt,
        llm=llm,
        semaforo=asyncio.Semaphore(ctx.MAX_CONCURRENCY),
        modelo_ner=None,
        modelo_re=None,
        modelo_att=None,
        tokenizer_re=None,
        tokenizer_att=None,
        node_list=None,
        modelo_icd10_head=None,
        modelo_icd10_prediction=None,
        label2id_NER=None,
        id2label_NER=None,
        label2id_RE=None,
        id2label_RE=None,
        label2id_ATT=None,
        id2label_ATT=None,
        label2id_ICD10=None,
        id2label_ICD10=None,
        id_no_hs_to_id_hs=None,
        re_token_distance_bins=None,
        re_negative_difficulty_by_bin=None,
        icd10_thresholds=None,
        root=None,
    )


def run_extractor_module(module_name: str, ctx: "PipelineContext", config: DOCXToJSONSConfig) -> None:
    """
    Function created so the different processess can be initialized and executed in separate threads
    Only used in the deterministic process

    Parameters
    ----------
        `module_name`: str
            - .py file name to be executed
        `ctx`: PipelineContext
            - Context with the types of variables to be used
        `config`: DOCXToJSONSConfig
            - Config previously initialized where the variables are going to be inputed

    Returns
    -------
        ``: None
            - The .py file is correctly executed. Nothing is returned
    """
    module = importlib.import_module(module_name)
    module.run(ctx, config)


def run_extractor_in_process(label: str, module_name: str, ctx: "PipelineContext", config: DOCXToJSONSConfig) -> None:
    """
    Function that allows to initialize parallel functions in different threads
    These functions are runned and the main process is blocked until finish
    Only used in the deterministic process

    Parameters
    ----------
        `label`: str
            - Label name of the .py file to be executed
        `module_name`: str
            - .py file name to be executed
        `ctx`: PipelineContext
            - Context with the types of variables to be used
        `config`: DOCXToJSONSConfig
            - Config previously initialized where the variables are going to be inputed

    Returns
    -------
        ``: None
            - The .py file is correctly executed. Nothing is returned
    """
    available = mp.get_all_start_methods()

    if "spawn" in available:
        start_method = "spawn"
    elif "forkserver" in available:
        start_method = "forkserver"
    else:
        start_method = None

    mp_context = mp.get_context(start_method) if start_method else mp.get_context()
    process = mp_context.Process(target=run_extractor_module, args=(module_name, ctx, config), name=f"SARVI-{label}")

    process.start()
    print(f"\n\033[1mStarted {label} extractor in PID {process.pid}\033[0m\n")
    process.join()

    if process.is_alive():
        process.terminate()
        process.join()

    if process.exitcode != 0:
        raise RuntimeError(f"{label} extractor failed with exit code {process.exitcode}\nStart method used: {start_method or 'default'}")

    print(f"\n\033[1mFinished {label} extractor; PID {process.pid} released\033[0m\n")

def run_s1_deterministic_sync(ctx: "PipelineContext"):
    """
    S1 Deterministic-Sync principal function. Each step is executed in different threads so, when completed, all the resources occupied are eliminated
    It initialize all the variables needed to run the process

    Parameters
    ----------
        `ctx`: PipelineContext
            - Context with the types of variables to be used

    Returns
    -------
        ``: None
            - The S1 process is completed
    """
    config = initialize_variables(ctx)

    print("\n\033[1m1.0. Load Files\033[0m\n\n")
    input_folder = ctx.paths.data_input / ctx.folder_and_archive_name
    if not any(input_folder.glob("*.docx")) and not any(input_folder.glob("*.txt")):
        raise FileNotFoundError("No .docx or .txt files found")

    for label, module_name in EXTRACTOR_MODULES:
        run_extractor_in_process(label, module_name, ctx, config)

    print("\n\033[1m1.5. Save to JSON\033[0m\n\n")

    dict_ann_icd = read_ann_list(ctx.paths.data_intermediate / ctx.folder_and_archive_name / "ANN" / "ICD10/Mix")
    df_ann_icd = pd.DataFrame([{"archivo_origen": file_name, **ann_data} for file_name, ann_data in dict_ann_icd.items()])

    file_col = find_file_column(df_ann_icd)
    if file_col is None:
        raise ValueError("No se encontró la columna con el nombre del archivo en las anotaciones ICD10")

    for _, row in df_ann_icd.iterrows():
        filename = row[file_col]
        try:
            respuesta_determinista = {"diagnosticos": result_ann_row_to_diagnosticos(row)}
            if not validate_json_created(respuesta_determinista, ctx.paths.docs_dir / "esquema_diagnosticos.json"):
                print(f"⚠️ Se omite {filename}: el JSON no sigue el esquema indicado")
                continue
            write_json(ctx, Path(f"{filename}.txt"), respuesta_determinista)
        except Exception as exc:
            print(f"⚠️ Se omite {filename}: error al crear el JSON: {exc!r}")


def run_s1_genrative_sync(ctx: "PipelineContext"):
    """
    S1 Generative-Sync principal function
    It initialize all the variables needed to run the process

    Parameters
    ----------
        `ctx`: PipelineContext
            - Context with the types of variables to be used

    Returns
    -------
        ``: None
            - The S1 process is completed
    """
    config = initialize_variables(ctx)

    for report in tqdm(config.report_list, desc="Extrayendo CIE10 de archivos...", unit="informe"):
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                informe_texto = read_docx_single(report)
                respuesta_llm = process_docx_SYNC(informe_texto, report, config.prompt, config.llm, ctx.json_parse, ctx.paths.docs_dir)
                write_json(ctx, report, respuesta_llm)
                break
            except Exception as e:
                print(f"⚠️ Error al procesar informe {report}: {e!r}")
                traceback.print_exc()
                if attempt < max_retries:
                    print(f"↻ Reintentando ({attempt}/{max_retries})...")
                else:
                    print(f"❌ Falló definitivamente el informe: {report}\n")


async def run_s1_deterministic_async(ctx: "PipelineContext"):
    """
    S1 Deterministic-Async principal function
    It initialize all the variables needed to run the process

    Parameters
    ----------
        `ctx`: PipelineContext
            - Context with the types of variables to be used

    Returns
    -------
        ``: None
            - The S1 process is completed
    """
    config = initialize_variables(ctx)
    pass


async def run_s1_genrative_async(ctx: "PipelineContext"):
    """
    S1 Generative-Async principal function
    It initialize all the variables needed to run the process

    Parameters
    ----------
        `ctx`: PipelineContext
            - Context with the types of variables to be used

    Returns
    -------
        ``: None
            - The S1 process is completed
    """
    from ..services.async_funcs.llm_funcs import procesar_docx as procesar_docx_ASYNC

    config = initialize_variables(ctx)

    async def procesar_con_reintentos(report, max_retries=5):
        for attempt in range(1, max_retries + 1):
            try:
                informe_texto = read_docx_single(report)
                respuesta_llm = await procesar_docx_ASYNC(informe_texto, report, config.prompt, config.llm, ctx.json_parse, ctx.paths.docs_dir, config.semaforo)
                write_json(ctx, report, respuesta_llm)
                break
            except Exception as e:
                print(f"⚠️ Error al procesar informe {report}: {e!r}")
                traceback.print_exc()
                if attempt < max_retries:
                    print(f"↻ Reintentando ({attempt}/{max_retries})...")
                else:
                    print(f"❌ Falló definitivamente el informe: {report}\n")

    tareas = [procesar_con_reintentos(report) for report in config.report_list]

    for future in tqdm(asyncio.as_completed(tareas), total=len(tareas), desc="Extrayendo CIE10 de archivos...", unit="informe"):
        await future
