from __future__ import annotations

import torch
import pandas as pd
from pathlib import Path
from typing import TYPE_CHECKING

from ..models.schemas import (
    JSONToXLSXConfig
)

if TYPE_CHECKING:
    from ..models.schemas import PipelineContext

from ..data_io.reader import (
    textwrap,
    defaultdict,
    read_excel_single,
    read_docx_list, read_txt_list,
    read_json_diagnosticos,
    read_ann_list
)
from ..data_io.writing import (
    write_excel_log,
    write_excel_final,
    write_ann_list
)

from ..services.common.llm_loader import (
    load_llm
)
from ..services.common.llm_funcs import (
    prompts as prompts_total,
    procesar_json_diagnosticos,
    clean_df_obtained_with_llm,
    completar_df_extra_data_for_analysis
)
from ..services.common.utils.json_utils import (
    model
)
from ..services.common.utils.ann_utils import (
    update_diso_comment_notes_from_df_final
)

from ..services.sync_funcs.llm_funcs import (
    completar_df_predicted_nearest_text_only as completar_df_predicted_nearest_text_only_SYNC,
    asistente_seleccionador_cie10 as asistente_seleccionador_cie10_SYNC,
    asistente_juzgador_cie10 as asistente_juzgador_cie10_SYNC,
    asistente_seleccionador_tratamiento_falsos_cie10 as asistente_seleccionador_tratamiento_falsos_cie10_SYNC
)
from ..services.async_funcs.llm_funcs import (
    asyncio,
    completar_df_predicted_nearest_text_only as completar_df_predicted_nearest_text_only_ASYNC,
    asistente_seleccionador_cie10 as asistente_seleccionador_cie10_ASYNC,
    asistente_juzgador_cie10 as asistente_juzgador_cie10_ASYNC,
    asistente_seleccionador_tratamiento_falsos_cie10 as asistente_seleccionador_tratamiento_falsos_cie10_ASYNC
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
        ``: JSONToXLSXConfig
            - Object with all the variables needed
    """
    #################################################################################################################

    if ctx.cie_10_version == "2018":
        df_reference = read_excel_single(ctx, "Diagnosticos_ES2018.xlsx", sheet_name="finales", header=0)
        df_reference = df_reference[['codigo', 'descripcion']].reset_index(drop=True).rename(columns={'codigo': 'Código', 'descripcion': 'Descripción'})

    elif ctx.cie_10_version == "2024":
        df_reference = read_excel_single(ctx, "Diagnosticos_ES2024.xlsx", sheet_name="ES2024 Completa + Marcadores", header=0)
        df_reference = df_reference.loc[df_reference["Nodo_Final"] == 1, ["Código", "Descripción"]].reset_index(drop=True)
        # df_reference_embeddings = read_torch_checkpoint(ctx, "descripcion_embeddings_nodo_final_all-MiniLM-L6-v2.pt")

    else:
        df_reference = read_excel_single(ctx, "Diagnosticos_ES2026.xlsx", sheet_name="ES2026 Completa + Marcadores", header=0)
        df_reference = df_reference.loc[df_reference["Nodo_Final"] == 1, ["Código", "Descripción"]].reset_index(drop=True)

    df_reference_embeddings = model.encode(df_reference["Descripción"].to_list(), show_progress_bar=True, convert_to_tensor=True)
    CIE10_full_list = df_reference["Código"].to_list()

    #################################################################################################################

    data_predicted = read_json_diagnosticos([ctx.paths.data_intermediate / ctx.folder_and_archive_name / "JSON"])
    df_predicted = procesar_json_diagnosticos(data_predicted)
    df_predicted = clean_df_obtained_with_llm(df_predicted, "CIE10_predicted")
    df_predicted_embeddings = model.encode(df_predicted["diagnostico_predicted"].to_list(), show_progress_bar=True, convert_to_tensor=True)

    #################################################################################################################

    llm = load_llm(ctx.llm_config)
    semaforo = asyncio.Semaphore(ctx.MAX_CONCURRENCY)

    input_folder = ctx.paths.data_input / ctx.folder_and_archive_name

    doc_lista = {}
    if any(input_folder.glob("*.docx")):
        doc_lista = read_docx_list(input_folder)
    elif any(input_folder.glob("*.txt")):
        doc_lista = read_txt_list(input_folder)

    #################################################################################################################

    prompts = defaultdict(str)
    prompts["prompt_CIE10_selector"] = textwrap.dedent(prompts_total["multiple_codes_to_selected"])
    prompts["prompt_CIE10_juzgador"] = textwrap.dedent(prompts_total["binary_code_confirmation"])
    prompts["prompt_CIE10_selector_tratamiento_falsos"] = textwrap.dedent(prompts_total["cie10_decider"])
    
    #################################################################################################################

    config = JSONToXLSXConfig(
        df_reference=df_reference,
        CIE10_full_list=CIE10_full_list,
        df_reference_embeddings=df_reference_embeddings,
        llm=llm,
        semaforo=semaforo,
        doc_lista=doc_lista,
        prompts=prompts
    )

    return config, df_predicted, df_predicted_embeddings

def run_s2_sync(ctx: "PipelineContext"):
    """
    S2 Sync principal function
    It initialize all the variables needed to run the process
    At each step the programm saves a log file with the corresponding data up to that point

    Parameters
    ----------
        `ctx`: PipelineContext
            - Context with the types of variables to be used

    Returns
    -------
        ``: None
            - The S2 process is completed
    """
    config, df_predicted, df_predicted_embeddings = initialize_variables(ctx)

    df_final = completar_df_predicted_nearest_text_only_SYNC(df_predicted, config.df_reference, config.df_reference_embeddings, df_predicted_embeddings, add_semantic_similarity=True)
    write_excel_log(df_final, ctx, 1)
    
    df_final = asistente_seleccionador_cie10_SYNC(df_final, config.CIE10_full_list, config.df_reference, config.prompts["prompt_CIE10_selector"], config.llm, ctx.paths.docs_dir, find_CIE10_similars_level = 0, model = model, add_semantic_similarity = True, tratamiento_fallos = False, json_parse=ctx.json_parse)
    write_excel_log(df_final, ctx, 2)
    
    df_final = completar_df_extra_data_for_analysis(df_final, tratamiento_fallos=False)
    write_excel_log(df_final, ctx, 3)
    
    df_final = asistente_juzgador_cie10_SYNC(df_final, config.prompts["prompt_CIE10_juzgador"], config.llm, config.doc_lista, ctx.paths.docs_dir, tratamiento_fallos=False, json_parse=ctx.json_parse)
    write_excel_log(df_final, ctx, 4)
    
    df_final = asistente_seleccionador_tratamiento_falsos_cie10_SYNC(df_final, config.prompts["prompt_CIE10_selector_tratamiento_falsos"], config.llm, config.doc_lista, ctx.paths.docs_dir, json_parse=ctx.json_parse)
    write_excel_log(df_final, ctx, 5)
    
    df_final = asistente_seleccionador_cie10_SYNC(df_final, config.CIE10_full_list, config.df_reference, config.prompts["prompt_CIE10_selector"], config.llm, ctx.paths.docs_dir, find_CIE10_similars_level = 0, model = model, add_semantic_similarity = True, tratamiento_fallos = True, json_parse=ctx.json_parse)
    write_excel_log(df_final, ctx, 6)
    
    df_final = completar_df_extra_data_for_analysis(df_final, tratamiento_fallos=True)
    write_excel_log(df_final, ctx, 7)
    
    df_final = asistente_juzgador_cie10_SYNC(df_final, config.prompts["prompt_CIE10_juzgador"], config.llm, config.doc_lista, ctx.paths.docs_dir, tratamiento_fallos=True, json_parse=ctx.json_parse)

    write_excel_final(df_final, ctx)

    dict_ann = read_ann_list(ctx.paths.data_intermediate / ctx.folder_and_archive_name / "ANN" / "ICD10/Mix")
    if dict_ann:
        df_ann = pd.DataFrame([{"archivo_origen": file_name, **ann_data}for file_name, ann_data in dict_ann.items()])
        df_ann = update_diso_comment_notes_from_df_final(df_ann, df_final)
        write_ann_list(ctx, df_ann, ctx.paths.data_output / ctx.folder_and_archive_name / "ANN")



async def run_s2_async(ctx: "PipelineContext"):
    """
    S2 Async principal function
    It initialize all the variables needed to run the process
    At each step the programm saves a log file with the corresponding data up to that point

    Parameters
    ----------
        `ctx`: PipelineContext
            - Context with the types of variables to be used

    Returns
    -------
        ``: None
            - The S2 process is completed
    """
    config, df_predicted, df_predicted_embeddings = initialize_variables(ctx)

    df_final = await completar_df_predicted_nearest_text_only_ASYNC(df_predicted, config.df_reference, config.df_reference_embeddings, df_predicted_embeddings, config.semaforo, add_semantic_similarity=True)
    write_excel_log(df_final, ctx, 1)
    
    df_final = await asistente_seleccionador_cie10_ASYNC(df_final, config.CIE10_full_list, config.df_reference, config.prompts["prompt_CIE10_selector"], config.llm, ctx.paths.docs_dir, config.semaforo, find_CIE10_similars_level = 0, model = model, add_semantic_similarity = True, tratamiento_fallos = False, json_parse=ctx.json_parse)
    write_excel_log(df_final, ctx, 2)
    
    df_final = completar_df_extra_data_for_analysis(df_final, tratamiento_fallos=False)
    write_excel_log(df_final, ctx, 3)
    
    df_final = await asistente_juzgador_cie10_ASYNC(df_final, config.prompts["prompt_CIE10_juzgador"], config.llm, config.doc_lista, ctx.paths.docs_dir, config.semaforo, tratamiento_fallos=False, json_parse=ctx.json_parse)
    write_excel_log(df_final, ctx, 4)
    
    df_final = await asistente_seleccionador_tratamiento_falsos_cie10_ASYNC(df_final, config.prompts["prompt_CIE10_selector_tratamiento_falsos"], config.llm, config.doc_lista, ctx.paths.docs_dir, config.semaforo, json_parse=ctx.json_parse)
    write_excel_log(df_final, ctx, 5)
    
    df_final = await asistente_seleccionador_cie10_ASYNC(df_final, config.CIE10_full_list, config.df_reference, config.prompts["prompt_CIE10_selector"], config.llm, ctx.paths.docs_dir, config.semaforo, find_CIE10_similars_level = 0, model = model, add_semantic_similarity = True, tratamiento_fallos = True, json_parse=ctx.json_parse)
    write_excel_log(df_final, ctx, 6)
    
    df_final = completar_df_extra_data_for_analysis(df_final, tratamiento_fallos=True)
    write_excel_log(df_final, ctx, 7)
    
    df_final = await asistente_juzgador_cie10_ASYNC(df_final, config.prompts["prompt_CIE10_juzgador"], config.llm, config.doc_lista, ctx.paths.docs_dir, config.semaforo, tratamiento_fallos=True, json_parse=ctx.json_parse)

    write_excel_final(df_final, ctx)
