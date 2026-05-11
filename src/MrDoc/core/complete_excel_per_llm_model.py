from ..models.schemas import (
    PipelineContext,
    JSONToXLSXConfig
)

from ..data_io.reader import (
    Path,
    textwrap,
    defaultdict,
    read_excel,
    cargar_docx_lista,
    read_json_diagnosticos,
    read_embeddings_pre_created
)
from ..data_io.writing import (
    create_output_folder_name,
    write_excel_log,
    write_excel_final
)

from ..services.common.llm_loader import load_llm
from ..services.common.common import (
    pd,
    torch,
    model,
    prompts as prompts_total,
    procesar_json_diagnosticos,
    clean_df_obtained_with_llm,
    completar_df_extra_data_for_analysis
)
from ..services.sync_functions.sync_funcs import (
    completar_df_predicted_nearest_text_only as completar_df_predicted_nearest_text_only_SYNC,
    asistente_seleccionador_cie10 as asistente_seleccionador_cie10_SYNC,
    asistente_juzgador_cie10 as asistente_juzgador_cie10_SYNC,
    asistente_seleccionador_tratamiento_falsos_cie10 as asistente_seleccionador_tratamiento_falsos_cie10_SYNC
)
from ..services.async_functions.async_funcs import (
    asyncio,
    completar_df_predicted_nearest_text_only as completar_df_predicted_nearest_text_only_ASYNC,
    asistente_seleccionador_cie10 as asistente_seleccionador_cie10_ASYNC,
    asistente_juzgador_cie10 as asistente_juzgador_cie10_ASYNC,
    asistente_seleccionador_tratamiento_falsos_cie10 as asistente_seleccionador_tratamiento_falsos_cie10_ASYNC
)


def initialize_variables(ctx: PipelineContext):
    create_output_folder_name(ctx)
    
    #################################################################################################################

    if ctx.cie_10_version == "2018":
        df_reference = read_excel(ctx, "Diagnosticos_ES2018.xlsx", sheet_name="finales", header=0)
        df_reference = df_reference[['codigo', 'descripcion']].reset_index(drop=True).rename(columns={'codigo': 'Código', 'descripcion': 'Descripción'})

    elif ctx.cie_10_version == "2024":
        df_reference = read_excel(ctx, "Diagnosticos_ES2024.xlsx", sheet_name="ES2024 Completa + Marcadores", header=0)
        df_reference = df_reference.loc[df_reference["Nodo_Final"] == 1, ["Código", "Descripción"]].reset_index(drop=True)
        # df_reference_embeddings = read_embeddings_pre_created(ctx, "descripcion_embeddings_nodo_final_all-MiniLM-L6-v2.pt")

    else:
        df_reference = read_excel(ctx, "Diagnosticos_ES2026.xlsx", sheet_name="ES2026 Completa + Marcadores", header=0)
        df_reference = df_reference.loc[df_reference["Nodo_Final"] == 1, ["Código", "Descripción"]].reset_index(drop=True)

    df_reference_embeddings = model.encode(df_reference["Descripción"].to_list(), show_progress_bar=True, convert_to_tensor=True)
    CIE10_full_list = df_reference["Código"].to_list()

    #################################################################################################################

    data_predicted = read_json_diagnosticos([ctx.paths.data_intermediate / f"Informes_JSON_{ctx.folder_and_archive_name}"])
    df_predicted = procesar_json_diagnosticos(data_predicted)
    df_predicted = clean_df_obtained_with_llm(df_predicted, "CIE10_predicted")
    df_predicted_embeddings = model.encode(df_predicted["diagnostico_predicted"].to_list(), show_progress_bar=True, convert_to_tensor=True)

    #################################################################################################################

    llm = load_llm(ctx.llm_config)
    semaforo = asyncio.Semaphore(ctx.MAX_CONCURRENCY)
    docx_lista = cargar_docx_lista(ctx.paths.data_input / ctx.folder_and_archive_name)

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
        docx_lista=docx_lista,
        prompts=prompts
    )

    return config, df_predicted, df_predicted_embeddings

def run_jsons_to_xlsx_sync(ctx: PipelineContext):
    config, df_predicted, df_predicted_embeddings = initialize_variables(ctx)

    df_final = completar_df_predicted_nearest_text_only_SYNC(df_predicted, config.df_reference, config.df_reference_embeddings, df_predicted_embeddings, add_semantic_similarity=True)
    write_excel_log(df_final, ctx, 1)
    
    df_final = asistente_seleccionador_cie10_SYNC(df_final, config.CIE10_full_list, config.df_reference, config.prompts["prompt_CIE10_selector"], config.llm, ctx.paths.docs_dir, find_CIE10_similars_level = 0, model = model, add_semantic_similarity = True, tratamiento_fallos = False, json_parse=ctx.json_parse)
    write_excel_log(df_final, ctx, 2)
    
    df_final = completar_df_extra_data_for_analysis(df_final, tratamiento_fallos=False)
    write_excel_log(df_final, ctx, 3)
    
    df_final = asistente_juzgador_cie10_SYNC(df_final, config.prompts["prompt_CIE10_juzgador"], config.llm, config.docx_lista, ctx.paths.docs_dir, tratamiento_fallos=False, json_parse=ctx.json_parse)
    write_excel_log(df_final, ctx, 4)
    
    df_final = asistente_seleccionador_tratamiento_falsos_cie10_SYNC(df_final, config.prompts["prompt_CIE10_selector_tratamiento_falsos"], config.llm, config.docx_lista, ctx.paths.docs_dir, json_parse=ctx.json_parse)
    write_excel_log(df_final, ctx, 5)
    
    df_final = asistente_seleccionador_cie10_SYNC(df_final, config.CIE10_full_list, config.df_reference, config.prompts["prompt_CIE10_selector"], config.llm, ctx.paths.docs_dir, find_CIE10_similars_level = 0, model = model, add_semantic_similarity = True, tratamiento_fallos = True, json_parse=ctx.json_parse)
    write_excel_log(df_final, ctx, 6)
    
    df_final = completar_df_extra_data_for_analysis(df_final, tratamiento_fallos=True)
    write_excel_log(df_final, ctx, 7)
    
    df_final = asistente_juzgador_cie10_SYNC(df_final, config.prompts["prompt_CIE10_juzgador"], config.llm, config.docx_lista, ctx.paths.docs_dir, tratamiento_fallos=True, json_parse=ctx.json_parse)

    write_excel_final(df_final, ctx)



async def run_jsons_to_xlsx_async(ctx: PipelineContext):
    config, df_predicted, df_predicted_embeddings = initialize_variables(ctx)

    df_final = await completar_df_predicted_nearest_text_only_ASYNC(df_predicted, config.df_reference, config.df_reference_embeddings, df_predicted_embeddings, config.semaforo, add_semantic_similarity=True)
    write_excel_log(df_final, ctx, 1)
    
    df_final = await asistente_seleccionador_cie10_ASYNC(df_final, config.CIE10_full_list, config.df_reference, config.prompts["prompt_CIE10_selector"], config.llm, ctx.paths.docs_dir, config.semaforo, find_CIE10_similars_level = 0, model = model, add_semantic_similarity = True, tratamiento_fallos = False, json_parse=ctx.json_parse)
    write_excel_log(df_final, ctx, 2)
    
    df_final = completar_df_extra_data_for_analysis(df_final, tratamiento_fallos=False)
    write_excel_log(df_final, ctx, 3)
    
    df_final = await asistente_juzgador_cie10_ASYNC(df_final, config.prompts["prompt_CIE10_juzgador"], config.llm, config.docx_lista, ctx.paths.docs_dir, config.semaforo, tratamiento_fallos=False, json_parse=ctx.json_parse)
    write_excel_log(df_final, ctx, 4)
    
    df_final = await asistente_seleccionador_tratamiento_falsos_cie10_ASYNC(df_final, config.prompts["prompt_CIE10_selector_tratamiento_falsos"], config.llm, config.docx_lista, ctx.paths.docs_dir, config.semaforo, json_parse=ctx.json_parse)
    write_excel_log(df_final, ctx, 5)
    
    df_final = await asistente_seleccionador_cie10_ASYNC(df_final, config.CIE10_full_list, config.df_reference, config.prompts["prompt_CIE10_selector"], config.llm, ctx.paths.docs_dir, config.semaforo, find_CIE10_similars_level = 0, model = model, add_semantic_similarity = True, tratamiento_fallos = True, json_parse=ctx.json_parse)
    write_excel_log(df_final, ctx, 6)
    
    df_final = completar_df_extra_data_for_analysis(df_final, tratamiento_fallos=True)
    write_excel_log(df_final, ctx, 7)
    
    df_final = await asistente_juzgador_cie10_ASYNC(df_final, config.prompts["prompt_CIE10_juzgador"], config.llm, config.docx_lista, ctx.paths.docs_dir, config.semaforo, tratamiento_fallos=True, json_parse=ctx.json_parse)

    write_excel_final(df_final, ctx)