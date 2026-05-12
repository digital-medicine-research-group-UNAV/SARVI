import random
import traceback
import pandas as pd

from pathlib import Path
from tqdm.auto import tqdm
from natsort import natsorted

from ..models.schemas import (
    PipelineContext,
    DOCXToJSONSConfig
)

from ..data_io.reader import (
    textwrap,
    cargar_docx_single, cargar_docx_lista,
    read_json_single,
    read_excel,
    read_parquet_file
)
from ..data_io.writing import (
    create_intermediate_folder_name,
    write_json
)

from ..services.common.llm_loader import (
    load_llm
)
from ..services.common.tree_funcs import (
    load_tree_hierarchical_module
)
from ..services.common.llm_funcs import (
    prompts as prompts_total
)
from ..services.common.ner_funcs import(
    build_ner_json,
    model_ner,
    tokenizer_ner,
    initialize_span_ner_model,
    id2label_ner,
    prepara_data_from_ner_pred_to_icd_pred,
    corrected_entities_to_df
)
from ..services.common.icd_pred_funcs import(
    initialize_icd10_hs_head_model,
    initialize_icd10_hs_prediction_model,
    initialize_icd10_no_hs_head_model,
    extract_flattened_predictions,
    result_item_to_diagnostico
)
from ..services.common.utils.icd_preds_utils import (
    remap_ids
)
from ..services.common.utils.json_utils import (
    validate_json_created
)

from ..services.sync_funcs.llm_funcs import (
    procesar_docx as procesar_docx_SYNC,
    correct_entities as correct_entities_SYNC,
)
from ..services.sync_funcs.ner_funcs import (
    prepare_data as prepare_data_SYNC,
    construct_loaders_ner as construct_loaders_ner_SYNC,
    run_ner_model as run_ner_model_SYNC,
    update_df_with_final_pred_entities as update_df_with_final_pred_entities_SYNC,
)
from ..services.sync_funcs.icd_pred_funcs import(
    construct_loaders_icd10 as construct_loaders_icd10_SYNC,
    run_icd10_hs_model as run_icd10_hs_model_SYNC,
    run_icd10_no_hs_model as run_icd10_no_hs_model_SYNC, 
    apply_thresholds_icd10 as apply_thresholds_icd10_SYNC,
)

from ..services.async_funcs.llm_funcs import (
    asyncio,
    procesar_docx as procesar_docx_ASYNC
)

def initialize_variables(ctx: PipelineContext):
    print("0. Initializing variables\n")
    create_intermediate_folder_name(ctx)

    #################################################################################################################

    if ctx.cie_10_version == "2018":
        df_reference = read_excel(ctx, "Diagnosticos_ES2018.xlsx", sheet_name="finales", header=0)
        df_reference = df_reference[['codigo', 'descripcion']].reset_index(drop=True).rename(columns={'codigo': 'Código', 'descripcion': 'Descripción'})
        df_reference["Código"] = (df_reference["Código"].astype(str).str.strip().str.replace("\xa0", "", regex=False))
        df_reference = df_reference[df_reference["Código"].str.match(r"^[A-Za-z]\d[A-Za-z0-9](?:\.[A-Za-z0-9]+)?(?:-[A-Za-z]\d[A-Za-z0-9](?:\.[A-Za-z0-9]+)?)?$")]

    elif ctx.cie_10_version == "2024":
        df_reference = read_excel(ctx, "Diagnosticos_ES2024.xlsx", sheet_name="ES2024 Completa + Marcadores", header=0)
        df_reference = df_reference[["Código", "Descripción"]].reset_index(drop=True)
        df_reference["Código"] = (df_reference["Código"].astype(str).str.strip().str.replace("\xa0", "", regex=False))
        df_reference = df_reference[df_reference["Código"].str.match(r"^[A-Za-z]\d[A-Za-z0-9](?:\.[A-Za-z0-9]+)?(?:-[A-Za-z]\d[A-Za-z0-9](?:\.[A-Za-z0-9]+)?)?$")]

    else:
        df_reference = read_excel(ctx, "Diagnosticos_ES2026.xlsx", sheet_name="ES2026 Completa + Marcadores", header=0)
        df_reference = df_reference[["Código", "Descripción"]].reset_index(drop=True)
        df_reference["Código"] = (df_reference["Código"].astype(str).str.strip().str.replace("\xa0", "", regex=False))
        df_reference = df_reference[df_reference["Código"].str.match(r"^[A-Za-z]\d[A-Za-z0-9](?:\.[A-Za-z0-9]+)?(?:-[A-Za-z]\d[A-Za-z0-9](?:\.[A-Za-z0-9]+)?)?$")]

    #################################################################################################################

    report_list = natsorted(p for p in (ctx.paths.data_input / ctx.folder_and_archive_name).iterdir() if p.is_file())
    
    if ctx.ussage == "generative":
        llm = load_llm(ctx.llm_config)
        prompt = textwrap.dedent(prompts_total["report_to_data"])
        modelo_ner = None
        node_list = None
        modelo_icd10_head = None
        modelo_icd10_prediction = None
        label2id_ICD10 = None
        id2label_ICD10 = None
        id_no_hs_to_id_hs = None
        icd10_thresholds = None
    else:
        if ctx.deterministic_use_llm_for_corrections:
            llm = load_llm(ctx.llm_config)
        else:
            llm = None
        prompt = textwrap.dedent(prompts_total["correct_entities"])
        modelo_ner = [initialize_span_ner_model(ctx, "NER_checkpoint.pt"), initialize_span_ner_model(ctx, "NER_checkpoint_FT.pt")]
        node_list = load_tree_hierarchical_module(df_reference)

        root = node_list["root"]
        root.set_indexes()
        label2id_hs = {key.name: value for key, value in root.node_to_id.items()}
        id2label_hs = {value: key for key, value in label2id_hs.items()}
        label2id_no_hs = {id2label_hs[value.item()]: i for i, value in enumerate(root.leaf_indexes)}
        id2label_no_hs = {value: key for key, value in label2id_no_hs.items()}
        label2id_ICD10 = [label2id_hs, label2id_no_hs]
        id2label_ICD10 = [id2label_hs, id2label_no_hs]
        id_no_hs_to_id_hs = {id_no_hs: label2id_hs[label_name] for label_name, id_no_hs in label2id_no_hs.items()}

        modelo_icd10_head = [initialize_icd10_hs_head_model(ctx, "ICD10_HS_checkpoint.pt", root), initialize_icd10_no_hs_head_model(ctx, "ICD10_NO_HS_checkpoint.pt", label2id_no_hs, root)]
        modelo_icd10_prediction = [initialize_icd10_hs_prediction_model(ctx, "ICD10_HS_checkpoint.pt", label2id_hs, root), None]
        icd10_thresholds = read_json_single(ctx.paths.docs_dir / "thresholds_ICD10.json")
    
    semaforo = asyncio.Semaphore(ctx.MAX_CONCURRENCY)

    config = DOCXToJSONSConfig(
        report_list=report_list,
        prompt=prompt,
        llm=llm,
        semaforo=semaforo,
        modelo_ner=modelo_ner,
        node_list=node_list,
        modelo_icd10_head=modelo_icd10_head,
        modelo_icd10_prediction=modelo_icd10_prediction,
        label2id_ICD10=label2id_ICD10,
        id2label_ICD10=id2label_ICD10,
        id_no_hs_to_id_hs=id_no_hs_to_id_hs,
        icd10_thresholds=icd10_thresholds
    )

    return config

def run_docx_to_jsons_deterministic_sync(ctx: PipelineContext):
    config = initialize_variables(ctx)

    # ------------------------------NER------------------------------
    print("1.1. Creating NER data")
    dict_data = cargar_docx_lista(ctx.paths.data_input / ctx.folder_and_archive_name)
    df_data = pd.DataFrame(list(dict_data.items()), columns=["archivo_origen", "Text"])

    df_data_prepared = prepare_data_SYNC(df_data, padding=False, tokenizer=tokenizer_ner, model=model_ner, device=ctx.device)
    df_data_full, data_loader_full = construct_loaders_ner_SYNC(data=df_data_prepared, tokenizer=tokenizer_ner)
    df_data_diags, data_loader_diags = construct_loaders_ner_SYNC(data=df_data_prepared, tokenizer=tokenizer_ner)

    print("1.2. Running NER model // FULL")
    all_pred_full, all_value_preds_full = run_ner_model_SYNC(config.modelo_ner[0], data_loader=data_loader_full, device=ctx.device)
    df_data_full["pred_id"] = all_pred_full
    df_data_full["pred_label"] = df_data_full["pred_id"].map(id2label_ner)
    df_data_full = update_df_with_final_pred_entities_SYNC(df_data_full)

    print("1.3. Running NER model // DIAGS")
    all_pred_diags, all_value_preds_diags = run_ner_model_SYNC(config.modelo_ner[1], data_loader=data_loader_diags, device=ctx.device)
    df_data_diags["pred_id"] = all_pred_diags
    df_data_diags["pred_label"] = df_data_diags["pred_id"].map(id2label_ner)
    df_data_diags = update_df_with_final_pred_entities_SYNC(df_data_diags)

    print("1.4. Mixing NER results")
    cols_check = [c for c in df_data_diags.columns if c not in ["CLS Embedding", "Embeddings", "pred_id", "pred_label"]]
    assert df_data_diags[cols_check].equals(df_data_full[cols_check])
    mask = ~df_data_full["pred_label"].isin([id2label_ner[1], id2label_ner[2]])
    df_final_ner = df_data_diags.copy()
    df_final_ner.loc[mask, ["CLS Embedding", "Embeddings", "pred_id", "pred_label"]] = df_data_full.loc[mask, ["CLS Embedding", "Embeddings", "pred_id", "pred_label"]]

    df_final_ner = df_final_ner[['File', 'Tokens', 'Decoded span', 'Token idx', 'Text instance', 'pred_id', 'pred_label']]

    print("1. -> 2. Transforming data to predict ICD10")
    df_final_ner = prepara_data_from_ner_pred_to_icd_pred(df_data, df_final_ner)

    # ------------------------------Correct NER------------------------------
    if ctx.deterministic_use_llm_for_corrections:
        print("1. -> 2. Fixing entities")
        codiesp_train = read_parquet_file(ctx, "codiesp_train.parquet")
        e3c_trian = read_parquet_file(ctx, "e3c_train.parquet")

        random_clinentity = list(set(random.sample([s for arr in codiesp_train["All Description"].dropna() for s in arr], k=min(30, len([s for arr in codiesp_train["All Description"].dropna() for s in arr])))))
        random_actor = list(set(random.sample([s for arr in e3c_trian["All ACTOR"].dropna() for s in arr], k=min(30, len([s for arr in e3c_trian["All ACTOR"].dropna() for s in arr])))))
        random_timex3 = list(set(random.sample([s for arr in e3c_trian["All TIMEX3"].dropna() for s in arr], k=min(30, len([s for arr in e3c_trian["All TIMEX3"].dropna() for s in arr])))))

        config.prompt = config.prompt.format(random_clinentity=random_clinentity, random_actor=random_actor, random_timex3=random_timex3)
        correct_entities_input = build_ner_json(df_final_ner, token_index_base=1)
        entities_correction_decision = correct_entities_SYNC(config.prompt, config.llm, correct_entities_input, ctx.paths.docs_dir, ctx.json_parse)
        df_final_ner = corrected_entities_to_df(entities_correction_decision, dict_data)

    # ------------------------------CIE10------------------------------
    print("2.1. Creating ICD10 prediction data")
    final_results = {}

    for i in tqdm(range(len(df_final_ner)), desc="Making ICD10 predictions", unit="text"):
        df_final_ner_file = df_final_ner.iloc[[i]]
        data_loader = construct_loaders_icd10_SYNC(df_final_ner_file)

        print(f"{i} // 2.2. Running ICD10 prediction data // YES HS")
        all_pred_yes_hs, all_value_preds_yes_hs, all_desc_global_yes_hs = run_icd10_hs_model_SYNC(config.modelo_icd10_head[0], config.modelo_icd10_prediction[0], data_loader)
        print(f"{i} // 2.3. Running ICD10 prediction data // NO HS")
        all_pred_no_hs, all_value_preds_no_hs, all_desc_global_no_hs = run_icd10_no_hs_model_SYNC(config.modelo_icd10_head[1], data_loader)

        all_pred_no_hs = [config.id_no_hs_to_id_hs[i] for i in all_pred_no_hs]
        all_value_preds_no_hs = remap_ids(all_value_preds_no_hs, config.id_no_hs_to_id_hs)
        assert all_desc_global_yes_hs == all_desc_global_no_hs, f"Descriptions arrays are different // Training has been done shuffled"

        print(f"{i} // 2.4. Creating final predictions")
        flattened_predictions_yes_hs = extract_flattened_predictions(all_value_preds_yes_hs, all_pred_yes_hs)
        flattened_predictions_no_hs = extract_flattened_predictions(all_value_preds_no_hs, all_pred_no_hs)

        results_yes_hd = apply_thresholds_icd10_SYNC(flattened_predictions_yes_hs, config.icd10_thresholds["YES_HS"])
        results_no_hd = apply_thresholds_icd10_SYNC(flattened_predictions_no_hs, config.icd10_thresholds["NO_HS"])

        results = [(all_desc_global_yes_hs[i], config.id2label_ICD10[0][all_pred_yes_hs[i]] if yes[0] else config.id2label_ICD10[0][all_pred_no_hs[i]] if no[0] else config.id2label_ICD10[0][all_pred_yes_hs[i]], yes if yes[0] else no if no[0] else yes) for i, (yes, no) in enumerate(zip(results_yes_hd, results_no_hd))]

        final_results[df_final_ner_file["Original File"][i]] = results

    for filename, items in final_results.items():
        respuesta_determinista = {"diagnosticos": [result_item_to_diagnostico(item) for item in items]}
        if not validate_json_created(respuesta_determinista, ctx.paths.docs_dir / "esquema_diagnosticos.json"):
            raise Exception(f"El JSON creado del documento {filename} no sigue el esquema indicado")
        write_json(ctx, Path(filename), respuesta_determinista)
    
    # return df_final_ner, data_loader, flattened_predictions_yes_hs, flattened_predictions_no_hs, final_results
    # return (df_final_ner, df_final_ner_corrected), (df_data_full, df_data_diags, tmp), (entities_correction_decision), (final_results, results_yes_hd, results_no_hd, flattened_predictions_yes_hs, flattened_predictions_no_hs)
    

def run_docx_to_jsons_genrative_sync(ctx: PipelineContext):
    config = initialize_variables(ctx)

    for report in tqdm(config.report_list, desc="Extrayendo CIE10 de archivos...", unit="informe"):
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                informe_texto = cargar_docx_single(report)
                respuesta_llm = procesar_docx_SYNC(informe_texto, report, config.prompt, config.llm, ctx.json_parse, ctx.paths.docs_dir)
                write_json(ctx, report, respuesta_llm)
                break
            except Exception as e:
                print(f"⚠️ Error al procesar informe {report}: {e!r}")
                traceback.print_exc()
                if attempt < max_retries:
                    print(f"↻ Reintentando ({attempt}/{max_retries})...")
                else:
                    print(f"❌ Falló definitivamente el informe: {report}\n")

async def run_docx_to_jsons_deterministic_async(ctx: PipelineContext):
    config = initialize_variables(ctx)
    pass

async def run_docx_to_jsons_genrative_async(ctx: PipelineContext):
    config = initialize_variables(ctx)

    async def procesar_con_reintentos(report, max_retries=5):
        for attempt in range(1, max_retries + 1):
            try:
                informe_texto = cargar_docx_single(report)
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