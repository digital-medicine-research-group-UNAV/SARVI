import traceback
from natsort import natsorted

from ..models import (
    PipelineContext,
    DOCXToJSONSConfig
)

from ..io.reader import (
    textwrap,
    cargar_docx_single,
    cargar_docx_lista,
    read_json_single,
    read_excel
)
from ..io.writing import (
    create_intermediate_folder_name,
    write_json
)

from ..services.llm_loader import load_llm
from ..services.common import (
    tqdm,
    ast,
    pd,
    device,
    tokenizer_ner,
    model_ner,
    merge_sentencepiece_words,
    initialize_span_ner_model,
    initialize_icd10_hs_head_model,
    initialize_icd10_hs_prediction_model,
    initialize_icd10_no_hs_head_model,
    id2label_ner,
    remap_ids,
    extract_flattened_predictions,
    prompts as prompts_total
)
from ..services.sync_funcs import (
    load_tree_hierarchical_module,
    procesar_docx as procesar_docx_SYNC,
    prepare_data as prepare_data_SYNC,
    construct_loaders_ner as construct_loaders_ner_SYNC,
    run_ner_model as run_ner_model_SYNC,
    update_df_with_final_pred_entities as update_df_with_final_pred_entities_SYNC,
    construct_loaders_icd10 as construct_loaders_icd10_SYNC,
    run_icd10_hs_model as run_icd10_hs_model_SYNC,
    run_icd10_no_hs_model as run_icd10_no_hs_model_SYNC, 
    apply_thresholds_icd10 as apply_thresholds_icd10_SYNC
)
from ..services.async_funcs import (
    asyncio,
    procesar_docx as procesar_docx_ASYNC
)

def initialize_variables(ctx: PipelineContext):
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
        llm = None
        prompt = None
        modelo_ner = [initialize_span_ner_model(ctx, "NER_checkpoint.pt"), initialize_span_ner_model(ctx, "NER_checkpoint_FT.pt")]
        node_list = load_tree_hierarchical_module(df_reference)

        root = root.set_indexes()
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
    dict_data = cargar_docx_lista(ctx.paths.data_input / ctx.folder_and_archive_name)
    df_data = pd.DataFrame(list(dict_data.items()), columns=["archivo_origen", "Text"])

    df_data = prepare_data_SYNC(df_data, padding=False, tokenizer=tokenizer_ner, model=model_ner, data_files_type="span", device=device)
    df_data_full, data_loader_full = construct_loaders_ner_SYNC(data=df_data, tokenizer=tokenizer_ner)
    df_data_diags, data_loader_diags = construct_loaders_ner_SYNC(data=df_data, tokenizer=tokenizer_ner)

    all_pred_full, all_value_preds_full = run_ner_model_SYNC(config.modelo_ner[0], data_loader=data_loader_full, device=device)
    df_data_full["pred_id"] = all_pred_full
    df_data_full["pred_label"] = df_data_full["pred_id"].map(id2label_ner)
    df_data_full = update_df_with_final_pred_entities_SYNC(df_data_full)

    all_pred_diags, all_value_preds_diags = run_ner_model_SYNC(config.modelo_ner[1], data_loader=data_loader_diags, device=device)
    df_data_diags["pred_id"] = all_pred_diags
    df_data_diags["pred_label"] = df_data_diags["pred_id"].map(id2label_ner)
    df_data_diags = update_df_with_final_pred_entities_SYNC(df_data_diags)

    cols_check = [c for c in df_data_diags.columns if c not in ["CLS Embedding", "Embeddings", "pred_id", "pred_label"]]
    assert df_data_diags[cols_check].equals(df_data_full[cols_check])
    mask = ~df_data_full["pred_label"].isin([id2label_ner[1], id2label_ner[2]])
    df_final_ner = df_data_diags.copy()
    df_final_ner.loc[mask, ["CLS Embedding", "Embeddings", "pred_id", "pred_label"]] = df_data_full.loc[mask, ["CLS Embedding", "Embeddings", "pred_id", "pred_label"]]

    df_final_ner = df_final_ner[['File', 'Tokens', 'Decoded span', 'Token idx', 'Text instance', 'pred_id', 'pred_label']]

    tmp = df_final_ner[df_final_ner["pred_id"].eq(1) | df_final_ner["pred_label"].eq("CLINENTITY")].copy()
    tmp["Tokens"] = tmp["Tokens"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    tmp["Token idx"] = tmp["Token idx"].apply(lambda x: tuple(ast.literal_eval(x) if isinstance(x, str) else x))
    tmp["entity"] = tmp["Tokens"].apply(lambda toks: " ".join(merge_sentencepiece_words(toks)))
    tmp = tmp.drop_duplicates(subset=["File", "Token idx", "entity"])

    entities_by_file = (tmp.groupby("File")["entity"].apply(list).reset_index(name="All Descriptions"))

    df_final_ner = (df_data[["archivo_origen", "Text"]].merge(entities_by_file, left_on="archivo_origen", right_on="File", how="left"))
    df_final_ner["All Descriptions"] = df_final_ner["All Descriptions"].apply(lambda x: x if isinstance(x, list) else [])
    df_final_ner = df_final_ner.rename(columns={"archivo_origen": "Original File"})[["Text", "All Descriptions", "Original File"]]

    # ------------------------------CIE10------------------------------
    data_loader = construct_loaders_icd10_SYNC(data=df_final_ner, tokenizer=tokenizer_ner)

    all_pred_yes_hs, all_value_preds_yes_hs = run_icd10_hs_model_SYNC(config.modelo_icd10_head[0], config.modelo_icd10_prediction[0], data_loader)
    all_pred_no_hs, all_value_preds_no_hs = run_icd10_no_hs_model_SYNC(config.modelo_icd10_head[1], data_loader)

    all_pred_no_hs = [config.id_no_hs_to_id_hs[i] for i in all_pred_no_hs]
    all_value_preds_no_hs = remap_ids(all_value_preds_no_hs, config.id_no_hs_to_id_hs)

    flattened_predictions_yes_hs = extract_flattened_predictions(all_value_preds_yes_hs, all_pred_yes_hs)
    flattened_predictions_no_hs = extract_flattened_predictions(all_value_preds_no_hs, all_pred_no_hs)

    results_yes_hd = apply_thresholds_icd10_SYNC(flattened_predictions_yes_hs, config.icd10_thresholds["YES_HS"])
    results_no_hd = apply_thresholds_icd10_SYNC(flattened_predictions_no_hs, config.icd10_thresholds["NO_HS"])

    final_results = [(all_pred_yes_hs[i] if yes[0] or not no[0] else all_pred_no_hs[i], yes) for i, (yes, no) in enumerate(zip(results_yes_hd, results_no_hd))]

    return final_results
    

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