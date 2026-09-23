from __future__ import annotations

import pandas as pd

from ...data_io.reader import read_ann_list, read_excel_single, read_json_single
from ...data_io.writing import write_ann_list
from ...models.schemas import DOCXToJSONSConfig
from ...services.common.icd_pred_funcs import (
    initialize_icd10_hs_head_model,
    initialize_icd10_hs_prediction_model,
    initialize_icd10_no_hs_head_model,
)
from ...services.common.tree_funcs import load_tree_hierarchical_module
from ...services.common.utils.ann_utils import icd_outputs_to_ann
from ...services.sync_funcs.icd_pred_funcs import (
    construct_loaders_icd as construct_loaders_icd_SYNC,
    create_mixed_icd10_threshold_results as create_mixed_icd10_threshold_results_SYNC,
    prepare_data as prepare_data_icd_SYNC,
    run_icd_classifier as run_icd_classifier_SYNC,
    add_standalone_icd_metadata as add_standalone_icd_metadata_SYNC
)
from .common import SEED, cleanup_cuda, load_input_dataframe, seed_everything


def read_icd10_reference(ctx) -> pd.DataFrame:
    """
    Read the ICD10 original dict

    Parameters
    ----------
        `ctx`: PipelineContext
            - Context with the types of variables to be used

    Returns
    -------
        `df_reference`: pd.DataFrame
            - DataFrame with the data from the original ICD dictionary
    """
    print("\n\033[1m0.1. Reading ICD10 dictionaries\033[0m\n\n")

    if ctx.cie_10_version == "2018":
        df_reference = read_excel_single(ctx, "Diagnosticos_ES2018.xlsx", sheet_name="finales", header=0)
        df_reference = df_reference[["codigo", "descripcion"]].reset_index(drop=True).rename(columns={"codigo": "Código", "descripcion": "Descripción"})
    elif ctx.cie_10_version == "2024":
        df_reference = read_excel_single(ctx, "Diagnosticos_ES2024.xlsx", sheet_name="ES2024 Completa + Marcadores", header=0)
        df_reference = df_reference[["Código", "Descripción"]].reset_index(drop=True)
    else:
        df_reference = read_excel_single(ctx, "Diagnosticos_ES2026.xlsx", sheet_name="ES2026 Completa + Marcadores", header=0)
        df_reference = df_reference[["Código", "Descripción"]].reset_index(drop=True)

    df_reference["Código"] = df_reference["Código"].astype(str).str.strip().str.replace("\xa0", "", regex=False)
    return df_reference[df_reference["Código"].str.match(r"^[A-Za-z]\d[A-Za-z0-9](?:\.[A-Za-z0-9]+)?(?:-[A-Za-z]\d[A-Za-z0-9](?:\.[A-Za-z0-9]+)?)?$")]


def initialize_variables(ctx, config: DOCXToJSONSConfig) -> DOCXToJSONSConfig:
    """
    Initialize the variables neeeded for the S1 ICD procedure

    Parameters
    ----------
        `ctx`: PipelineContext
            - Context with the types of variables to be used
        `config`: DOCXToJSONSConfig
            - Config with the empty variables where the initialize data must be introduced

    Returns
    -------
        ``: DOCXToJSONSConfig
            - Same config object with the necessary variables inputed
    """
    print("\n\033[1m0.5. Initializing ICD10 Models\033[0m\n\n")

    df_reference = read_icd10_reference(ctx)
    config.node_list = load_tree_hierarchical_module(df_reference)
    config.root = config.node_list["root"]
    config.root.set_indexes()

    id2label_icd_nohs = read_json_single(ctx.paths.docs_dir / "ICDTraining/NO_HS/id2label_icd_pred_nohs.json")
    id2label_icd_nohs = {int(k): v for k, v in id2label_icd_nohs.items()}
    label2id_icd_nohs = {v: int(k) for k, v in id2label_icd_nohs.items()}

    id2label_icd_hs = read_json_single(ctx.paths.docs_dir / "ICDTraining/HS/id2label_icd_pred_hs.json")
    id2label_icd_hs = {int(k): v for k, v in id2label_icd_hs.items()}
    label2id_icd_hs = {v: int(k) for k, v in id2label_icd_hs.items()}

    config.id_no_hs_to_id_hs = {id_no_hs: label2id_icd_hs[label_name] for label_name, id_no_hs in label2id_icd_nohs.items()}
    config.id2label_ICD10 = [id2label_icd_nohs, id2label_icd_hs]
    config.label2id_ICD10 = [label2id_icd_nohs, label2id_icd_hs]
    config.icd10_thresholds = read_json_single(ctx.paths.docs_dir / "thresholds_ICD10.json")

    config.modelo_icd10_head = [
        initialize_icd10_no_hs_head_model(ctx, None, None, ctx.paths.docs_dir / "ICDTraining/NO_HS/icd_pred_nohs_checkpoint.pt", label2id_icd_nohs, config.root, base_encoder_name=ctx.base_encoder_name),
        initialize_icd10_hs_head_model(ctx, None, None, ctx.paths.docs_dir / "ICDTraining/HS/icd_pred_hs_checkpoint.pt", config.root, base_encoder_name=ctx.base_encoder_name),
    ]
    config.modelo_icd10_prediction = [
        None,
        initialize_icd10_hs_prediction_model(ctx, ctx.paths.docs_dir / "ICDTraining/HS/icd_pred_hs_predictor_checkpoint.pt", label2id_icd_hs, config.root),
    ]

    return config


def run(ctx, config: DOCXToJSONSConfig) -> None:
    """
    S1 Deterministic-Sync ICD principal function
    Only executed in a different thread from the original process. All the resources occupied will be liverated later

    Parameters
    ----------
        `ctx`: PipelineContext
            - Context with the types of variables to be used
        `config`: DOCXToJSONSConfig
            - Config with the empty variables where the initialize data must be introduced

    Returns
    -------
        ``: None
            - The S1 ICD process is completed
    """
    seed_everything(SEED)
    config = initialize_variables(ctx, config)
    df_data = load_input_dataframe(ctx)

    print("\n\033[1m1.4. ICD10\033[0m\n\n")

    dict_ann_icd = read_ann_list(ctx.paths.data_intermediate / ctx.folder_and_archive_name / "ANN" / "ATT/Status")
    df_ann_icd = pd.DataFrame([{"archivo_origen": file_name, **ann_data} for file_name, ann_data in dict_ann_icd.items()])

    data_prepared_icd, all_window_labels_icd, file_names_icd = prepare_data_icd_SYNC(df_data, data_ann=df_ann_icd)
    _, data_loader_full_icd_nohs, _, _ = construct_loaders_icd_SYNC(data_prepared_icd, None, file_names_icd, label2id=config.label2id_ICD10[0], id2label=config.id2label_ICD10[0], seed=SEED, batch_size=32, hs=False, root=config.root, shuffle=False)
    _, data_loader_full_icd_hs, _, _ = construct_loaders_icd_SYNC(data_prepared_icd, None, file_names_icd, label2id=config.label2id_ICD10[1], id2label=config.id2label_ICD10[1], seed=SEED, batch_size=32, hs=True, root=config.root, shuffle=False)

    results_icd_nohs = run_icd_classifier_SYNC(config.modelo_icd10_head[0], data_loader_full_icd_nohs, ctx.device, config.id2label_ICD10[0], hs=False, train=False)
    results_icd_hs = run_icd_classifier_SYNC(config.modelo_icd10_head[1], data_loader_full_icd_hs, ctx.device, config.id2label_ICD10[1], hs=True, train=False, criterion=config.modelo_icd10_prediction[1])
    mixed_icd_results = create_mixed_icd10_threshold_results_SYNC(results_icd_hs, results_icd_nohs, config.icd10_thresholds["YES_HS"], config.icd10_thresholds["NO_HS"], id2label_hs=config.id2label_ICD10[1], id_no_hs_to_id_hs=config.id_no_hs_to_id_hs)

    icd_ann_format_nohs = icd_outputs_to_ann(add_standalone_icd_metadata_SYNC(results_icd_nohs, config.id2label_ICD10[0], "NO_HS", config.node_list), config.id2label_ICD10[0], data_loader=data_loader_full_icd_nohs, original_ann=df_ann_icd, node_list=config.node_list)
    icd_ann_format_hs = icd_outputs_to_ann(add_standalone_icd_metadata_SYNC(results_icd_hs, config.id2label_ICD10[1], "HS", config.node_list), config.id2label_ICD10[1], data_loader=data_loader_full_icd_hs, original_ann=df_ann_icd, node_list=config.node_list)
    icd_ann_format_mix = icd_outputs_to_ann(mixed_icd_results, config.id2label_ICD10[1], data_loader=data_loader_full_icd_hs, original_ann=df_ann_icd, node_list=config.node_list)

    write_ann_list(ctx, icd_ann_format_nohs, ctx.paths.data_intermediate / ctx.folder_and_archive_name / "ANN" / "ICD10/NO_HS")
    write_ann_list(ctx, icd_ann_format_hs, ctx.paths.data_intermediate / ctx.folder_and_archive_name / "ANN" / "ICD10/HS")
    write_ann_list(ctx, icd_ann_format_mix, ctx.paths.data_intermediate / ctx.folder_and_archive_name / "ANN" / "ICD10/Mix")

    cleanup_cuda(config.modelo_icd10_head, config.modelo_icd10_prediction, results_icd_nohs, results_icd_hs, data_loader_full_icd_nohs, data_loader_full_icd_hs)
    config.modelo_icd10_head, config.modelo_icd10_prediction, results_icd_nohs, results_icd_hs, data_loader_full_icd_nohs, data_loader_full_icd_hs = None, None, None, None, None, None
