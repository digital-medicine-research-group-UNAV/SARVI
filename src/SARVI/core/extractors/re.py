from __future__ import annotations

import pandas as pd

from ...data_io.reader import read_ann_list, read_json_single
from ...data_io.writing import write_ann_list
from ...models.schemas import DOCXToJSONSConfig
from ...services.common.re_funcs import initialize_re_model
from ...services.common.utils.ann_utils import re_outputs_to_ann
from ...services.common.utils.nn_utils import ensure_entity_marker_tokens
from ...services.sync_funcs.re_funcs import (
    construct_loader_re as construct_loader_re_SYNC,
    prepare_data as prepare_data_re_SYNC,
    run_re_classifier as run_re_classifier_SYNC,
)
from .common import SEED, cleanup_cuda, load_input_dataframe, seed_everything


def initialize_variables(ctx, config: DOCXToJSONSConfig) -> DOCXToJSONSConfig:
    """
    Initialize the variables neeeded for the S1 RE procedure

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
    print("\n\033[1m0.3. Initializing RE Models\033[0m\n\n")

    config.re_token_distance_bins = {0: (0, 0), 1: (1, 2), 2: (3, 5), 3: (6, 10), 4: (11, 20), 5: (21, 40), 6: (41, 80), 7: (81, 160), 8: (161, float("inf"))}
    config.re_negative_difficulty_by_bin = {0: "hard", 1: "hard", 2: "hard", 3: "hard", 4: "medium", 5: "medium", 6: "easy", 7: "easy", 8: "easy"}

    id2label_re_experience = read_json_single(ctx.paths.docs_dir / "RETraining/DISO_LIVB/id2label_diso_livb_reclassifier.json")
    id2label_re_experience = {int(k): v for k, v in id2label_re_experience.items()}
    label2id_re_experience = {v: int(k) for k, v in id2label_re_experience.items()}

    id2label_re_negation = read_json_single(ctx.paths.docs_dir / "RETraining/DISO_Negcue/id2label_diso_negcue_reclassifier.json")
    id2label_re_negation = {int(k): v for k, v in id2label_re_negation.items()}
    label2id_re_negation = {v: int(k) for k, v in id2label_re_negation.items()}

    id2label_re_speculation = read_json_single(ctx.paths.docs_dir / "RETraining/DISO_Speccue/id2label_diso_speccue_reclassifier.json")
    id2label_re_speculation = {int(k): v for k, v in id2label_re_speculation.items()}
    label2id_re_speculation = {v: int(k) for k, v in id2label_re_speculation.items()}

    id2label_re_time = read_json_single(ctx.paths.docs_dir / "RETraining/DISO_Date/id2label_diso_date_reclassifier.json")
    id2label_re_time = {int(k): v for k, v in id2label_re_time.items()}
    label2id_re_time = {v: int(k) for k, v in id2label_re_time.items()}

    config.id2label_RE = [id2label_re_experience, id2label_re_negation, id2label_re_speculation, id2label_re_time]
    config.label2id_RE = [label2id_re_experience, label2id_re_negation, label2id_re_speculation, label2id_re_time]

    config.tokenizer_re = [
        ensure_entity_marker_tokens(None, ["DISO", "LIVB"], "re", base_encoder_name=ctx.base_encoder_name),
        ensure_entity_marker_tokens(None, ["DISO", "Neg_cue"], "re", base_encoder_name=ctx.base_encoder_name),
        ensure_entity_marker_tokens(None, ["DISO", "Spec_cue"], "re", base_encoder_name=ctx.base_encoder_name),
        ensure_entity_marker_tokens(None, ["DISO", "Date"], "re", base_encoder_name=ctx.base_encoder_name),
    ]

    config.modelo_re = [
        initialize_re_model(ctx, None, len(id2label_re_experience), len(config.re_token_distance_bins), "RETraining/DISO_LIVB/diso_livb_reclassifier_checkpoint.pt", base_encoder_name=ctx.base_encoder_name, tokenizer=config.tokenizer_re[0]),
        initialize_re_model(ctx, None, len(id2label_re_negation), len(config.re_token_distance_bins), "RETraining/DISO_Negcue/diso_negcue_reclassifier_checkpoint.pt", base_encoder_name=ctx.base_encoder_name, tokenizer=config.tokenizer_re[1]),
        initialize_re_model(ctx, None, len(id2label_re_speculation), len(config.re_token_distance_bins), "RETraining/DISO_Speccue/diso_speccue_reclassifier_checkpoint.pt", base_encoder_name=ctx.base_encoder_name, tokenizer=config.tokenizer_re[2]),
        initialize_re_model(ctx, None, len(id2label_re_time), len(config.re_token_distance_bins), "RETraining/DISO_Date/diso_date_reclassifier_checkpoint.pt", base_encoder_name=ctx.base_encoder_name, tokenizer=config.tokenizer_re[3]),
    ]

    return config


def run_re_models_deterministc_sync(ctx, config: DOCXToJSONSConfig, df_data: pd.DataFrame, entity_label_1: str, entity_label_2: str, idx: int, folder_input: str, folder_output: str):
    """
    S1 Deterministic-Sync RE function to construct the necesarry data to run the different models

    Parameters
    ----------
        `ctx`: PipelineContext
            - Context with the types of variables to be used
        `config`: DOCXToJSONSConfig
            - Config with the empty variables where the initialize data must be introduced
        `df_data`: pd.DataFrame
            - 

    Returns
    -------
        ``: None
            - The S1 RE process is completed
    """
    dict_ann = read_ann_list(ctx.paths.data_intermediate / ctx.folder_and_archive_name / "ANN" / folder_input)
    df_ann = pd.DataFrame([{"archivo_origen": file_name, **ann_data} for file_name, ann_data in dict_ann.items()])

    data_prepared_re, all_relation_labels_re, file_names_re, _ = prepare_data_re_SYNC(data=df_data, data_ann=df_ann, entity_label_1=entity_label_1, entity_label_2=entity_label_2, tokenizer=config.tokenizer_re[idx], w=2, v=1, token_distance_bins=config.re_token_distance_bins, negative_difficulty_by_bin=config.re_negative_difficulty_by_bin, print_warnings=False, return_tokenizer=False)
    _, data_loader_full_re, _, _ = construct_loader_re_SYNC(data_prepared_re, None, file_names=file_names_re, label2id=config.label2id_RE[idx], id2label=config.id2label_RE[idx], seed=SEED, batch_size=32)

    results_re = run_re_classifier_SYNC(config.modelo_re[idx], data_loader=data_loader_full_re, device=ctx.device, id2label=config.id2label_RE[idx], train=False)

    re_ann_format = re_outputs_to_ann(results_re, config.id2label_RE[idx], data_loader=data_loader_full_re, original_ann=df_ann)
    write_ann_list(ctx, re_ann_format, ctx.paths.data_intermediate / ctx.folder_and_archive_name / "ANN" / folder_output)

    cleanup_cuda(config.modelo_re[idx], results_re, data_loader_full_re, data_prepared_re)
    config.modelo_re[idx], results_re, data_loader_full_re, data_prepared_re = None, None, None, None


def run(ctx, config: DOCXToJSONSConfig) -> None:
    """
    S1 Deterministic-Sync RE principal function
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
            - The S1 RE process is completed
    """
    seed_everything(SEED)
    config = initialize_variables(ctx, config)
    df_data = load_input_dataframe(ctx)

    print("\n\033[1m1.2. RE\033[0m\n\n")

    print("\n\033[1m1.2.1. RE - Experience\033[0m\n\n")
    run_re_models_deterministc_sync(ctx, config, df_data, entity_label_1="DISO", entity_label_2="LIVB", idx=0, folder_input="NER/Mix", folder_output="RE/Experience")

    print("\n\033[1m1.2.2. RE - Negation\033[0m\n\n")
    run_re_models_deterministc_sync(ctx, config, df_data, entity_label_1="DISO", entity_label_2="Neg_cue", idx=1, folder_input="RE/Experience", folder_output="RE/Negation")

    print("\n\033[1m1.2.3. RE - Speculation\033[0m\n\n")
    run_re_models_deterministc_sync(ctx, config, df_data, entity_label_1="DISO", entity_label_2="Spec_cue", idx=2, folder_input="RE/Negation", folder_output="RE/Speculation")

    print("\n\033[1m1.2.4. RE - Time\033[0m\n\n")
    run_re_models_deterministc_sync(ctx, config, df_data, entity_label_1="DISO", entity_label_2="Date", idx=3, folder_input="RE/Speculation", folder_output="RE/Time")

    cleanup_cuda(config.modelo_re)
    config.modelo_re, config.tokenizer_re = None, None
