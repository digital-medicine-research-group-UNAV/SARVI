from __future__ import annotations

import pandas as pd

from ...data_io.reader import read_ann_list, read_json_single
from ...data_io.writing import write_ann_list
from ...models.schemas import DOCXToJSONSConfig
from ...services.common.att_funcs import initialize_att_model
from ...services.common.utils.ann_utils import att_outputs_to_ann
from ...services.common.utils.nn_utils import ensure_entity_marker_tokens
from ...services.sync_funcs.att_funcs import (
    construct_loader_att as construct_loader_att_SYNC,
    prepare_data as prepare_data_att_SYNC,
    run_att_classifier as run_att_classifier_SYNC,
)
from .common import SEED, cleanup_cuda, load_input_dataframe, seed_everything


def initialize_variables(ctx, config: DOCXToJSONSConfig) -> DOCXToJSONSConfig:
    """
    Initialize the variables neeeded for the S1 ATT procedure

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
    print("\n\033[1m0.4. Initializing ATT Model\033[0m\n\n")

    id2label_att_status = read_json_single(ctx.paths.docs_dir / "ATTTraining/Status/id2label_status_attclassifier.json")
    id2label_att_status = {int(k): v for k, v in id2label_att_status.items()}
    label2id_att_status = {v: int(k) for k, v in id2label_att_status.items()}

    config.id2label_ATT = [id2label_att_status]
    config.label2id_ATT = [label2id_att_status]
    config.tokenizer_att = [
        ensure_entity_marker_tokens(None, ["DISO"], "att", base_encoder_name=ctx.base_encoder_name)
    ]
    config.modelo_att = [
        initialize_att_model(ctx, None, len(id2label_att_status), "ATTTraining/Status/status_attclassifier_checkpoint.pt", base_encoder_name=ctx.base_encoder_name, tokenizer=config.tokenizer_att[0])
    ]

    return config


def run(ctx, config: DOCXToJSONSConfig) -> None:
    """
    S1 Deterministic-Sync ATT principal function
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
            - The S1 ATT process is completed
    """
    seed_everything(SEED)
    config = initialize_variables(ctx, config)
    df_data = load_input_dataframe(ctx)

    print("\n\033[1m1.3. ATT\033[0m\n\n")

    dict_ann_att = read_ann_list(ctx.paths.data_intermediate / ctx.folder_and_archive_name / "ANN" / "RE/Time")
    df_ann_att = pd.DataFrame([{"archivo_origen": file_name, **ann_data} for file_name, ann_data in dict_ann_att.items()])

    data_prepared_att, all_attribute_labels_att, file_names_att, _ = prepare_data_att_SYNC(data=df_data, data_ann=df_ann_att, entity_label="DISO", tokenizer=config.tokenizer_att[0], w=2, print_warnings=False, return_tokenizer=False)
    _, data_loader_full_att, _, _ = construct_loader_att_SYNC(data_prepared_att, None, file_names=file_names_att, label2id=config.label2id_ATT[0], id2label=config.id2label_ATT[0], seed=SEED, batch_size=32)

    results_att = run_att_classifier_SYNC(config.modelo_att[0], data_loader=data_loader_full_att, device=ctx.device, id2label=config.id2label_ATT[0], train=False)

    att_ann_format = att_outputs_to_ann(results_att, config.id2label_ATT[0], data_loader=data_loader_full_att, original_ann=df_ann_att)

    write_ann_list(ctx, att_ann_format, ctx.paths.data_intermediate / ctx.folder_and_archive_name / "ANN" / "ATT/Status")

    cleanup_cuda(config.modelo_att, results_att, data_loader_full_att, data_prepared_att)
    config.modelo_att, results_att, data_loader_full_att, data_prepared_att = None, None, None, None
