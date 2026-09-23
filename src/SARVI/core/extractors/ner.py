from __future__ import annotations

import pandas as pd

from ...data_io.reader import read_ann_list, read_json_single
from ...data_io.writing import write_ann_list
from ...models.schemas import DOCXToJSONSConfig
from ...services.common.ner_funcs import (
    create_ultimate_df_ann,
    initialize_bio_ner_model,
    initialize_span_ner_model,
    model_ner,
    tokenizer_ner,
)
from ...services.common.utils.ann_utils import ner_outputs_to_ann
from ...services.sync_funcs.ner_funcs import (
    construct_loaders_ner as construct_loaders_ner_SYNC,
    prepare_data as prepare_data_ner_SYNC,
    run_bio_nerclassifier as run_bio_nerclassifier_SYNC,
    run_span_nerclassifier as run_span_nerclassifier_SYNC,
)
from .common import SEED, cleanup_cuda, load_input_dataframe, seed_everything


def initialize_variables(ctx, config: DOCXToJSONSConfig) -> DOCXToJSONSConfig:
    """
    Initialize the variables neeeded for the S1 NER procedure

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
    print("\n\033[1m0.2. Initializing NER Models\033[0m\n\n")

    id2label_ner_bio = read_json_single(ctx.paths.docs_dir / "NERTraining/BIO/CT-EBM-SP/id2label_bio_nerclassifier.json")
    id2label_ner_bio = {int(k): v for k, v in id2label_ner_bio.items()}
    label2id_ner_bio = {v: int(k) for k, v in id2label_ner_bio.items()}

    id2label_ner_span = read_json_single(ctx.paths.docs_dir / "NERTraining/Span/CT-EBM-SP/id2label_span_nerclassifier.json")
    id2label_ner_span = {int(k): v for k, v in id2label_ner_span.items()}
    label2id_ner_span = {v: int(k) for k, v in id2label_ner_span.items()}

    config.id2label_NER = [id2label_ner_bio, id2label_ner_span]
    config.label2id_NER = [label2id_ner_bio, label2id_ner_span]
    config.modelo_ner = [
        initialize_bio_ner_model(ctx, None, len(id2label_ner_bio), "NERTraining/BIO/CT-EBM-SP/bio_nerclassifier_checkpoint.pt", base_encoder_name=ctx.base_encoder_name),
        initialize_span_ner_model(ctx, None, len(id2label_ner_span), id2label_ner_span, label2id_ner_span, None, "NERTraining/Span/CT-EBM-SP/span_nerclassifier_checkpoint.pt", True, base_encoder_name=ctx.base_encoder_name),
    ]

    return config


def run(ctx, config: DOCXToJSONSConfig) -> None:
    """
    S1 Deterministic-Sync NER principal function
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
            - The S1 NER process is completed
    """
    seed_everything(SEED)
    config = initialize_variables(ctx, config)
    df_data = load_input_dataframe(ctx)

    print("\n\033[1m1.1. NER\033[0m\n\n")

    data_prepared, all_window_labels, file_names = prepare_data_ner_SYNC(df_data, padding=True, tokenizer=tokenizer_ner, model=model_ner, device=ctx.device)

    _, data_loader_full_bio, _, _ = construct_loaders_ner_SYNC(data_prepared, all_window_labels, file_names, ner_type="bio", label2id=config.label2id_NER[0], id2label=config.id2label_NER[0], seed=SEED, batch_size=32)
    _, data_loader_full_span, _, _ = construct_loaders_ner_SYNC(data_prepared, all_window_labels, file_names, ner_type="span", label2id=config.label2id_NER[1], id2label=config.id2label_NER[1], seed=SEED, ignore_o_labels=False, batch_size=32)

    results_bio = run_bio_nerclassifier_SYNC(config.modelo_ner[0], data_loader=data_loader_full_bio, device=ctx.device, id2label=config.id2label_NER[0], train=False)
    results_span = run_span_nerclassifier_SYNC(config.modelo_ner[1], data_loader=data_loader_full_span, device=ctx.device, id2label=config.id2label_NER[1], train=False)

    ner_ann_format_bio = ner_outputs_to_ann(results_bio, config.id2label_NER[0], data_loader=data_loader_full_bio, original_texts=df_data, tokenizer=tokenizer_ner)
    ner_ann_format_span = ner_outputs_to_ann(results_span, config.id2label_NER[1], data_loader=data_loader_full_span, original_texts=df_data, tokenizer=tokenizer_ner)

    write_ann_list(ctx, ner_ann_format_bio, ctx.paths.data_intermediate / ctx.folder_and_archive_name / "ANN" / "NER/BIO")
    write_ann_list(ctx, ner_ann_format_span, ctx.paths.data_intermediate / ctx.folder_and_archive_name / "ANN" / "NER/Span")

    ner_ann_format_ultimate = create_ultimate_df_ann(
        principal_df_ann=pd.DataFrame([{"archivo_origen": file_name, **ann_data} for file_name, ann_data in read_ann_list(ctx.paths.data_intermediate / ctx.folder_and_archive_name / "ANN" / "NER/BIO").items()]),
        secondary_df_ann=pd.DataFrame([{"archivo_origen": file_name, **ann_data} for file_name, ann_data in read_ann_list(ctx.paths.data_intermediate / ctx.folder_and_archive_name / "ANN" / "NER/Span").items()]),
        overlapped_entities=False,
        extend_entities=True,
        add_extra_entities=False,
        return_type="ann_list",
        original_texts=df_data,
    )

    write_ann_list(ctx, ner_ann_format_ultimate, ctx.paths.data_intermediate / ctx.folder_and_archive_name / "ANN" / "NER/Mix")

    cleanup_cuda(config.modelo_ner, results_bio, results_span, data_loader_full_bio, data_loader_full_span, data_prepared)
    config.modelo_ner, results_bio, results_span, data_loader_full_bio, data_loader_full_span, data_prepared = None, None, None, None, None, None
