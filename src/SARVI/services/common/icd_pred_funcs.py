from __future__ import annotations

import torch
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Any, TYPE_CHECKING

from sklearn.metrics import (
    f1_score as f1_score_sklearn,
    precision_score as precision_score_sklearn,
    recall_score as recall_score_sklearn,
    classification_report as classification_report_sklearn
)

from transformers import AutoModel, AutoTokenizer

from .utils.nn_utils import (
    tensor_to_python
)
from .utils.icd_preds_utils import (
    extract_brat_note_codes, has_value, value_to_int
)
from .utils.ann_utils import (
    normalize_ann_lines, parse_brat_text_boundaries, ann_parts, related_ent
)

from .ner_funcs import (
    tokenizer_ner, model_ner
)

from ...data_io.reader import (
    read_torch_checkpoint, read_parquet_file
)

from ...models.neural_networks import (
    ICD10Predictor_HS_Head, ICD10Predictor_HS_CrossEntropyLoss, ICD10Predictor_NO_HS
)

if TYPE_CHECKING:
    from hierarchicalsoftmax import SoftmaxNode
    from ...models.schemas import PipelineContext

###
device = "cuda" if torch.cuda.is_available() else "cpu"
###

def embed_texts(texts: list, batch_size: int = 64, max_length: int = 512, show_tqdm: bool = False):
    """
    Genera embeddings para una lista de textos usando el tokenizer y el modelo NER, aplicando pooling medio sobre los estados ocultos de los tokens.

    Parameters
    ----------
        `texts`: list
            - Lista de textos que se quieren convertir en embeddings

        `batch_size`: int
            - Número de textos procesados en cada batch. Por defecto es **64**

        `max_length`: int
            - Longitud máxima de tokens permitida por texto durante la tokenización. Por defecto es **512**

        `show_tqdm`: bool
            - Indica si se debe mostrar una barra de progreso con **tqdm**. Por defecto es **False**

    Returns
    -------
        `embeddings`: torch.Tensor
            - Tensor con los embeddings de todos los textos, reconstruidos en el mismo orden que la lista original de entrada
    """
    cache = {}

    new_texts = [t for t in texts if t not in cache]
    new_texts = list(set(new_texts))

    with torch.no_grad():
        iterator = range(0, len(new_texts), batch_size)
        if show_tqdm:
            iterator = tqdm(iterator, desc="Embedding texts", unit="text")
        for i in iterator:
            batch = new_texts[i:i+batch_size]
            
            tokens = tokenizer_ner(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt", add_special_tokens=False)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            outputs = model_ner(**tokens)
            hidden = outputs.last_hidden_state

            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)

            pooled = pooled.cpu()

            for text, emb in zip(batch, pooled):
                cache[text] = emb

    # rebuild embeddings in original order
    embeddings = [cache[t] for t in texts]
    return torch.stack(embeddings)

def create_all_labels_desc(ctx: "PipelineContext", label2id: dict):
    """
    Crea un diccionario de embeddings de descripciones para todas las etiquetas, combinando información de distintos conjuntos de datos.

    Parameters
    ----------
        `ctx`: PipelineContext
            - Contexto del modelo completo
        `label2id`: dict
            - Diccionario con la traduccion de códigos a ids

    Returns
    -------
        `all_labels_desc`: dict
            - Diccionario donde cada clave es el ID numérico de una etiqueta y cada valor es un tensor con los embeddings de las descripciones asociadas a esa etiqueta
    """
    # df = read_parquet_file(ctx, "old/full_icd10_2026.parquet")
    # df["target_id"] = df["Code Perceiver"].map(label2id)

    # all_labels_desc_ORIGINAL = defaultdict(list)
    # for code in tqdm(df.groupby(["target_id"], group_keys=True)[["Description"]], desc="Gathering diags definitions: Original dictionary", unit="diag"):
    #     all_labels_desc_ORIGINAL[code[0][0]] = code[1]["Description"].to_list()

    # ####################################

    # df = read_parquet_file(ctx, "old/codiesp_train.parquet")
    # df = df[df["All Code Full"].str.len() > 0]
    # df["target_id"] = df["All Code Perceiver"].apply(lambda x: [label2id[c] for c in x])

    # all_labels_desc_CODIESP = defaultdict(list)
    # for _,row in tqdm(df.iterrows(), total=len(df), desc="Gathering diags definitions: CodiEsp", unit="diag"):
    #     for code,desc in zip(row["target_id"], row["All Description"]):
    #         all_labels_desc_CODIESP[code].append(desc)

    # ####################################

    # df = read_parquet_file(ctx, "old/cares_train.parquet")
    # df = df[df["All Code Full"].str.len() > 0]
    # df["target_id"] = df["All Code Perceiver"].apply(lambda x: [label2id[c] for c in x])

    # all_labels_desc_CARES = defaultdict(list)
    # for _,row in tqdm(df.iterrows(), total=len(df), desc="Gathering diags definitions: CARES", unit="diag"):
    #     for code,desc in zip(row["target_id"], row["All Description"]):
    #         all_labels_desc_CARES[code].append(desc)

    # ####################################

    # df = read_parquet_file(ctx, "old/cares_test.parquet")
    # df = df[df["All Code Full"].str.len() > 0]
    # df["target_id"] = df["All Code Perceiver"].apply(lambda x: [label2id[c] for c in x])

    # for _,row in tqdm(df.iterrows(), total=len(df), desc="Gathering diags definitions: CARES", unit="diag"):
    #     for code,desc in zip(row["target_id"], row["All Description"]):
    #         all_labels_desc_CARES[code].append(desc)

    # ####################################

    # df = read_parquet_file(ctx, "old/not_codiesp_cares_general.parquet")
    # all_labels_desc_REST = dict(zip(df["All Code Full"].map(label2id), df["All Description"]))
    
    # ####################################

    # all_labels_desc = {}

    # for code in tqdm(all_labels_desc_ORIGINAL.keys(), total=len(all_labels_desc_ORIGINAL), desc="Gathering diags definitions: Embedding definitions", unit="diag"):
    #     codes_desc_REAL = list(set(all_labels_desc_CODIESP[code] + all_labels_desc_CARES[code]))
    #     if codes_desc_REAL == []:
    #         codes_desc_REAL = list(set(all_labels_desc_REST[code]))
    #     codes_desc_REAL = list(set(codes_desc_REAL + [all_labels_desc_ORIGINAL[code][0]]))
    #     all_labels_desc[code] = embed_texts(codes_desc_REAL, show_tqdm=False)

    df = read_parquet_file(ctx, "all_label_description.parquet")
    all_labels_desc_df = dict(zip(df["All Code Full"].map(label2id), df["All Description"]))

    all_labels_desc = {}
    for code, desc in tqdm(all_labels_desc_df.items(), total=len(all_labels_desc_df), desc="Gathering diags definitions: Embedding definitions", unit="diag"):
        all_labels_desc[code] = embed_texts(desc, show_tqdm=False)

    return all_labels_desc

def extract_flattened_predictions(data: list, all_preds: list):
    """
    Extrae y aplana las predicciones guardadas en una estructura anidada, validando que coincidan con las predicciones globales recibidas.

    Parameters
    ----------
        `data`: list
            - Estructura anidada que contiene grupos de datos. Cada grupo debe contener, en su segunda posición, listas de arrays o tuplas con valores de predicción

        `all_preds`: list
            - Lista o array con las predicciones globales que se usan para comprobar la coherencia con los valores extraídos de `data`

    Returns
    -------
        `flattened_predictions`: list
            - Lista de series de predicciones aplanadas. Cada elemento contiene los valores convertidos a **float** correspondientes a una secuencia extraída
    """
    flattened_predictions = []
    indice_global = 0

    for grupo_idx, grupo in enumerate(data):
        if len(grupo) <= 1:
            continue

        lista_arrays_tuplas = grupo[1]

        for arr_idx, arr_tuplas in enumerate(lista_arrays_tuplas):
            if len(arr_tuplas) == 0:
                continue

            valores = []
            for tupla_paso in arr_tuplas:
                value = tensor_to_python(tupla_paso[0])
                while isinstance(value, (list, tuple)):
                    value = value[0]
                valores.append(float(value))

            flattened_predictions.append(valores)

            if indice_global >= len(all_preds):
                warnings.warn(f"No hay suficientes elementos en all_targets/all_preds para la serie {indice_global}.")
                break

            ultimo_segundo_valor = tensor_to_python(arr_tuplas[-1][1], squeeze_single=True)
            pred_actual = tensor_to_python(all_preds[indice_global], squeeze_single=True)

            if ultimo_segundo_valor != pred_actual:
                warnings.warn(f"Desajuste en índice {indice_global}: all_preds[{indice_global}]={pred_actual} pero el segundo valor de la última tupla es {ultimo_segundo_valor}. (grupo={grupo_idx}, elemento={arr_idx})")

            indice_global += 1

    if len(all_preds) != len(flattened_predictions):
        warnings.warn(f"Número de series ({len(flattened_predictions)}) distinto de ({len(all_preds)}).")

    return flattened_predictions

def result_ann_row_to_diagnosticos(row):
    text, label, code, rs, ats = ann_parts(row)

    diagnosticos = []

    for target_id, ent_label in label.items():
        if ent_label != "DISO":
            continue

        diagnostico_extraido = text.get(target_id, "")
        codigo_cie10 = code.get(target_id, "")

        fecha_alta = (related_ent(target_id, "Before", "Date", text, label, rs) or related_ent(target_id, "Overlap", "Date", text, label, rs) or "none")
        fecha_baja = (related_ent(target_id, "After", "Date", text, label, rs) or related_ent(target_id, "Overlap", "Date", text, label, rs) or "none")

        pertenencia = "personal"
        for line in ats:
            p = line.split("\t")
            if len(p) >= 2:
                bits = p[1].split()
                if len(bits) >= 2 and bits[0] == "Family_history_of" and bits[1] == target_id:
                    pertenencia = "familiar"
                    break

        if pertenencia == "personal":
            livb = related_ent(target_id, "Experiences", "LIVB", text, label, rs)
            if livb:
                pertenencia = livb

        diagnosticos.append({"diagnostico_extraido": diagnostico_extraido, "codigo_CIE10": codigo_cie10, "fecha_alta": fecha_alta, "fecha_baja": fecha_baja, "pertenencia": pertenencia, "ann_ent_id": target_id})

    return diagnosticos

def extract_diso_diagnoses(text: str, ann_row: pd.Series, t_col: str = "T", code_col: str = "#", output_only_max_conf: bool = False, output_only_judged_if_possible: bool = False) -> tuple[list[str], list[str] | None]:
    note_codes = extract_brat_note_codes(ann_row[code_col]) if code_col in ann_row and has_value(ann_row[code_col]) else None
    diagnoses = []
    icd_codes = [] if note_codes is not None else None

    for line in normalize_ann_lines(ann_row.get(t_col)):
        parts = str(line).split("\t")
        if len(parts) < 2:
            continue

        ann_id = parts[0]
        label_and_boundaries = parts[1].split(maxsplit=1)
        if len(label_and_boundaries) != 2:
            continue

        label, boundary_text = label_and_boundaries
        if label != "DISO":
            continue

        char_spans = parse_brat_text_boundaries(boundary_text)
        if not char_spans:
            continue

        diagnosis = " ".join(text[start:end] for start, end in char_spans).strip()

        if icd_codes is not None:
            code, confidence_source, judger = note_codes.get(ann_id)
            if output_only_max_conf and confidence_source == "HS_FALLBACK":
                continue
            if output_only_judged_if_possible and judger == "False":
                continue
            icd_codes.append(code)
            diagnoses.append(diagnosis)
        else:
            diagnoses.append(diagnosis)

    if icd_codes is not None and any(code is None for code in icd_codes):
        icd_codes = None

    return diagnoses, icd_codes

def classification_report_icd_codiesp(id2label, seed, targets=None, preds=None, file_names=None, results=None, dataset_icds=None, level="category", shuffle=False):
    """
    @author: antonio

    COPIED DIRECLY FROM https://github.com/TeMU-BSC/codiesp-evaluation-script/blob/master/comp_f1_diag_proc.py

    - level: `diagnosis` or `category`
    """
    def dataloader_order_indices(dataset_size, seed=seed, shuffle=shuffle):
        if not shuffle:
            return list(range(dataset_size))
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.randperm(dataset_size, generator=generator).tolist()
    def code_from_id(label_id):
        label_id = int(label_id)
        if label_id == -1:
            return None
        return str(id2label[label_id]).lower()
    def category_from_code(code):
        if code is None:
            return None
        return str(code).split(".")[0].lower()
    def codiesp_dataframe(label_ids, file_names, level="diagnosis"):
        rows = []
        for file_name, label_id in zip(file_names, label_ids):
            code = code_from_id(label_id)
            if code is None:
                continue
            if level == "category":
                code = category_from_code(code)
            elif level != "diagnosis":
                raise ValueError(f"Unsupported CodiEsp level: {level}")
            rows.append({"clinical_case": str(file_name), "code": code})
        return pd.DataFrame(rows, columns=["clinical_case", "code"])

    if targets is None and preds is None:
        order = dataloader_order_indices(len(dataset_icds))
        targets = [int(value) for value in np.asarray([dataset_icds.icd_codes[idx] for idx in order]).reshape(-1).tolist()]
        preds = [int(value) for value in np.asarray(results["all_pred"]).reshape(-1).tolist()]
        file_names = [dataset_icds.file_names[idx] for idx in order]

    df_gs = codiesp_dataframe(targets, file_names, level=level)
    df_pred = codiesp_dataframe(preds, file_names, level=level)

    Pred_Pos_per_cc = df_pred.drop_duplicates(subset=['clinical_case', "code"]).groupby("clinical_case")["code"].count()
    Pred_Pos = df_pred.drop_duplicates(subset=['clinical_case', "code"]).shape[0]
    
    # Gold Standard Positives:
    GS_Pos_per_cc = df_gs.drop_duplicates(subset=['clinical_case', "code"]).groupby("clinical_case")["code"].count()
    GS_Pos = df_gs.drop_duplicates(subset=['clinical_case', "code"]).shape[0]
    cc = set(df_gs.clinical_case.tolist())
    TP_per_cc = pd.Series(dtype=float)
    for c in cc:
        pred = set(df_pred.loc[df_pred['clinical_case']==c,'code'].values)
        gs = set(df_gs.loc[df_gs['clinical_case']==c,'code'].values)
        TP_per_cc[c] = len(pred.intersection(gs))
        
    TP = sum(TP_per_cc.values)
    
    # Calculate Final Metrics:
    P_per_cc =  TP_per_cc / Pred_Pos_per_cc
    P = TP / Pred_Pos
    R_per_cc = TP_per_cc / GS_Pos_per_cc
    R = TP / GS_Pos
    F1_per_cc = (2 * P_per_cc * R_per_cc) / (P_per_cc + R_per_cc)
    if (P+R) == 0:
        F1 = 0
        warnings.warn('Global F1 score automatically set to zero to avoid division by zero')
        return P_per_cc, P, R_per_cc, R, F1_per_cc, F1
    F1 = (2 * P * R) / (P + R)
    
    return P_per_cc, P, R_per_cc, R, F1_per_cc, F1

def classification_report_icd_flatten(y_true: list, y_pred: list, print_results: bool=False, print_full_report_level: int = 0):
    precision_macro = precision_score_sklearn(y_true, y_pred, average="macro", zero_division=0)
    precision_micro = precision_score_sklearn(y_true, y_pred, average="micro", zero_division=0)
    recall_macro = recall_score_sklearn(y_true, y_pred, average="macro", zero_division=0)
    recall_micro = recall_score_sklearn(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score_sklearn(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score_sklearn(y_true, y_pred, average="micro", zero_division=0)

    if print_results:
        print(f"Precision (macro): {precision_macro:.4f} | Precision (micro): {precision_micro:.4f} | Recall (macro): {recall_macro:.4f} | Recall (micro): {recall_micro:.4f} | F1 (macro): {macro_f1:.4f} | F1 (micro): {micro_f1:.4f}")

    if print_full_report_level != 0:
        y_true = [elem[0:print_full_report_level] for elem in y_true]
        y_pred = [elem[0:print_full_report_level] for elem in y_pred]
        print(classification_report_sklearn(y_true, y_pred, zero_division=0))

    return precision_macro, precision_micro, recall_macro, recall_micro, macro_f1, micro_f1

def build_icd_label_maps(root, hs: bool, label2id: dict | None = None, id2label: dict | None = None):
    if label2id is not None:
        label2id = {key: value_to_int(value) for key, value in label2id.items()}
        if id2label is None:
            id2label = {value: key for key, value in label2id.items()}
        else:
            id2label = {value_to_int(key): value for key, value in id2label.items()}
        return label2id, id2label

    if root is None:
        raise ValueError("root is required when label2id is not provided")

    label2id = {node.name: value_to_int(value) for node, value in root.node_to_id.items()}
    id2label = {value: key for key, value in label2id.items()}

    if not hs:
        label2id = {id2label[value_to_int(value)]: dense_idx for dense_idx, value in enumerate(root.leaf_indexes)}
        id2label = {value: key for key, value in label2id.items()}

    return label2id, id2label

def initialize_icd10_hs_head_model(ctx: "PipelineContext", encoder: Any,  tokenizer: Any, checkpoint_name: str|None, root: "SoftmaxNode", freeze_encoder=False, base_encoder_name: str = None) -> ICD10Predictor_HS_Head:
    if base_encoder_name is None:
        encoder_use = encoder
        tokenizer_use = tokenizer
    else:
        encoder_use = AutoModel.from_pretrained(base_encoder_name).to(device)
        tokenizer_use = AutoTokenizer.from_pretrained(base_encoder_name, trim_offsets=False, use_fast=True)

    model = ICD10Predictor_HS_Head(root, device=ctx.device, encoder=encoder_use, tokenizer=tokenizer_use, freeze_encoder=freeze_encoder).to(device)
    if checkpoint_name is not None:
        checkpoint = read_torch_checkpoint(ctx, checkpoint_name)
        model.load_state_dict(checkpoint, strict=False)

    return model

def initialize_icd10_hs_prediction_model(ctx: "PipelineContext", checkpoint_name: str|None, label2id: dict, root: "SoftmaxNode", internal_K: int = 3) -> ICD10Predictor_HS_CrossEntropyLoss:
    all_labels_desc = create_all_labels_desc(ctx, label2id)
    model = ICD10Predictor_HS_CrossEntropyLoss(root, all_labels_desc, device, internal_K=internal_K).to(device)
    if checkpoint_name is not None:
        checkpoint = read_torch_checkpoint(ctx, checkpoint_name)
        model.load_state_dict(checkpoint, strict=False)

    return model

def initialize_icd10_no_hs_head_model(ctx: "PipelineContext", encoder: Any,  tokenizer: Any, checkpoint_name: str|None, label2id: dict, root: "SoftmaxNode", freeze_encoder=False, base_encoder_name: str = None) -> ICD10Predictor_NO_HS:
    if base_encoder_name is None:
        encoder_use = encoder
        tokenizer_use = tokenizer
    else:
        encoder_use = AutoModel.from_pretrained(base_encoder_name).to(device)
        tokenizer_use = AutoTokenizer.from_pretrained(base_encoder_name, trim_offsets=False, use_fast=True)
    
    all_labels_desc = create_all_labels_desc(ctx, label2id)
    model = ICD10Predictor_NO_HS(root, ctx.device, all_labels_desc, encoder=encoder_use, tokenizer=tokenizer_use, freeze_encoder=freeze_encoder).to(device)
    if checkpoint_name is not None:
        checkpoint = read_torch_checkpoint(ctx, checkpoint_name)
        model.load_state_dict(checkpoint, strict=False)

    return model
