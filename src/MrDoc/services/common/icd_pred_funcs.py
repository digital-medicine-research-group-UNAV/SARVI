import torch
import warnings
from tqdm.auto import tqdm
from collections import defaultdict
from hierarchicalsoftmax import SoftmaxNode
from transformers import AutoTokenizer, AutoModel

from .utils.icd_preds_utils import (
    tensor_a_float, valor_normal
)

from ...data_io.reader import (
    read_torch_checkpoint, read_parquet_file
)
from ...models.schemas import (
    PipelineContext
)
from ...models.neural_networks import (
    ICD10Predictor_HS_Head, ICD10Predictor_HS_CrossEntropyLoss, ICD10Predictor_NO_HS
)

###
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer_ner = AutoTokenizer.from_pretrained("IIC/RigoBERTa-Clinical", trim_offsets=False, use_fast=True)
model_ner = AutoModel.from_pretrained("IIC/RigoBERTa-Clinical").to(device)
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
            iterator = tqdm(iterator)
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

def create_all_labels_desc(ctx: PipelineContext, label2id: dict):
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
    df = read_parquet_file(ctx, "full_icd10_2026.parquet")
    df["target_id"] = df["Code Perceiver"].map(label2id)

    all_labels_desc_ORIGINAL = defaultdict(list)
    for code in tqdm(df.groupby(["target_id"], group_keys=True)[["Description"]]):
        all_labels_desc_ORIGINAL[code[0][0]] = code[1]["Description"].to_list()

    ####################################

    df = read_parquet_file(ctx, "codiesp_train.parquet")
    df = df[df["All Code Full"].str.len() > 0]
    df["target_id"] = df["All Code Perceiver"].apply(lambda x: [label2id[c] for c in x])

    all_labels_desc_CODIESP = defaultdict(list)
    for _,row in tqdm(df.iterrows(), total=len(df)):
        for code,desc in zip(row["target_id"], row["All Description"]):
            all_labels_desc_CODIESP[code].append(desc)

    ####################################

    df = read_parquet_file(ctx, "cares_train.parquet")
    df = df[df["All Code Full"].str.len() > 0]
    df["target_id"] = df["All Code Perceiver"].apply(lambda x: [label2id[c] for c in x])

    all_labels_desc_CARES = defaultdict(list)
    for _,row in tqdm(df.iterrows(), total=len(df)):
        for code,desc in zip(row["target_id"], row["All Description"]):
            all_labels_desc_CARES[code].append(desc)

    ####################################

    df = read_parquet_file(ctx, "cares_test.parquet")
    df = df[df["All Code Full"].str.len() > 0]
    df["target_id"] = df["All Code Perceiver"].apply(lambda x: [label2id[c] for c in x])

    for _,row in tqdm(df.iterrows(), total=len(df)):
        for code,desc in zip(row["target_id"], row["All Description"]):
            all_labels_desc_CARES[code].append(desc)

    ####################################

    df = read_parquet_file(ctx, "not_codiesp_cares_general.parquet")
    all_labels_desc_REST = dict(zip(df["All Code Full"].map(label2id), df["All Description"]))
    
    ####################################

    all_labels_desc = {}

    for code in tqdm(all_labels_desc_ORIGINAL.keys(), total=len(all_labels_desc_ORIGINAL)):
        codes_desc_REAL = list(set(all_labels_desc_CODIESP[code] + all_labels_desc_CARES[code]))
        if codes_desc_REAL == []:
            codes_desc_REAL = list(set(all_labels_desc_REST[code]))
        codes_desc_REAL = list(set(codes_desc_REAL + [all_labels_desc_ORIGINAL[code][0]]))
        all_labels_desc[code] = embed_texts(codes_desc_REAL, show_tqdm=False)

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
                valores.append(tensor_a_float(tupla_paso[0]))

            flattened_predictions.append(valores)

            if indice_global >= len(all_preds):
                warnings.warn(f"No hay suficientes elementos en all_targets/all_preds para la serie {indice_global}.")
                break

            ultimo_segundo_valor = valor_normal(arr_tuplas[-1][1])
            pred_actual = valor_normal(all_preds[indice_global])

            if ultimo_segundo_valor != pred_actual:
                warnings.warn(f"Desajuste en índice {indice_global}: all_preds[{indice_global}]={pred_actual} pero el segundo valor de la última tupla es {ultimo_segundo_valor}. (grupo={grupo_idx}, elemento={arr_idx})")

            indice_global += 1

    if len(all_preds) != len(flattened_predictions):
        warnings.warn(f"Número de series ({len(flattened_predictions)}) distinto de ({len(all_preds)}).")

    return flattened_predictions

def initialize_icd10_hs_head_model(ctx: PipelineContext, name: str, root: SoftmaxNode) -> ICD10Predictor_HS_Head:
    checkpoint = read_torch_checkpoint(ctx, name)

    model = ICD10Predictor_HS_Head(root).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return model

def initialize_icd10_hs_prediction_model(ctx: PipelineContext, name: str, label2id: dict, root: SoftmaxNode, internal_K: int = 3) -> ICD10Predictor_HS_CrossEntropyLoss:
    checkpoint = read_torch_checkpoint(ctx, name)

    all_labels_desc = create_all_labels_desc(ctx, label2id)

    model = ICD10Predictor_HS_CrossEntropyLoss(root, all_labels_desc, internal_K).to(device)
    model.load_state_dict(checkpoint["optimizer_state_dict"], strict=False)

    return model

def initialize_icd10_no_hs_head_model(ctx: PipelineContext, name: str, label2id: dict, root: SoftmaxNode) -> ICD10Predictor_NO_HS:
    checkpoint = read_torch_checkpoint(ctx, name)

    all_labels_desc = create_all_labels_desc(ctx, label2id)

    model = ICD10Predictor_NO_HS(root, all_labels_desc).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return model