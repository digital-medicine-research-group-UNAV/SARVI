import torch
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from ..common.icd_pred_funcs import (
    device, embed_texts,
)
from ..common.utils.icd_preds_utils import(
    tensor_items_same_structure
)
from ...models.datasets import (
    ICD10Dataset
)
from ...models.neural_networks import (
    ICD10Predictor_HS_Head, ICD10Predictor_HS_CrossEntropyLoss, ICD10Predictor_NO_HS
)

def construct_dataset_icd10(df: pd.DataFrame):
    """
    Construye los tensores de entrada y las queries para un dataset ICD-10, generando embeddings a partir del texto principal y de las descripciones asociadas.

    Parameters
    ----------
        `df`: pd.DataFrame
            - DataFrame con los datos de entrada. Debe contener las columnas **Text** y **All Description**

    Returns
    -------
        `inputs`, `queries`: tuple
            - Tupla formada por:
                - `inputs`: tensor con los embeddings de los textos del DataFrame
                - `queries`: lista de tensores con los embeddings de las descripciones asociadas a cada fila
    """
    inputs = embed_texts(df["Text"].tolist()).half()

    all_descriptions = [d for row in df["All Description"] for d in row]
    all_query_embeddings = embed_texts(all_descriptions).half()
    queries = []
    all_desc = []
    idx = 0
    for desc_list in df["All Description"]:
        all_desc.append(desc_list)
        q_len = len(desc_list)
        queries.append(all_query_embeddings[idx:idx + q_len])
        idx += q_len

    return inputs, queries, all_desc

def construct_loaders_icd10(df: pd.DataFrame):
    """
    Crea un DataLoader para datos ICD-10 a partir de un DataFrame.

    Parameters
    ----------
        `df`: pd.DataFrame
            - DataFrame con los datos que se quieren convertir en dataset y loader. Debe contener las columnas necesarias para **construct_dataset_icd10**

    Returns
    -------
        `data_loader`: DataLoader
            - DataLoader construido a partir de `ICD10Dataset`, con batches de tamaño **1**, mezcla aleatoria y memoria fijada activada
    """
    inputs, queries, all_desc = construct_dataset_icd10(df)
    dataset = ICD10Dataset(inputs, queries, all_desc)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    return data_loader

def run_icd10_hs_model(model: ICD10Predictor_HS_Head, predictor: ICD10Predictor_HS_CrossEntropyLoss, data_loader: DataLoader):
    """
    Ejecuta un modelo ICD-10 usando HierarchicalSoftmax y obtiene las predicciones para cada query del DataLoader.

    Parameters
    ----------
        `model`: ICD10Predictor_NO_HS
            - Modelo ICD-10 que recibe las queries como entrada y devuelve las puntuaciones de clasificación

        `predictor`: ICD10Predictor_HS_CrossEntropyLoss
            - Clase predictora que obtiene el valor final del diagnostico

        `data_loader`: DataLoader
            - DataLoader que proporciona batches con **inputs** y **queries**

    Returns
    -------
        `all_pred`, `all_value_preds`: tuple
            - Tupla formada por:
                - `all_pred`: array con las clases predichas por el modelo.
                - `all_value_preds`: array con las probabilidades predichas para cada clase.
    """
    dtype = next(model.parameters()).dtype
    model.eval()

    all_preds = []
    all_value_preds = []
    all_desc_global = []

    with torch.no_grad():

        for inputs, queries, all_desc in tqdm(data_loader, total=len(data_loader)):

            inputs = inputs.to(device, dtype=dtype, non_blocking=True)
            queries = queries.to(device, dtype=dtype, non_blocking=True)
        
            B, Q, _ = queries.shape
            
            outputs = model(queries)
            outputs, path_results = predictor.predict(outputs)

            all_preds.append(outputs)
            all_value_preds.append((None,path_results))
            all_desc_global.extend(all_desc)

    all_preds = [x for sublist in all_preds for x in sublist]
    all_value_preds = tensor_items_same_structure(all_value_preds)

    return all_preds, all_value_preds, all_desc_global

def run_icd10_no_hs_model(model: ICD10Predictor_NO_HS, data_loader: DataLoader):
    """
    Ejecuta un modelo ICD-10 sin usar HierarchicalSoftmax y obtiene las predicciones para cada query del DataLoader.

    Parameters
    ----------
        `model`: ICD10Predictor_NO_HS
            - Modelo ICD-10 que recibe las queries como entrada y devuelve las puntuaciones de clasificación

        `data_loader`: DataLoader
            - DataLoader que proporciona batches con **inputs** y **queries**

    Returns
    -------
        `all_pred`, `all_value_preds`: tuple
            - Tupla formada por:
                - `all_pred`: array con las clases predichas por el modelo.
                - `all_value_preds`: array con las probabilidades predichas para cada clase.
    """
    dtype = next(model.parameters()).dtype
    model.eval()

    all_preds = []
    all_value_preds = []
    all_desc_global = []

    with torch.no_grad():

        for inputs, queries, all_desc in tqdm(data_loader, total=len(data_loader)):

            inputs = inputs.to(device, dtype=dtype, non_blocking=True)
            queries = queries.to(device, dtype=dtype, non_blocking=True)
        
            B, Q, _ = queries.shape
            
            outputs = model(queries)
            all_preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
            max_vals, argmax_vals = torch.max(outputs, dim=1)

            all_value_preds.append((None, [[(max_vals[i].item(), argmax_vals[i].item())] for i in range(outputs.size(0))]))
            all_desc_global.extend(all_desc)
            
    all_preds = np.concatenate(all_preds)

    return all_preds, all_value_preds, all_desc_global

def apply_thresholds_icd10(flattened_predictions: list, thresholds: list):
    """
    Aplica una lista de umbrales a secuencias de predicciones para determinar en qué paso debe detenerse cada secuencia.

    Parameters
    ----------
        `flattened_predictions`: list
            - Lista de secuencias de predicciones. Cada secuencia contiene los valores obtenidos en distintos pasos o niveles

        `thresholds`: list
            - Lista de umbrales que se comparan con los valores de cada secuencia. Cada posición representa el umbral correspondiente a ese paso

    Returns
    -------
        `result`: list
            - Lista de tuplas `(reached_end, stop_step, stop_value)`. `reached_end` indica si se ha llegado al final de su secuencia correctamente, `stop_step` indica el último paso aceptado antes de caer por debajo del umbral, y `stop_value` contiene el valor que provocó la parada o el último valor evaluado si no se detuvo antes.
    """
    result = []

    for seq in flattened_predictions:
        if len(seq) == 0:
            result.append((False, None, None))
            continue

        max_steps = min(len(seq), len(thresholds))

        stop_step = None
        stop_value = None
        reached_end = True

        for step in range(max_steps):
            value = seq[step]

            if value < thresholds[step]:
                stop_step = step - 1
                stop_value = value
                reached_end = False
                break

        if stop_step is None:
            stop_step = max_steps - 1
            stop_value = seq[stop_step]

        result.append((reached_end, stop_step, stop_value))

    return result