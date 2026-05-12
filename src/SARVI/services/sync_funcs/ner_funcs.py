import torch
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from ..common.ner_funcs import (
    series_to_striding_ner_windows, average_overlapping_hidden_states_checked, generate_sequences, is_valid_decoder, span_collate_fn
)
from ..common.utils.ner_utils import (
    normalize_list, get_final_entity_components_from_group
)

from ...models.schemas import (
    Any
)
from ...models.datasets import (
    SpanDataset
)
from ...models.neural_networks import (
    SpanClassifier
)

def prepare_data(data: pd.DataFrame, padding: bool, tokenizer: Any, model: Any, device: torch.device, window_tokens_no_special: int = 510, stride: int = 128, strict: bool = False):
    """
    Prepara el conjunto de datos a utilizar de manera determinista para la extracción de entidades. 
    Genera ventanas con stride, calculando los embeddings de cada ventana con el modelo y fusionando los estados ocultos solapados.

    Parameters
    ----------
        `data`: pd.DataFrame
            - DataFrame con los datos de entrada. Debe contar unicamente con las columnas `archivo_origen` y `Text`

        `padding`: bool
            - Indica si se debe aplicar padding a las ventanas generadas por el tokenizer

        `tokenizer`: Any
            - Tokenizer utilizado para transformar el texto en tokens y para generar las ventanas de entrada del modelo

        `model`: Any
            - Modelo utilizado para calcular los estados ocultos de cada ventana

        `device`: torch.device
            - Dispositivo donde se ejecutará el modelo, por ejemplo **cpu** o **cuda**

        `window_tokens_no_special`: int
            - Número máximo de tokens por ventana sin contar tokens especiales. Por defecto es **510** *(512-2 tokens especiales)*.

        `stride`: int
            - Número de tokens de solapamiento o desplazamiento entre ventanas consecutivas. Por defecto es **128*.

        `strict`: bool
            - Si es `True`, aplica comprobaciones estrictas al fusionar los estados ocultos solapados. Por defecto es **False**.

    Returns
    -------
        ``: list
            - Lista con las ventanas preparadas para cada fila del DataFrame. Cada ventana incluye su embedding calculado y el nombre del archivo de origen.
    """
    iterator = data.iterrows()
    
    data_prepared = []

    for i, info in tqdm(iterator, total=len(data)):
        windows = series_to_striding_ner_windows(info, tokenizer=tokenizer, window_tokens=window_tokens_no_special, stride=stride, padding=padding)

        last_hidden_state_list = []
        for w in windows:
            inputs = {
                "input_ids": torch.tensor([w["input_ids"]], device=device),
                "attention_mask": torch.tensor([w["attention_mask"]], device=device),
            }
            with torch.no_grad():
                outputs = model(**inputs)

            last_hidden_states = outputs.last_hidden_state
            last_hidden_state_list.append(last_hidden_states)
            
        merged, logs = average_overlapping_hidden_states_checked(windows=windows, last_hidden_state_list=last_hidden_state_list, window_tokens_no_special=window_tokens_no_special, stride=stride, strict=strict)

        for w,m in zip(windows, merged):
            w["embedding"] = m
            w["file_name"] = info["archivo_origen"]

        data_prepared.append(windows)

    return data_prepared

def construct_dataset_ner(data: list, tokenizer: Any, skip_incomplete_spans: bool = True):
    """
    Construye un dataset a partir de una lista de ventanas procesadas, generando secuencias de spans de tokens y asociándolas con sus embeddings correspondientes.

    Parameters
    ----------
        `data`: list
            - Lista con los datos de entrada ya preparados. Cada elemento contiene subinstancias con **input_ids**, embeddings y el nombre del archivo de origen

        `tokenizer`: Any
            - Tokenizer utilizado para generar secuencias de tokens, convertir IDs a tokens legibles y validar los spans generados

        `skip_incomplete_spans`: bool
            - Indica si se deben omitir spans incompletos o no válidos según **is_valid_decoder**. Por defecto es **True**

    Returns
    -------
        ``: pd.DataFrame
            - DataFrame construido a partir de los spans generados. Cada fila contiene el archivo de origen, los tokens del span, el span decodificado, el embedding del token ``CLS``, los embeddings del span, los índices tokens y la instancia de texto correspondiente
    """
    rows = []

    for instance in tqdm(data):

        for i,subinstance in enumerate(instance):

            token_span_sequences = generate_sequences(tokenizer, subinstance["input_ids"], max_len=15)

            for token_span in token_span_sequences:
                if skip_incomplete_spans and not is_valid_decoder(tokenizer.convert_ids_to_tokens(token_span[1]), tokenizer.convert_ids_to_tokens(token_span[2]), tokenizer):
                    continue
                
                decoded_span = tokenizer.convert_ids_to_tokens(token_span[1])

                if (len(token_span[0]) == 1):
                    subsequence = {"File": subinstance["file_name"], "Tokens": token_span[1], 
                                "Decoded span": decoded_span,
                                "CLS Embedding": subinstance["embedding"][0][0], 
                                "Embeddings": subinstance["embedding"][0][token_span[0][0]],
                                "Token idx": token_span[0], "Text instance": i}
                else:
                    subsequence = {"File": subinstance["file_name"], "Tokens": token_span[1], 
                                "Decoded span": decoded_span,
                                "CLS Embedding": subinstance["embedding"][0][0],  
                                "Embeddings": subinstance["embedding"][0][token_span[0][0]:token_span[0][-1]],
                                "Token idx": token_span[0], "Text instance": i}

                rows.append(subsequence)
    return pd.DataFrame(rows)

def construct_loaders_ner(data: list, tokenizer: Any, skip_incomplete_spans: bool = True):
    """
    Construye el data loader para poder realizar la predicción de entidades.

    Parameters
    ----------
        `data`: list
            - Lista con los datos de entrada que se usarán para construir el dataset

        `tokenizer`: Any
            - Tokenizer utilizado por `construct_dataset` para procesar los datos y generar las representaciones necesarias

        `skip_incomplete_spans`: bool
            - Indica si se deben omitir spans incompletos durante la construcción del dataset. Por defecto es **True**

    Returns
    -------
        `dataframe`, `data_loader`: tuple
            - Tupla formada por:
                - `dataframe`: DataFrame construido a partir de los datos de entrada
                - `test_loader`: DataLoader que permite iterar sobre el dataset por lotes
    """
    dataframe = construct_dataset_ner(data, tokenizer, skip_incomplete_spans=skip_incomplete_spans)

    batch_size = 32

    data_dataset = SpanDataset(dataframe)
    data_loader = DataLoader(
        data_dataset,
        batch_size=batch_size,
        collate_fn=span_collate_fn
    )

    return dataframe, data_loader

def run_ner_model(model: SpanClassifier, data_loader: DataLoader, device: torch.device):
    """
    Ejecuta el model NER sobre un conjunto de datos y obtiene las etiquetas predichas y sus probabilidades

    Parameters
    ----------
        `model`: SpanClassifier
            - Modelo de clasificación de spans

        `data_loader`: DataLoader
            - DataLoader que proporciona los batches de datos

        `device`: torch.device
            - Dispositivo donde se ejecutará el modelo, por ejemplo **cpu** o **cuda**

    Returns
    -------
        `all_pred`, `all_value_preds`: tuple
            - Tupla formada por:
                - `all_pred`: array con las clases predichas por el modelo.
                - `all_value_preds`: array con las probabilidades predichas para cada clase.
    """
    model.eval()

    all_pred = []
    all_value_preds = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            span_repr, cls_repr, span_widths = batch

            span_repr = span_repr.to(device)
            cls_repr = cls_repr.to(device)
            span_widths = span_widths.to(device)

            logits = model(span_repr, cls_repr, span_widths)
            value_preds = torch.softmax(logits, dim=-1)
            preds = torch.argmax(value_preds, dim=-1)

            all_pred.append(preds.cpu().numpy())
            all_value_preds.append(value_preds.cpu().numpy())
            
    all_pred = np.concatenate(all_pred)
    all_value_preds = np.concatenate(all_value_preds)

    return all_pred, all_value_preds

def update_df_with_final_pred_entities(df: pd.DataFrame, file_col: str = "File", token_idx_col: str = "Token idx", text_instance_col: str = "Text instance", label_col: str = "pred_label", id_col: str = "pred_id", outside_label: str = "O", outside_id: int = 2, fallback_when_final_span_missing: str = "max_existing"):
    """
    Actualiza un DataFrame de predicciones uniendo spans solapados o adyacentes que pertenecen a la misma entidad final.

    Parameters
    ----------
        `df`: pd.DataFrame
            - DataFrame original con las predicciones de spans. Debe contener columnas para archivo, instancia de texto, índices de tokens, etiqueta predicha e ID de predicción

        `file_col`: str
            - Nombre de la columna que identifica el archivo de origen. Por defecto es **File**

        `token_idx_col`: str
            - Nombre de la columna que contiene los índices de tokens del span. Por defecto es **Token idx**

        `text_instance_col`: str
            - Nombre de la columna que identifica la instancia de texto. Por defecto es **Text instance**

        `label_col`: str
            - Nombre de la columna que contiene la etiqueta predicha. Por defecto es **pred_label**

        `id_col`: str
            - Nombre de la columna que contiene el ID de la etiqueta predicha. Por defecto es **pred_id**

        `outside_label`: str
            - Etiqueta usada para marcar spans que no forman parte de ninguna entidad. Por defecto es **O**

        `outside_id`: int
            - ID asociado a la etiqueta **outside_label**. Por defecto es **2**

        `fallback_when_final_span_missing`: str
            - Estrategia a usar cuando el span final unido no existe como fila en el DataFrame. Puede ser **keep_original** para mantener las predicciones originales o **max_existing** para etiquetar el span existente más largo dentro del componente. Por defecto es **max_existing**

    Returns
    -------
        `df_out`: pd.DataFrame
            - Copia del DataFrame original con las columnas de predicción actualizadas. Mantiene exactamente las mismas filas, columnas, índice y forma que el DataFrame de entrada
    """
    if fallback_when_final_span_missing not in {"keep_original", "max_existing"}: raise ValueError("fallback_when_final_span_missing must be 'keep_original' or 'max_existing'")
    df_out = df.copy()
    label_to_id = df[df[label_col].ne(outside_label)].drop_duplicates(subset=[label_col]).set_index(label_col)[id_col].to_dict()
    normalized_token_idx = {idx: tuple(int(x) for x in normalize_list(row[token_idx_col])) for idx, row in df.iterrows()}
    token_sets = {idx: set(tok) for idx, tok in normalized_token_idx.items()}
    ent_df = df[df[label_col].ne(outside_label)].copy()
    decisions = []
    for (file_name, text_instance, entity_label), group in ent_df.groupby([file_col, text_instance_col, label_col], sort=False):
        components = get_final_entity_components_from_group(group=group, token_idx_col=token_idx_col)
        entity_id = label_to_id[entity_label]
        mask_same_context = (df[file_col].eq(file_name) & df[text_instance_col].eq(text_instance))
        context_indices = list(df[mask_same_context].index)
        for component in components:
            source_indices = component["source_row_indices"]
            if len(source_indices) <= 1: continue
            final_token_idx = tuple(component["final_token_idx"])
            final_set = set(final_token_idx)
            exact_candidates = [idx for idx in context_indices if normalized_token_idx[idx] == final_token_idx]
            if exact_candidates:
                final_row_idx = exact_candidates[0]
                decisions.append({"label": entity_label, "id": entity_id, "final_idx": final_row_idx, "source_indices": source_indices, "final_len": len(final_token_idx)})
                continue
            if fallback_when_final_span_missing == "keep_original": continue
            possible_candidates = [idx for idx in context_indices if token_sets[idx] and token_sets[idx].issubset(final_set)]
            if not possible_candidates: continue
            best_idx = max(possible_candidates, key=lambda idx: (len(token_sets[idx]), -abs(min(token_sets[idx]) - min(final_set))))
            best_len = len(token_sets[best_idx])
            max_source_len = max(len(token_sets[idx]) for idx in source_indices)
            if best_len < max_source_len: continue
            decisions.append({"label": entity_label, "id": entity_id, "final_idx": best_idx, "source_indices": source_indices, "final_len": best_len})
    decisions = sorted(decisions, key=lambda x: x["final_len"], reverse=True)
    protected_final_indices = set()
    for decision in decisions:
        final_idx = decision["final_idx"]
        source_indices = decision["source_indices"]
        entity_label = decision["label"]
        entity_id = decision["id"]
        if final_idx in protected_final_indices: continue
        for idx in source_indices:
            if idx != final_idx:
                df_out.at[idx, label_col] = outside_label
                df_out.at[idx, id_col] = outside_id
        df_out.at[final_idx, label_col] = entity_label
        df_out.at[final_idx, id_col] = entity_id
        protected_final_indices.add(final_idx)
    assert list(df_out.columns) == list(df.columns)
    assert list(df_out.index) == list(df.index)
    assert df_out.shape == df.shape
    return df_out