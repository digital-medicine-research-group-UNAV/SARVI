# 💸 Agente Mr.Doc: OpenAI API
# 💸 = correr el código tiene coste

import traceback
from ftfy import fix_text
from langchain_core.messages import SystemMessage, HumanMessage

from ..common.llm_funcs import (
    re,
    np,
    pd,
    Any,
    tqdm,
    Path,
    json,
    torch,
    time,
    cos_sim,
    SentenceTransformer,
    SpanDataset,
    SpanClassifier,
    DataLoader,
    series_to_striding_ner_windows,
    SoftmaxNode,
    is_code_in_range,
    normalize_code,
    parse_range,
    is_valid,
    add_min_consecutive_subgroups,
    defaultdict,
    average_overlapping_hidden_states_checked,
    embed_texts,
    ICD10Dataset,
    ICD10Predictor_HS_Head,
    ICD10Predictor_HS_CrossEntropyLoss,
    ICD10Predictor_NO_HS,
    tensor_items_same_structure,
    get_missing_entities,
    char_span_to_token_span,
    generate_sequences,
    is_valid_decoder,
    span_collate_fn,
    normalize_list,
    get_final_entity_components_from_group,
    validate_complete_objects_from_truncated_output,
    clean_objects_with_schema,
    clean_single_cie10_value,
    find_CIE10_similars,
    validate_json_created,
    cie10_judger_tokenizer,
    cie10_judger_model,
    device
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

def load_tree_hierarchical_module(df_reference: pd.DataFrame):
    """
    Construye un árbol jerárquico de códigos a partir de un DataFrame de referencia.

    Parameters
    ----------
        `df_reference`: pd.DataFrame
            - DataFrame con los códigos y descripciones de referencia. Debe contener las columnas **Código** y **Descripción**

    Returns
    -------
        `node_dict`: dict
            - Diccionario con todos los nodos del árbol jerárquico. Incluye el nodo raíz, los nodos por letra, los rangos, las hojas y los nodos sintéticos creados para subgrupos consecutivos
    """
    node_dict = {}

    root = SoftmaxNode("Root", description="Root", alpha=10.0, gamma=2.0)
    node_dict["root"] = root

    # -------------------------
    # CLEAN DATA
    # -------------------------
    rows = []
    for _, row in tqdm(df_reference.iterrows(), total=len(df_reference)):
        code = str(row["Código"]).strip().upper()

        if "." in code:
            continue

        if is_valid(code):
            rows.append({"code": code, "description": row["Descripción"]})

    # -------------------------
    # CREATE LETTER NODES
    # -------------------------
    letters = sorted(set(r["code"][0] for r in rows))
    for letter in letters:
        node_dict[letter] = SoftmaxNode(letter, parent=root, description=f"Chapter {letter}", alpha=5.0, gamma=2.0)

    # -------------------------
    # SPLIT TYPES
    # -------------------------
    ranges = []
    leaves = []

    for r in rows:
        code = r["code"]

        if "-" in code:
            ranges.append(r)
            if code not in node_dict:
                node_dict[code] = SoftmaxNode(code, parent=None, description=r["description"], alpha=2.0, gamma=2.0)
        else:
            leaves.append(r)
            if code not in node_dict:
                node_dict[code] = SoftmaxNode(code, parent=None, description=r["description"])

    # Agrupar por letra
    groups = defaultdict(list)
    for r in ranges:
        letter, start_num, end_num = parse_range(r["code"])
        groups[letter].append((r, start_num, end_num))

    filtered_ranges = []

    for letter, items in groups.items():
        min_start = min(x[1] for x in items)
        max_end = max(x[2] for x in items)

        for r, start_num, end_num in items:
            # Eliminamos SOLO el rango que cubre todo
            if not (start_num == min_start and end_num == max_end):
                filtered_ranges.append(r)

    ranges = filtered_ranges

    # -------------------------
    # BUILD TREE
    # -------------------------
    range_codes = [r["code"] for r in ranges]
    leaf_codes = [r["code"] for r in leaves]

    # ---- 1. RANGE → RANGE
    for child_code in range_codes:
        possible_parents = []

        for parent_code in range_codes:
            if child_code == parent_code:
                continue

            if (is_code_in_range(child_code.split("-")[0], parent_code) and is_code_in_range(child_code.split("-")[1], parent_code)):
                possible_parents.append(parent_code)

        if possible_parents:
            parent_code = min(possible_parents, key=lambda x: normalize_code(x.split("-")[1])[1] - normalize_code(x.split("-")[0])[1])
            node_dict[child_code].parent = node_dict[parent_code]
        else:
            letter = child_code[0]
            node_dict[child_code].parent = node_dict[letter]

    # ---- 2. LEAF → RANGE
    for leaf_code in leaf_codes:
        possible_parents = []

        for range_code in range_codes:
            if is_code_in_range(leaf_code, range_code):
                possible_parents.append(range_code)

        if possible_parents:
            parent_code = min(possible_parents, key=lambda x: normalize_code(x.split("-")[1])[1] - normalize_code(x.split("-")[0])[1])
            node_dict[leaf_code].parent = node_dict[parent_code]
        else:
            letter = leaf_code[0]
            node_dict[leaf_code].parent = node_dict[letter]

    # -------------------------
    # Rrefinar el árbol con subgrupos mínimos consecutivos
    # -------------------------
    add_min_consecutive_subgroups(node_dict)

    return node_dict

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

def correct_entities(prompt: str, llm, correct_entities_input: dict, docs_dir: Path, json_parse: bool = False, max_attempts: int = 40, max_entities_per_call: int = 15):

    def chunk_list(items: list, chunk_size: int = 30):
        for i in range(0, len(items), chunk_size):
            yield items[i:i + chunk_size]

    def make_keep_answer(file_id: str, entities: list[dict]) -> dict:
        return {
            "data": [
                {
                    "file_id": file_id,
                    "entities": [
                        {
                            "ent_id": ent["entity"]["ent_id"],
                            "original_text": ent["entity"]["text"],
                            "original_start": ent["entity"]["start"],
                            "original_end": ent["entity"]["end"],
                            "original_label": ent["target_label"],
                            "decision": "KEEP",
                            "corrected_text": ent["entity"]["text"],
                            "corrected_start": ent["entity"]["start"],
                            "corrected_end": ent["entity"]["end"],
                            "corrected_label": ent["target_label"],
                        }
                        for ent in entities
                    ],
                }
            ]
        }
        
    answer_global = {"data": []}
    schema_path = docs_dir / "esquema_correcion_entidades.json"

    for i,data in tqdm(enumerate(correct_entities_input["data"]), desc="Correcting entities", total=len(correct_entities_input["data"])):
        original_data = data
        accumulated_answer = None
        entities_to_process = original_data["entities"]

        for attempt in range(1, max_attempts + 1):
            start_time = time.perf_counter()

            try:
                for entities_chunk in chunk_list(entities_to_process, max_entities_per_call):
                    current_input = {**original_data, "entities": entities_chunk}

                    messages = [prompt, str({"data": current_input})]
                    raw_answer = llm.invoke(messages, json_schema=schema_path)

                    if json_parse:
                        m = re.findall(r"```(?:json)?\s*(.*?)\s*```", raw_answer, re.DOTALL | re.IGNORECASE)

                        json_texto = (m[-1] if m else raw_answer).strip()
                        objetos = validate_complete_objects_from_truncated_output(json_texto)
                        objetos_limpios = clean_objects_with_schema(objetos, schema_path)

                        if not objetos_limpios:
                            raise ValueError("No valid objects found after schema cleaning.")

                        answer = {"data": objetos_limpios}

                    else:
                        answer = json.loads(raw_answer)

                    if accumulated_answer is None:
                        accumulated_answer = answer
                    else:
                        accumulated_answer["data"][0]["entities"].extend(answer["data"][0]["entities"])

                corrected_entities = accumulated_answer["data"][0]["entities"]
 
                missing_entities = get_missing_entities(original_data["entities"], corrected_entities, original_data["text"])

                elapsed = time.perf_counter() - start_time

                if not missing_entities:
                    tqdm.write(f"Attempt {i} / {attempt} succeeded in {elapsed:.2f}s")
                    break

                tqdm.write(f"Attempt c{attempt} partially succeeded in {elapsed:.2f}s. Missing entities: {len(missing_entities)}")

                entities_to_process = missing_entities

                if attempt == max_attempts:
                    fallback_answer = make_keep_answer(file_id=original_data["file_id"], entities=missing_entities)

                    if accumulated_answer is None:
                        accumulated_answer = fallback_answer
                    else:
                        accumulated_answer["data"][0]["entities"].extend(fallback_answer["data"][0]["entities"])

                    tqdm.write(f"Max attempts reached for {i} / {original_data['file_id']}. Added {len(missing_entities)} missing entities as KEEP.")


            except Exception as e:
                elapsed = time.perf_counter() - start_time
                tqdm.write(f"Attempt {i} / {attempt} failed in {elapsed:.2f}s: {e}")

                if attempt == max_attempts:
                    fallback_answer = make_keep_answer(file_id=original_data["file_id"], entities=missing_entities)

                    if accumulated_answer is None:
                        accumulated_answer = fallback_answer
                    else:
                        accumulated_answer["data"][0]["entities"].extend(fallback_answer["data"][0]["entities"])

                    tqdm.write(f"Max attempts reached for {i} / {original_data['file_id']}. Added {len(missing_entities)} missing entities as KEEP.")

        if accumulated_answer is None:
            raise ValueError("No valid answer was produced by the LLM.")

        answer_global["data"].append(accumulated_answer["data"][0])

    return answer_global

def corrected_entities_to_df(answer_global: dict, original_texts: dict[str, str]) -> pd.DataFrame: 
    rows = []

    labels = ["ACTOR", "CLINENTITY", "TIMEX3"]

    for item in answer_global["data"]:
        file_id = item["file_id"]
        text = original_texts[file_id]

        row = {"Text": text, "Original File": file_id}

        grouped = {label: {"texts": [], "token_idxs": []} for label in labels}

        for ent in item["entities"]:
            if ent["decision"] == "REJECT":
                continue

            corrected_text = ent["corrected_text"]
            corrected_start = ent["corrected_start"]
            corrected_end = ent["corrected_end"]
            corrected_label = ent["corrected_label"]

            if corrected_label is None:
                continue

            if corrected_label not in grouped:
                continue

            token_idxs = char_span_to_token_span(text=text, start=corrected_start, end=corrected_end)

            grouped[corrected_label]["texts"].append(corrected_text)
            grouped[corrected_label]["token_idxs"].append(token_idxs)

        row["All ACTOR"] = grouped["ACTOR"]["texts"]
        row["All ACTOR Token idx"] = grouped["ACTOR"]["token_idxs"]

        row["All Description"] = grouped["CLINENTITY"]["texts"]
        row["All Description Token idx"] = grouped["CLINENTITY"]["token_idxs"]

        row["All TIMEX3"] = grouped["TIMEX3"]["texts"]
        row["All TIMEX3 Token idx"] = grouped["TIMEX3"]["token_idxs"]

        rows.append(row)

    columns = ["Text", "All ACTOR", "All ACTOR Token idx", "All Description", "All Description Token idx", "All TIMEX3", "All TIMEX3 Token idx", "Original File"]

    return pd.DataFrame(rows, columns=columns)

def procesar_docx(informe_texto: str, report: Path, prompt: str, llm, json_parse: bool, docs_dir: Path):
    """
    Procesa un informe DOCX y guarda el resultado en un JSON.
    Versión sincrónica.
    """
    respuesta_llm = procesar_informe(informe_texto, prompt, llm, docs_dir, json_parse)

    if not validate_json_created(respuesta_llm, docs_dir / "esquema_diagnosticos.json"):
        raise Exception(f"El JSON creado del documento {report.stem} no sigue el esquema indicado")
    
    return respuesta_llm


def procesar_informe(informe_raw: str, prompt: str, llm, docs_dir: Path, json_parse: bool = False) -> dict:
    """
    💸💸💸

    Procesa el informe médico contenido en `informe_raw` utilizando un modelo OpenAI para extraer información
    estructurada y guardarla en un archivo JSON.

    Parameters
    ----------
        `informe_raw`: str
            - El informe en texto plano
        `prompt`: str
            - Prompt de instrucciones usado por el LLM para saber como dirigir su tarea
        `llm`
            - Objeto LLM mediante el cual poder hacer las llamadas
        `json_parse`: bool
            - Booleano para saber si es necesario realizar un parse de los resultados de del LLM
            - Se realiza con LLMs que NO sean de OpenAI

    Returns
    -------
        `respuesta_llm`: dict
            - Respuesta del LLM directamente en formato dict/json
    """
    if json_parse:
        messages = [prompt, informe_raw]
    else:
        messages = [SystemMessage(content=prompt),
                    HumanMessage(content=informe_raw)]

    answer = llm.invoke(messages, json_schema=docs_dir / "esquema_diagnosticos.json")

    if json_parse:
        m = re.findall(r'```(?:json)?\s*(.*?)\s*```', answer, re.DOTALL | re.IGNORECASE)
        json_texto = (m[-1] if m else answer).strip()
        # try:
        #     answer = json.loads(json_texto)
        # except Exception as e:
        objetos = validate_complete_objects_from_truncated_output(json_texto)
        objetos_limpios = clean_objects_with_schema(objetos, docs_dir / "esquema_diagnosticos.json")
        if objetos_limpios:
            answer = {"diagnosticos": objetos_limpios}
    else:
        answer = json.loads(answer)

    return answer

def seleccionar_CIE10_lista(enfermedad: str, CIE10_dict: dict, prompt: str, llm, docs_dir: Path, json_parse: bool = False) -> dict:
    """
    💸💸💸

    Selecciona el código CIE10 correspondiente a la enfermedad dada como parámetro de entrada mediante una lista de códigos CIE10 también introducida como parámetro

    Parameters
    ----------
        `enfermedad`: str
            - Enfermed de la cual se quiere obtener el código CIE10
        `CIE10_dict`: dict
            - Diccionario de los posibles códigos CIE10 y su descripción sobre el cual la enfermedad puede basarse
        `prompt`: str
            - Prompt de instrucciones usado por el LLM para saber como dirigir su tarea
        `llm`
            - Objeto LLM mediante el cual poder hacer las llamadas
        `json_parse`: bool
            - Booleano para saber si es necesario realizar un parse de los resultados de del LLM
            - Se realiza con LLMs que NO sean de OpenAI

    Returns
    -------
        `respuesta_llm`: dict
            - Respuesta del LLM directamente en formato dict/json
    """
    if json_parse:
        messages = [prompt,
                    f"""
                        ```json
                        {{
                            "enfermedad": "{enfermedad}"
                            "dict_codigos_CIE10": {json.dumps(CIE10_dict, ensure_ascii=False)}
                        }}
                        ```
                    """]

    else:
        messages = [SystemMessage(content=prompt),
                    HumanMessage(content=f"""
                                            ```json
                                            {{
                                                "enfermedad": "{enfermedad}"
                                                "dict_codigos_CIE10": {json.dumps(CIE10_dict, ensure_ascii=False)}
                                            }}
                                            ```
                                        """)]

    answer = llm.invoke(messages, json_schema=docs_dir / "esquema_selected_and_decider.json")
    
    if json_parse:
        m = re.findall(r'```(?:json)?\s*(.*?)\s*```', answer, re.DOTALL | re.IGNORECASE)
        json_texto = (m[-1] if m else answer).strip()
        # try:
        #     answer = json.loads(json_texto)
        # except Exception as e:
        answer = validate_complete_objects_from_truncated_output(json_texto)
        answer = clean_objects_with_schema(answer, docs_dir / "esquema_selected_and_decider.json")
        answer = answer[0]
    else:
        answer = json.loads(answer.content)
    
    return answer


# def juzgar_CIE10(CIE10_codigo: str, CIE10_descripción: str, diagnostico_extraido: str, contexto: str, prompt: str, llm, docs_dir: Path, json_parse: bool = False) -> dict:
def juzgar_CIE10(CIE10_codigo: str, CIE10_descripción: str, diagnostico_extraido: str, prompt: str, docs_dir: Path, json_parse: bool = False) -> dict:
    """
    💸💸💸

    Juzga si el código CIE10 seleccionado es correcto respecto al diagnostico original

    Parameters
    ----------
        `CIE10_codigo`: str
            - Código CIE10 seleccionado a juzgar
        `CIE10_descripción`: str
            - Descripción del código CIE10 seleccionado a juzgar
        `diagnostico_extraido`: str
            - Diagnostico original del cual juzgar si el CIE10 es correcto o no
        `contexto`: str
            - Informe completo desde el cual se ha extraido el diagnostico
        `prompt`: str
            - Prompt de instrucciones usado por el LLM para saber como dirigir su tarea
        `llm`
            - Objeto LLM mediante el cual poder hacer las llamadas
        `json_parse`: bool
            - Booleano para saber si es necesario realizar un parse de los resultados de del LLM
            - Se realiza con LLMs que NO sean de OpenAI

    Returns
    -------
        `respuesta_llm`: dict
            - Respuesta del LLM directamente en formato dict/json
    """
    # if json_parse:
    #     messages = [prompt,
    #                 f"""
    #                     ```json
    #                     {{
    #                         "CIE10_codigo": "{CIE10_codigo}"
    #                         "CIE10_descripción": "{CIE10_descripción}"
    #                         "diagnostico_extraido": "{diagnostico_extraido}"
    #                         "contexto": "{contexto}"
    #                     }}
    #                     ```
    #                 """]

    # else:
    #     messages = [SystemMessage(content=prompt),
    #                 HumanMessage(content=f"""
    #                                         ```json
    #                                         {{
    #                                             "CIE10_codigo": "{CIE10_codigo}"
    #                                             "CIE10_descripción": "{CIE10_descripción}"
    #                                             "diagnostico_extraido": "{diagnostico_extraido}"
    #                                             "contexto": "{contexto}"
    #                                         }}
    #                                         ```
    #                                     """)]

    # answer = llm.invoke(messages, json_schema=docs_dir / "esquema_juzgar.json")

    prompt = f"[REF]{diagnostico_extraido.lower()}[CODE]{CIE10_codigo.upper()}[DESC]{CIE10_descripción.lower()}"
    inputs = cie10_judger_tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=256).to(device)
    with torch.no_grad():
        outputs = cie10_judger_model(**inputs)
        logits = outputs.logits
        prediction = logits.argmax(dim=-1).item()

    answer = f"""```json
    {json.dumps({"resultado": True if prediction==1 else False})}
    ```"""
    
    if json_parse:
        m = re.findall(r'```(?:json)?\s*(.*?)\s*```', answer, re.DOTALL | re.IGNORECASE)
        json_texto = (m[-1] if m else answer).strip()
        # try:
        #     answer = json.loads(json_texto)
        # except Exception as e:
        answer = validate_complete_objects_from_truncated_output(json_texto)
        answer = clean_objects_with_schema(answer, docs_dir / "esquema_juzgar.json")[0]
    else:
        answer = json.loads(answer.content)

    return answer


def decidir_CIE10(diagnostico_extraido: str, contexto: str, prompt: str, llm, docs_dir: Path, json_parse: bool = False) -> dict:
    """
    💸💸💸

    Decide el código CIE10 de un diagnostico extriado

    Parameters
    ----------
        `diagnostico_extraido`: str
            - Diagnostico original del cual juzgar si el CIE10 es correcto o no
        `contexto`: str
            - Informe completo desde el cual se ha extraido el diagnostico
        `prompt`: str
            - Prompt de instrucciones usado por el LLM para saber como dirigir su tarea
        `llm`
            - Objeto LLM mediante el cual poder hacer las llamadas
        `json_parse`: bool
            - Booleano para saber si es necesario realizar un parse de los resultados de del LLM
            - Se realiza con LLMs que NO sean de OpenAI

    Returns
    -------
        `respuesta_llm`: dict
            - Respuesta del LLM directamente en formato dict/json
    """
    if json_parse:
        messages = [prompt,
                    f"""
                        ```json
                        {{
                            "diagnostico_extraido": "{diagnostico_extraido}"
                            "contexto": "{contexto}"
                        }}
                        ```
                    """]

    else:
        messages = [SystemMessage(content=prompt),
                    HumanMessage(content=f"""
                                            ```json
                                            {{
                                                "diagnostico_extraido": "{diagnostico_extraido}"
                                                "contexto": "{contexto}"
                                            }}
                                            ```
                                        """)]

    answer = llm.invoke(messages, json_schema=docs_dir / "esquema_selected_and_decider.json")
    
    if json_parse:
        m = re.findall(r'```(?:json)?\s*(.*?)\s*```', answer, re.DOTALL | re.IGNORECASE)
        json_texto = (m[-1] if m else answer).strip()
        # try:
        #     answer = json.loads(json_texto)
        # except Exception as e:
        answer = validate_complete_objects_from_truncated_output(json_texto)
        answer = clean_objects_with_schema(answer, docs_dir / "esquema_selected_and_decider.json")[0]
    else:
        answer = json.loads(answer.content)

    return answer


def completar_df_predicted_nearest_text_only(df_predicted: pd.DataFrame, df_reference: pd.DataFrame, df_nearest_embeddings: torch.Tensor, df_predicted_embeddings: torch.Tensor, add_semantic_similarity: bool = False) -> pd.DataFrame:
    """
    Función que se encarga de averiguar cual es la enfermedad más cercana a la obtenida mediante el LLM anteriormente. Realiza una similaridad semántica de embeddings entre lo predicho y todas las descripciones reales de referencia. Aquella más similar es la que se añade al DataFrame respuesta.

    El código CIE10 añadido como referencia es el relacionado con la enfermedad más parecida semánticamente

    NO toma en cuenta el código CIE10 relacionado con la enfermedad ni en el predicho ni en la referencia

    Parameters
    ----------
        `df_predicted`: pd.DataFrame
            - DataFrame con los **datos predichos** sobre los informes. Obtenidos anteriormente por algún **LLM**
        `df_reference`: pd.DataFrame
            - DataFrame con todos los códigos CIE10 originales y sus respectivas descripciones. Tienen que estár en castellano
        `df_nearest_embeddings`: torch.Tensor
            - Embeddings ya procesados anteriormente. Se trata de cada una de las descripciones de enfermedades obtenidas anteriormente por un LLM
        `df_predicted_embeddings`: torch.Tensor
            - Embeddings ya procesados anteriormente. Se trata de cada una de las descripciones de enfermedades reales en los códigos CIE10
        `add_semantic_similarity`: bool
            - Booleano para saber si se quiere añadir el valor de similitud semántica obtenido

    Returns
    -------
        `df_final`: pd.DataFrame
            - Una extensión del `df_predicted` donde se añaden nuevas columnas para indicar cual es la enfermedad y su código correspondiente más similar
    """
    df_final = df_predicted.copy()
    max_vals, max_idx = torch.max(cos_sim(df_nearest_embeddings, df_predicted_embeddings), dim=0)

    resultados = []

    for i, (max_id, max_val) in tqdm(enumerate(zip(max_idx, max_vals)), total=len(max_idx), desc="Completando el DF (nearest)", unit="diagnostico"):
        resultado = {"idx": i, "diagnostico_nearest": df_reference.loc[int(max_id), "Descripción"], "CIE10_nearest": df_reference.loc[int(max_id), "Código"]}
        if add_semantic_similarity:
            resultado["similarity_predicted_nearest"] = float(max_val)
        resultados.append(resultado)

    for r in resultados:
        i = r.pop("idx")
        for col, val in r.items():
            df_final.loc[i, col] = val
    
    df_final = df_final.reset_index(drop=True)

    return df_final


def asistente_seleccionador_cie10(df_final: pd.DataFrame, CIE10_full_list: list, df_reference: pd.DataFrame, prompt_CIE10_selector: str, llm, docs_dir: Path, find_CIE10_similars_level: int = 0, model: SentenceTransformer = None, add_semantic_similarity: bool = False, json_parse: bool = False, tratamiento_fallos: bool = False) -> pd.DataFrame:
    """
    Función que permite que el asistene evaluador lleve a cabo su tarea. Se realizan una serie de pasos para cada enfermedad:

        1) Se obtienen todos los códigos CIE10 similares respecto a un nivel del obtenido por el LLM inicialmente y del semanticamente similar entre enfermedades

        2) Se le pasa al LLM juzgador todos estos códigos junto con sus descripciones reales. Se le pide que, dada como valor de entrada el diagnostico extraido por el LLM inicial, devuelva a cual se asemeja realmente entre todos los posibles 

        3) Se añade al DataFrame dos columnas nuevas las cuales indican el código CIE10 y diagnostico seleccionado

    Si el valor de la variable `tratamiento_fallos` es `True`, solamente se hará todo esto con los diagnosticos que su valor en `tree_5` sea `False`

    Parameters
    ----------
        `df_final`: pd.DataFrame
            - DataFrame con lo obtenido mediante el LLM anteriormente y una posterior similitud de coseno entre diagnosticos
        `CIE10_full_list`: list
            - Lista con todos los códigos CIE10. Únicamente usa los códigos, sin su descripción
        `df_reference`: pd.DataFrame
            - DataFrame con todos los códigos CIE10 originales y sus respectivas descripciones. Tienen que estár en castellano
        `prompt_CIE10_selector`: str
            - Prompt para que el asistente evaluador pueda saber como realizar su trabajo
        `llm`
            - Objeto LLM mediante el cual poder hacer las llamadas
        `find_CIE10_similars_level`: int
            - Número entero que indica sobre que nivel realizar la búsqueda de códigos CIE10 similares. Por defecto (`cie10_similar_level` = 0) dicta que lo anterior al punto es fijo. Valores postivos aumentan lo fijado por la derecha del punto y valores negativos reducen lo fijado por la izquierda del punto
            - Investigar la función `find_CIE10_similars` para más información
        `model`: SentenceTransformer
            - Encoder que permite hacer embeddings de los diagnosticos si se considera necesario
        `add_semantic_similarity`: bool
            - Booleano para saber si se quiere añadir el valor de similitud semántica obtenido
        `json_parse`: bool
            - Booleano para saber si es necesario realizar un parse de los resultados de del LLM
            - Se realiza con LLMs que NO sean de OpenAI
        `tratamiento_fallos`: bool
            - Booleano para saber si realizar el analisis con los que no se han llegado a decidir como correcto anteriormente
            - Únicamente se realiza el proceso con aquellos que su variable `tree_5` sea `False`

    Returns
    -------
        `df_final`: pd.DataFrame
            - DataFrame final con el código CIE10 y su enfermedad correspondiente asociada elegida tras la decisión del asistente evaluador
    """
    if(tratamiento_fallos):
        desc = "Completando el DF (selected) - Tratamiento fallos"
        sufijo = "_V2"
    else:
        desc = "Completando el DF (selected)"
        sufijo = ""

    resultados = []
    failed = []

    for i in tqdm(range(len(df_final)), total=len(df_final), desc=desc, unit="diagnostico"):
        if (tratamiento_fallos and not df_final.loc[i, "tree_5"]) or (not tratamiento_fallos):
            max_retries = 5
            for attempt in range(1, max_retries + 1):
                try:
                    aux = find_CIE10_similars_level
                    # preparación (similaridades)
                    CIE10_sim_predicted = find_CIE10_similars(df_final.loc[i, f"CIE10_predicted{sufijo}"], CIE10_full_list, level=find_CIE10_similars_level)
                    CIE10_sim_nearest = find_CIE10_similars(df_final.loc[i, "CIE10_nearest"], CIE10_full_list, level=find_CIE10_similars_level)
                    CIE10_sim = list(set(CIE10_sim_predicted + CIE10_sim_nearest))
                    while(len(CIE10_sim) > 70 and aux < 8):
                        aux += 1
                        CIE10_sim_predicted = find_CIE10_similars(df_final.loc[i, f"CIE10_predicted{sufijo}"], CIE10_full_list, level=aux)
                        CIE10_sim_nearest = find_CIE10_similars(df_final.loc[i, "CIE10_nearest"], CIE10_full_list, level=aux)
                        CIE10_sim = list(set(CIE10_sim_predicted + CIE10_sim_nearest))
                    CIE10_sim_dict = df_reference[df_reference["Código"].isin(CIE10_sim)].set_index("Código")["Descripción"].to_dict()

                    # llamada al LLM
                    if (aux < 8):
                        max_retries = 5
                        for attempt in range(1, max_retries + 1):
                            try:
                                CIE10_final = seleccionar_CIE10_lista(df_final.loc[i, "diagnostico_predicted"], CIE10_sim_dict, prompt_CIE10_selector, llm, docs_dir, json_parse)
                                if not validate_json_created(CIE10_final, docs_dir / "esquema_selected_and_decider.json"):
                                    raise Exception(f"El JSON creado del documento para la fila {i} no sigue el esquema indicado")
                                break
                            except Exception as e:
                                print(f"⚠️ Error al seleccionar (LLM) la fila {i}: {e!r}")
                                traceback.print_exc()
                                if attempt < max_retries:
                                    print("↻ Reintentando...")
                                else:
                                    print(f"❌ Falló definitivamente al seleccionar (LLM) la fila {i}\n")
                        
                    else:
                        CIE10_final = {"CIE10": df_final.loc[i, "CIE10_nearest"]}

                    # calculamos embeddings
                    similarity_val = None
                    if add_semantic_similarity and model is not None:
                        emb1 = model.encode(df_final.loc[i, "diagnostico_predicted"], convert_to_tensor=True)
                        emb2 = model.encode(df_reference.loc[df_reference["Código"] == CIE10_final["CIE10"], "Descripción"].iloc[0], convert_to_tensor=True)
                        similarity_val = float(cos_sim(emb1, emb2))

                    resultados.append((i, CIE10_final["CIE10"], df_reference.loc[df_reference["Código"] == CIE10_final["CIE10"], "Descripción"].iloc[0], similarity_val))
                    break
                except Exception as e:
                    print(f"⚠️ Error al seleccionar la fila {i} : {e!r}")
                    traceback.print_exc()
                    if attempt < max_retries:
                        print("↻ Reintentando...")
                    else:
                        if tratamiento_fallos:
                            resultados.append((i, df_final.loc[i, "CIE10_selected"], df_final.loc[i, "diagnostico_selected"], df_final.loc[i, "similarity_predicted_selected"]))
                        else:
                            failed.append(i)
                        print(f"❌ Falló definitivamente al seleccionar la fila {i}\n")
        else:
            resultados.append((i, df_final.loc[i, "CIE10_selected"], df_final.loc[i, "diagnostico_selected"], df_final.loc[i, "similarity_predicted_selected"]))


    for i, cie10_sel, diag_sel, sim_val in resultados:
        df_final.loc[i, f"CIE10_selected{sufijo}"] = cie10_sel
        df_final.loc[i, f"diagnostico_selected{sufijo}"] = diag_sel
        if sim_val is not None:
            df_final.loc[i, f"similarity_predicted_selected{sufijo}"] = sim_val

    df_final = df_final.drop(index=failed, errors="ignore")
    df_final = df_final.reset_index(drop=True)

    return df_final


def asistente_juzgador_cie10(df_final: pd.DataFrame, prompt_CIE10_juzgador: str, llm, contexts: dict[str, str], docs_dir: Path, json_parse: bool = False, tratamiento_fallos: bool = False) -> pd.DataFrame:
    """
    Función que permite que el asistene juzgador lleve a cabo su tarea. Navega por todo el DataFrame añadiendo una nueva columna fina `tree_5`:
    
        1) Si la columna `tree_4` es `True`, no se evalúan los valores y continua siendo `True`
        
        2) Si la columna `tree_4` es `False` trata de juzgar si el código CIE10, y su correspondiente descripción, corresponden y tienen sentido respecto al diagnostico de la enfermedad. Si el juzgador lo considera oportuno, este nuevo valor será el de `True`

    Parameters
    ----------
        `df_final`: pd.DataFrame
            - DataFrame con lo obtenido mediante el LLM anteriormente y una posterior similitud de coseno entre diagnosticos
        `prompt_CIE10_juzgador`: str
            - Prompt para que el asistente juzgador pueda saber como realizar su trabajo
        `llm`
            - Objeto LLM mediante el cual poder hacer las llamadas
        `contexts`: dict[str, str]:
            - Diccionario de todos los documentos leidos anteriormente. Key es el nombre, value el contenido
        `json_parse`: bool
            - Booleano para saber si es necesario realizar un parse de los resultados de del LLM
            - Se realiza con LLMs que NO sean de OpenAI
        `tratamiento_fallos`: bool
            - Booleano para saber si realizar el analisis con los que no se han llegado a decidir como correcto anteriormente
            - Unicamente se realizara la llamada al LLM cuando el valor original de `tree_5` es False

    Returns
    -------
        `df_final`: pd.DataFrame
            - DataFrame final con la última columna completada
    """
    if(tratamiento_fallos):
        sufijo = "_V2"
        desc = "Completando el DF (juzgador) - Tratamiento fallos"
    else:
        sufijo = ""
        desc = "Completando el DF (juzgador)"
    
    df_final["diagnostico_predicted"] = df_final["diagnostico_predicted"].map(lambda x: fix_text(x) if isinstance(x, str) else x)

    resultados = []
    failed = []

    for idx, row in tqdm(df_final.iterrows(), total=len(df_final), desc=desc, unit="diagnostico"):
        if (row["tree_4"] is True) or (tratamiento_fallos and row["tree_5"] is True):
            resultado = True
        else:
            max_retries = 5
            for attempt in range(1, max_retries + 1):
                try:
                    # raw_result = juzgar_CIE10(row[f"CIE10_selected{sufijo}"], row[f"diagnostico_selected{sufijo}"], row["diagnostico_predicted"], contexts[row["nombre_archivo"]], prompt_CIE10_juzgador, llm, docs_dir, json_parse)
                    raw_result = juzgar_CIE10(row[f"CIE10_selected{sufijo}"], row[f"diagnostico_selected{sufijo}"], row["diagnostico_predicted"], prompt_CIE10_juzgador, docs_dir, json_parse)
                    if not validate_json_created(raw_result, docs_dir / "esquema_juzgar.json"):
                        raise Exception(f"El JSON creado del documento para la fila {idx} no sigue el esquema indicado")
                    result = raw_result["resultado"]
                    if isinstance(result, str):
                        result = result.replace('"', '').strip()
                        result = result.lower() == "true"
                    resultado = result
                    break
                except Exception as e:
                    print(f"⚠️ Error al juzgar la fila {idx} : {e!r}")
                    traceback.print_exc()
                    if attempt < max_retries:
                        print("↻ Reintentando...")
                    else:
                        failed.append(idx)
                        print(f"❌ Falló definitivamente al juzgar la fila {idx}\n")
        resultados.append((idx, resultado))

    for idx, resultado in resultados:
        df_final.loc[idx, f"tree_5{sufijo}"] = resultado

    df_final = df_final.drop(index=failed, errors="ignore")
    df_final = df_final.reset_index(drop=True)

    return df_final

def asistente_seleccionador_tratamiento_falsos_cie10(df_final: pd.DataFrame, prompt_CIE10_seleccionador: str, llm, contexts: dict[str, str], docs_dir: Path, json_parse: bool = False) -> pd.DataFrame:
    """
    Función que permite volver a realizar el proceso de selección de códigos CIE10 a aquellos diagnosticos que no han podido ser declarados como correctos en procesos anteriores

    Parameters
    ----------
        `df_final`: pd.DataFrame
            - DataFrame con lo obtenido mediante el LLM anteriormente y una posterior similitud de coseno entre diagnosticos
        `prompt_CIE10_seleccionador`: str
            - Prompt para que el asistente juzgador pueda saber como realizar su trabajo
        `llm`
            - Objeto LLM mediante el cual poder hacer las llamadas
        `contexts`: dict[str, str]:
            - Diccionario de todos los documentos leidos anteriormente. Key es el nombre, value el contenido
        `semaforo`: asyncio.Semaphore
            - Semaforo para poder hacer toda la actividad de forma asincrona
        `json_parse`: bool
            - Booleano para saber si es necesario realizar un parse de los resultados de del LLM
            - Se realiza con LLMs que NO sean de OpenAI

    Returns
    -------
        `df_final`: pd.DataFrame
            - DataFrame final con la última columna completada
    """
    resultados = []
        
    for idx, row in tqdm(df_final.iterrows(), total=len(df_final), desc="Completando el DF (reselección)", unit="diagnostico"):
        if row["tree_5"] is True:
            resultado = row["CIE10_selected"]
        else:
            max_retries = 5
            for attempt in range(1, max_retries + 1):
                try:
                    raw_result = decidir_CIE10(row["diagnostico_predicted"], contexts[row["nombre_archivo"]], prompt_CIE10_seleccionador, llm, docs_dir, json_parse)
                    if not validate_json_created(raw_result, docs_dir / "esquema_selected_and_decider.json"):
                        raise Exception(f"El JSON creado del documento para la fila {idx} no sigue el esquema indicado")
                    resultado = raw_result["CIE10"]
                    break
                except Exception as e:
                    print(f"⚠️ Error al reseleccionar la fila {idx}: {e!r}")
                    traceback.print_exc()
                    if attempt < max_retries:
                        print("↻ Reintentando...")
                    else:
                        resultado = None
                        print(f"❌ Falló definitivamente la fila {idx}\n")
            if resultado == None:
                resultado = row["CIE10_selected"]
            else:
                resultado = clean_single_cie10_value(resultado)
                if resultado == None:
                    resultado = row["CIE10_selected"]
        resultados.append((idx, resultado))


    for idx, resultado in resultados:
        df_final.loc[idx, "CIE10_predicted_V2"] = resultado

    df_final = df_final.reset_index(drop=True)

    return df_final