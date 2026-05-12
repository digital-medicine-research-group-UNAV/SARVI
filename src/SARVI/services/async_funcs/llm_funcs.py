# 💸 Agente Mr.Doc: OpenAI API
# 💸 = correr el código tiene coste
import re
import json
import torch
import asyncio
import traceback
import pandas as pd

from pathlib import Path
from ftfy import fix_text
from tqdm.auto import tqdm
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer
from langchain_core.messages import SystemMessage, HumanMessage
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..common.llm_funcs import (
    clean_single_cie10_value, find_CIE10_similars, device
)

from ..common.utils.json_utils import (
    validate_json_created, validate_complete_objects_from_truncated_output, clean_objects_with_schema
)

#################################################################################################################
cie10_judger_tokenizer = AutoTokenizer.from_pretrained("JulenRM/RigoBERTa-Clinical_CIE10Judger")
cie10_judger_model = AutoModelForSequenceClassification.from_pretrained("JulenRM/RigoBERTa-Clinical_CIE10Judger").to(device)

async def retry_async(fn, *args, retries=3, delay=1, **kwargs):
    for attempt in range(retries):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            print(f"⚠️ Error en {fn.__name__} (intento {attempt + 1}/{retries}): {e}")
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay)
#################################################################################################################

async def procesar_docx(informe_texto: str, report: Path, prompt: str, llm, json_parse: bool, docs_dir: Path, semaforo: asyncio.Semaphore):
    """
    Procesa un informe DOCX y guarda el resultado en un JSON.
    Versión sincrónica.
    """
    async with semaforo:
        respuesta_llm = await procesar_informe(informe_texto, prompt, llm, docs_dir, json_parse)

        if not validate_json_created(respuesta_llm, docs_dir / "esquema_diagnosticos.json"):
            raise Exception(f"El JSON creado del documento {report.stem} no sigue el esquema indicado")
        
        return respuesta_llm

async def procesar_informe(informe_raw: str, prompt: str, llm, docs_dir: Path, json_parse: bool = False) -> dict:
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
    messages = [SystemMessage(content=prompt),
                HumanMessage(content=informe_raw)]

    answer = await llm.ainvoke(messages, json_schema=docs_dir / "esquema_diagnosticos.json")
    answer = answer.content

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
    

async def seleccionar_CIE10_lista(enfermedad: str, CIE10_dict: dict, prompt: str, llm, docs_dir: Path, json_parse: bool = False) -> dict:
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
    messages = [SystemMessage(content=prompt),
                HumanMessage(content=f"""
                                    ```json
                                    {{
                                        "enfermedad": "{enfermedad}"
                                        "dict_codigos_CIE10": {json.dumps(CIE10_dict, ensure_ascii=False)}
                                    }}
                                    ```
                                    """)]

    answer = await llm.ainvoke(messages, json_schema=docs_dir / "esquema_selected_and_decider.json")
    answer = answer.content

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


# async def juzgar_CIE10(CIE10_codigo: str, CIE10_descripción: str, diagnostico_extraido: str, contexto: str, prompt: str, llm, docs_dir: Path, json_parse: bool = False) -> dict:
async def juzgar_CIE10(CIE10_codigo: str, CIE10_descripción: str, diagnostico_extraido: str, prompt: str, docs_dir: Path, json_parse: bool = False) -> dict:
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
    # messages = [SystemMessage(content=prompt),
    #             HumanMessage(content=f"""
    #                                 ```json
    #                                 {{
    #                                     "CIE10_codigo": "{CIE10_codigo}"
    #                                     "CIE10_descripción": "{CIE10_descripción}"
    #                                     "diagnostico_extraido": "{diagnostico_extraido}"
    #                                     "contexto": "{contexto}"
    #                                 }}
    #                                 ```
    #                                 """)]

    # answer = await llm.ainvoke(messages, json_schema=docs_dir / "esquema_juzgar.json")
    # answer = answer.content

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


async def decidir_CIE10(diagnostico_extraido: str, contexto: str, prompt: str, llm, docs_dir: Path, json_parse: bool = False) -> dict:
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
    messages = [SystemMessage(content=prompt),
                HumanMessage(content=f"""
                                    ```json
                                    {{
                                        "diagnostico_extraido": "{diagnostico_extraido}"
                                        "contexto": "{contexto}"
                                    }}
                                    ```
                                    """)]

    answer = await llm.ainvoke(messages, json_schema=docs_dir / "esquema_selected_and_decider.json")
    answer = answer.content
    
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


async def completar_df_predicted_nearest_text_only(df_predicted: pd.DataFrame, df_reference: pd.DataFrame, df_nearest_embeddings: torch.Tensor, df_predicted_embeddings: torch.Tensor, semaforo: asyncio.Semaphore, add_semantic_similarity: bool = False) -> pd.DataFrame:
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
        `semaforo`: asyncio.Semaphore
            - Semaforo para poder hacer toda la actividad de forma asincrona
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

    async def procesar_fila(i: int, max_id: torch.Tensor, max_val: torch.Tensor):
        async with semaforo:
            resultado = {"idx": i, "diagnostico_nearest": df_reference.loc[int(max_id), "Descripción"], "CIE10_nearest": df_reference.loc[int(max_id), "Código"]}
            if add_semantic_similarity:
                resultado["similarity_predicted_nearest"] = float(max_val)
            resultados.append(resultado)

    tareas = [procesar_fila(i, max_id, max_val) for i, (max_id, max_val) in enumerate(zip(max_idx, max_vals))]

    for future in tqdm(asyncio.as_completed(tareas), total=len(tareas), desc="Completando el DF (nearest)", unit="diagnostico"):
        await future

    for r in resultados:
        i = r.pop("idx")
        for col, val in r.items():
            df_final.loc[i, col] = val

    df_final = df_final.reset_index(drop=True)

    return df_final


async def asistente_seleccionador_cie10(df_final: pd.DataFrame, CIE10_full_list: list, df_reference: pd.DataFrame, prompt_CIE10_selector: str, llm, docs_dir: Path, semaforo: asyncio.Semaphore, find_CIE10_similars_level: int = 0, model: SentenceTransformer = None, add_semantic_similarity: bool = False, json_parse: bool = False, tratamiento_fallos: bool = False) -> pd.DataFrame:
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
        `semaforo`: asyncio.Semaphore
            - Semaforo para poder hacer toda la actividad de forma asincrona
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

    async def procesar_fila(i: int, find_CIE10_similars_level: int):
        if (tratamiento_fallos and not df_final.loc[i, "tree_5"]) or (not tratamiento_fallos):
            max_retries = 5
            for attempt in range(1, max_retries + 1):
                try:
                    async with semaforo:
                        aux = find_CIE10_similars_level
                        # preparación (similaridades)
                        CIE10_sim_predicted = find_CIE10_similars(df_final.loc[i, f"CIE10_predicted{sufijo}"], CIE10_full_list, level=find_CIE10_similars_level)
                        CIE10_sim_nearest = find_CIE10_similars(df_final.loc[i, "CIE10_nearest"], CIE10_full_list, level=find_CIE10_similars_level)
                        CIE10_sim = list(set(CIE10_sim_predicted + CIE10_sim_nearest))
                        while(len(CIE10_sim) > 100 and aux < 8):
                            aux += 1
                            CIE10_sim_predicted = find_CIE10_similars(df_final.loc[i, f"CIE10_predicted{sufijo}"], CIE10_full_list, level=find_CIE10_similars_level)
                            CIE10_sim_nearest = find_CIE10_similars(df_final.loc[i, "CIE10_nearest"], CIE10_full_list, level=find_CIE10_similars_level)
                            CIE10_sim = list(set(CIE10_sim_predicted + CIE10_sim_nearest))
                        CIE10_sim_dict = df_reference[df_reference["Código"].isin(CIE10_sim)].set_index("Código")["Descripción"].to_dict()

                        # llamada al LLM
                        if (aux < 8):
                            max_retries = 5
                            for attempt in range(1, max_retries + 1):
                                try:
                                    CIE10_final = await seleccionar_CIE10_lista(df_final.loc[i, "diagnostico_predicted"], CIE10_sim_dict, prompt_CIE10_selector, llm, docs_dir, json_parse)
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

                    # calculamos embeddings fuera del semáforo
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

    tareas = [retry_async(procesar_fila, i, find_CIE10_similars_level, retries=5) for i in range(len(df_final))]

    for future in tqdm(asyncio.as_completed(tareas), total=len(tareas), desc=desc, unit="diagnostico"):
        await future

    for i, cie10_sel, diag_sel, sim_val in resultados:
        df_final.loc[i, f"CIE10_selected{sufijo}"] = cie10_sel
        df_final.loc[i, f"diagnostico_selected{sufijo}"] = diag_sel
        if sim_val is not None:
            df_final.loc[i, f"similarity_predicted_selected{sufijo}"] = sim_val

    df_final = df_final.drop(index=failed, errors="ignore")
    df_final = df_final.reset_index(drop=True)

    return df_final


async def asistente_juzgador_cie10(df_final: pd.DataFrame, prompt_CIE10_juzgador: str, llm, contexts: dict[str, str], docs_dir: Path, semaforo = asyncio.Semaphore, json_parse: bool = False, tratamiento_fallos: bool = False) -> pd.DataFrame:
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
        `semaforo`: asyncio.Semaphore
            - Semaforo para poder hacer toda la actividad de forma asincrona
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

    async def procesar_fila(idx: int, row: pd.Series):
        if (row["tree_4"] is True) or (tratamiento_fallos and row["tree_5"] is True):
            resultado = True
        else:
            max_retries = 5
            async with semaforo:
                for attempt in range(1, max_retries + 1):
                    try:
                        # raw_result = await juzgar_CIE10(row[f"CIE10_selected{sufijo}"], row[f"diagnostico_selected{sufijo}"], row["diagnostico_predicted"], contexts[row["nombre_archivo"]], prompt_CIE10_juzgador, llm, docs_dir, json_parse)
                        raw_result = await juzgar_CIE10(row[f"CIE10_selected{sufijo}"], row[f"diagnostico_selected{sufijo}"], row["diagnostico_predicted"], prompt_CIE10_juzgador, docs_dir, json_parse)
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

    tareas = [procesar_fila(idx, row) for idx, row in df_final.iterrows()]

    for future in tqdm(asyncio.as_completed(tareas), total=len(tareas), desc=desc, unit="diagnostico"):
        await future

    for idx, resultado in resultados:
        df_final.loc[idx, f"tree_5{sufijo}"] = resultado

    df_final = df_final.drop(index=failed, errors="ignore")
    df_final = df_final.reset_index(drop=True)

    return df_final

async def asistente_seleccionador_tratamiento_falsos_cie10(df_final: pd.DataFrame, prompt_CIE10_seleccionador: str, llm, contexts: dict[str, str], docs_dir: Path, semaforo: asyncio.Semaphore, json_parse: bool = False) -> pd.DataFrame:
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

    async def procesar_fila(idx: int, row: pd.Series):
        if row["tree_5"] is True:
            resultado = row["CIE10_selected"]
        else:
            max_retries = 5
            async with semaforo:
                for attempt in range(1, max_retries + 1):
                    try:
                        raw_result = await decidir_CIE10(row["diagnostico_predicted"], contexts[row["nombre_archivo"]], prompt_CIE10_seleccionador, llm, docs_dir, json_parse)
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

    tareas = [procesar_fila(idx, row) for idx, row in df_final.iterrows()]

    for future in tqdm(asyncio.as_completed(tareas), total=len(tareas), desc="Completando el DF (reselección)", unit="diagnostico"):
        await future

    for idx, resultado in resultados:
        df_final.loc[idx, "CIE10_predicted_V2"] = resultado

    df_final = df_final.reset_index(drop=True)
    
    return df_final