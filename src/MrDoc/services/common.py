import re
import json
import yaml
import torch
import pandas as pd
import plotly.express as px
from pathlib import Path
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
from IPython.display import HTML, display
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from ..io.reader import load_schema_info
from ..config import BASE_DIR
from ..models import Any

###
with open(BASE_DIR / "docs" / "prompts.yml", 'r', encoding='utf-8') as file:
    prompts = yaml.safe_load(file)
CODE_RE = re.compile(r"^[A-Z][A-Za-z0-9]{2}(?:\.[A-Za-z0-9]{1,})?$", re.I)
PARTIAL_RE = re.compile(r"[A-Z][A-Za-z0-9]{1,2}(?:\.[A-Za-z0-9]{1,})?", re.I)
LLM_TRUNCATED_OUTPUT = re.compile(r'\{(?:[^{}"]|"(?:(?:\\.)|[^"\\])*")*\}', flags=re.DOTALL)

model = SentenceTransformer('all-MiniLM-L6-v2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cie10_judger_tokenizer = AutoTokenizer.from_pretrained("JulenRM/RigoBERTa-Clinical_CIE10Judger")
cie10_judger_model = AutoModelForSequenceClassification.from_pretrained("JulenRM/RigoBERTa-Clinical_CIE10Judger").to(device)

tokenizer_ner = AutoTokenizer.from_pretrained("IIC/RigoBERTa-Clinical", trim_offsets=False, use_fast=True)
model_ner = AutoModel.from_pretrained("IIC/RigoBERTa-Clinical").to(device)
###

def procesar_json_diagnosticos(data: dict[str, dict]) -> pd.DataFrame:
    """
    Extract the data from the multiple `dict` and creates a DataFrame with them

    Parameters
    ----------
        `data`: dict[str, dict]
            - Dict with all the data to tabulate. Keys are document name and value is the content in dict format

    Returns
    -------
        `df`: pd.DataFrame
            - DataFrame constructed with all the data
    """
    diagnosticos_consolidado = []

    for key,value in data.items():
        if 'diagnosticos' in value:
            for diagnostico in value['diagnosticos']:
                diagnostico['nombre_archivo'] = key
                diagnosticos_consolidado.append(diagnostico)

    df = pd.DataFrame(diagnosticos_consolidado)
    df = df.rename(columns={
        "diagnostico_extraido": "diagnostico_predicted",
        "codigo_CIE10": "CIE10_predicted"
    })

    return df


def validate_json_created(json_created: dict, schema: Path) -> bool:
    """
    Validate if the json created follows the schema proposed. If not, return false

    Parameters
    ----------
        `json`: dict
            - The JSON created
        `schema`: Path
            - Direction of the file
    Returns
    -------
        `result`: bool
            - Boolean result if the JSON follow the schema proposed
    """
    with open(schema, "r", encoding="utf-8") as f:
        diagnosticos_json_schema = json.load(f)
        diagnosticos_json_schema_validator = Draft202012Validator(diagnosticos_json_schema)

    try:
        diagnosticos_json_schema_validator.validate(json_created)
        return True
    except ValidationError:
        return False
    
def validate_complete_objects_from_truncated_output(txt: str) -> list:
    """
    Extract only the correct objects from the truncated output of an LLM

    Parameters
    ----------
        `txt`: str
            - The whole output

    Returns
    -------
        `objetos`: list
            - List with all the correct objects from the output
    """
    objetos = []
    try:
        txt = bytes(txt, "utf-8").decode("unicode_escape")
    except Exception:
        pass
    
    for _, m in enumerate(LLM_TRUNCATED_OUTPUT.finditer(txt), start=1):
        bloque = m.group(0)
        try:
            if "'" in bloque and '"' not in bloque:
                bloque = bloque.replace("'", '"')
            cargado = json.loads(bloque)
            if isinstance(cargado, dict):
                objetos.append(cargado)
            elif isinstance(cargado, list):
                objetos.extend([x for x in cargado if isinstance(x, dict)])
        except json.JSONDecodeError:
            continue
    return objetos





def _best_match(key: str, candidates: set[str], threshold: float = 0.6) -> str | None:
    """
    Find the best matching key from a set of candidates using fuzzy similarity.

    Parameters
    ----------
        `key`: str
            - The key name to compare.
        `candidates`: set[str]
            - Set of valid key names to compare against.
        `threshold`: float, optional
            - Minimum similarity ratio to consider a match (default = 0.75).

    Returns
    -------
        `match`: str | None
            - The best matching key name if similarity >= threshold, otherwise None.
    """
    if not candidates:
        return None

    # Encode both the key and the candidates
    key_emb = model.encode(key, convert_to_tensor=True)
    cand_embs = model.encode(list(candidates), convert_to_tensor=True)

    # Compute cosine similarities
    cos_scores = cos_sim(key_emb, cand_embs)[0]

    # Find the best-scoring candidate
    best_idx = int(cos_scores.argmax())
    best_score = float(cos_scores[best_idx])
    best = list(candidates)[best_idx]

    return best if best_score >= threshold else None


def clean_objects_with_schema(objetos: list[dict], schema_path: Path) -> list[dict]:
    """
    Clean and validate extracted objects against a schema definition.

    Parameters
    ----------
        `objetos`: list[dict]
            - List of objects returned by `validate_complete_objects_from_truncated_output`.
        `schema_path`: Path
            - Path to the JSON schema file (e.g., 'esquema.json').

    Returns
    -------
        `objetos_limpios`: list[dict]
            - List of cleaned and validated dictionaries.
            Each dictionary:
                * Contains only keys defined in the schema.
                * Has similar/translated keys renamed to the correct schema names.
                * Is kept only if all required fields are present.
    """
    valid_keys, required, allow_extra = load_schema_info(schema_path)

    resultado = []
    for obj in objetos:
        limpio = {}
        for k, v in obj.items():
            match = _best_match(k, valid_keys)
            if match:
                limpio[match] = v
        if required.issubset(limpio.keys()):
            resultado.append(limpio)
    return resultado


def clean_single_cie10_value(value: str) -> str | None:
    """
    Clean or validate a single CIE10 code string.

    Parameters
    ----------
        value : str
            - Raw value (possibly messy) representing a CIE10 code.

    Returns
    -------
        str | None
            - Cleaned CIE10 code string if valid, otherwise None.
    """
    if pd.isna(value):
        return None

    s = str(value).strip()
    if CODE_RE.match(s):
        return s
    else:
        m = PARTIAL_RE.search(s)
        if m:
            return m.group(0)
    return None


def clean_df_obtained_with_llm(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Clean the DataFrame previously constructed by an LLM.
    Keeps only rows with valid and clean CIE10 codes.

    Parameters
    ----------
        df : pd.DataFrame
            - DataFrame to clean
        col_name : str
            - Column name where the CIE10 codes are located

    Returns
    -------
        pd.DataFrame
            - Cleaned DataFrame
    """
    keep_idx = []

    for idx, val in df[col_name].items():
        cleaned = clean_single_cie10_value(val)
        if cleaned is not None:
            df.at[idx, col_name] = cleaned
            keep_idx.append(idx)

    return df.loc[keep_idx].reset_index(drop=True)


def show_scrollable(df: pd.DataFrame, height: int = 400):
    """
    Displays a DataFrame as a scrollable table

    Parameters
    ----------
        `df`: pd.DataFrame
            - DataFrame to display
        `height`: int = 400
            - Maximum hight of the HTML window created

    Returns
    -------
        `None`
    """
    html = f'''
    <div style="max-height:{height}px; overflow:auto; border:1px solid #ddd; border-radius:6px;">
        {df.to_html(border=0)}
    </div>
    '''
    display(HTML(html))

def _parse(CIE10: str) -> tuple:
    """
    Given a CIE10 code, it parses and returns into three different pieces **left value**, **right value** and if it **has a dot** (which separates the values)

    Parameters
    ----------
        `CIE10`: str
            - CIE10 code to parse

    Returns
    -------
        ``: tuple
            - Tuple of **left value**, **right value** and if it **has a dot** values
    """
    c = CIE10.strip().upper()
    if not CODE_RE.match(c):
        raise ValueError(f"Invalid CIE-10 code: {CIE10!r}")
    if "." in c:
        left, post = c.split(".", 1)
        return left, post, True
    return c, "", False

def find_CIE10_similars(CIE10: str, CIE10_full_list: list, level: int = 0) -> list:
    """
    Giving a CIE10 code, this functions returns all the similar codes by the level wanted

    The value given to the variable `level` indicates where to cut the fixed values. Level *0* means to start after the *dot*.

    **Example**: 
        `CIE10` = K75.23
        `level` = 0
    **CIE10 FIXED**
        K75.xx

    ---

    **Example**: 
        `CIE10` = K75.23
        `level` = -2
    **CIE10 FIXED**
        Kxx.xx

    ---

    Some examples are the following

    ---

    **Example**: 
        `CIE10` = K75.23
        `level` = 1
    **Return**:
        `CIE10_rest` = [K75.20, K75.21, ...]

    ---

    **Example**: 
        `CIE10` = K75.23
        `level` = -1
    **Return**:
        `CIE10_rest` = [K70.00, K70.01, ...]

    ---
        
    Parameters
    ----------
        `CIE10`: str
            - CIE10 code to start the search
        `CIE10_full_list`: list
            - Full list of CIE10 codes
        `level`: int = 0
            - Level indicating the fixed values

    Returns
    -------
        `CIE10_rest`: list
            - List with all the CIE10 codes founded
    """
    # Parse CIE10 code
    left, post, has_dot = _parse(CIE10)

    # Build matching prefix
    if level < 0:
        # Broaden before the dot
        # cut inside the 3-char left block
        cut = max(1, min(3 + level, 3))  # e.g., -1 -> 2 chars ('A0'), -2 -> 1 char ('A')
        target = left[:cut]
        include_non_dotted = True
    else:
        if has_dot:
            # Seed has dot: 'LDD.' + post[:level]
            target = left + "."
            if level >= 1:
                target += post[:level]
            include_non_dotted = False  # matching a dotted prefix
        else:
            # Seed has no dot
            if level == 0:
                # Include the seed itself + all dotted children
                target = left  # 'A01'
                include_non_dotted = True
            else:
                # level >= 1: only dotted family members
                target = left + "."  # 'A01.'
                include_non_dotted = False

    # Collect matches, preserving input order
    CIE10_rest = []
    for c in CIE10_full_list:
        try:
            l2, p2, d2 = _parse(c)
        except ValueError:
            continue
        cand = l2 + ("." + p2 if d2 else "")

        # If we're matching a dotted prefix, skip non-dotted candidates
        if not include_non_dotted and not d2:
            # Exception: level==0 with seed no dot should still include the seed itself
            if level == 0 and not has_dot and l2 == left and not d2:
                CIE10_rest.append(c)  # include the seed 'A01'
            continue

        if cand.startswith(target):
            # For level==0 with seed no dot, we already handled the seed above,
            # but this condition won't re-add it because cand ('A01') does start
            # with target ('A01'), and we do want it included exactly once.
            if level == 0 and not has_dot:
                # Avoid duplicates: we might add the seed twice if present
                if c not in CIE10_rest:
                    CIE10_rest.append(c)
            else:
                CIE10_rest.append(c)

    return CIE10_rest


def completar_df_extra_data_for_analysis(df: pd.DataFrame, sim_threshold: float = 0.6, tratamiento_fallos: bool = False) -> pd.DataFrame:
    """
    Función que se encarga de añadir nuevas columnas para completar la información que se puede averiguar de lo obtenido anteriormente y dictar que valores son correctos e incorrectos

    No realiza ninguna busqueda nueva ni elimina contenido anterior. Solamente expande añadiendo nuevas métricas. En este caso son las siguientes:

        1) CIE10 (True si lo anterior al punto coincide)
            1.1) Predicted w/ Nearest
            1.2) Predicted w/ Selected
            1.3) Nearest w/ Selected
            1.4) Predicted w/ Selected wnot/ Nearest
            1.5) Nearest w/ Selected wnot/ Predicted
            1.6) Predicted w/ Nearest w/ Selected

        2) Columnas `tree_x` -> Utilizadas para decidir si un valor es correcto o incorrecto

    Parameters
    ----------
        `df`: pd.DataFrame
            - DataFrame con los datos obtenidos previamente
        `sim_threshold` : float
            - Umbral para similarity_predicted_nearest y similarity_predicted_selected
        `tratamiento_fallos`: bool
            - Booleano para saber si realizar el analisis con los que no se han llegado a decidir como correcto anteriormente

    Returns
    -------
        `df`: pd.DataFrame
            - DataFrame con las nuevas columnas añadidas correctamente
    """
    if(tratamiento_fallos):
        sufijo = "_V2"
    else:
        sufijo = ""

    cie10_sim_binary = [("predicted", "nearest"), ("predicted", "selected"), ("nearest", "selected")]

    for comp in cie10_sim_binary:
        df[f"similarity_{comp[0]}_{comp[1]}_CIE10{sufijo}"] = (
            df[f"CIE10_{comp[0]}"].astype("string").str.strip().str.upper().str.split(".", n=1).str[0]
            ==
            df[f"CIE10_{comp[1]}"].astype("string").str.strip().str.upper().str.split(".", n=1).str[0]
        )
    
    cie10_sim_ternary = [("predicted", "selected", "nearest"), ("nearest", "selected", "predicted"), ("predicted", "nearest", "selected")]

    for i, comp in enumerate(cie10_sim_ternary):
        if i != len(cie10_sim_ternary)-1:
            df[f"similarity_{comp[0]}_and_{comp[1]}_not_{comp[2]}_CIE10{sufijo}"] = (
                (
                    df[f"CIE10_{comp[0]}"].astype("string").str.strip().str.upper().str.split(".", n=1).str[0]
                    == 
                    df[f"CIE10_{comp[1]}"].astype("string").str.strip().str.upper().str.split(".", n=1).str[0]
                )
                &
                (
                    df[f"CIE10_{comp[2]}"].astype("string").str.strip().str.upper().str.split(".", n=1).str[0]
                    !=
                    df[f"CIE10_{comp[1]}"].astype("string").str.strip().str.upper().str.split(".", n=1).str[0]
                )
            )
        else:
            df[f"similarity_{comp[0]}_and_{comp[1]}_and_{comp[2]}_CIE10{sufijo}"] = (
                (
                    df[f"CIE10_{comp[0]}"].astype("string").str.strip().str.upper().str.split(".", n=1).str[0]
                    == 
                    df[f"CIE10_{comp[1]}"].astype("string").str.strip().str.upper().str.split(".", n=1).str[0]
                )
                &
                (
                    df[f"CIE10_{comp[2]}"].astype("string").str.strip().str.upper().str.split(".", n=1).str[0]
                    ==
                    df[f"CIE10_{comp[1]}"].astype("string").str.strip().str.upper().str.split(".", n=1).str[0]
                )
            )

    S = df[f'similarity_predicted_and_nearest_and_selected_CIE10{sufijo}']      # bool
    I = df['similarity_predicted_nearest']                                      # num
    L = df[f'similarity_predicted_selected{sufijo}']                            # num
    O = df[f'similarity_nearest_selected_CIE10{sufijo}']                        # bool

    df[f'tree_1{sufijo}'] = (S & (I > sim_threshold) & (L > sim_threshold))
    df[f'tree_2{sufijo}'] = (df[f'tree_1{sufijo}'] | (S & (L > sim_threshold)))
    df[f'tree_3{sufijo}'] = (df[f'tree_2{sufijo}'] | (S & (I > sim_threshold)))
    df[f'tree_4{sufijo}'] = (df[f'tree_3{sufijo}'] | O)

    df = df.reset_index(drop=True)

    return df


def plot_histogram_with_binary_data(df: pd.DataFrame, column_name_emb_sim: str, column_name_cie10_sim: str, title: str, umbral: float = 0.6, y_max: float = 10, start: float = 0, end: float = 1, size: float = 0.01):
    """
    Función que plotea un histograma con los datos de similaridad de embedding que se le indique. 

    También incluye información visual de como se distribuye la similaridad de embeddings entre codigos CIE10 según la columna que se le indique

    Parameters
    ----------
        `df`: pd.DataFrame
            - DataFrame con los datos obtenidos previamente
        `column_name_emb_sim`: str
            - Nombre de la columna de donde se quieren obtener los datos de similaridad de embeddings
        `column_name_cie10_sim`: str
            - Nombre de la columna de donde se quieren obtener los datos de similaridad de codigos CIE10
        `title`: str
            - Title of the histogram
        `umbral`: float
            - Valor arbitrario para poder mostrar una linea vertical sobre el histrograma y poder establecer un umbral
        `y_max`: float
            - Valor con el cual se establece la máxima altura del eje y
        `start`: float
            - Valor donde empieza el ploteo del histograma
        `end`: float
            - Valor donde acaba el ploteo del histograma
        `size`: float
            - Valor de división de bins del histograma
            
    Returns
    -------
        - `None`
    """
    x = pd.to_numeric(df[column_name_emb_sim], errors="coerce").dropna()
    media = float(x.mean())

    # --- Datos limpios para que leyenda y hist coincidan ---
    df_valid = df.copy()
    df_valid["_xnum_"] = pd.to_numeric(df_valid[column_name_emb_sim], errors="coerce")
    df_valid = df_valid[
        df_valid["_xnum_"].notna() & df_valid[column_name_cie10_sim].isin([True, False])
    ]

    # Recuentos para la leyenda
    n_true  = int((df_valid[column_name_cie10_sim] == True).sum())
    n_false = int((df_valid[column_name_cie10_sim] == False).sum())

    # Histograma coloreado por booleano
    fig = px.histogram(
        df_valid,
        x="_xnum_",
        color=column_name_cie10_sim,
        color_discrete_map={True: "green", False: "red", "True": "green", "False": "red"},
        labels={column_name_cie10_sim: column_name_cie10_sim, "_xnum_": column_name_cie10_sim},
        title=title
    )
    fig.update_traces(xbins=dict(start=start, end=end, size=size))
    fig.update_xaxes(range=[start, end], title=column_name_emb_sim)

    # Fijar y mantener el máximo del eje Y
    if y_max is not None:
        fig.update_yaxes(range=[0, y_max], autorange=False, title="Frecuencia")
    else:
        fig.update_yaxes(autorange=True, title="Frecuencia")

    # Leyenda con cantidades
    def _rename_trace(t):
        name = str(t.name).strip().lower()
        if name in ["true", "1.0", "1"]:
            t.name = f"True ({n_true})"
        elif name in ["false", "0.0", "0"]:
            t.name = f"False ({n_false})"
    fig.for_each_trace(_rename_trace)
    fig.update_layout(legend_title_text=column_name_cie10_sim)

    # Presentación
    fig.update_layout(
        barmode="relative",  # "overlay" para superponer si lo prefieres
        bargap=0,
        height=520,
        margin=dict(t=90)
    )

    # Líneas de referencia
    fig.add_vline(x=umbral, line_width=2, line_dash="dash", line_color="red",
                    annotation_text=f"Umbral {umbral}", annotation_position="top")
    fig.add_vline(x=media, line_width=2, line_color="black",
                    annotation_text=f"Media {media:.2f}", annotation_position="top left")

    # ---- Conteos a cada lado del umbral (sobre el total de x) ----
    n = int(len(x))
    izq_u = int((x < umbral).sum())
    der_u = int((x >= umbral).sum())

    fig.add_annotation(x=umbral, y=1.13, yref="paper", xanchor="right",
                        text=f"< {umbral:.2f}: {izq_u} ({izq_u/n:.1%})",
                        showarrow=False)
    fig.add_annotation(x=umbral, y=1.13, yref="paper", xanchor="left",
                        text=f"| ≥ {umbral:.2f}: {der_u} ({der_u/n:.1%})",
                        showarrow=False)

    fig.show()

def plot_true_evolution(df: pd.DataFrame, title: str = None):
    """
    Función que plotea la evolución del porcentaje de valores True en columnas
    con nombres del tipo 'tree_1', 'tree_2', ..., 'tree_1_V2', etc.
    
    La función identifica automáticamente las columnas, las ordena en el
    orden lógico (1..5, luego 1_V2..5_V2, etc.) y muestra un gráfico de línea
    con marcadores y etiquetas de valores encima de cada punto.

    Parameters
    ----------
        `df`: pd.DataFrame
            - DataFrame con las columnas booleanas cuyo patrón sea 'tree_X' o 'tree_X_VY'
        `title`: str
            - Title to display
    Returns
    -------
        - `None`
    """

    # Patrón para capturar número y versión
    pattern = re.compile(r"^tree_(\d)(?:_V(\d+))?$")

    # Extraer y ordenar las columnas
    parsed = []
    for col in df.columns:
        m = pattern.match(col)
        if m:
            num = int(m.group(1))
            ver = int(m.group(2)) if m.group(2) else 1
            parsed.append((col, num, ver))

    if not parsed:
        raise ValueError("No se encontraron columnas que coincidan con el patrón 'tree_X' o 'tree_X_VY'.")

    ordered = [c for c, _, _ in sorted(parsed, key=lambda x: (x[2], x[1]))]

    # Calcular proporción de True por columna
    true_rates = df[ordered].astype(bool).mean().reset_index()
    true_rates.columns = ["columna", "proporcion_true"]

    # Crear gráfico con Plotly
    fig = px.line(
        true_rates,
        x="columna",
        y="proporcion_true",
        markers=True,
        text=true_rates["proporcion_true"].round(2),
        title="Evolución del porcentaje de True por columna" if title is None else title
    )

    # Ajustes visuales
    fig.update_traces(textposition="top center")
    fig.update_yaxes(range=[0, 1], title="Proporción de aciertos")
    fig.update_xaxes(title="Columna - tree_x")
    fig.update_layout(template="plotly_white", title_x=0.5)

    fig.show()

def average_overlapping_hidden_states_checked(windows: list[dict[str, Any]], last_hidden_state_list: list[torch.Tensor], *, window_tokens_no_special: int, stride: int, strict: bool = True) -> tuple[list[torch.Tensor], list[str]]:
    """
    Solapa los estados ocultos de los tokens solapados entre ventanas consecutivas, comprobando que las dimensiones, máscaras y parámetros sean coherentes.

    Parameters
    ----------
        `windows`: list[dict[str, Any]]
            - Lista de ventanas generadas previamente. Cada ventana debe contener, al menos, la clave `attention_mask` para identificar los tokens reales

        `last_hidden_state_list`: list[torch.Tensor]
            - Lista de tensores con los estados ocultos generados por el modelo para cada ventana. Cada tensor debe tener forma **(1, seq_len, hidden_dim)**

        `window_tokens_no_special`: int
            - Número máximo de tokens reales por ventana, sin contar tokens especiales como `CLS` y `SEP`

        `stride`: int
            - Desplazamiento entre ventanas consecutivas. Se usa junto con `window_tokens_no_special` para calcular el solapamiento.

        `strict`: bool
            - Si es **True**, lanza errores cuando detecta inconsistencias en los datos, dimensiones o solapamientos. Si es **False**, registra advertencias en **logs** y continúa cuando sea posible. Por defecto es **True**.

    Returns
    -------
        out, logs: tuple[list[torch.Tensor], list[str]]
            - Tupla formada por:
            
                - `out`: lista de tensores con los estados ocultos actualizados, donde los tokens solapados entre ventanas han sido promediados.
                - `logs`: lista de mensajes informativos o advertencias sobre el proceso.
    """

    logs = []

    if len(windows) != len(last_hidden_state_list):
        raise ValueError("windows y last_hidden_state_list deben tener la misma longitud")

    if window_tokens_no_special <= 0:
        raise ValueError("window_tokens_no_special debe ser > 0")
    if stride <= 0:
        raise ValueError("stride debe ser > 0")

    overlap = window_tokens_no_special - stride
    if overlap < 0:
        msg = f"overlap negativo: window_tokens_no_special({window_tokens_no_special}) - stride({stride}) = {overlap}"
        if strict:
            raise ValueError(msg)
        logs.append("WARNING: " + msg)
        # sin solape útil
        return [h.clone() for h in last_hidden_state_list], logs

    out = [h.clone() for h in last_hidden_state_list]

    for i in range(len(out) - 1):
        h_i = out[i]      # (1, Li, H)
        h_j = out[i + 1]  # (1, Lj, H)

        if h_i.dim() != 3 or h_j.dim() != 3 or h_i.size(0) != 1 or h_j.size(0) != 1:
            msg = f"Ventanas deben ser (1, seq_len, hidden_dim). Got {tuple(h_i.shape)} and {tuple(h_j.shape)} at i={i}"
            if strict:
                raise ValueError(msg)
            logs.append("WARNING: " + msg)
            continue

        Li = h_i.shape[1]
        Lj = h_j.shape[1]

        mask_i = torch.tensor(windows[i]["attention_mask"])
        mask_j = torch.tensor(windows[i + 1]["attention_mask"])

        # contar solo tokens activos
        real_i = int(mask_i.sum().item()) - 2  # quitamos CLS y SEP
        real_j = int(mask_j.sum().item()) - 2

        if strict:
            if real_i <= 0:
                raise ValueError(f"Ventana {i} no tiene tokens reales: seq_len={Li}")
            if real_j <= 0:
                raise ValueError(f"Ventana {i+1} no tiene tokens reales: seq_len={Lj}")
        else:
            if real_i <= 0 or real_j <= 0:
                logs.append(f"WARNING: ventana {i} o {i+1} sin tokens reales (Li={Li}, Lj={Lj}). Se omite.")
                continue

        # Si overlap == 0 no hay nada que promediar
        if overlap == 0:
            logs.append("INFO: overlap=0, no se promedia nada.")
            return out, logs

        # solape efectivo (la última ventana puede ser corta)
        ov = min(overlap, real_i, real_j)
        if ov <= 0:
            msg = f"Sin solape efectivo en par (i={i}, i+1={i+1}): overlap={overlap}, real_i={real_i}, real_j={real_j}"
            if strict:
                raise ValueError(msg)
            logs.append("WARNING: " + msg)
            continue

        # Sanity check: si no es la última ventana, normalmente real_i debería ser == window_tokens_no_special
        # (excepto quizá la última).
        if strict and i < len(out) - 2:
            # esta condición depende de cómo generaste las ventanas; si tu extractor puede cortar antes,
            # pon strict=False.
            if real_i != window_tokens_no_special:
                raise ValueError(
                    f"Ventana {i} tiene {real_i} tokens reales, esperado {window_tokens_no_special}. "
                    "Si tu extractor produce ventanas más cortas, usa strict=False."
                )

        # indices dentro del tensor (saltando specials):
        # reales de i: [1 .. 1+real_i)
        # últimos ov reales:
        i_start = 1 + (real_i - ov)
        i_end   = 1 + real_i

        # reales de j: [1 .. 1+real_j)
        # primeros ov reales:
        j_start = 1
        j_end   = 1 + ov

        # Promedio simétrico
        avg = 0.5 * (h_i[:, i_start:i_end, :] + h_j[:, j_start:j_end, :])

        h_i[:, i_start:i_end, :] = avg
        h_j[:, j_start:j_end, :] = avg

        logs.append(
            f"OK pair {i}-{i+1}: ov={ov} | "
            f"i[{i_start}:{i_end}] <-> j[{j_start}:{j_end}] | "
            f"shapes Li={Li}, Lj={Lj}"
        )

    return out, logs

def series_to_striding_ner_windows(serie, tokenizer: Any, *, window_tokens: int = 512, stride: int = 128, padding: bool = False) -> list[dict[str, Any]]:
    """
    Divide el texto de una serie en ventanas de tokens con solapamiento, usando stride, y tokeniza cada ventana para preparar entradas compatibles con el modelo.

    Parameters
    ----------
        `serie`: pd.Series
            - Fila o serie que contiene el texto a procesar. Debe incluir la clave `Text` con el contenido completo del documento

        `tokenizer`: Any
            - Tokenizer utilizado para convertir el texto en tokens, obtener los `input_ids`, las máscaras de atención y los offsets de caracteres

        `window_tokens`: int
            - Número máximo de tokens por ventana, sin contar tokens especiales. Por defecto es **512**

        `stride`: int
            - Número de tokens de solapamiento entre ventanas consecutivas. Por defecto es **128**

        `padding`: bool
            - Indica si se debe aplicar padding hasta la longitud máxima `window_tokens + 2` para incluir tokens especiales. Por defecto es **False**.

    Returns
    -------
        `windows`: list[dict[str, Any]]
            - Lista de diccionarios, donde cada diccionario representa una ventana del texto. Cada ventana incluye su índice, posiciones de caracteres, texto, **input_ids**, **attention_mask** y **offset_mapping**
    """
    text = serie["Text"]

    # Tokenize full doc no special tokens
    enc_full = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    full_ids = enc_full["input_ids"]
    full_offsets = enc_full["offset_mapping"]
    if not full_ids:
        return []

    step = window_tokens - stride
    if step <= 0:
        raise ValueError("Invalid stride/window size")

    windows = []
    token_start = 0
    widx = 0

    while token_start < len(full_ids):
        token_end = min(token_start + window_tokens, len(full_ids))
        win_char_start = full_offsets[token_start][0]
        win_char_end = full_offsets[token_end - 1][1]
        win_text = text[win_char_start:win_char_end]

        # Original token slice
        win_ids_slice = full_ids[token_start:token_end]
        win_offsets_slice = full_offsets[token_start:token_end]

        # Prepare tokenized window WITH specials
        if padding:
            retok = tokenizer(win_text, truncation=True, padding="max_length", max_length=window_tokens + 2, return_offsets_mapping=True)
        else:
            retok = tokenizer(win_text, truncation=True, padding=False, return_offsets_mapping=True)

        # Now align original token mapping to retokenized
        # Build map: char -> retokenized token index
        tok_char_to_new_idx = {}
        for idx, (s_off, e_off) in enumerate(retok["offset_mapping"]):
            for cpos in range(s_off, e_off):
                tok_char_to_new_idx[cpos] = idx

        windows.append({"window_index": widx, "char_start": win_char_start, "char_end": win_char_end, "text": win_text, "input_ids": retok["input_ids"], "attention_mask": retok["attention_mask"], "offset_mapping": retok["offset_mapping"]})

        if token_end == len(full_ids):
            break
        token_start += step
        widx += 1

    return windows
