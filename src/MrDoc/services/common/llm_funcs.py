import re
import yaml
import torch
import pandas as pd

from ...config import (
    BASE_DIR
)

###
device = "cuda" if torch.cuda.is_available() else "cpu"

with open(BASE_DIR / "docs" / "prompts.yml", 'r', encoding='utf-8') as file:
    prompts = yaml.safe_load(file)
CODE_RE = re.compile(r"^[A-Z][A-Za-z0-9]{2}(?:\.[A-Za-z0-9]{1,})?$", re.I)
PARTIAL_RE = re.compile(r"[A-Z][A-Za-z0-9]{1,2}(?:\.[A-Za-z0-9]{1,})?", re.I)
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