import os
import ast
import string
import contextlib
import numpy as np
import pandas as pd

from text_to_num import text2num

from ....models.schemas import (
    Any
)

###
@contextlib.contextmanager
def suppress_stderr():
    old_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(devnull)
###

def is_number(token: str):
    """
    Comprueba si un token representa un número, ya sea en formato numérico o escrito en texto en inglés o español.

    Parameters
    ----------
        `token`: str
            - Token que se quiere evaluar como posible número

    Returns
    -------
        ``: bool
            - `True` si el token puede interpretarse como número. En caso contrario, devuelve `False`
    """
    token = str(token).strip("▁").strip()
    if not token:
        return False
    try:
        float(token.replace(",", "."))
        return True
    except Exception:
        pass

    lowered = token.lower().replace("-", " ").strip()
    if not lowered:
        return False

    for lang in ("en", "es"):
        try:
            with suppress_stderr():
                text2num(lowered, lang)
            return True
        except Exception:
            pass

    return False

def merge_sentencepiece_words(tokens: list):
    """
    Reconstruye palabras completas a partir de tokens generados por un tokenizer tipo SentencePiece

    Parameters
    ----------
        `tokens`: list
            - Lista de tokens. Los tokens que empiezan por `▁` se interpretan como el inicio de una nueva palabra

    Returns
    -------
        `words`: list
            - Lista de palabras reconstruidas a partir de los tokens originales
    """
    words = []
    current = []

    for tok in tokens:
        if tok.startswith("▁"):
            if current:
                words.append("".join(current))
            current = [tok[1:]]  # remove ▁
        else:
            if current:
                current.append(tok)
            else:
                current = [tok]

    if current:
        words.append("".join(current))

    return words
    
def is_punct(word: str):
    """
    Comprueba si una palabra está formada únicamente por signos de puntuación.

    Parameters
    ----------
        `word`: str
            - Palabra o token que se quiere comprobar

    Returns
    -------
        ``: bool
            - `True` si todos los caracteres de `word` son signos de puntuación y la cadena no está vacía. En caso contrario, devuelve `False`
    """
    return all(ch in string.punctuation for ch in word) and len(word) > 0

def is_stopword_punct_or_number(word: str, stopwords_es: list):
    """
    Comprueba si una palabra es una stopword, un signo de puntuación o un número.

    Parameters
    ----------
        `word`: str
            - Palabra que se quiere evaluar

        `stopwords_es`: list
            - Lista de stopwords en español usadas como criterio de filtrado

    Returns
    -------
        ``: bool
            - `True` si la palabra es una stopword, puntuación o número. En caso contrario, devuelve `False`
    """
    word = word.strip().lower()
    return (word in stopwords_es or is_number(word) or is_punct(word))


def has_bad_surrounding(decoder:list, stopwords_es: list, limit_stopwords_surround: int):
    """
    Evalúa si los tokens de contenido de un span están rodeados por demasiadas stopwords, números o signos de puntuación.

    Parameters
    ----------
        `decoder`: list
            - Lista de tokens decodificados que forman el span

        `stopwords_es`: list
            - Lista de stopwords en español usadas para detectar palabras poco informativas

        `limit_stopwords_surround`: int
            - Número máximo de palabras de contexto que se revisan alrededor de cada palabra de contenido

    Returns
    -------
        ``: bool
            - `True` si el span presenta un contexto considerado problemático. Devuelve `False` si encuentra una palabra de contenido con contexto aceptable
    """
    words = merge_sentencepiece_words(decoder)
    span = limit_stopwords_surround + 1

    for i, word in enumerate(words):
        if is_stopword_punct_or_number(word, stopwords_es):
            continue
        
        # check right
        if i - span >= 0:
            left_words = words[i - span:i]
            if all(is_stopword_punct_or_number(w, stopwords_es) for w in left_words):
                return False

        # check left
        if i + span < len(words):
            right_words = words[i + 1:i + 1 + span]
            if all(is_stopword_punct_or_number(w, stopwords_es) for w in right_words):
                return False

    return True

def get_edge_words(decoder: list):
    """
    Obtiene la primera y la última palabra de un span tras reconstruir las palabras completas desde tokens tipo SentencePiece.

    Parameters
    ----------
        `decoder`: list
            - Lista de tokens decodificados que forman el span

    Returns
    -------
        ``: tuple
            - Tupla con la primera y la última palabra en minúsculas. Si no hay palabras, devuelve **("", "")**
    """
    words = merge_sentencepiece_words(decoder)

    if not words:
        return "", ""

    return words[0].lower(), words[-1].lower()


def normalize_list(value: Any):
    """
    Normaliza un valor de entrada para devolverlo siempre como una lista.

    Parameters
    ----------
        `value`: Any
            - Valor que se quiere convertir a lista. Puede ser `None`, un `np.ndarray`, una lista, una tupla, una cadena de texto u otro objeto iterable

    Returns
    -------
        ``: list
            - Lista normalizada. Si el valor es **None** devuelve una lista vacía. Si es una cadena con formato de lista, intenta interpretarla como tal. Si no es iterable, devuelve una lista con el propio valor
    """
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, str):
        value = value.strip()

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
        except Exception:
            pass

        value = value.strip("[]")
        if not value:
            return []

        return [x.strip().strip("'").strip('"') for x in value.split(",")]

    try:
        return list(value)
    except TypeError:
        return [value]

def token_spans_are_adjacent(idx_a: list, idx_b: list):
    """
    Comprueba si dos spans de tokens son directamente consecutivos

    Parameters
    ----------
        `idx_a`: list
            - Índices de tokens del primer span.

        `idx_b`: list
            - Índices de tokens del segundo span.

    Returns
    -------
        ``: bool
            - **True** si los dos spans son adyacentes, es decir, si el último token de uno está justo antes del primer token del otro. En caso contrario, devuelve **False**
    """
    if not idx_a or not idx_b:
        return False

    return max(idx_a) + 1 == min(idx_b) or max(idx_b) + 1 == min(idx_a)

def same_entity_component(row_a: dict, row_b: dict):
    """
    Comprueba si dos spans deben formar parte de la misma entidad final.

    Parameters
    ----------
        `row_a`: dict
            - Diccionario con la información del primer span. Debe contener **token_set** y **token_idx**

        `row_b`: dict
            - Diccionario con la información del segundo span. Debe contener **token_set** y **token_idx**

    Returns
    -------
        ``: bool
            - **True** si los spans se solapan o si son directamente adyacentes. En caso contrario, devuelve **False**
    """
    if row_a["token_set"] & row_b["token_set"]:
        return True

    if token_spans_are_adjacent(row_a["token_idx"], row_b["token_idx"]):
        return True

    return False

def build_entity_token_idx_from_component(component: list):
    """
    Construye la lista final de índices de tokens de una entidad a partir de un componente de spans.

    Parameters
    ----------
        `component`: list
            - Lista de spans que forman parte de la misma entidad. Cada elemento debe contener la clave **token_idx**

    Returns
    -------
        `all_idx`: list
            - Lista ordenada con todos los índices de tokens que forman la entidad final, sin duplicados
    """
    all_idx = set()

    for row in component:
        all_idx.update(row["token_idx"])

    return sorted(all_idx)

def get_final_entity_components_from_group(group: pd.DataFrame, token_idx_col: str):
    """
    Obtiene los componentes finales de entidades dentro de un grupo de spans con el mismo archivo, instancia de texto y etiqueta predicha.

    Parameters
    ----------
        `group`: pd.DataFrame
            - Grupo de filas del DataFrame correspondiente a una misma combinación de archivo, instancia de texto y etiqueta predicha

        `token_idx_col`: str
            - Nombre de la columna que contiene los índices de tokens de cada span

    Returns
    -------
        `components`: list
            - Lista de componentes. Cada componente incluye: **source_row_indices** con los índices de las filas originales que forman el componente, y **final_token_idx** con los índices unidos de la entidad
    """
    rows = []

    for row_index, row in group.iterrows():
        token_idx = normalize_list(row[token_idx_col])
        token_idx = [int(x) for x in token_idx]

        if len(token_idx) == 0:
            continue

        rows.append({"row_index": row_index, "token_idx": token_idx, "token_set": set(token_idx)})

    visited = [False] * len(rows)
    components = []

    for i in range(len(rows)):
        if visited[i]:
            continue

        stack = [i]
        visited[i] = True
        component = []

        while stack:
            idx = stack.pop()
            component.append(rows[idx])

            for j in range(len(rows)):
                if visited[j]:
                    continue

                if same_entity_component(rows[idx], rows[j]):
                    visited[j] = True
                    stack.append(j)

        final_token_idx = build_entity_token_idx_from_component(component)

        components.append({"source_row_indices": [x["row_index"] for x in component], "final_token_idx": final_token_idx})

    return components

def char_span_to_token_span(text: str, start: int, end: int, tokenizer_ner: Any):
    encoding = tokenizer_ner(text, return_offsets_mapping=True, add_special_tokens=False)

    token_idxs = []

    for i, (tok_start, tok_end) in enumerate(encoding["offset_mapping"]):
        if tok_start < end and tok_end > start:
            token_idxs.append(i)

    return token_idxs

def lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x