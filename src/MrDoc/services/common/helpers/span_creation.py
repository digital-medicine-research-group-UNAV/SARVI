import os
import string
import contextlib

from text_to_num import text2num

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