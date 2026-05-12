import re
import json
from pathlib import Path
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer

from ....data_io.reader import (
    load_schema_info
)

###
LLM_TRUNCATED_OUTPUT = re.compile(r'\{(?:[^{}"]|"(?:(?:\\.)|[^"\\])*")*\}', flags=re.DOTALL)

model = SentenceTransformer('all-MiniLM-L6-v2')
###

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