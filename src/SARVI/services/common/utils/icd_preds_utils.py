import torch
import numpy as np

from ....models.schemas import (
    Any
)

def tensor_to_item(x: Any):
    """
    Convierte tensores de PyTorch en valores escalares, manteniendo la estructura original de listas, tuplas y diccionarios.

    Parameters
    ----------
        `x`: Any
            - Objeto que se quiere convertir. Puede ser un tensor, una lista, una tupla, un diccionario u otro tipo de dato

    Returns
    -------
        ``: Any
            - Objeto convertido. Si es un tensor, devuelve su valor mediante **item()**. Si es una estructura anidada, convierte recursivamente sus elementos.
    """
    if isinstance(x, torch.Tensor):
        return x.item()

    elif isinstance(x, list):
        return [tensor_to_item(item) for item in x]

    elif isinstance(x, tuple):
        return tuple(tensor_to_item(item) for item in x)

    elif isinstance(x, dict):
        return {k: tensor_to_item(v) for k, v in x.items()}

    else:
        return x

def tensor_items_same_structure(obj: Any):
    """
    Convierte tensores de PyTorch a valores nativos de Python o listas, conservando la misma estructura del objeto original.

    Parameters
    ----------
        `obj`: Any
            - Objeto que se quiere convertir. Puede ser un tensor, una lista, una tupla, un diccionario u otro tipo de dato

    Returns
    -------
        ``: Any
            - Objeto con la misma estructura que la entrada. Los tensores escalares se convierten con **item()** y los tensores no escalares se convierten a listas usando **detach().cpu().tolist()**
    """
    if isinstance(obj, torch.Tensor):
        if obj.dim() == 0:
            return obj.item()
        return obj.detach().cpu().tolist()

    elif isinstance(obj, tuple):
        return tuple(tensor_items_same_structure(x) for x in obj)

    elif isinstance(obj, list):
        return [tensor_items_same_structure(x) for x in obj]

    elif isinstance(obj, dict):
        return {k: tensor_items_same_structure(v) for k, v in obj.items()}

    return obj

def remap_ids(x: Any, id_no_hs_to_id_hs: dict):
    """
    Remapea IDs de forma recursiva usando un diccionario de correspondencias.

    Parameters
    ----------
        `x`: Any
            - Objeto que contiene los IDs que se quieren remapear. Puede ser un entero, una lista, una tupla, un diccionario u otro tipo de dato

        `id_no_hs_to_id_hs`: dict
            - Diccionario que mapea IDs originales a nuevos IDs

    Returns
    -------
        ``: Any
            - Objeto con la misma estructura que la entrada, pero con los IDs enteros sustituidos por sus valores correspondientes en `id_no_hs_to_id_hs`. Los tipos no contemplados se devuelven sin modificar
    """
    if isinstance(x, int):
        return id_no_hs_to_id_hs[x]
    elif isinstance(x, list):
        return [remap_ids(item, id_no_hs_to_id_hs) for item in x]
    elif isinstance(x, tuple):
        return tuple(remap_ids(item, id_no_hs_to_id_hs) for item in x)
    elif isinstance(x, dict):
        return {k: remap_ids(v, id_no_hs_to_id_hs) for k, v in x.items()}
    else:
        return x
    
def tensor_a_float(x: Any):
    """
    Convierte un valor a **float**, contemplando tensores, listas y escalares.

    Parameters
    ----------
        `x`: Any
            - Valor que se quiere convertir. Puede ser un tensor de PyTorch, una lista o un valor escalar convertible a **float**

    Returns
    -------
        ``: float
            - Valor convertido a ``float``. Si es un tensor, se toma el primer elemento tras moverlo a CPU. Si es una lista, se convierte su primer elemento
    """
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().reshape(-1)[0].item())
    if isinstance(x, list):
        return float(x[0])
    return float(x)

def valor_normal(x: Any):
    """
    Normaliza valores para facilitar comparaciones entre tensores, arrays, listas, tuplas y escalares

    Parameters
    ----------
        `x`: Any
            - Valor que se quiere normalizar. Puede ser un tensor de PyTorch, un array de NumPy, una lista, una tupla, un escalar de NumPy o un escalar normal

    Returns
    -------
        ``: Any
            - Valor normalizado. Si contiene un único elemento, devuelve el escalar correspondiente. Si contiene varios elementos, devuelve una lista con los valores normalizados
    """
    # Tensor de torch
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            return x.reshape(-1)[0].item()
        return x.tolist()

    # Array de numpy
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.reshape(-1)[0].item()
        return x.tolist()

    # Lista o tupla de un elemento
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return valor_normal(x[0])
        return [valor_normal(v) for v in x]

    # Escalares numpy
    if hasattr(x, "item") and callable(x.item):
        try:
            return x.item()
        except:
            pass

    return x