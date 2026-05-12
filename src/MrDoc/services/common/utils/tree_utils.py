from hierarchicalsoftmax import SoftmaxNode

def parse_range(code: str):
    """
    Extrae la letra inicial y los límites numéricos de un código de rango.

    Parameters
    ----------
        `code`: str
            - Código de rango con formato tipo **M00-M25**

    Returns
    -------
        `letter`, `start_num`, `end_num`: tuple
            - Tupla formada por la letra del código, el número inicial y el número final del rango
    """
    start, end = code.split("-")
    letter = start[0]
    start_num = int("".join(c if c.isdigit() else "9" for c in start[1:]))
    end_num = int("".join(c if c.isdigit() else "9" for c in end[1:]))
    return letter, start_num, end_num

def is_valid(code: str):
    """
    Comprueba si un código individual o de rango tiene un formato válido.

    Parameters
    ----------
        `code`: str
            - Código que se quiere validar. Puede ser un código simple, como **M01**, o un rango, como **M00-M25**

    Returns
    -------
        ``: bool
            - ``True`` si el código es válido. En caso contrario, devuelve ``False``.
    """
    if len(code) == 3:
        return True
    elif "-" not in code:
        return False

    a, b = code.split("-")
    if a == b or a[0] != b[0]:
        return False
    else:
        return True
    
def normalize_code(code: str):
    """
    Normaliza un código para poder ordenarlo o compararlo numéricamente.

    Parameters
    ----------
        `code`: str
            - Código que se quiere normalizar. Si contiene letras en la parte numérica estas se sustituyen por **9** para facilitar la comparación

    Returns
    -------
        `letter`, `num`: tuple
            - Tupla formada por la letra inicial del código y su valor numérico normalizado
    """
    letter = code[0]
    rest = code[1:]

    if rest.isdigit():
        return letter, int(rest)

    num = ""
    for c in rest:
        if c.isdigit():
            num += c
        else:
            num += "9"

    return letter, int(num)

def is_code_in_range(code: str, range_code: str):
    """
    Comprueba si un código pertenece a un rango determinado.

    Parameters
    ----------
        `code`: str
            - Código individual que se quiere comprobar, por ejemplo **M01** o **M1A**

        `range_code`: str
            - Código de rango contra el que se compara, por ejemplo **M00-M25**

    Returns
    -------
        ``: bool
            - **True** si el código está dentro del rango indicado. En caso contrario, devuelve **False**
    """
    start, end = range_code.split("-")

    if not code[1:].isdigit():
        return code == start or code == end

    l1, v = normalize_code(code)
    l2, v_start = normalize_code(start)
    l3, v_end = normalize_code(end)

    if l1 != l2 or l1 != l3:
        return False

    return v_start <= v <= v_end

def is_numeric_leaf_code(code: str):
    """
    Comprueba si un código es una hoja numérica simple.

    Parameters
    ----------
        `code`: str
            - Código que se quiere comprobar.

    Returns
    -------
        ``: bool
            - **True** si el código no es un rango, empieza por una letra y el resto está formado solo por dígitos. En caso contrario, devuelve **False**.
    """
    return "-" not in code and len(code) >= 2 and code[0].isalpha() and code[1:].isdigit()


def are_consecutive_codes(a: str, b: str):
    """
    Comprueba si dos códigos numéricos simples son consecutivos.

    Parameters
    ----------
        `a`: str
            - Primer código.

        `b`: str
            - Segundo código.

    Returns
    -------
        ``: bool
            - ``True`` si ambos códigos pertenecen a la misma letra y el segundo código
              es exactamente el siguiente al primero. En caso contrario, devuelve
              ``False``.
    """
    if not (is_numeric_leaf_code(a) and is_numeric_leaf_code(b)):
        return False
    la, va = normalize_code(a)
    lb, vb = normalize_code(b)
    return la == lb and vb == va + 1

def make_range_name(codes: list):
    """
    Genera el nombre de un rango a partir de una lista de códigos numéricos.

    Parameters
    ----------
        `codes`: list
            - Lista de códigos numéricos simples pertenecientes a la misma letra

    Returns
    -------
        ``: str
            - Nombre del rango formado por el mínimo y el máximo código de la lista, manteniendo el ancho numérico original
    """
    letter = codes[0][0]
    nums = [int(c[1:]) for c in codes]
    width = max(len(c) - 1 for c in codes)
    return f"{letter}{min(nums):0{width}d}-{letter}{max(nums):0{width}d}"

def range_matches_children(parent_name: str, child_names: list):
    """
    Comprueba si un nodo padre de tipo rango representa exactamente el rango natural cubierto por sus hijos numéricos.

    Parameters
    ----------
        `parent_name`: str
            - Nombre del nodo padre, normalmente un rango como **A00-A09**

        `child_names`: list
            - Lista de nombres de nodos hijos numéricos

    Returns
    -------
        ``: bool
            - **True** si el rango del padre coincide con el mínimo y máximo de sus hijos. En caso contrario, devuelve **False**
    """
    if "-" not in parent_name or not child_names:
        return False

    letter, start_num, end_num = parse_range(parent_name)

    if not all(is_numeric_leaf_code(c) and c[0] == letter for c in child_names):
        return False

    values = [int(c[1:]) for c in child_names]
    return min(values) == start_num and max(values) == end_num

def create_synthetic_node(node_dict: dict, visible_name: str, description:str = None, alpha: float = 2.0, gamma: float = 2.0):
    """
    Crea un nodo sintético nuevo y lo añade al diccionario de nodos.

    Parameters
    ----------
        `node_dict`: dict
            - Diccionario donde se almacenan los nodos del árbol

        `visible_name`: str
            - Nombre visible del nodo que aparecerá en el árbol

        `description`: str
            - Descripción asociada al nodo. Si no se indica, se usa **visible_name**

        `alpha`: float
            - Valor de `alpha` usado al crear el `SoftmaxNode`. Por defecto es **2.0**

        `gamma`: float
            - Valor de `gamma` usado al crear el `SoftmaxNode`. Por defecto es **2.0**

    Returns
    -------
        `node`: SoftmaxNode
            - Nodo sintético creado y añadido a `node_dict`
    """
    key = visible_name
    if key in node_dict:
        i = 1
        while f"__synthetic__{visible_name}__{i}" in node_dict:
            i += 1
        key = f"__synthetic__{visible_name}__{i}"

    node = SoftmaxNode(visible_name, parent=None, description=description or visible_name, alpha=alpha, gamma=gamma)
    node_dict[key] = node
    return node