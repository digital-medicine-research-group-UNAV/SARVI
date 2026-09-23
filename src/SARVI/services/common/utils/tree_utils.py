from hierarchicalsoftmax import SoftmaxNode

def parse_range(code: str):
    """
    Extracts the initial letter and the numerical limits from a range code.

    Parameters
    ----------
        `code`: str
            - Range code in the format **M00-M25**

    Returns
    -------
        `letter`, `start_num`, `end_num`: tuple
            - A tuple consisting of the letter in the code, the starting number, and the ending number of the range
    """
    start, end = code.split("-")
    letter = start[0]
    start_num = int("".join(c if c.isdigit() else "9" for c in start[1:]))
    end_num = int("".join(c if c.isdigit() else "9" for c in end[1:]))
    return letter, start_num, end_num

def is_valid(code: str):
    """
    Checks whether an individual code or a range of codes is in a valid format.

    Parameters
    ----------
        `code`: str
            - The code to be validated. It can be a single code, such as **M01**, or a range, such as **M00-M25**

    Returns
    -------
        ``: bool
            - ``True`` if the code is valid. Otherwise, returns ``False``.
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
    Normalizes a code so it can be sorted or compared numerically.

    Parameters
    ----------
        `code`: str
            - The code to be normalized. If it contains letters in the numeric portion, they are replaced with **9** to facilitate comparison

    Returns
    -------
        `letter`, `num`: tuple
            - A tuple consisting of the initial letter of the code and its normalized numerical value
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
    Checks whether a code belongs to a specific range.

    Parameters
    ----------
        `code`: str
            - Individual code to be checked, for example **M01** or **M1A**

        `range_code`: str
            - Range code against which the code is compared, for example **M00-M25**

    Returns
    -------
        ``: bool
            - **True** if the code is within the specified range. Otherwise, returns **False**
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
    Checks whether a code is a simple numeric string.

    Parameters
    ----------
        `code`: str
            - The code to be checked.

    Returns
    -------
        ``: bool
            - **True** if the code is not a range, starts with a letter, and consists solely of digits. Otherwise, returns **False**.
    """
    return "-" not in code and len(code) >= 2 and code[0].isalpha() and code[1:].isdigit()


def are_consecutive_codes(a: str, b: str):
    """
    Checks whether two simple numeric codes are consecutive.

    Parameters
    ----------
        `a`: str
            - First code.

        `b`: str
            - Second code.

    Returns
    -------
        ``: bool
            - ``True`` if both codes correspond to the same letter and the second code is exactly the one following the first. Otherwise, returns ``False``.
    """
    if not (is_numeric_leaf_code(a) and is_numeric_leaf_code(b)):
        return False
    la, va = normalize_code(a)
    lb, vb = normalize_code(b)
    return la == lb and vb == va + 1

def make_range_name(codes: list):
    """
    Generates the name of a range based on a list of numeric codes.

    Parameters
    ----------
        `codes`: list
            - List of single numeric codes corresponding to the same letter

    Returns
    -------
        ``: str
            - Name of the range formed by the minimum and maximum codes in the list, maintaining the original numerical range
    """
    letter = codes[0][0]
    nums = [int(c[1:]) for c in codes]
    width = max(len(c) - 1 for c in codes)
    return f"{letter}{min(nums):0{width}d}-{letter}{max(nums):0{width}d}"

def range_matches_children(parent_name: str, child_names: list):
    """
    Checks whether a parent node of type “range” exactly represents the natural range covered by its numeric children.

    Parameters
    ----------
        `parent_name`: str
            - Name of the parent node, typically a range such as **A00-A09**

        `child_names`: list
            - List of names of numeric child nodes

    Returns
    -------
        ``: bool
            - **True** if the parent's range matches the minimum and maximum of its children. Otherwise, returns **False**
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
    Creates a new synthetic node and adds it to the node dictionary.

    Parameters
    ----------
        `node_dict`: dict
            - Dictionary where the tree's nodes are stored

        `visible_name`: str
            - Visible name of the node that will appear in the tree

        `description`: str
            - Description associated with the node. If not specified, **visible_name** is used

        `alpha`: float
            - `alpha` value used when creating the `SoftmaxNode`. The default is **2.0**

        `gamma`: float
            - `gamma` value used when creating the `SoftmaxNode`. The default is **2.0**

    Returns
    -------
        `node`: SoftmaxNode
            - Synthetic node created and added to `node_dict`
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
