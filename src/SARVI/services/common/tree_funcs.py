import pandas as pd
from tqdm.auto import tqdm
from hierarchicalsoftmax import SoftmaxNode
from collections import defaultdict

from .utils.tree_utils import (
    is_numeric_leaf_code, range_matches_children, make_range_name, create_synthetic_node, are_consecutive_codes, is_valid, is_code_in_range, parse_range, normalize_code
)

def add_min_consecutive_subgroups(node_dict: dict):
    """
    Refina un árbol jerárquico creando subgrupos mínimos de códigos consecutivos.

    Parameters
    ----------
        `node_dict`: dict
            - Diccionario con todos los nodos del árbol. Debe contener la clave **root**

    Returns
    -------
        ``: None
            - Modifica directamente los hijos de los nodos dentro de `node_dict`
    """
    root = node_dict["root"]

    def recurse(node):
        # 1) Primero bajar: nivel más bajo posible
        for child in list(node.children):
            recurse(child)

        children = list(node.children)

        # Leaves numéricos directos de este nodo
        numeric_leaves = [child for child in children if len(child.children) == 0 and is_numeric_leaf_code(child.name)]

        if len(numeric_leaves) < 2:
            return

        numeric_names = [child.name for child in numeric_leaves]

        # ¿Este nodo ya es el contenedor correcto de esos leaves?
        parent_already_is_container = ("-" in node.name and len(children) == len(numeric_leaves) and range_matches_children(node.name, numeric_names))
        working_node = node

        # 2) Si no lo es, crear contenedor intermedio
        #    Ej: bajo M, agrupar M95,M96,M97,M99 -> M95-M99
        if not parent_already_is_container:
            container_name = make_range_name(numeric_names)
            container = create_synthetic_node(node_dict, visible_name=container_name, description=container_name, alpha=2.0, gamma=2.0)

            new_children = []
            inserted = False

            for child in children:
                if child in numeric_leaves:
                    if not inserted:
                        new_children.append(container)
                        inserted = True
                    continue
                new_children.append(child)

            node.children = tuple(new_children)
            container.children = tuple(numeric_leaves)
            working_node = container

        # 3) Dentro del contenedor correcto, agrupar pares consecutivos
        children = list(working_node.children)
        new_children = []
        i = 0

        while i < len(children):
            current_child = children[i]

            can_pair = (i + 1 < len(children) and len(current_child.children) == 0 and is_numeric_leaf_code(current_child.name) and len(children[i + 1].children) == 0 and is_numeric_leaf_code(children[i + 1].name) and are_consecutive_codes(current_child.name, children[i + 1].name))

            if can_pair:
                next_child = children[i + 1]
                pair_name = make_range_name([current_child.name, next_child.name])

                # Evita duplicar algo como:
                # M26-M27
                #   └── M26-M27
                #       ├── M26
                #       └── M27
                if not (pair_name == working_node.name and len(children) == 2):
                    pair_node = create_synthetic_node(node_dict, visible_name=pair_name, description=pair_name, alpha=2.0, gamma=2.0)
                    pair_node.children = (current_child, next_child)
                    new_children.append(pair_node)
                    i += 2
                    continue

            new_children.append(current_child)
            i += 1

        working_node.children = tuple(new_children)

    recurse(root)

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
    for _, row in tqdm(df_reference.iterrows(), total=len(df_reference), desc="Hierarchical tree construction: Cleaning data", unit="code"):
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