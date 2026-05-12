import re
import pandas as pd
import plotly.express as px

from IPython.display import HTML, display

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