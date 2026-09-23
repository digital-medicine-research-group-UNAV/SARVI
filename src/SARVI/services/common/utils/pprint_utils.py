import re
import string
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from numpy.linalg import LinAlgError
from matplotlib.colors import to_rgb
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
    A function that plots a histogram using the specified embedding similarity data.
    It also provides a visual representation of how embedding similarity is distributed across CIE10 codes based on the specified column.

    Parameters
    ----------
        `df`: pd.DataFrame
            - Input dataframe containing the records to process.
        `column_name_emb_sim`: str
            - Column containing embedding similarity values.
        `column_name_cie10_sim`: str
            - Column containing ICD-10 similarity values.
        `title`: str
            - Plot title.
        `umbral`: float
            - Threshold drawn on the plot.
        `y_max`: float
            - Maximum y-axis value.
        `start`: float
            - Start value or position.
        `end`: float
            - End value or position.
        `size`: float
            - Step or marker size.

    Returns
    -------
        `None`
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

    # Patrón para capturar número y versión
    """
    Function that plots the trend of the percentage of “True” values in columns with names such as `tree_1`, `tree_2`, ..., `tree_1_V2`, etc.
    The function automatically identifies the columns, sorts them in logical order (1..5, then 1_V2..5_V2, etc.), and displays a line chart with markers and value labels above each point.

    Parameters
    ----------
        `df`: pd.DataFrame
            - Input dataframe containing the records to process.
        `title`: str
            - Plot title.

    Returns
    -------
        `None`
    """
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

def lighten_color(color, amount=0.70):
    """
    Returns a lighter version of a color.
    amount=0.0 -> original co,or
    amount=1.0 -> white

    Parameters
    ----------
        `color`: Any
            - Color value to adjust.
        `amount`: Any
            - Lightening factor applied to the color.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    rgb = np.asarray(to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def draw_kde_region(ax, points, color, mass=0.80, gridsize=150, lighten_amount=0.70, fill_alpha=0.65, linewidth=1.2, kde_padding=0.50):
    points = np.asarray(points)

    if len(points) < 8:
        return None

    if np.unique(points, axis=0).shape[0] < 3:
        return None

    try:
        kde = gaussian_kde(points.T, bw_method="scott")
    except (LinAlgError, ValueError):
        return None

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    # Este padding solo se usa para calcular correctamente la KDE.
    # No se utilizará directamente como límite del gráfico.
    x_padding = kde_padding * x_range
    y_padding = kde_padding * y_range

    xx, yy = np.meshgrid(np.linspace(x_min - x_padding, x_max + x_padding, gridsize), np.linspace(y_min - y_padding, y_max + y_padding, gridsize))

    positions = np.vstack([xx.ravel(), yy.ravel()])

    density = kde(positions).reshape(xx.shape)

    sorted_density = np.sort(density.ravel())[::-1]
    cumulative_mass = np.cumsum(sorted_density)
    cumulative_mass /= cumulative_mass[-1]

    threshold_index = np.searchsorted(cumulative_mass, mass)

    threshold_index = min(threshold_index, len(sorted_density) - 1)

    threshold = sorted_density[threshold_index]

    if not np.isfinite(threshold):
        return None

    if threshold >= density.max():
        return None

    light_color = lighten_color(color, amount=lighten_amount)
    ax.contourf(xx, yy, density, levels=[threshold, density.max()], colors=[light_color], alpha=fill_alpha, zorder=1)
    contour = ax.contour(xx, yy, density, levels=[threshold], colors=[color], linewidths=linewidth, alpha=0.75, zorder=2)

    # Obtener únicamente los límites del contorno real,
    # no los límites de toda la cuadrícula KDE.
    vertices = []

    for path in contour.get_paths():
        if len(path.vertices) > 0:
            vertices.append(path.vertices)

    if not vertices:
        return None

    vertices = np.vstack(vertices)

    return (vertices[:, 0].min(), vertices[:, 0].max(), vertices[:, 1].min(), vertices[:, 1].max())

def plot_arcface_results(supertitle, labels_defs, labels_preds, classes, unique_labels, emb_defs, emb_preds, label_to_color, density_mass=0.90, lighten_amount=0.95, fill_alpha=0.8, linewidth=0.5, kde_padding=0.5, save=False):
    """
    Plots embedding and prediction distributions for ArcFace results.

    Parameters
    ----------
        `supertitle`: Any
            - Argument controlling supertitle.
        `labels_defs`: Any
            - Entity or relation labels used by the model.
        `labels_preds`: Any
            - Entity or relation labels used by the model.
        `classes`: Any
            - Argument controlling classes.
        `unique_labels`: Any
            - Entity or relation labels used by the model.
        `emb_defs`: Any
            - Argument controlling emb defs.
        `emb_preds`: Any
            - Argument controlling emb preds.
        `label_to_color`: Any
            - Entity or relation labels used by the model.
        `density_mass`: Any
            - Argument controlling density mass.
        `lighten_amount`: Any
            - Argument controlling lighten amount.
        `fill_alpha`: Any
            - Argument controlling fill alpha.
        `linewidth`: Any
            - Identifier values used to link or index records.
        `kde_padding`: Any
            - Argument controlling kde padding.
        `save`: Any
            - Argument controlling save.

    Returns
    -------
        `Any`
            - Derived value produced by the operation.
    """
    labels_defs = np.asarray(labels_defs)
    labels_preds = np.asarray(labels_preds)

    panel_specs = [
        ("Correct", 1),
        ("Incorrect", 0),
    ]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    page_width = 7.16
    target_figure_height = page_width / 2

    panel_label_space = 0.15
    initial_legend_space = 0.35

    # Keep the original initial size while plotting and measuring
    # the legend. The width remains exactly page_width.
    initial_figure_height = (
        target_figure_height
        + panel_label_space
        + initial_legend_space
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(page_width, initial_figure_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    all_bounds = []

    counts = {
        lab: {
            "Correct": 0,
            "Incorrect": 0,
        }
        for lab in unique_labels
    }

    for ax, (panel_title, correctness_value) in zip(
        axes.flat,
        panel_specs,
    ):
        for lab in tqdm(unique_labels, desc=panel_title):
            mask_defs = labels_defs == lab

            mask_preds = (
                (labels_preds == lab)
                & (classes == correctness_value)
            )

            points_defs = emb_defs[mask_defs]
            points_preds = emb_preds[mask_preds]

            color = label_to_color[lab]
            counts[lab][panel_title] = len(points_preds)

            bounds = draw_kde_region(
                ax=ax,
                points=points_defs,
                color=color,
                mass=density_mass,
                lighten_amount=lighten_amount,
                fill_alpha=fill_alpha,
                linewidth=linewidth,
                kde_padding=kde_padding,
            )

            if bounds is not None:
                all_bounds.append(bounds)

            if len(points_preds) > 0:
                all_bounds.append((
                    points_preds[:, 0].min(),
                    points_preds[:, 0].max(),
                    points_preds[:, 1].min(),
                    points_preds[:, 1].max(),
                ))

            ax.scatter(
                points_preds[:, 0],
                points_preds[:, 1],
                color=color,
                s=35,
                alpha=0.7,
                edgecolors="none",
                zorder=3,
            )

        ax.set_title(panel_title)
        ax.set_xlabel("UMAP 1")

        if panel_title == "Correct":
            ax.set_ylabel("UMAP 2")

    if all_bounds:
        x_min = min(bounds[0] for bounds in all_bounds)
        x_max = max(bounds[1] for bounds in all_bounds)
        y_min = min(bounds[2] for bounds in all_bounds)
        y_max = max(bounds[3] for bounds in all_bounds)

        x_range = max(x_max - x_min, 1e-6)
        y_range = max(y_max - y_min, 1e-6)

        x_margin = 0.025 * x_range
        y_margin = 0.025 * y_range

        for ax in axes.flat:
            ax.set_xlim(
                x_min - x_margin,
                x_max + x_margin,
            )

            ax.set_ylim(
                y_min - y_margin,
                y_max + y_margin,
            )

    legend_handles = []
    legend_labels = []

    for lab in unique_labels:
        color = label_to_color[lab]

        n_correct = counts[lab]["Correct"]
        n_incorrect = counts[lab]["Incorrect"]

        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=6,
            )
        )

        legend_labels.append(
            f"{lab} (C={n_correct}, I={n_incorrect})"
        )

    # Fixed final figure size.
    figure_height = page_width / 2
    fig.set_size_inches(page_width, figure_height, forward=True)

    # Fixed axes placement matching the reference PDF.
    # Do not use tight_layout here.
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.31,
        wspace=0.16,
    )

    # Panel labels: keep them clearly above the legend.
    panel_label_y = 0.185

    for i, ax in enumerate(axes.flat):
        position = ax.get_position()

        fig.text(
            (position.x0 + position.x1) / 2,
            panel_label_y,
            f"({string.ascii_lowercase[i]})",
            ha="center",
            va="center",
            fontsize=8,
            fontfamily="Times New Roman",
        )

    # Let the legend use almost the complete figure width.
    # This removes the large empty area on the left and gives each
    # column more horizontal room.
    legend_left = 0.025
    legend_right = 0.99
    legend_bottom = 0.002
    legend_width = legend_right - legend_left

    legend = fig.legend(
        legend_handles,
        legend_labels,
        title="Label · C = correct · I = incorrect",
        title_fontproperties={
            "weight": "bold",
            "size": 8,
            "family": "Times New Roman",
        },
        loc="lower left",
        bbox_to_anchor=(
            legend_left,
            legend_bottom,
            legend_width,
            0.13,
        ),
        bbox_transform=fig.transFigure,
        mode="expand",
        ncols=7,
        fontsize=7,
        frameon=False,

        # More separation between legend columns.
        columnspacing=1.25,

        # Slightly more separation between marker and text.
        handletextpad=0.35,

        labelspacing=0.28,
        borderaxespad=0,
    )

    if save:
        fig.savefig(
            f"{supertitle}.pdf",
            format="pdf",
            bbox_inches=None,
            pad_inches=0,
        )

    plt.show()
