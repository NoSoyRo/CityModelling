#!/usr/bin/env python3
"""
Diagrama horizontal (tesis Cap. 2): tensor RGB -> tensor de caracteristicas -> mapa binario.
Salida PNG fija en caps_larraga/figures/pipeline_rgb_features_binary_cap02.png.

Solo usa matplotlib + numpy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

# Salida FIJA junto al resto de figuras LaTeX (no usar otra carpeta).
_FIGURES_DIR = Path(__file__).resolve().parents[1]  # …/caps_larraga/figures
OUTPUT = _FIGURES_DIR / "pipeline_rgb_features_binary_cap02.png"

# Proporcion ancho/alto igual a rejilla de trabajo (conceptual, no escala territorial).
ASPECT = 3024 / 1792


NAVY = "#203864"
BLUE = "#365f91"
TEXT = "#1f2d3d"
MUTED = "#53606f"
LIGHT_BORDER = "#cfd8e3"


def add_label_box(ax, x, y, title, subtitle, width=2.65):
    box = FancyBboxPatch(
        (x - width / 2, y - 0.34),
        width,
        0.68,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        linewidth=0.9,
        edgecolor="#d4dbe6",
        facecolor="#f7f9fc",
        zorder=10,
    )
    ax.add_patch(box)
    ax.text(x, y + 0.11, title, ha="center", va="center", fontsize=10.2, fontweight="bold", color=NAVY, zorder=11)
    ax.text(x, y - 0.15, subtitle, ha="center", va="center", fontsize=8.6, color=TEXT, zorder=11)


def arrow_between(ax, x0, x1, y, title, subtitle):
    arrow = FancyArrowPatch(
        (x0, y),
        (x1, y),
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=2.3,
        color=BLUE,
        shrinkA=5,
        shrinkB=5,
        zorder=4,
    )
    ax.add_patch(arrow)
    add_label_box(ax, (x0 + x1) / 2, y + 1.05, title, subtitle, width=2.85)


def draw_tensor(ax, x, y, w, h, depth, colors, edge=NAVY, layer_dx=0.11, layer_dy=0.08, label=None):
    """Dibuja un volumen laminar con capas desplazadas."""
    for k in range(depth - 1, -1, -1):
        dx = layer_dx * k
        dy = layer_dy * k
        rect = FancyBboxPatch(
            (x + dx, y + dy),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.045",
            linewidth=1.35,
            edgecolor=edge,
            facecolor=colors[k % len(colors)],
            alpha=0.94,
            zorder=2 + k,
        )
        ax.add_patch(rect)
    if label:
        ax.text(x + w + depth * layer_dx + 0.15, y + h + depth * layer_dy - 0.05, label, ha="left", va="top", fontsize=9, color=MUTED)


def draw_feature_tensor(ax, x, y, w, h, n_layers=18):
    """Tensor de caracteristicas: muchas capas azules, separadas pero no encimadas con texto."""
    colors = plt.cm.Blues(np.linspace(0.32, 0.86, n_layers))[:, :3]
    draw_tensor(ax, x, y, w, h, n_layers, list(colors), edge="#1b4d7a", layer_dx=0.055, layer_dy=0.045)


def make_binary_pattern(rows=18, cols=28):
    """Patron urbano/no urbano legible: nucleos, corredores y crecimiento disperso."""
    yy, xx = np.mgrid[0:rows, 0:cols]
    pattern = np.zeros((rows, cols), dtype=int)

    def ellipse(cx, cy, rx, ry):
        return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1

    pattern |= ellipse(10.5, 9.5, 5.7, 3.8)
    pattern |= ellipse(16.5, 8.0, 4.0, 3.0)
    pattern |= ellipse(13.8, 12.8, 4.8, 2.5)
    pattern[8:11, 4:24] = 1
    pattern[5:15, 13:16] = 1

    # Puntos dispersos/periurbanos para que se entienda el mapa binario.
    points = [(3, 3), (5, 15), (7, 4), (9, 16), (21, 3), (23, 6), (24, 14), (6, 13), (18, 15), (25, 10)]
    for px, py in points:
        pattern[max(py - 1, 0) : min(py + 2, rows), max(px - 1, 0) : min(px + 2, cols)] = 1
    return pattern


def draw_binary_map(ax, x, y, w, h):
    data = make_binary_pattern()
    rows, cols = data.shape
    cw, ch = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            face = "#203864" if data[r, c] else "#edf2f7"
            ax.add_patch(
                Rectangle(
                    (x + c * cw, y + (rows - 1 - r) * ch),
                    cw,
                    ch,
                    facecolor=face,
                    edgecolor="#ffffff",
                    linewidth=0.28,
                    zorder=3,
                )
            )
    frame = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.045",
        linewidth=2.2,
        edgecolor=NAVY,
        facecolor=(0, 0, 0, 0),
        zorder=6,
    )
    ax.add_patch(frame)

    # Mini leyenda clara, separada del raster.
    lx, ly = x, y + h + 0.28
    ax.add_patch(Rectangle((lx, ly), 0.18, 0.18, facecolor="#edf2f7", edgecolor=LIGHT_BORDER, linewidth=0.6))
    ax.text(lx + 0.25, ly + 0.09, "0 no urbano", ha="left", va="center", fontsize=8.4, color=MUTED)
    ax.add_patch(Rectangle((lx + 1.35, ly), 0.18, 0.18, facecolor=NAVY, edgecolor=NAVY, linewidth=0.6))
    ax.text(lx + 1.60, ly + 0.09, "1 urbano", ha="left", va="center", fontsize=8.4, color=MUTED)


def draw_panel_header(ax, x, y, title, subtitle, width):
    ax.text(x + width / 2, y, title, ha="center", va="top", fontsize=12.2, fontweight="bold", color=TEXT)
    ax.text(x + width / 2, y - 0.34, subtitle, ha="center", va="top", fontsize=9.8, color=MUTED)


def main():
    fig_w, fig_h = 17.4, 7.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_xlim(0, 18.2)
    ax.set_ylim(0, 7.4)

    fig.text(
        0.5,
        0.955,
        "Flujo conceptual del preprocesamiento: RGB -> espacio de caracteristicas -> mapa binario",
        ha="center",
        fontsize=15,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.5,
        0.918,
        "Las dimensiones espaciales (1792 x 3024) se conservan; cambia la profundidad de informacion por celda.",
        ha="center",
        fontsize=10.8,
        color=MUTED,
    )

    # Medidas comunes. La forma respeta la proporcion 3024/1792.
    w, h = 2.85, 1.68
    base_y = 3.2
    x1, x2, x3 = 0.85, 7.35, 14.20

    draw_panel_header(ax, x1, 1.32, "Paso 1 - tensor RGB", r"$1792 \times 3024 \times 3$", w)
    draw_panel_header(ax, x2, 1.32, "Paso 2 - tensor de rasgos", r"$1792 \times 3024 \times d$  ($d\approx15$--$23$)", w + 0.85)
    draw_panel_header(ax, x3, 1.32, "Paso 3 - mapa binario", r"$M_t\in\{0,1\}^{1792\times3024}$", w)

    draw_tensor(
        ax,
        x1,
        base_y,
        w,
        h,
        3,
        colors=["#e85d5d", "#61b86b", "#5d80d7"],
        edge=NAVY,
        layer_dx=0.16,
        layer_dy=0.12,
        label="3 canales\nR, G, B",
    )

    draw_feature_tensor(ax, x2, base_y, w, h, n_layers=18)
    draw_binary_map(ax, x3, base_y, w, h)

    arrow_between(
        ax,
        x1 + w + 0.75,
        x2 - 0.62,
        base_y + h * 0.55,
        "Transformacion A",
        "Extraccion de rasgos\nLBP + indices + derivados",
    )
    arrow_between(
        ax,
        x2 + w + 1.35,
        x3 - 0.70,
        base_y + h * 0.55,
        "Transformacion B",
        "K-Means + SVM/PCA\ncolapso a {0,1}",
    )

    # Notas breves separadas de las figuras para evitar encimamientos.
    notes = [
        (x1 + w / 2, 2.55, "Entrada anual homogénea\ndesde captura PNG"),
        (x2 + w / 2 + 0.35, 2.55, "Aumenta la profundidad:\nde 3 canales a decenas de rasgos"),
        (x3 + w / 2, 2.55, "Conserva filas y columnas;\ncada celda queda etiquetada"),
    ]
    for x, y, txt in notes:
        ax.text(x, y, txt, ha="center", va="top", fontsize=9.2, color=MUTED)

    fig.subplots_adjust(left=0.02, right=0.985, top=0.87, bottom=0.08)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Guardado: {OUTPUT}")


if __name__ == "__main__":
    main()
