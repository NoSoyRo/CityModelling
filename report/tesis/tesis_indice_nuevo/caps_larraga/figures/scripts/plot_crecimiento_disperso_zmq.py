#!/usr/bin/env python3
"""
Mapa sintético de expansión dispersa — ZMQ (tesis WoE–AC).

Fuentes (repositorio):
  data/processed/standardized_maps/1984.npy
  data/processed/standardized_maps/2020.npy

Salida:
  caps_larraga/figures/crecimiento_disperso_zmq_1984_2020.png

Solo usa numpy + matplotlib (sin scipy).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# .../CityModelling/report/tesis/tesis_indice_nuevo/caps_larraga/figures/scripts/file.py → parents[6]
REPO_ROOT = Path(__file__).resolve().parents[6]
FIGURES_DIR = Path(__file__).resolve().parents[1]
MAPS_DIR = REPO_ROOT / "data" / "processed" / "standardized_maps"
OUTPUT = FIGURES_DIR / "crecimiento_disperso_zmq_1984_2020.png"


def tile_aggregate(mask: np.ndarray, factor: int) -> np.ndarray:
    """Promedio binario por bloques factor×factor (sin pad; recorta borde)."""
    h, w = mask.shape
    h2, w2 = h // factor * factor, w // factor * factor
    m = mask[:h2, :w2].astype(np.float64)
    nh, nw = h2 // factor, w2 // factor
    blocked = m.reshape(nh, factor, nw, factor)
    return blocked.mean(axis=(1, 3))


def main():
    if not MAPS_DIR.is_dir():
        raise FileNotFoundError(f"No existe MAPS_DIR: {MAPS_DIR}")

    prev = Path(MAPS_DIR / "1984.npy")
    curr = Path(MAPS_DIR / "2020.npy")
    m0 = np.load(prev)
    m1 = np.load(curr)
    if m0.shape != m1.shape:
        raise ValueError(f"Shapes distintos: {m0.shape} vs {m1.shape}")

    urban0 = m0 == 1
    urban1 = m1 == 1
    new_urban = urban1 & ~urban0

    n_new = int(new_urban.sum())
    print(f"Tiles map shape {m0.shape}; nuevas urbanas (2020∩¬1984): {n_new:_}")

    # Vista principal: dispersión del tejido nuevo sobre contexto liviano
    h, w = m0.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    # fondo tierra muy claro
    rgb[:, :] = np.array([0.94, 0.93, 0.91], dtype=np.float32)
    # urbe estable 1984 (azul apagado)
    rgb[urban0] = np.array([0.25, 0.42, 0.68], dtype=np.float32)
    # expansión 1984→2020 (naranja intenso, encima donde no pisaba estable)
    rgb[new_urban] = np.array([0.93, 0.38, 0.12], dtype=np.float32)

    # Mapa auxiliar agregado: proporción de celdas nuevas por bloque
    FACTOR = 32
    frac_new = tile_aggregate(new_urban, FACTOR)

    fig = plt.figure(figsize=(13.6, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.72], wspace=0.05)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(rgb, origin="upper")
    ax0.set_axis_off()
    ax0.set_title(
        "Urbano estable (1984) y expansión neta hasta 2020\n"
        "(mapas binarios derivados del pipeline de clasificación)",
        fontsize=11,
        pad=8,
    )

    ax1 = fig.add_subplot(gs[0, 1])
    him = ax1.imshow(frac_new, cmap="YlOrRd", origin="upper", vmin=0, vmax=max(frac_new.max(), 1e-6))
    ax1.set_axis_off()
    ax1.set_title(
        "Intensidad de expansión (bloques {:.0f}×{:.0f} celdas)\n".format(FACTOR, FACTOR),
        fontsize=11,
        pad=8,
    )
    cb = plt.colorbar(him, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label(
        f"Fracción de celdas nuevas en bloques {FACTOR}×{FACTOR}",
        fontsize=9,
    )

    # Leyenda discreta tipo patch (matplotlib)
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=[0.25, 0.42, 0.68], edgecolor="k", linewidth=0.3, label="Urbano existente en 1984"),
        Patch(facecolor=[0.93, 0.38, 0.12], edgecolor="k", linewidth=0.3, label="Nuevo urbano desde 1984 hasta 2020"),
        Patch(facecolor=[0.94, 0.93, 0.91], edgecolor="gray", linewidth=0.3, label="No urbano (2020)"),
    ]
    ax0.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.92,
    )

    footer = (
        "Fuente: CityModelling — data/processed/standardized_maps/1984.npy, "
        "2020.npy · ZMQ análisis temporal 1984–2020 (tesis MSc)"
    )
    fig.text(0.5, 0.02, footer, ha="center", fontsize=8, color="0.35")

    fig.subplots_adjust(bottom=0.08, left=0.02, right=0.97, top=0.90)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Guardado: {OUTPUT}")


if __name__ == "__main__":
    main()
