"""Mapa de error de la ventana 2015-2020 con el par elegido por el pliegue interno.

Clasifica cada celda con la misma convencion del FoM de `metricas`: entre las
celdas no urbanas en 2015, aciertos (B), falsas alarmas (A), omisiones (C) y
permanencia no urbana; las urbanas en 2015 se pintan aparte. Antes de dibujar
comprueba que A, B y C coincidan con evaluacion_final_cinco_ventanas.json.

Uso:

    .venv/bin/python analysis/calibracion_pliegue_interno/mapa_error.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tools"))
sys.path.insert(0, str(AQUI))

from calibracion_pliegue_interno import cargar_woe, simular  # noqa: E402

MAPAS = RAIZ / "data/processed/standardized_maps"
SALIDA = RAIZ / "report/tesis/tesis_indice_nuevo/caps_larraga/figures/F12_mapa_error_2015_2020.png"
RESULTADOS = AQUI / "resultados/evaluacion_final_cinco_ventanas.json"

ANIO0, ANIO1 = 2015, 2020
THETA, ALPHA = 0.85, 0.60
SEMILLA = 20260918
PIXEL_M = 26.7
GIRO_NORTE_GRADOS = 22.0
LADO_ZOOM = 560

COLORES = {
    0: (0.925, 0.933, 0.945),  # no urbana en ambos
    1: (0.502, 0.525, 0.549),  # urbana en 2015
    2: (0.106, 0.478, 0.282),  # B, acierto
    3: (0.769, 0.518, 0.071),  # A, falsa alarma
    4: (0.600, 0.157, 0.157),  # C, omision
}


def clasificar(inicial, observado, predicho):
    trans_obs = (observado == 1) & (inicial == 0)
    trans_pred = (predicho == 1) & (inicial == 0)
    clase = np.zeros(inicial.shape, dtype=np.uint8)
    clase[inicial == 1] = 1
    clase[trans_obs & trans_pred] = 2
    clase[~trans_obs & trans_pred] = 3
    clase[trans_obs & ~trans_pred] = 4
    return clase


def comprobar(clase):
    ref = json.loads(RESULTADOS.read_text())
    fom = ref["pares"]["pliegue_interno"]["ventanas"][f"{ANIO0}-{ANIO1}"]["fom"]
    conteos = {"B_aciertos": int((clase == 2).sum()),
               "A_falsas_alarmas": int((clase == 3).sum()),
               "C_omisiones": int((clase == 4).sum())}
    for clave, valor in conteos.items():
        if valor != fom[clave]:
            raise SystemExit(f"{clave}: {valor} no coincide con el JSON ({fom[clave]})")
    return conteos, fom["fom"]


def flecha_norte(ax, x, y, largo):
    giro = np.deg2rad(GIRO_NORTE_GRADOS)
    dx, dy = largo * np.sin(giro), -largo * np.cos(giro)
    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.1))
    ax.text(x + 1.25 * dx, y + 1.25 * dy, "N", ha="center", va="center", fontsize=8)


def barra_escala(ax, x, y, km):
    largo = km * 1000 / PIXEL_M
    ax.add_patch(Rectangle((x, y), largo, largo * 0.06, color="black"))
    ax.text(x + largo / 2, y - largo * 0.12, f"{km} km", ha="center", va="baseline", fontsize=7)


def main() -> None:
    calc, pesos = cargar_woe()
    inicial = np.load(MAPAS / f"{ANIO0}.npy")
    observado = np.load(MAPAS / f"{ANIO1}.npy")
    predicho = simular(inicial, ANIO1 - ANIO0, THETA, ALPHA, SEMILLA, calc, pesos)

    clase = clasificar(inicial, observado, predicho)
    conteos, fom = comprobar(clase)
    print(f"conteos verificados contra el JSON: {conteos}, FoM {fom:.4f}")

    cmap = ListedColormap([COLORES[k] for k in range(5)])
    alto, ancho = clase.shape
    f0, c0 = alto // 2 - LADO_ZOOM // 2, ancho // 2 - LADO_ZOOM // 2

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.95),
                                   gridspec_kw={"width_ratios": [ancho / alto, 1]})
    ax1.imshow(clase, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax1.add_patch(Rectangle((c0, f0), LADO_ZOOM, LADO_ZOOM, fill=False, ec="black", lw=0.9))
    ax1.set_title("Ventana completa (80,8 × 47,9 km)", fontsize=8)
    flecha_norte(ax1, ancho * 0.94, alto * 0.30, alto * 0.13)
    barra_escala(ax1, ancho * 0.04, alto * 0.90, 10)

    ax2.imshow(clase[f0:f0 + LADO_ZOOM, c0:c0 + LADO_ZOOM], cmap=cmap, vmin=0, vmax=4,
               interpolation="nearest")
    ax2.set_title(f"Detalle central ({LADO_ZOOM * PIXEL_M / 1000:.0f} × "
                  f"{LADO_ZOOM * PIXEL_M / 1000:.0f} km)", fontsize=8)
    barra_escala(ax2, LADO_ZOOM * 0.06, LADO_ZOOM * 0.90, 3)

    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])

    def miles(n):
        return f"{n:,}".replace(",", "\u2009")

    leyenda = [
        Patch(color=COLORES[2], label=f"B, acierto ({miles(conteos['B_aciertos'])})"),
        Patch(color=COLORES[3], label=f"A, falsa alarma ({miles(conteos['A_falsas_alarmas'])})"),
        Patch(color=COLORES[4], label=f"C, omisión ({miles(conteos['C_omisiones'])})"),
        Patch(color=COLORES[1], label="Urbana en 2015"),
        Patch(facecolor=COLORES[0], edgecolor="0.6", lw=0.4, label="No urbana en 2015 y 2020"),
    ]
    fig.legend(handles=leyenda, loc="lower center", ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.17, wspace=0.04)
    fig.savefig(SALIDA, dpi=300)
    print(f"escrito: {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
