"""Genera los datos del panel (b) de las figuras de lectura intuitiva (Cap. 5).

Para una porcion de 36 x 36 celdas del mapa de 2011 escribe, por variable, el
color de cada celda segun su peso WoE, y aparte las celdas que se urbanizaron
entre 2011 y 2016. El macro \\campoWoE de campo_woe.tex los dibuja.

El pickle del WoE solo se lee.

Uso:
    python tools/generar_campo_woe.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "tools"))
from variables_rapidas import variables_espaciales  # noqa: E402

PKL = RAIZ / "data/processed/woe_pooled_1984_2010.pkl"
MAPAS = RAIZ / "data/processed/standardized_maps"
DESTINO = RAIZ / "report/tesis/tesis_indice_nuevo/caps_larraga/cap06_resultados_analisis/datos"

FILA0, COL0, N = 312, 2590, 36
LIMITE = 2.0
VARIABLES = (
    "distance_urban",
    "neighbor_density_3x3",
    "local_fragmentation",
    "urban_gradient",
    "nearest_cluster_size",
)

FAVORECE = np.array([27, 122, 72])     # Acerto
DESALIENTA = np.array([176, 48, 42])   # Transicion
NEUTRO = np.array([250, 250, 247])
URBANA = np.array([27, 42, 74])        # Urbano


def color(peso: float) -> np.ndarray:
    t = min(abs(peso), LIMITE) / LIMITE
    extremo = FAVORECE if peso > 0 else DESALIENTA
    return NEUTRO + t * (extremo - NEUTRO)


def main() -> None:
    with open(PKL, "rb") as f:
        calc = pickle.load(f)["woe_calculator"]
    m11 = np.load(MAPAS / "2011.npy")
    m16 = np.load(MAPAS / "2016.npy")
    variables = variables_espaciales(m11)
    ventana = (slice(FILA0, FILA0 + N), slice(COL0, COL0 + N))
    urbana = m11[ventana] == 1
    transita = (m11[ventana] == 0) & (m16[ventana] == 1)

    DESTINO.mkdir(parents=True, exist_ok=True)
    for nombre in VARIABLES:
        pesos = calc.apply_woe_transform(variables[nombre].flatten(), nombre)
        pesos = pesos.reshape(m11.shape)[ventana]
        filas = ["x y c"]
        for y in range(N):
            for x in range(N):
                c = URBANA if urbana[y, x] else color(pesos[y, x])
                r, g, b = (int(round(v)) for v in c)
                filas.append(f"{x} {y} rgb255={r},{g},{b}")
        (DESTINO / f"campo_{nombre}.dat").write_text("\n".join(filas) + "\n")

    ys, xs = np.nonzero(transita)
    filas = ["x y"] + [f"{x} {y}" for y, x in zip(ys, xs)]
    (DESTINO / "transiciones_2011_2016.dat").write_text("\n".join(filas) + "\n")
    print(f"{len(VARIABLES)} campos y {len(xs)} transiciones en {DESTINO.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
