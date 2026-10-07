"""Regenera las cinco figuras summary_* con el par elegido por el pliegue interno.

No reimplementa el dibujo: importa visualize_summary de
run_all_quinquenal_validations.py, que es la funcion que produjo las figuras
publicadas. Lo unico que cambia es de donde sale el mapa predicho, que aqui lo
produce el harness vectorizado con semilla en lugar del bucle celda a celda.

Escribe en figures/ del documento, sobre los mismos nombres de archivo, para no
tocar ningun \\includegraphics.

Uso:

    .venv/bin/python analysis/calibracion_pliegue_interno/figuras_resumen.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibracion_pliegue_interno import cargar_woe, simular  # noqa: E402

MAPAS = RAIZ / "data/processed/standardized_maps"
FIGURAS = RAIZ / "report/tesis/tesis_indice_nuevo/caps_larraga/figures"

VENTANAS = [(2011, 2016), (2012, 2017), (2013, 2018), (2014, 2019), (2015, 2020)]
THETA, ALPHA = 0.85, 0.60
SEMILLA = 20260918


def carga_dibujante():
    """Importa visualize_summary sin ejecutar la validacion completa."""
    ruta = RAIZ / "run_all_quinquenal_validations.py"
    spec = importlib.util.spec_from_file_location("validaciones_publicadas", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo.visualize_summary


def main() -> None:
    dibuja = carga_dibujante()
    calc, pesos = cargar_woe()

    for anio0, anio1 in VENTANAS:
        inicial = np.load(MAPAS / f"{anio0}.npy")
        observado = np.load(MAPAS / f"{anio1}.npy")
        predicho = simular(inicial, anio1 - anio0, THETA, ALPHA, SEMILLA, calc, pesos)

        TP = int(np.sum((predicho == 1) & (observado == 1)))
        FP = int(np.sum((predicho == 1) & (observado == 0)))
        FN = int(np.sum((predicho == 0) & (observado == 1)))

        destino = FIGURAS / f"summary_{anio0}_to_{anio1}.png"
        dibuja(inicial, observado, predicho, anio0, anio1, destino, TP, FP, FN)
        print(f"  {destino.name}  TP {TP:,}  FP {FP:,}  FN {FN:,}")


if __name__ == "__main__":
    main()
