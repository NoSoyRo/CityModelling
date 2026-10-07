"""Evalua un par (theta, alpha) en las cinco ventanas quinquenales publicadas.

Produce la tanda completa de metricas que reporta el Capitulo 5: matriz de
confusion, metricas pixel a pixel, Figure of Merit con sus metricas de cambio,
la descomposicion de Pontius y las estadisticas de crecimiento. Las
definiciones son las de validate_quinquenal_2011_2016.py, verificadas linea por
linea; la unica diferencia es que aqui el sorteo lleva semilla.

Uso:

    .venv/bin/python analysis/calibracion_pliegue_interno/evaluacion_final.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibracion_pliegue_interno import cargar_woe, simular  # noqa: E402

MAPAS = RAIZ / "data/processed/standardized_maps"
SALIDA = Path(__file__).resolve().parent / "resultados"

VENTANAS = [(2011, 2016), (2012, 2017), (2013, 2018), (2014, 2019), (2015, 2020)]
SEMILLA = 20260918

PARES = {
    "publicado": (0.75, 0.50),
    "pliegue_interno": (0.85, 0.60),
}


def metricas_completas(inicial: np.ndarray, observado: np.ndarray,
                       predicho: np.ndarray) -> dict:
    """Replica la tanda completa de metricas del script publicado."""
    obs, pred = observado.ravel(), predicho.ravel()
    TP = int(np.sum((pred == 1) & (obs == 1)))
    TN = int(np.sum((pred == 0) & (obs == 0)))
    FP = int(np.sum((pred == 1) & (obs == 0)))
    FN = int(np.sum((pred == 0) & (obs == 1)))
    N = TP + TN + FP + FN

    exactitud = (TP + TN) / N
    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = TP / (TP + FP + FN) if TP + FP + FN else 0.0
    pe = ((TP + FP) * (TP + FN) + (TN + FN) * (TN + FP)) / N**2
    kappa = (exactitud - pe) / (1 - pe) if 1 - pe else 0.0

    trans_obs = (observado == 1) & (inicial == 0)
    trans_pred = (predicho == 1) & (inicial == 0)
    B = int(np.sum(trans_obs & trans_pred))
    C = int(np.sum(trans_obs & ~trans_pred))
    A = int(np.sum(~trans_obs & trans_pred))
    fom = B / (A + B + C) if A + B + C else 0.0
    cambio_precision = B / (A + B) if A + B else 0.0
    cambio_recall = B / (B + C) if B + C else 0.0
    cambio_f1 = 2 * B / (2 * B + A + C) if 2 * B + A + C else 0.0

    desacuerdo_cantidad = abs(FP - FN)
    desacuerdo_asignacion = 2 * min(FP, FN)
    desacuerdo_total = desacuerdo_cantidad + desacuerdo_asignacion

    urbanas_inicio = int(inicial.sum())
    urbanas_obs = int(observado.sum())
    urbanas_pred = int(predicho.sum())
    crecimiento_obs = urbanas_obs - urbanas_inicio
    crecimiento_pred = urbanas_pred - urbanas_inicio

    return {
        "confusion": {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
        "pixel": {
            "exactitud": exactitud, "iou": iou, "precision": precision,
            "recall": recall, "f1": f1, "kappa": kappa,
        },
        "fom": {
            "fom": fom, "A_falsas_alarmas": A, "B_aciertos": B, "C_omisiones": C,
            "cambio_precision": cambio_precision, "cambio_recall": cambio_recall,
            "cambio_f1": cambio_f1,
        },
        "pontius": {
            "cantidad": desacuerdo_cantidad,
            "asignacion": desacuerdo_asignacion,
            "fraccion_cantidad": desacuerdo_cantidad / desacuerdo_total
            if desacuerdo_total else 0.0,
        },
        "crecimiento": {
            "urbanas_inicio": urbanas_inicio,
            "urbanas_observadas": urbanas_obs,
            "urbanas_predichas": urbanas_pred,
            "crecimiento_observado": crecimiento_obs,
            "crecimiento_predicho": crecimiento_pred,
            "tasa_observada_pct": 100 * crecimiento_obs / urbanas_inicio,
            "tasa_predicha_pct": 100 * crecimiento_pred / urbanas_inicio,
        },
    }


def evalua_par(theta: float, alpha: float, calc, pesos) -> dict:
    filas = {}
    for anio0, anio1 in VENTANAS:
        inicial = np.load(MAPAS / f"{anio0}.npy")
        observado = np.load(MAPAS / f"{anio1}.npy")
        predicho = simular(inicial, anio1 - anio0, theta, alpha, SEMILLA, calc, pesos)
        filas[f"{anio0}-{anio1}"] = metricas_completas(inicial, observado, predicho)

    def media(ruta: tuple[str, str]) -> float:
        a, b = ruta
        return sum(f[a][b] for f in filas.values()) / len(filas)

    promedio = {
        "exactitud": media(("pixel", "exactitud")),
        "iou": media(("pixel", "iou")),
        "precision": media(("pixel", "precision")),
        "recall": media(("pixel", "recall")),
        "f1": media(("pixel", "f1")),
        "kappa": media(("pixel", "kappa")),
        "fom": media(("fom", "fom")),
        "fraccion_cantidad_media": media(("pontius", "fraccion_cantidad")),
    }
    # La tesis reporta la fraccion de cantidad como razon de sumas, no como
    # promedio de razones. Se dejan las dos para que el texto pueda citar la
    # que corresponda sin recalcular.
    suma_cant = sum(f["pontius"]["cantidad"] for f in filas.values())
    suma_asig = sum(f["pontius"]["asignacion"] for f in filas.values())
    promedio["fraccion_cantidad_razon_de_sumas"] = suma_cant / (suma_cant + suma_asig)

    foms = sorted(f["fom"]["fom"] for f in filas.values())
    promedio["fom_min"], promedio["fom_max"] = foms[0], foms[-1]
    promedio["fom_factor"] = foms[-1] / foms[0]
    kappas = sorted(f["pixel"]["kappa"] for f in filas.values())
    promedio["kappa_min"], promedio["kappa_max"] = kappas[0], kappas[-1]

    return {"theta": theta, "alpha": alpha, "ventanas": filas, "promedio": promedio}


def main() -> None:
    calc, pesos = cargar_woe()
    salida = {"semilla": SEMILLA, "pares": {}}

    for nombre, (theta, alpha) in PARES.items():
        print(f"\n=== {nombre}: theta {theta}, alpha {alpha} ===")
        r = evalua_par(theta, alpha, calc, pesos)
        salida["pares"][nombre] = r
        print(f"{'ventana':<12}{'FoM':>9}{'Kappa':>9}{'IoU':>9}"
              f"{'Exact':>9}{'Prec':>9}{'Recall':>9}{'F1':>9}{'%cant':>8}")
        for v, f in r["ventanas"].items():
            print(f"{v:<12}{f['fom']['fom']:>9.4f}{f['pixel']['kappa']:>9.4f}"
                  f"{f['pixel']['iou']:>9.4f}{f['pixel']['exactitud']:>9.4f}"
                  f"{f['pixel']['precision']:>9.4f}{f['pixel']['recall']:>9.4f}"
                  f"{f['pixel']['f1']:>9.4f}"
                  f"{100 * f['pontius']['fraccion_cantidad']:>7.1f}%")
        p = r["promedio"]
        print(f"{'PROMEDIO':<12}{p['fom']:>9.4f}{p['kappa']:>9.4f}{p['iou']:>9.4f}"
              f"{p['exactitud']:>9.4f}{p['precision']:>9.4f}{p['recall']:>9.4f}"
              f"{p['f1']:>9.4f}{100 * p['fraccion_cantidad_razon_de_sumas']:>7.1f}%")
        print(f"  FoM de {p['fom_min']:.4f} a {p['fom_max']:.4f}, "
              f"factor {p['fom_factor']:.2f}")

    SALIDA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA / "evaluacion_final_cinco_ventanas.json"
    destino.write_text(json.dumps(salida, indent=1))
    print(f"\nguardado en {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
