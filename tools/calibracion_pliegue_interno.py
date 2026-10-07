#!/usr/bin/env python3
"""Calibra el umbral y el peso de vecindad usando solo datos de 1984 a 2010.

Motivo. En el modelo publicado los pesos WoE y sus Information Value se
estimaron unicamente con 1984-2010, pero el umbral se fijo a mano despues de
observar sobre-prediccion en la primera ventana de evaluacion (2011-2016). Eso
impide describir 2011-2020 como hold-out del modelo completo.

Este script elige el par (theta, alpha) con ventanas internas que terminan en
2010 o antes, sin tocar 2011-2020. Los pesos WoE se usan tal como estan en
data/processed/woe_pooled_1984_2010.pkl: no se reestiman.

Alcance de lo que esto demuestra. Los pesos se ajustaron con el periodo que
incluye las ventanas internas, de modo que esto es seleccion de
hiperparametros sobre el conjunto de entrenamiento, no validacion cruzada
anidada. Lo que se gana es concreto y limitado: 2011-2020 deja de intervenir en
la eleccion de theta y alpha.

Dos diferencias de implementacion respecto al script publicado, ambas
deliberadas:

- Las siete variables se calculan con tools/variables_rapidas.py, que es
  equivalente salvo el desempate de nearest_cluster_size (1,2% del peso). El
  modo "verificar" comprueba que el FoM publicado se reproduce.
- La aleatoriedad se siembra. Sin semilla las corridas no son repetibles, que
  es justo lo que impide comparar configuraciones entre si.

Uso:
    python tools/calibracion_pliegue_interno.py verificar
    python tools/calibracion_pliegue_interno.py calibrar
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tools"))

MAPAS = RAIZ / "data/processed/standardized_maps"
PICKLE = RAIZ / "data/processed/woe_pooled_1984_2010.pkl"
SALIDA = RAIZ / "data/processed/calibracion_pliegue_interno"

# Ventanas de cinco pasos contenidas por completo en el periodo de
# entrenamiento. La mas reciente termina en 2010 para quedar lo mas cerca
# posible del regimen de crecimiento que se evalua despues.
VENTANAS_INTERNAS = [(2000, 2005), (2002, 2007), (2005, 2010)]

# Par publicado, que es la hipotesis a confirmar o refutar.
THETA_PUBLICADO = 0.75
ALPHA_PUBLICADO = 0.50

SEMILLAS = (20260918, 7, 31415)


def cargar_woe():
    """Devuelve el calculador publicado y los pesos IV normalizados."""
    with open(PICKLE, "rb") as f:
        datos = pickle.load(f)
    calc = datos["woe_calculator"]
    iv = {n: float(r.iv_total) for n, r in calc.woe_results.items()}
    total = sum(iv.values())
    return calc, {n: v / total for n, v in iv.items()}


def aptitud(estado: np.ndarray, calc, pesos: dict[str, float]) -> np.ndarray:
    """Sigmoide de la suma de WoE ponderada por IV, sobre el estado actual."""
    from variables_rapidas import variables_espaciales

    variables = variables_espaciales(estado)
    total_woe = np.zeros(estado.shape, dtype=float)
    for nombre, mapa in variables.items():
        woe = calc.apply_woe_transform(mapa.flatten(), nombre)
        total_woe += pesos[nombre] * woe.reshape(estado.shape)
    return 1.0 / (1.0 + np.exp(-total_woe))


def simular(inicial: np.ndarray, pasos: int, theta: float, alpha: float,
            semilla: int, calc, pesos) -> np.ndarray:
    """Corre el AC con la misma regla del script publicado, con semilla."""
    rng = np.random.default_rng(semilla)
    estado = inicial.copy()
    nucleo = np.ones((3, 3))
    nucleo[1, 1] = 0

    for _ in range(pasos):
        prob = aptitud(estado, calc, pesos)
        vecinos = convolve(estado.astype(float), nucleo, mode="constant", cval=0) / 8.0
        combinada = np.clip(prob + alpha * vecinos, 0, 1)
        transita = (combinada > theta) & (rng.random(estado.shape) < combinada)
        estado[transita & (estado == 0)] = 1
    return estado


def metricas(inicial: np.ndarray, observado: np.ndarray,
             predicho: np.ndarray) -> dict[str, float]:
    """Replica exactamente las metricas del script publicado."""
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

    return dict(fom=fom, kappa=kappa, iou=iou, exactitud=exactitud,
                precision=precision, recall=recall, f1=f1,
                B=B, C=C, A=A, urbanas_predichas=int(predicho.sum()),
                urbanas_observadas=int(observado.sum()))


def corre_ventana(anio0: int, anio1: int, theta: float, alpha: float,
                  semilla: int, calc, pesos) -> dict[str, float]:
    inicial = np.load(MAPAS / f"{anio0}.npy")
    observado = np.load(MAPAS / f"{anio1}.npy")
    predicho = simular(inicial, anio1 - anio0, theta, alpha, semilla, calc, pesos)
    return metricas(inicial, observado, predicho)


def verificar(calc, pesos) -> None:
    """Reproduce la ventana 2011-2016 publicada y mide la dispersion por semilla."""
    print("Reproduccion de la ventana publicada 2011-2016 "
          f"(theta={THETA_PUBLICADO}, alpha={ALPHA_PUBLICADO})")
    print("El valor publicado es FoM = 0,2219 con aleatoriedad sin sembrar.\n")
    print(f"{'semilla':>10} {'FoM':>8} {'Kappa':>8} {'IoU':>8} {'urbanas':>12} {'s':>7}")
    foms = []
    for semilla in SEMILLAS:
        t0 = time.time()
        m = corre_ventana(2011, 2016, THETA_PUBLICADO, ALPHA_PUBLICADO,
                          semilla, calc, pesos)
        foms.append(m["fom"])
        print(f"{semilla:>10} {m['fom']:>8.4f} {m['kappa']:>8.4f} {m['iou']:>8.4f} "
              f"{m['urbanas_predichas']:>12,} {time.time()-t0:>7.1f}")
    foms = np.array(foms)
    print(f"\nmedia {foms.mean():.4f}  desv {foms.std(ddof=1):.4f}  "
          f"rango {foms.min():.4f} a {foms.max():.4f}")
    print(f"diferencia con el publicado (0,2219): {foms.mean()-0.22193:+.4f}")


def calibrar(calc, pesos) -> None:
    """Barre theta y alpha sobre las ventanas internas y elige el mejor par."""
    thetas = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    alphas = [0.30, 0.40, 0.50, 0.60]
    pares = list(product(thetas, alphas))

    print(f"Ventanas internas (todas terminan en 2010 o antes): "
          f"{', '.join(f'{a}-{b}' for a, b in VENTANAS_INTERNAS)}")
    print(f"Barrido: {len(thetas)} thetas x {len(alphas)} alphas = {len(pares)} pares")
    print(f"Semillas por par y ventana: {len(SEMILLAS)}")
    print(f"Total de simulaciones: "
          f"{len(pares)*len(VENTANAS_INTERNAS)*len(SEMILLAS)}\n")

    registro = []
    t_inicio = time.time()
    for i, (theta, alpha) in enumerate(pares, 1):
        por_ventana = {}
        for anio0, anio1 in VENTANAS_INTERNAS:
            foms = [corre_ventana(anio0, anio1, theta, alpha, s, calc, pesos)["fom"]
                    for s in SEMILLAS]
            por_ventana[f"{anio0}-{anio1}"] = float(np.mean(foms))
        medio = float(np.mean(list(por_ventana.values())))
        registro.append(dict(theta=theta, alpha=alpha, fom_medio=medio,
                             por_ventana=por_ventana))
        transcurrido = time.time() - t_inicio
        print(f"[{i:>2}/{len(pares)}] theta={theta:.2f} alpha={alpha:.2f}  "
              f"FoM medio={medio:.4f}  "
              f"({', '.join(f'{v:.3f}' for v in por_ventana.values())})  "
              f"{transcurrido/60:.1f} min")

    registro.sort(key=lambda r: -r["fom_medio"])
    mejor = registro[0]
    publicado = next(r for r in registro
                     if r["theta"] == THETA_PUBLICADO and r["alpha"] == ALPHA_PUBLICADO)

    print("\nMejores cinco pares")
    for r in registro[:5]:
        print(f"  theta={r['theta']:.2f} alpha={r['alpha']:.2f}  "
              f"FoM medio={r['fom_medio']:.4f}")
    print(f"\nPar elegido por el pliegue interno: theta={mejor['theta']:.2f}, "
          f"alpha={mejor['alpha']:.2f}  (FoM medio {mejor['fom_medio']:.4f})")
    print(f"Par publicado:                      theta={THETA_PUBLICADO:.2f}, "
          f"alpha={ALPHA_PUBLICADO:.2f}  (FoM medio {publicado['fom_medio']:.4f})")
    coincide = (mejor["theta"] == THETA_PUBLICADO and mejor["alpha"] == ALPHA_PUBLICADO)
    print(f"\nEl pliegue interno reproduce el par publicado: {coincide}")
    if not coincide:
        posicion = registro.index(publicado) + 1
        print(f"El par publicado queda en la posicion {posicion} de {len(registro)}.")

    SALIDA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA / "barrido.json"
    with open(destino, "w") as f:
        json.dump(dict(
            descripcion="Calibracion de theta y alpha con ventanas internas "
                        "contenidas en 1984-2010. No interviene 2011-2020.",
            pesos_woe="data/processed/woe_pooled_1984_2010.pkl (sin reestimar)",
            ventanas_internas=[f"{a}-{b}" for a, b in VENTANAS_INTERNAS],
            semillas=list(SEMILLAS),
            par_publicado=dict(theta=THETA_PUBLICADO, alpha=ALPHA_PUBLICADO,
                               fom_medio=publicado["fom_medio"]),
            par_elegido=dict(theta=mejor["theta"], alpha=mejor["alpha"],
                             fom_medio=mejor["fom_medio"]),
            reproduce_el_publicado=coincide,
            barrido=registro,
        ), f, indent=2, ensure_ascii=False)
    print(f"\nResultados en {destino.relative_to(RAIZ)}")


def main() -> None:
    modo = sys.argv[1] if len(sys.argv) > 1 else "verificar"
    calc, pesos = cargar_woe()
    if modo == "verificar":
        verificar(calc, pesos)
    elif modo == "calibrar":
        calibrar(calc, pesos)
    else:
        raise SystemExit("modos: verificar | calibrar")


if __name__ == "__main__":
    main()
