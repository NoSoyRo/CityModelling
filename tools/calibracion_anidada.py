#!/usr/bin/env python3
"""Compara la calibracion de theta y alpha con y sin fuga del WoE.

Dos barridos sobre las mismas seis ventanas deslizantes de 2000 a 2010:

A. Con el WoE publicado, estimado con 1984-2010. Las ventanas de calibracion
   caen dentro de ese periodo, de modo que sus transiciones contribuyeron a los
   conteos por bin. Es seleccion de hiperparametros sobre el conjunto de
   entrenamiento.

B. Con un WoE reestimado unicamente con 1984-1999, por el mismo procedimiento
   del script publicado (promedio de variables, bins por cuantiles,
   min_bin_size=50, max_bins=10). Aqui las transiciones de 2000-2010 nunca
   intervinieron en los pesos, asi que la seleccion es anidada de verdad.

Si ambos barridos eligen el mismo par, queda demostrado que la fuga no mueve la
eleccion, que es la objecion de fondo. Si eligen pares distintos, la magnitud de
la diferencia dice cuanto importaba.

Uso:
    python tools/calibracion_anidada.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tools"))

logging.disable(logging.INFO)

from calibracion_pliegue_interno import cargar_woe, corre_ventana  # noqa: E402
from variables_rapidas import variables_espaciales  # noqa: E402
from tesis_ac.woe.woe import WoECalculator  # noqa: E402

MAPAS = RAIZ / "data/processed/standardized_maps"
SALIDA = RAIZ / "data/processed/calibracion_pliegue_interno"

VENTANAS = [(a, a + 5) for a in range(2000, 2006)]
SEMILLA = 20260918
THETAS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
ALPHAS = [0.30, 0.40, 0.50, 0.60, 0.70]
PUBLICADO = (0.75, 0.50)

# El ultimo anio de entrenamiento del WoE anidado. Con 1999 la ultima transicion
# es 1998->1999, de modo que el mapa de 2000 no interviene y la primera ventana
# de calibracion (2000-2005) parte de un estado no visto.
FIN_ENTRENAMIENTO_ANIDADO = 1999


def entrenar_woe(anio_inicial: int, anio_final: int):
    """Replica train_woe_pooled.py sobre un rango de anios arbitrario."""
    anios = list(range(anio_inicial, anio_final + 1))
    print(f"  reentrenando WoE con {anios[0]}-{anios[-1]} "
          f"({len(anios)-1} transiciones)", flush=True)
    t0 = time.time()

    mapas = {a: np.load(MAPAS / f"{a}.npy") for a in anios}
    acumulado: dict[str, list[np.ndarray]] = {}
    transiciones = []
    for i in range(len(anios) - 1):
        g0, g1 = mapas[anios[i]], mapas[anios[i + 1]]
        transiciones.append(((g0 == 0) & (g1 == 1)).flatten())
        for nombre, dato in variables_espaciales(g0).items():
            acumulado.setdefault(nombre, []).append(dato)

    objetivo = np.concatenate(transiciones)
    del transiciones
    promedio = {n: np.mean(v, axis=0) for n, v in acumulado.items()}
    del acumulado

    calc = WoECalculator(min_bin_size=50, max_bins=10)
    resultados = {}
    for nombre, mapa in promedio.items():
        grande = np.tile(mapa.flatten(), len(anios) - 1)
        resultados[nombre] = calc.calculate_woe(grande, objetivo, nombre)
        del grande
    calc.woe_results = resultados

    iv = {n: float(r.iv_total) for n, r in resultados.items()}
    total = sum(iv.values())
    pesos = {n: v / total for n, v in iv.items()}
    print(f"  listo en {time.time()-t0:.0f} s  "
          f"(transiciones={int(objetivo.sum()):,})", flush=True)
    return calc, pesos


def barrer(etiqueta: str, calc, pesos) -> list[dict]:
    pares = list(product(THETAS, ALPHAS))
    print(f"\n=== {etiqueta} ===", flush=True)
    print(f"{len(pares)} pares x {len(VENTANAS)} ventanas", flush=True)
    registro = []
    t0 = time.time()
    for i, (theta, alpha) in enumerate(pares, 1):
        pv = {f"{a0}-{a1}": corre_ventana(a0, a1, theta, alpha, SEMILLA, calc, pesos)["fom"]
              for a0, a1 in VENTANAS}
        medio = float(np.mean(list(pv.values())))
        registro.append(dict(theta=theta, alpha=alpha, fom_medio=medio, por_ventana=pv))
        if i % 5 == 0 or i == len(pares):
            print(f"  [{i:>2}/{len(pares)}] ultimo theta={theta:.2f} alpha={alpha:.2f} "
                  f"FoM={medio:.4f}   {(time.time()-t0)/60:.1f} min", flush=True)
    registro.sort(key=lambda r: -r["fom_medio"])
    return registro


def informar(etiqueta: str, registro: list[dict]) -> dict:
    pub = next(r for r in registro if (r["theta"], r["alpha"]) == PUBLICADO)
    mejor = registro[0]
    print(f"\n{etiqueta}: mejores cuatro")
    for r in registro[:4]:
        print(f"   theta={r['theta']:.2f} alpha={r['alpha']:.2f}  FoM={r['fom_medio']:.4f}")
    print(f"   par publicado (0.75, 0.50): FoM={pub['fom_medio']:.4f}, "
          f"posicion {registro.index(pub)+1} de {len(registro)}")
    return dict(mejor=mejor, publicado=pub)


def main() -> None:
    print("Comparacion de la calibracion con y sin fuga del WoE")
    print(f"Ventanas: {', '.join(f'{a}-{b}' for a, b in VENTANAS)}\n")

    calc_a, pesos_a = cargar_woe()
    reg_a = barrer("A. WoE publicado 1984-2010 (las ventanas estan dentro)",
                   calc_a, pesos_a)
    res_a = informar("A", reg_a)

    print()
    calc_b, pesos_b = entrenar_woe(1984, FIN_ENTRENAMIENTO_ANIDADO)
    reg_b = barrer(f"B. WoE reestimado 1984-{FIN_ENTRENAMIENTO_ANIDADO} "
                   "(ventanas no vistas)", calc_b, pesos_b)
    res_b = informar("B", reg_b)

    par_a = (res_a["mejor"]["theta"], res_a["mejor"]["alpha"])
    par_b = (res_b["mejor"]["theta"], res_b["mejor"]["alpha"])
    print("\n" + "=" * 60)
    print(f"A, con fuga:  theta={par_a[0]:.2f}, alpha={par_a[1]:.2f}")
    print(f"B, sin fuga:  theta={par_b[0]:.2f}, alpha={par_b[1]:.2f}")
    print(f"Coinciden: {par_a == par_b}")
    if par_a != par_b:
        en_a = next(r for r in reg_a if (r["theta"], r["alpha"]) == par_b)
        print(f"El par de B, evaluado en el barrido A, da FoM={en_a['fom_medio']:.4f} "
              f"contra {res_a['mejor']['fom_medio']:.4f} del ganador de A "
              f"(diferencia {res_a['mejor']['fom_medio']-en_a['fom_medio']:+.4f}).")

    print("\nPesos IV: publicado contra reestimado")
    print(f"  {'variable':<24} {'1984-2010':>10} {'1984-1999':>10}")
    for n in sorted(pesos_a, key=lambda x: -pesos_a[x]):
        print(f"  {n:<24} {pesos_a[n]:>10.4f} {pesos_b[n]:>10.4f}")

    SALIDA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA / "calibracion_anidada.json"
    with open(destino, "w") as f:
        json.dump(dict(
            descripcion="Calibracion de theta y alpha con el WoE publicado y con "
                        "un WoE reestimado sin las ventanas de calibracion.",
            ventanas=[f"{a}-{b}" for a, b in VENTANAS],
            semilla=SEMILLA,
            A_woe_publicado=dict(entrenamiento="1984-2010", par=par_a,
                                 fom=res_a["mejor"]["fom_medio"], barrido=reg_a),
            B_woe_anidado=dict(entrenamiento=f"1984-{FIN_ENTRENAMIENTO_ANIDADO}",
                               par=par_b, fom=res_b["mejor"]["fom_medio"],
                               barrido=reg_b),
            coinciden=par_a == par_b,
            pesos_iv=dict(publicado=pesos_a, anidado=pesos_b),
        ), f, indent=2, ensure_ascii=False)
    print(f"\nguardado en {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
