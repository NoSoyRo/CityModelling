#!/usr/bin/env python3
"""Ablación del esquema de ponderación de los pesos de evidencia.

Responde a una sola pregunta: ¿la ponderación por Information Value normalizado
mejora el desempeño frente a las dos alternativas obvias?

Se comparan tres formas de combinar los pesos WoE por celda, con el modelo WoE
de 1984-2010 intacto. No se reentrena nada; solo cambia el vector de pesos.

    iv     w_k = IV_k / sum_j IV_j      (lo que usa la tesis; los pesos suman 1)
    equal  w_k = 1/K                    (promedio simple; también suman 1)
    sum    w_k = 1                       (suma del WoE clásico; interpretación log-momios)

El umbral de transición no es comparable entre esquemas: con `sum` el score es
del orden de K veces mayor, así que la sigmoide satura y un umbral de 0,75 deja
pasar casi todo. Por eso se barre el umbral dentro de cada esquema y se reporta
su mejor FoM, además del valor a umbral fijo.

Uso:
    python tools/ablacion_ponderacion.py                 # barrido completo
    python tools/ablacion_ponderacion.py --cronometrar   # mide un paso y sale
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "tools"))

from variables_rapidas import variables_espaciales  # noqa: E402

MODELO_WOE = RAIZ / "data/processed/woe_pooled_1984_2010.pkl"
MAPAS = RAIZ / "data/processed/standardized_maps"
SALIDA = RAIZ / "data/processed/ablacion_ponderacion"

# Las cinco ventanas quinquenales de la validación publicada.
VENTANAS = [(2011, 2016), (2012, 2017), (2013, 2018), (2014, 2019), (2015, 2020)]

PESO_VECINAL = 0.50  # neighbor_weight de quinquenal_best_config.json
UMBRAL_TESIS = 0.75  # threshold de quinquenal_best_config.json

# La validación original no fija semilla, así que sus cifras vienen de una
# corrida estocástica irrepetible. Aquí se promedian tres semillas y se reporta
# la dispersión, para separar el efecto del esquema de pesos del ruido del
# sorteo aleatorio.
SEMILLAS = [20260918, 7, 31415]

# Barridos por esquema. Los esquemas convexos (iv, equal) viven en un rango
# estrecho alrededor de 0,75; `sum` satura y necesita umbrales mucho más altos.
UMBRALES = {
    "iv": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    "equal": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
    "sum": [0.75, 0.90, 0.95, 0.99, 0.999, 0.9999],
}


def cargar_woe():
    with open(MODELO_WOE, "rb") as f:
        return pickle.load(f)["woe_calculator"]


def vector_pesos(esquema: str, calculador) -> dict[str, float]:
    ivs = {k: r.iv_total for k, r in calculador.woe_results.items()}
    if esquema == "iv":
        total = sum(ivs.values())
        return {k: v / total for k, v in ivs.items()}
    if esquema == "equal":
        return {k: 1.0 / len(ivs) for k in ivs}
    if esquema == "sum":
        return {k: 1.0 for k in ivs}
    raise ValueError(esquema)


def simular(mapa_inicial, calculador, pesos, umbral, pasos, semilla):
    """Corre el autómata y devuelve el mapa final.

    Réplica exacta del bucle de validate_quinquenal_2011_2016.py, con el vector
    de pesos como único parámetro libre.
    """
    rng = np.random.default_rng(semilla)
    estado = mapa_inicial.copy()

    for _ in range(pasos):
        variables = variables_espaciales(estado)

        woe_total = np.zeros(estado.shape, dtype=float)
        for nombre, mapa_var in variables.items():
            woe = calculador.apply_woe_transform(mapa_var.flatten(), nombre)
            woe_total += pesos[nombre] * woe.reshape(estado.shape)

        prob = 1.0 / (1.0 + np.exp(-woe_total))

        kernel = np.ones((3, 3))
        kernel[1, 1] = 0
        vecinos = convolve(estado.astype(float), kernel, mode="constant", cval=0) / 8.0

        combinada = np.clip(prob + PESO_VECINAL * vecinos, 0, 1)

        sorteo = rng.random(estado.shape)
        transita = (combinada > umbral) & (sorteo < combinada) & (estado == 0)
        estado[transita] = 1

    return estado


def metricas(inicial, predicho, observado) -> dict[str, float]:
    """FoM, Kappa, exactitud e IoU sobre el mismo criterio de la validación."""
    cambio_obs = (observado == 1) & (inicial == 0)
    cambio_pred = (predicho == 1) & (inicial == 0)

    aciertos = int(np.sum(cambio_obs & cambio_pred))
    perdidos = int(np.sum(cambio_obs & ~cambio_pred))
    falsas = int(np.sum(~cambio_obs & cambio_pred))
    denominador = aciertos + perdidos + falsas
    fom = aciertos / denominador if denominador else 1.0

    p, o = predicho.ravel() == 1, observado.ravel() == 1
    vp = int(np.sum(p & o))
    vn = int(np.sum(~p & ~o))
    fp = int(np.sum(p & ~o))
    fn = int(np.sum(~p & o))
    n = vp + vn + fp + fn

    exactitud = (vp + vn) / n
    esperado = ((vp + fp) * (vp + fn) + (vn + fn) * (vn + fp)) / (n * n)
    kappa = (exactitud - esperado) / (1 - esperado) if esperado < 1 else 0.0
    union = vp + fp + fn
    iou = vp / union if union else 0.0

    return {
        "fom": fom,
        "kappa": kappa,
        "accuracy": exactitud,
        "iou": iou,
        "aciertos": aciertos,
        "perdidos": perdidos,
        "falsas_alarmas": falsas,
        "celdas_nuevas_predichas": int(np.sum(cambio_pred)),
        "celdas_nuevas_observadas": int(np.sum(cambio_obs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cronometrar", action="store_true",
                        help="mide el costo de un paso y termina")
    args = parser.parse_args()

    calculador = cargar_woe()

    if args.cronometrar:
        mapa = np.load(MAPAS / "2011.npy")
        pesos = vector_pesos("iv", calculador)
        t0 = time.time()
        simular(mapa, calculador, pesos, UMBRAL_TESIS, 1, SEMILLAS[0])
        print(f"un paso: {time.time() - t0:.1f} s")
        total = sum(len(v) for v in UMBRALES.values()) * len(VENTANAS) * 5 * len(SEMILLAS)
        print(f"pasos del barrido completo: {total}")
        return

    SALIDA.mkdir(parents=True, exist_ok=True)
    resultados = []
    t_inicio = time.time()

    for esquema, umbrales in UMBRALES.items():
        pesos = vector_pesos(esquema, calculador)
        print(f"\n=== esquema {esquema} ===")
        print("  pesos: " + ", ".join(f"{k}={v:.4f}" for k, v in sorted(pesos.items())))

        for umbral in umbrales:
            por_ventana = []
            for inicio, fin in VENTANAS:
                mapa_ini = np.load(MAPAS / f"{inicio}.npy")
                mapa_obs = np.load(MAPAS / f"{fin}.npy")

                corridas = []
                for semilla in SEMILLAS:
                    predicho = simular(mapa_ini, calculador, pesos, umbral,
                                       fin - inicio, semilla)
                    corridas.append(metricas(mapa_ini, predicho, mapa_obs))

                promedio = {clave: float(np.mean([c[clave] for c in corridas]))
                            for clave in corridas[0]}
                promedio["ventana"] = f"{inicio}-{fin}"
                promedio["fom_desv_semillas"] = float(
                    np.std([c["fom"] for c in corridas]))
                por_ventana.append(promedio)

            fom = float(np.mean([m["fom"] for m in por_ventana]))
            kappa = float(np.mean([m["kappa"] for m in por_ventana]))
            resultados.append({
                "esquema": esquema,
                "umbral": umbral,
                "fom_promedio": fom,
                "kappa_promedio": kappa,
                "iou_promedio": float(np.mean([m["iou"] for m in por_ventana])),
                "accuracy_promedio": float(np.mean([m["accuracy"] for m in por_ventana])),
                "fom_min": float(np.min([m["fom"] for m in por_ventana])),
                "fom_max": float(np.max([m["fom"] for m in por_ventana])),
                "fom_desv_semillas_max": float(
                    np.max([m["fom_desv_semillas"] for m in por_ventana])),
                "por_ventana": por_ventana,
            })
            print(f"  umbral {umbral:<7} FoM={fom:.4f}  Kappa={kappa:.4f}"
                  f"  ({time.time() - t_inicio:.0f} s acumulados)")

    destino = SALIDA / "ablacion_resultados.json"
    destino.write_text(json.dumps({
        "semillas": SEMILLAS,
        "peso_vecinal": PESO_VECINAL,
        "umbral_de_la_tesis": UMBRAL_TESIS,
        "modelo_woe": str(MODELO_WOE.relative_to(RAIZ)),
        "nota": "El modelo WoE no se reentrena. Solo cambia el vector de pesos "
                "con que se combinan los pesos de evidencia por celda.",
        "resultados": resultados,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n--- mejor umbral por esquema ---")
    for esquema in UMBRALES:
        propios = [r for r in resultados if r["esquema"] == esquema]
        mejor = max(propios, key=lambda r: r["fom_promedio"])
        fijo = next((r for r in propios if r["umbral"] == UMBRAL_TESIS), None)
        linea = (f"  {esquema:<6} mejor: FoM={mejor['fom_promedio']:.4f} "
                 f"Kappa={mejor['kappa_promedio']:.4f} en umbral {mejor['umbral']}")
        if fijo is not None:
            linea += f"   |  a umbral 0,75: FoM={fijo['fom_promedio']:.4f}"
        print(linea)

    print(f"\nresultados en {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
