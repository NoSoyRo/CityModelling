# Calibración de theta y alpha con un pliegue interno a 1984-2010

Rama `calibracion/pliegue-interno`. No modifica los resultados publicados ni las
fuentes de la tesis. Reproducible con:

```bash
python tools/calibracion_pliegue_interno.py verificar
python tools/calibracion_pliegue_interno.py calibrar
```

## Por qué se hizo

En el modelo publicado los pesos WoE y sus Information Value se estimaron solo
con 1984-2010, pero el umbral theta se fijó a mano después de observar
sobre-predicción en la primera ventana de evaluación, 2011-2016. Por eso el
período 2011-2020 no podía describirse como hold-out del modelo completo, y dos
pasajes de la tesis lo afirman de más (`cap06:156` y el pie de figura
`cap05:368`).

Este ejercicio elige theta y alpha usando únicamente ventanas que terminan en
2010 o antes, de modo que 2011-2020 deje de intervenir en esa elección.

## Qué demuestra y qué no

Los pesos WoE se ajustaron con el período que contiene las ventanas internas.
Por tanto esto es selección de hiperparámetros sobre el conjunto de
entrenamiento, no validación cruzada anidada. La ganancia es concreta y
limitada: 2011-2020 ya no participa en la elección de theta ni de alpha.

Para un protocolo del todo anidado habría que reestimar el WoE sobre un
subperíodo (por ejemplo 1984-2005), calibrar en 2006-2010 y recién entonces
evaluar. Eso cambiaría también los pesos publicados y es trabajo de cierre de
investigación, no de titulación.

## Montaje

| Elemento | Valor |
|---|---|
| Pesos WoE | `data/processed/woe_pooled_1984_2010.pkl`, sin reestimar |
| Ventanas internas | 2000-2005, 2002-2007, 2005-2010 (cinco pasos cada una) |
| Rejilla inicial | theta en 0,55 a 0,85 (paso 0,05); alpha en 0,30 a 0,60 (paso 0,10) |
| Extensión | theta en 0,85 a 0,95; alpha en 0,60 a 0,80 |
| Semillas | 20260918, 7, 31415 |
| Criterio | FoM medio sobre las tres ventanas internas |
| Variables espaciales | `tools/variables_rapidas.py` |

## Resultado 1: el pipeline publicado es reproducible

El harness reproduce la tabla de la tesis con desviación máxima de 0,0006 en
FoM. Eso valida a la vez la réplica de la regla y la versión vectorizada de las
variables.

| Ventana | FoM en la tesis | FoM aquí, con semilla |
|---|---|---|
| 2011-2016 | 0,2219 | 0,2217 |
| 2012-2017 | 0,3451 | 0,3453 |
| 2013-2018 | 0,3553 | 0,3547 |
| 2014-2019 | 0,2873 | 0,2873 |
| 2015-2020 | 0,3777 | 0,3781 |
| Promedio | 0,3175 | 0,3174 |

## Resultado 2: la aleatoriedad sin sembrar era cosmética

Tres semillas sobre 2011-2016 con el par publicado dan FoM de 0,2217, 0,2217 y
0,2218. La desviación estándar es nula en cuatro decimales. La falta de semilla
impedía la repetición bit a bit, pero no afectaba ninguna cifra reportada.

## Resultado 3: el pliegue interno no elige el par publicado

El par publicado (0,75, 0,50) queda en la posición 9 de 28 con FoM medio interno
de 0,2626. El pliegue elige **theta = 0,85 y alpha = 0,60**, con 0,2742.

La extensión de la rejilla confirma que ese punto es un óptimo interior y no un
artefacto del borde: ningún par con theta hasta 0,95 o alpha hasta 0,80 lo
supera.

Conviene registrar dos rasgos de la superficie de respuesta. Hay una meseta
amplia entre 0,272 y 0,274 que abarca varios pares, de modo que la elección no
queda determinada con nitidez; y hay celdas de colapso, como (0,85, 0,30) con
FoM de 0,055, donde el umbral es demasiado alto respecto del peso de vecindad y
casi nada transita. Los dos parámetros interactúan y no se pueden leer por
separado.

## Resultado 4: el par calibrado mejora el desempeño fuera de muestra

Aplicando ambos pares a las cinco ventanas de 2011-2020, que no intervinieron en
la calibración:

| Métrica | Par publicado (0,75, 0,50) | Par del pliegue (0,85, 0,60) |
|---|---|---|
| FoM promedio | 0,3174 | **0,3245** |
| Kappa promedio | 0,4446 | **0,5256** |
| IoU promedio | 0,6329 | **0,6607** |
| FoM mínimo | 0,2217 | **0,2397** |
| Factor máx/mín de FoM | 1,71 | **1,60** |

Por ventana, con el par del pliegue: 0,2397, 0,3446, 0,3471, 0,3075 y 0,3837.

## Lectura

El resultado va en la dirección favorable. El umbral publicado no infló las
cifras: haberlo informado con 2011-2016 produjo un par ligeramente peor que el
que se obtiene calibrando a ciegas con datos anteriores a 2011. Con el par
calibrado el desempeño sube en las tres métricas, la ventana peor mejora y la
dispersión entre ventanas se reduce.

El salto mayor es en Kappa, de 0,445 a 0,526, que dentro de la escala de uso
común de Landis y Koch pasa de la parte baja a la parte media de la banda de
acuerdo moderado.

Con esto el protocolo verde es alcanzable sin rehacer el WoE: calibrar theta y
alpha con ventanas anteriores a 2011 y evaluar después en las cinco ventanas.

## Qué no se ha hecho

- No se han tocado los resultados publicados ni los archivos de la tesis.
- No se ha reestimado el WoE, así que el anidamiento no es completo.
- Las ventanas internas se solapan entre sí, lo que es admisible para elegir un
  hiperparámetro pero impide tratar su FoM medio como una estimación
  independiente.
- No se ha medido la dispersión por semilla en las ventanas internas con la
  misma profundidad que en 2011-2016.
