# Calibración de theta y alpha con un pliegue interno a 1984-2010

Rama `calibracion/pliegue-interno`. No modifica las fuentes de la tesis ni los
resultados publicados. El pickle `woe_pooled_1984_2010.pkl` se abre solo en
lectura y su hash sigue siendo idéntico al de `main`.

```bash
python tools/calibracion_pliegue_interno.py verificar   # reproduce la tesis
python tools/calibracion_pliegue_interno.py calibrar    # primer barrido, 3 ventanas
python tools/calibracion_anidada.py                     # 6 ventanas, con y sin fuga
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

## Montaje

| Elemento | Valor |
|---|---|
| Ventanas internas | Las seis deslizantes de 2000-2005 a 2005-2010 |
| Rejilla | theta de 0,55 a 0,90 (paso 0,05); alpha de 0,30 a 0,70 (paso 0,10) |
| Criterio | FoM medio sobre las seis ventanas internas |
| Semilla | 20260918 |
| Variables espaciales | `tools/variables_rapidas.py` |

Las seis ventanas deslizantes replican la misma estructura del protocolo
externo. El primer barrido usaba solo tres ventanas sueltas, elegidas para
acotar el tiempo de cómputo antes de saber que cada corrida tarda 3,3 segundos.
La rejilla se amplió hasta theta 0,90 y alpha 0,70 porque en el primer barrido
el ganador caía en el borde.

## Resultado 1: el pipeline publicado es reproducible

El harness reproduce la tabla de la tesis con desviación máxima de 0,0006 en
FoM. Eso valida a la vez la réplica de la regla de transición y la versión
vectorizada de las variables espaciales.

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
impedía la repetición bit a bit pero no afectaba ninguna cifra reportada.

## Resultado 3: la fuga del WoE no cambia la elección

Objeción pertinente: las ventanas internas caen dentro del período con el que se
estimaron los pesos WoE, así que sus transiciones contribuyeron a los conteos
por bin. Formalmente es selección de hiperparámetros sobre el conjunto de
entrenamiento, no validación cruzada anidada.

Se midió en lugar de discutirla. Se repitió el barrido completo con un WoE
reestimado únicamente con 1984-1999, por el mismo procedimiento del script
publicado, de modo que las transiciones de 2000-2010 nunca intervinieron en los
pesos.

| Barrido | WoE | Par elegido | FoM interno |
|---|---|---|---|
| A, con fuga | 1984-2010 | theta 0,85, alpha 0,60 | 0,2766 |
| B, sin fuga | 1984-1999 | theta 0,85, alpha 0,50 | 0,2765 |

Los dos coinciden en theta = 0,85 y difieren un solo paso de rejilla en alpha. El
par de B, evaluado en el barrido A, da 0,2757 contra 0,2766 del ganador de A: una
diferencia de 0,0009, que es tres órdenes de magnitud mayor que el ruido de
semilla pero despreciable frente a la distancia al par publicado.

El par publicado queda en la posición 12 de 40 en A y 14 de 40 en B. La
conclusión no depende de la fuga.

## Resultado 4: los tres pares sobre las cinco ventanas de evaluación

Aplicando cada par a 2011-2020, que no intervino en ninguna calibración:

| Par | Origen | FoM | Kappa | IoU | Factor máx/mín de FoM |
|---|---|---|---|---|---|
| 0,75 / 0,50 | publicado, umbral informado por 2011-2016 | 0,3174 | 0,4446 | 0,6329 | 1,71 |
| 0,85 / 0,60 | barrido con fuga | **0,3245** | 0,5256 | 0,6607 | 1,60 |
| 0,85 / 0,50 | barrido sin fuga | 0,3211 | **0,5381** | **0,6644** | **1,57** |

Cualquiera de los dos pares calibrados mejora al publicado en las tres métricas y
reduce la dispersión entre ventanas. El par sin fuga es el mejor en Kappa, en IoU
y en estabilidad.

Por ventana, con el par sin fuga: 0,2411, 0,3384, 0,3398, 0,3085 y 0,3776.

## Lectura

El resultado va en la dirección favorable. El umbral publicado no infló las
cifras: haberlo informado con 2011-2016 produjo un par ligeramente **peor** que
el que se obtiene calibrando a ciegas con datos anteriores a 2011.

El salto mayor es en Kappa, de 0,445 a 0,538, que dentro de la escala de uso
común de Landis y Koch pasa de la parte baja a la parte media de la banda de
acuerdo moderado.

Con esto el protocolo verde es alcanzable sin rehacer el WoE publicado:
calibrar theta y alpha con ventanas anteriores a 2011 y evaluar después en las
cinco ventanas.

## Hallazgo lateral que conviene revisar: el IV no es estable

Al reestimar el WoE con 1984-1999 en lugar de 1984-2010, la ponderación por
Information Value cambia de forma marcada en una variable:

| Variable | IV normalizado, 1984-2010 | IV normalizado, 1984-1999 |
|---|---|---|
| distance_urban | 0,2815 | **0,0822** |
| neighbor_density_3x3 | 0,2495 | 0,3171 |
| neighbor_density_5x5 | 0,1445 | 0,2016 |
| neighbor_density_7x7 | 0,1067 | 0,1478 |
| local_fragmentation | 0,1062 | 0,1307 |
| urban_gradient | 0,0996 | 0,1100 |
| nearest_cluster_size | 0,0120 | 0,0105 |

`distance_urban` pasa de ser la variable de mayor peso a una de las menores, un
factor de 3,4. El orden de importancia que la tesis presenta como resultado
interpretable no se sostiene al cambiar el período de estimación.

Hipótesis no verificada: el IV de `distance_urban` está dominado por el bin
degenerado de distancia cero, que corresponde a celdas ya urbanas y cuyo WoE es
-14,28. La proporción de celdas urbanas difiere entre 1984-1999 y 1984-2010, así
que ese bin pesa distinto. No se ha comprobado el mecanismo.

Lo tranquilizador es que la calibración apenas se mueve pese a ese cambio, lo que
indica que el comportamiento del autómata está dominado por las densidades de
vecindad y no por la distancia al frente urbano.

## Qué no se ha hecho

- No se han tocado los resultados publicados ni los archivos de la tesis.
- El WoE reestimado con 1984-1999 vive solo en memoria; no se guardó a disco.
- El par del barrido B se eligió con un WoE distinto del publicado y después se
  aplicó al modelo con el WoE publicado. Es una prueba de robustez, no un
  protocolo anidado completo, que exigiría también evaluar con el WoE de
  1984-1999.
- Las ventanas internas se solapan entre sí, lo que es admisible para elegir un
  hiperparámetro pero impide tratar su FoM medio como estimación independiente.
- No se ha confirmado el mecanismo detrás de la inestabilidad del IV.
