Asunto: Capítulo 4 — corrí el pliegue interno que le propuse, y cambia una cosa

Estimada Dra. Lárraga:

En mi correo anterior le planteé el pliegue interno dentro de 1984–2010 como
trabajo de culminación, fuera del alcance de titulación. Lo corrí. Resultó mucho
más barato de lo que suponía y preferí traerle el resultado antes de reescribir
nada del Capítulo 4.

Le adelanto la conclusión, porque es la que importa: el barrido **no** reproduce
el umbral de 0,75 de la tesis. Elige 0,85. Pero el par que elige se comporta
**mejor** que el publicado en las cinco ventanas de 2011–2020, no peor. El umbral
que fijé a mano mirando 2011–2016 no infló las cifras de la tesis; las dejó un
poco por debajo de donde habrían quedado calibrando a ciegas.

Por qué ahora sí se pudo. El cálculo de las siete variables espaciales estaba
implementado celda por celda. Lo reescribí en forma vectorizada, con el mismo
resultado numérico. Una ventana de cinco años pasó de unas tres horas a 3,3
segundos. Eso es lo que volvió factible un barrido de 40 combinaciones sobre seis
ventanas, dos veces.

1. Montaje

Las seis ventanas deslizantes que terminan en 2010 o antes: 2000–2005,
2001–2006, 2002–2007, 2003–2008, 2004–2009 y 2005–2010. No tres sueltas como le
propuse en el correo anterior, sino las seis, para que el pliegue interno tenga
la misma estructura que el protocolo externo de la tesis.

Rejilla de theta de 0,55 a 0,90 en pasos de 0,05 y alfa de 0,30 a 0,70 en pasos
de 0,10. Cuarenta combinaciones. Criterio: FoM medio sobre las seis ventanas. La
regla de transición es exactamente la publicada. Añadí una semilla al sorteo
aleatorio, que en el código original no la tenía.

2. El montaje reproduce la tesis

Antes de barrer nada verifiqué que mi réplica dé las cifras publicadas. La
desviación máxima es de 0,0006 en FoM sobre las cinco ventanas. Eso valida a la
vez la réplica de la regla y la versión vectorizada de las variables.

Aprovecho para cerrar un punto menor. Como el script original no sembraba la
aleatoriedad, sus resultados no eran repetibles bit a bit. Corrí tres semillas
sobre 2011–2016: FoM 0,2217, 0,2217 y 0,2218. La desviación es nula en cuatro
decimales. La omisión impedía la repetición exacta pero no afectaba ninguna cifra
reportada.

3. El barrido no reproduce 0,75

| Barrido | Par elegido | FoM interno |
|---|---|---|
| Con el WoE publicado | theta 0,85, alfa 0,60 | 0,2766 |

El par publicado (0,75 y 0,50) da 0,2633 y queda en la posición 12 de 40.

Dicho de otro modo: si hubiera calibrado el umbral con datos anteriores a 2011,
como corresponde, no habría llegado a 0,75.

4. Anticipo su objeción sobre ese barrido

Las seis ventanas internas caen dentro del período con el que se estimaron los
pesos WoE, así que sus transiciones contribuyeron a los conteos por bin.
Formalmente eso es selección de hiperparámetros sobre el conjunto de
entrenamiento, no validación cruzada anidada. Es la misma clase de objeción que
usted me hizo sobre 2011–2020, un nivel más abajo.

En lugar de argumentarla la medí. Repetí el barrido completo con un WoE
reestimado únicamente con 1984–1999, por el mismo procedimiento del script
publicado, de modo que ninguna transición de 2000–2010 participara en los pesos.
Son 4 430 952 transiciones y el reentrenamiento tarda 30 segundos.

| Barrido | WoE estimado con | Par elegido | FoM interno |
|---|---|---|---|
| A, con fuga | 1984–2010 | theta 0,85, alfa 0,60 | 0,2766 |
| B, sin fuga | 1984–1999 | theta 0,85, alfa 0,50 | 0,2765 |

Los dos coinciden en theta = 0,85 y difieren un solo paso de rejilla en alfa. La
superficie de respuesta es plana en esa región: en el barrido sin fuga, las dos
primeras posiciones son justamente esos dos pares, separados por 0,0001. El par
publicado queda en la posición 14 de 40.

La conclusión no depende de la fuga. Y vale la pena notar que el barrido sin fuga
aterriza exactamente en alfa = 0,50, el valor heredado del ensayo exploratorio.
Lo que estaba desalineado era el umbral, no el peso de vecindad.

5. Qué pasa en las cinco ventanas de evaluación

Apliqué los tres pares a 2011–2020, que no intervino en ninguno de los dos
barridos.

| Par | Origen | FoM | Kappa | IoU |
|---|---|---|---|---|
| 0,75 / 0,50 | publicado | 0,317 | 0,445 | 0,633 |
| 0,85 / 0,60 | barrido con fuga | 0,325 | 0,526 | 0,661 |
| 0,85 / 0,50 | barrido sin fuga | 0,321 | 0,538 | 0,664 |

Cualquiera de los dos pares calibrados mejora al publicado en las tres métricas.
El salto grande es en Kappa: de 0,445 a 0,538, que en la escala de Landis y Koch
pasa de la parte baja a la parte media del acuerdo moderado.

También mejora lo que a mí más me interesaba, porque es la debilidad que el
Capítulo 5 documenta. Con el par sin fuga, la Precision sube de 0,659 a 0,718 y
el Recall baja de 0,944 a 0,904. La razón entre celdas urbanas predichas y
observadas cae de 1,44 a 1,27 en promedio. Es decir, el modelo sigue
sobre-prediciendo cuánto crece la ciudad, pero bastante menos. Eso es coherente
con un umbral más exigente, y es el mecanismo esperado.

Y mejora la estabilidad temporal, que era su pregunta de fondo. El factor entre
la mejor y la peor ventana baja de 1,70 a 1,57 en FoM y de 1,43 a 1,30 en Kappa.
La dispersión sigue siendo real; no voy a llamarla estabilidad estricta.

6. Qué se puede afirmar ahora

Mantengo lo que le escribí: en la tesis tal como está, el umbral no es hold-out
de 2011–2020, y las dos frases que lo afirman de más se corrigen. Son el párrafo
de calibración del Capítulo 5 y el pie de la figura del Capítulo 4. Eso no
cambia.

Lo que cambia es que ahora tengo evidencia de la dirección del sesgo, y va en
contra del riesgo que usted señaló. El renglón rojo de su comparativo describe un
desempeño que «puede verse optimista». Aquí es lo contrario: el par informado por
2011–2016 rinde por debajo del que se obtiene sin mirar ese período. El FoM de
0,317 no está inflado por la forma de calibrar.

También queda demostrado que el protocolo verde es alcanzable sin rehacer el WoE:
calibrar theta y alfa con ventanas anteriores a 2011 y evaluar después en las
cinco.

7. Un hallazgo que no buscaba y que sí me preocupa

Al reestimar el WoE con 1984–1999 en lugar de 1984–2010, la ponderación por
Information Value cambia mucho en una variable:

| Variable | IV normalizado, 1984–2010 | IV normalizado, 1984–1999 |
|---|---|---|
| distance_urban | 0,2815 | 0,0822 |
| neighbor_density_3x3 | 0,2495 | 0,3171 |
| neighbor_density_5x5 | 0,1445 | 0,2016 |
| neighbor_density_7x7 | 0,1067 | 0,1478 |
| local_fragmentation | 0,1062 | 0,1307 |
| urban_gradient | 0,0996 | 0,1100 |
| nearest_cluster_size | 0,0120 | 0,0105 |

La distancia al área urbana pasa de ser la variable de mayor peso a una de las
menores, un factor de 3,4. Las demás se mueven poco y conservan su orden. Es
decir, el orden de importancia entre variables, que la tesis presenta como
resultado interpretable, no se sostiene al cambiar el período de estimación.

Mi hipótesis, y la marco como no verificada, es que esto viene del punto que ya
habíamos discutido: el muestreo WoE incluye las celdas ya urbanas, y por eso
`distance_urban` tiene un bin degenerado en distancia cero con WoE de −14,28. La
proporción de celdas urbanas difiere entre los dos períodos, así que ese bin pesa
distinto. No he comprobado el mecanismo.

Lo tranquilizador es que la calibración de theta y alfa apenas se mueve pese a ese
cambio, lo que sugiere que el comportamiento del autómata está gobernado por las
densidades de vecindad y no por la distancia al frente urbano. Pero la lectura
interpretativa del IV sí queda tocada, y preferí decírselo antes de que lo
encuentre un sinodal.

8. Qué propongo

No he modificado nada de la tesis ni de los resultados publicados. El ejercicio
está en una rama aparte del repositorio, con su propia carpeta, su bitácora y los
archivos de salida. El pickle de pesos de la tesis conserva su huella digital
original y el WoE reentrenado no se guardó en disco.

Sobre qué hacer con esto, tengo una duda genuina y es suya la decisión.

La primera opción es dejar la tesis como está, corregir solo las dos frases que
afirman de más, y reservar todo este ejercicio para la defensa y para el cierre de
investigación. Es lo que yo mismo le propuse y es lo más conservador.

La segunda es incorporarlo como una sección corta de validación adicional: el
protocolo real queda descrito con precisión, y a continuación se muestra que una
calibración limpia habría dado un par ligeramente distinto y un desempeño
ligeramente mejor. Tiene la ventaja de desarmar la objeción antes de que la
hagan, y el costo de abrir una pregunta nueva sobre la estabilidad del IV, que no
tengo resuelta.

Lo que no haría es cambiar las cifras de la tesis por las del par nuevo. Las
cinco ventanas publicadas se corrieron con el par publicado, y sustituirlas a
estas alturas me parece peor que explicar bien lo que hay.

Quedo pendiente de su opinión para seguir con el Capítulo 4.

Saludos cordiales,

José Rodrigo Moreno López
