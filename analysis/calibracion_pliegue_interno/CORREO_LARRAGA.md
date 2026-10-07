Asunto: Corrí el pliegue interno que le propuse y cambié los capítulos en consecuencia

Estimada Dra. Lárraga:

En mi correo anterior le planteé el pliegue interno dentro de 1984–2010 como
trabajo de culminación, fuera del alcance de titulación. Lo corrí. Resultó mucho
más barato de lo que suponía, y el resultado fue lo bastante claro como para que
ya no tuviera sentido dejarlo fuera: actualicé con él los Capítulos 4, 5 y 6 y el
resumen. Le detallo todo abajo, incluido lo que cambió de narrativa y no solo de
cifra.

Le adelanto la conclusión, porque es la que importa: el barrido no reproduce el
umbral de 0,75 de la tesis. Elige 0,85. Pero el par que elige se comporta mejor
que el publicado en las cinco ventanas de 2011–2020, no peor. El umbral
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

El par ganador es theta = 0,85 y alfa = 0,60, con un FoM interno de 0,2766. El
par publicado, 0,75 y 0,50, da 0,2633 y queda en la posición 12 de 40.

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

Los dos barridos, uno al lado del otro:

- Con fuga, WoE de 1984–2010: elige theta 0,85 y alfa 0,60, FoM interno 0,2766.
- Sin fuga, WoE de 1984–1999: elige theta 0,85 y alfa 0,50, FoM interno 0,2765.

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

- Par publicado, 0,75 y 0,50: FoM 0,317, Kappa 0,445, IoU 0,633.
- Par del barrido con fuga, 0,85 y 0,60: FoM 0,325, Kappa 0,526, IoU 0,661.
- Par del barrido sin fuga, 0,85 y 0,50: FoM 0,321, Kappa 0,538, IoU 0,664.

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

Hay un efecto que no es de magnitud sino de interpretación, y es el que más me
hizo pensar. Con el par publicado, el desacuerdo de cantidad concentraba el 79,3
por ciento del desacuerdo total y dominaba en las cinco ventanas, lo que permitía
la lectura cómoda de que el modelo acierta dónde y falla en cuánto. Con el par
calibrado baja al 62,4 por ciento, y el promedio deja de ser representativo: la
cantidad sigue dominando en las dos ventanas de menor crecimiento observado, con
86,2 y 77,5 por ciento, pero en las tres ventanas de crecimiento positivo cae al
rango de 42,7 a 48,6 y el desacuerdo de asignación pasa a ser el mayoritario.

Eso obligó a reescribir la sección de descomposición del error, no solo a
renumerarla. La lectura que queda es más matizada y, me parece, más honesta:
cuando la señal de cambio es débil el margen de mejora está en estimar la
magnitud del crecimiento, y cuando es clara el límite lo pone la regla que decide
dónde ocurre.

6. Qué se puede afirmar ahora

Le había escrito que en la tesis tal como estaba el umbral no era hold-out de
2011–2020, y que las dos frases que lo afirmaban de más había que corregirlas. Ya
no hay nada que corregir en ese punto, porque el umbral ahora sí se elige con
ventanas anteriores a 2011. El protocolo que usted marcó en verde resultó
alcanzable sin rehacer el WoE, y es el que está descrito en los capítulos.

Sobre la dirección del sesgo, que era el riesgo que usted señaló: el renglón rojo
de su comparativo describe un desempeño que «puede verse optimista». Aquí ocurre
lo contrario. El par informado por 2011–2016 rinde por debajo del que se obtiene
sin mirar ese período, de modo que el 0,317 que reportaba la versión anterior no
estaba inflado por la forma de calibrar, sino al revés.

7. Un hallazgo que no buscaba y que sí me preocupa

Al reestimar el WoE con 1984–1999 en lugar de 1984–2010, la ponderación por
Information Value cambia mucho en una variable. Cada renglón da el IV normalizado
con 1984–2010 y después con 1984–1999.

- distance_urban: 0,2815 y 0,0822.
- neighbor_density_3x3: 0,2495 y 0,3171.
- neighbor_density_5x5: 0,1445 y 0,2016.
- neighbor_density_7x7: 0,1067 y 0,1478.
- local_fragmentation: 0,1062 y 0,1307.
- urban_gradient: 0,0996 y 0,1100.
- nearest_cluster_size: 0,0120 y 0,0105.

La distancia al área urbana pasa de ser la variable de mayor peso a una de las
menores, un factor de 3,4. Las demás se mueven poco y conservan su orden. Es
decir, el orden de importancia entre variables, que la tesis presenta como
resultado interpretable, no se sostiene al cambiar el período de estimación.

Mi hipótesis, y la marco como no verificada, es que esto viene del punto que ya
habíamos discutido: el muestreo WoE incluye las celdas ya urbanas, y por eso
distance_urban tiene un bin degenerado en distancia cero con WoE de −14,28. La
proporción de celdas urbanas difiere entre los dos períodos, así que ese bin pesa
distinto. No he comprobado el mecanismo.

Lo tranquilizador es que la calibración de theta y alfa apenas se mueve pese a ese
cambio, lo que sugiere que el comportamiento del autómata está gobernado por las
densidades de vecindad y no por la distancia al frente urbano. Pero la lectura
interpretativa del IV sí queda tocada, y preferí decírselo antes de que lo
encuentre un sinodal.

8. Qué cambié en el documento

Adopté el protocolo de dos etapas y el par theta = 0,85, alfa = 0,60. Los pesos
de evidencia y su Information Value se estiman con las 26 transiciones anuales de
1984 a 2010; con esos pesos ya fijos, el umbral y el peso de vecindad se eligen
por búsqueda en cuadrícula sobre las seis ventanas internas a ese mismo período;
y el par resultante se aplica después, sin reajuste, a las cinco ventanas de
2011–2020. Ninguna de las cinco interviene en la elección de los parámetros.

Lo que esto tocó:

- Capítulo 4: el umbral, el peso de vecindad y el pie de la figura del protocolo
  de validación, que ahora dice con precisión qué se estima y qué se elige con el
  período de ajuste.
- Capítulo 5: la tanda completa de métricas recalculada, la sección de
  calibración reescrita para describir las dos etapas, y la sección de
  descomposición del error reescrita por lo que le comenté en el punto 5.
- Capítulo 6 y resumen: las cifras de síntesis.
- Las cinco figuras de resumen por ventana, regeneradas con el mismo diseño.

Las cifras de síntesis quedan así: FoM de 0,317 a 0,325, Kappa de 0,445 a 0,526,
exactitud de 0,722 a 0,763, IoU de 0,633 a 0,661.

Lo que no entró en el documento es el barrido sin fuga del punto 4. No porque lo
esconda, sino porque no forma parte del protocolo que la tesis describe: es la
verificación de que ese protocolo no depende de la fuga, y su lugar natural es la
defensa o el cierre de investigación. Si usted prefiere que aparezca como una
sección corta de validación adicional, lo escribo; mi reserva es que abriría la
pregunta sobre la estabilidad del IV del punto 7, que no tengo resuelta.

El ejercicio completo está en una rama aparte del repositorio, con su propia
carpeta, su bitácora y los archivos de salida. El pickle de pesos conserva su
huella digital original y el WoE reentrenado nunca se guardó en disco.

Me queda una duda de forma sobre el Capítulo 1, aparte de todo lo anterior. Al
sustituir las cuatro hipótesis particulares por la hipótesis general, los
Capítulos 5 y 6 quedaron remitiendo a unas H1 a H4 que ya no estaban enunciadas.
Seguí la salida que usted misma apunta en su comentario y las convertí en cuatro
criterios de evaluación, C1 a C4, uno por cada pregunta de investigación. Si
prefiere otra solución, el cambio es acotado.

Quedo pendiente de su opinión.

Saludos cordiales,

José Rodrigo Moreno López
