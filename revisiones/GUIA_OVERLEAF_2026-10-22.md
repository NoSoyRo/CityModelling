# Guía para aplicar a mano en Overleaf los cambios del 22 de octubre

Los números de línea son los de tu versión actual en Overleaf (la que subiste con los mapas de calor).
Dentro de cada archivo los cambios van **de abajo hacia arriba**: hazlos en ese orden y los números no se mueven.
Cada cambio dice qué líneas borrar y qué texto pegar en su lugar. Para ubicarte, también va el inicio del texto que reemplazas.
En total son 51 cambios en 7 archivos. Lo que no aparece aquí no cambió.


## Resumen — `front/resumen.tex`

2 cambios.

### Resumen, cambio 1

Empieza con: «El modelo obtuvo un FoM promedio de \textcolor{red}{$0{,}325$}, un Kappa promedio de \text…»

**Reemplaza la línea 15** por:

```latex
El modelo obtuvo un FoM promedio de \textcolor{red}{$0{,}325$}, un Kappa promedio de \textcolor{red}{$0{,}526$}, una exactitud de \textcolor{red}{$0{,}763$} y un IoU de \textcolor{red}{$0{,}661$} en las cinco ventanas de evaluación. Los resultados muestran que el modelo reproduce parte del patrón espacial de crecimiento observado y mantiene un desempeño comparable entre los periodos evaluados \textcolor{red}{(FoM entre $0{,}240$ y $0{,}384$)}. La función de transición basada en WoE permite identificar la contribución de las variables utilizadas y el procesamiento posterior a la adquisición de las imágenes puede ejecutarse con herramientas de código abierto y CPU, sin requerir GPU ni software propietario. La principal aportación del trabajo es el protocolo de evaluación multitemporal aplicado al modelo WoE-AC y su implementación reproducible para el análisis del crecimiento urbano de la ZMQ.
```

### Resumen, cambio 2

Empieza con: «% calibración anterior (0.317, 0.445, 0.722, 0.633). Valores vigentes en…»

**Reemplaza la línea 10** por:

```latex
% calibración anterior (0.317, 0.445, 0.722, 0.633). Después se añadió, también
% en rojo, el rango del FoM entre ventanas junto a «desempeño comparable», para
% que la frase concuerde con el criterio C2, satisfecho de forma parcial. Valores vigentes en
```

## Abstract — `front/abstract.tex`

5 cambios.

### Abstract, cambio 1

Empieza con: «\vspace{0.6cm}…»

**Reemplaza la línea 35** por:

```latex
\vspace{0.3cm}
```

### Abstract, cambio 2

Empieza con: «The contribution lies not in the WoE-CA coupling, well documented in the…»

**Reemplaza las líneas 25–33** por:

```latex
% Versión anterior:
%
% A strategy for the integration, calibration and validation of a
% reproducible urban growth model is presented. The model is built from freely
% viewable historical imagery and assessed through a multi-temporal
% validation protocol with five shifted five-year windows. It couples Weights of
% Evidence (WoE) with a cellular automaton (CA) and is applied, as a case study,
% to the Querétaro Metropolitan Area over a 37-year annual series (1984--2020).
%
% RGB captures from Google Earth's historical imagery tool were converted into
% binary urban/non-urban maps through an open classification pipeline: local
% binary patterns (LBP), K-Means clustering and a linear SVM. The Weights of
% Evidence quantify, without linearity assumptions, the spatial influence of
% seven variables derived from the historical binary map, feeding alongside the
% Moore neighborhood a probabilistic transition rule. The transition threshold
% and the neighborhood weight were chosen by grid search over windows prior to
% 2011, and the model thus fixed was then evaluated, without readjustment, over
% five shifted five-year windows (2011--2016 to 2015--2020), a protocol designed
% to distinguish stable behavior from overfitting to a single period.
%
% The contribution lies not in the WoE-CA coupling, well documented in the
% literature, but in the multi-temporal evaluation protocol, the
% variable-by-variable auditability and the full reproducibility of the
% processing on CPU, without GPU or proprietary software. Average performance
% reaches a Figure of Merit of 0.325, within the intermediate band of the
% multi-site spectrum documented by pontius2008comparing (six of thirteen
% applications with FoM below 0.15 and only one above 0.50), together with a
% Kappa coefficient of 0.526, an accuracy of 0.763 and an IoU of 0.661,
% sustained across all five windows.
```

### Abstract, cambio 3

Empieza con: «RGB captures from Google Earth's historical imagery tool were converted into…»

**Reemplaza las líneas 14–23** por:

```latex
The model achieved an average FoM of $0.325$, an average Kappa of $0.526$, an
accuracy of $0.763$ and an IoU of $0.661$ over the five evaluation windows. The
results show that the model reproduces part of the observed spatial growth
pattern and maintains comparable performance across the evaluated periods (FoM
between $0.240$ and $0.384$). The WoE-based transition function makes it
possible to identify the contribution of the variables used, and the processing
after image acquisition can be run with open-source tools on a CPU, without
requiring a GPU or proprietary software. The main contribution of this work is
the multi-temporal evaluation protocol applied to the WoE-CA model and its
reproducible implementation for the analysis of urban growth in the ZMQ.
}
```

### Abstract, cambio 4

Empieza con: «A strategy for the integration, calibration and validation of a…»

**Reemplaza las líneas 7–12** por:

```latex
% CAMBIO RM: El abstract se retradujo a partir del resumen vigente de la
% Dra. Lárraga; el anterior traducía la versión vieja del resumen. Se conserva
% comentado más abajo.
{\color{red}
Dispersed urban growth hinders territorial planning, especially when only
limited historical information is available to analyze the expansion of cities.
In this context, models are needed that make it possible to interpret the
drivers of growth, evaluate their performance over different periods and
reproduce the process with accessible data and tools. In this direction, the
objective of this work is to design and implement an urban growth model based on
cellular automata and Weights of Evidence (WoE), and to evaluate its performance
through a multi-temporal validation protocol. The Querétaro Metropolitan Area
(ZMQ, for its Spanish acronym) was used as a case study, for which an annual
historical series of binary urban/non-urban maps for 1984--2020 was built from
Google Earth historical imagery. The images were processed with local binary
patterns (LBP), K-Means clustering and a linear SVM classifier. Seven spatial
variables were computed from the resulting maps and used as evidence for the
WoE model. These variables, together with a Moore neighborhood, were
incorporated into a probabilistic cellular automaton to simulate urban
expansion. The model parameters were calibrated through a systematic search
over windows prior to 2011, and its performance was evaluated, without
readjustment, over five shifted five-year windows, from 2011--2016 to
2015--2020, using the Figure of Merit (FoM), Kappa and Intersection over Union
(IoU).
```

### Abstract, cambio 5

**Después de la línea 5**, pega:

```latex
\vspace*{-1.2cm}
```

## Cap. 2 Marco teórico — `cap02_marco_teorico/cap02_marco_teorico.tex`

9 cambios.

### Cap. 2 Marco teórico, cambio 1

Empieza con: «% CAMBIO RM: Se precisa qué suma entra a la sigmoide en el modelo de la tesis…»

**Reemplaza las líneas 313–315** por:

```latex
% CAMBIO RM: Se retiró la precisión sobre qué suma entra a la sigmoide en el
% modelo de la tesis; describe el modelo propio y ya figura en el Capítulo 4
% (cuarta decisión de la Sección "Definición del modelo").
```

### Cap. 2 Marco teórico, cambio 2

Empieza con: «\textcolor{blue}{En un autómata celular, la regla de transición determina la probabilidad …»

**Reemplaza la línea 311** por:

```latex
\textcolor{blue}{En un autómata celular, la regla de transición determina la probabilidad de que una celda cambie de estado a partir de las características de su entorno.} En este contexto, el WoE permite cuantificar la relación entre los factores geográficos y la ocurrencia del cambio, \textcolor{blue}{asignando un valor a cada celda de acuerdo con las evidencias asociadas a las variables consideradas.} La ecuación~\ref{eq:woe_sigmoid} \textcolor{blue}{transforma la combinación de estas evidencias en un valor de probabilidad en el intervalo $[0,1]$, que puede utilizarse como parte de la regla de transición del autómata.}
```

### Cap. 2 Marco teórico, cambio 3

Empieza con: «Los pesos $\omega_k$ suman uno por construcción y no son parámetros libres: se derivan de …»

**Reemplaza la línea 299** por:

```latex
Los pesos $\omega_k$ suman uno por construcción y no son parámetros libres: se derivan de los mismos datos históricos que producen los $W^+(b)$.

% CAMBIO RM: Se retiró la frase que indicaba que esta es la forma adoptada en
% la tesis; el Capítulo 4 ya lo establece al combinar los pesos.
```

### Cap. 2 Marco teórico, cambio 4

Empieza con: «El IV es no negativo y crece cuando las dos distribuciones se separan. La escala convencio…»

**Reemplaza la línea 290** por:

```latex
El IV es no negativo y crece cuando las dos distribuciones se separan. La escala convencional de interpretación, tomada de esa misma práctica \parencite{Siddiqi2006}, lo divide en cinco tramos: por debajo de $0{,}02$ la variable no tiene valor predictivo; entre $0{,}02$ y $0{,}10$ es débil; entre $0{,}10$ y $0{,}30$, medio; entre $0{,}30$ y $0{,}50$, fuerte; y por encima de $0{,}50$ la propia fuente lo califica de sospechoso, porque en su ámbito de origen un valor tan alto suele delatar que la variable ya contiene la respuesta. Esa última advertencia se formuló para el crédito, donde el predictor y el evento provienen de registros distintos\textcolor{red}{, y su lectura depende de que esa separación exista en la aplicación}.

% CAMBIO RM: La continuación de esta frase hablaba de los predictores del
% modelo propio; se trasladó al Capítulo 4, junto a la ponderación por IV.
```

### Cap. 2 Marco teórico, cambio 5

Empieza con: «La Figura~\ref{fig:concepto-woe} ilustra el procedimiento sobre una variable típica. La di…»

**Reemplaza la línea 276** por:

```latex
\textcolor{red}{Una variable típica ilustra el procedimiento.} La distancia al frente urbano se discretiza en intervalos; los intervalos próximos al frente acumulan pesos positivos, porque en ellos la transición se observa con más frecuencia de la que correspondería a su superficie, y los intervalos lejanos acumulan pesos negativos. \textcolor{red}{Lo relevante del ejemplo} es que la relación entre distancia y probabilidad no es lineal: los pesos caen con rapidez en los primeros intervalos y después se estabilizan, un comportamiento que un modelo con forma funcional impuesta reproduciría mal.
```

### Cap. 2 Marco teórico, cambio 6

Empieza con: «\begin{figure}[htbp]…»

**Reemplaza las líneas 230–274** por:

```latex
% CAMBIO RM: La figura del ejemplo de cálculo (fig:concepto-woe) se trasladó al
% Capítulo 4, junto a la regla de transición, porque su panel derecho usa los
% valores calibrados del modelo propio (omega, alpha y theta).
```

### Cap. 2 Marco teórico, cambio 7

Empieza con: «% CAMBIO RM: La ecuación de f es deliberadamente general. Se añade cómo se…»

**Reemplaza las líneas 187–190** por:

```latex
% CAMBIO RM: Por indicación de la Dra. Lárraga, el marco teórico solo define
% conceptos. Se retiró el párrafo que describía la forma concreta de f en el
% modelo de la tesis; esa forma ya está en el Capítulo 4 (regla de transición,
% ecuación eq:transicion).
```

### Cap. 2 Marco teórico, cambio 8

Empieza con: «\textcolor{red}{En el modelo de esta tesis, $f$ toma una forma explícita. La idoneidad WoE…»

**Reemplaza la línea 185** por:

```latex
% CAMBIO RM: Se retiró la remisión a los valores adoptados en el caso de
% estudio; el marco teórico solo presenta el concepto.
```

### Cap. 2 Marco teórico, cambio 9

Empieza con: «donde $W_{i,j}$ es la idoneidad espacial asignada por el método WoE (definida a continuaci…»

**Reemplaza la línea 183** por:

```latex
donde $W_{i,j}$ es la idoneidad espacial asignada por el método WoE (definida a continuación), $\Omega_{i,j}(r)$ es el efecto de vecindad, y $\varepsilon_{i,j}$ es un término estocástico que introduce variabilidad aleatoria en la simulación. La función $f$ combina estos tres factores según parámetros que se fijan en la calibración. Esta función es la que define la transición de una celda no-urbana a una urbana.
```

## Cap. 3 Estado del arte — `cap03_estado_del_arte/cap03_estado_del_arte.tex`

23 cambios.

### Cap. 3 Estado del arte, cambio 1

Empieza con: «El capítulo siguiente toma tres decisiones que esta revisión deja abiertas.…»

**Reemplaza las líneas 796–814** por:

```latex
\textcolor{red}{Los huecos que deja la revisión pueden enunciarse de forma
concreta. En la calibración, los marcos más difundidos ajustan varios
coeficientes en espacios de búsqueda muy amplios, con un costo computacional
alto y con varias parametrizaciones capaces de un ajuste equivalente. En la
validación, buena parte de las aplicaciones se evalúa en un único período, a
menudo con datos que también intervinieron en el ajuste, y el FoM depende de la
magnitud del cambio, lo que limita la comparación de cifras entre sitios. En la
interpretabilidad, los modelos de aprendizaje automático no expresan de manera
directa la contribución de cada variable a la transición de una celda. En la
reproducibilidad, varias plataformas LUCC se distribuyen sin costo, pero sin un
código fuente que permita verificar la implementación. A estos cuatro problemas
se suman las limitaciones del propio WoE: la independencia condicional, la
sensibilidad a la discretización, la redundancia entre variables
correlacionadas y la autocorrelación espacial, para la que la literatura no
ofrece una corrección estándar. En el contexto mexicano, por último, la Zona
Metropolitana de Querétaro ya fue modelada con autómatas celulares, con una
regla elegida y evaluada sobre el mismo mapa observado.}

\subsection{\textcolor{red}{Cómo atiende este trabajo los huecos identificados}}

\textcolor{red}{Frente a estos huecos, el modelo del
Capítulo~\ref{ch:modelo-crecimiento-urbano} toma un conjunto de decisiones de
diseño. La primera concierne a la calibración. Los pesos de evidencia y su
ponderación por Information Value se estiman una sola vez con las transiciones
de 1984 a 2010 y quedan fijos. La búsqueda recorre solo dos parámetros, el
umbral de transición $\theta$ y el peso de vecindad $\alpha$, de modo que la
ambigüedad entre parametrizaciones queda confinada a esos dos valores y no
altera la lectura de cada variable. Esa búsqueda se hace sobre ventanas
internas al período de estimación, y ninguna de las ventanas de evaluación
interviene en ella (Sección~\ref{ssec:calibracion-parametros}).}

\textcolor{red}{La segunda concierne a la validación. El modelo se evalúa, sin
reajuste, en cinco ventanas quinquenales desplazadas, de 2011--2016 a
2015--2020, con el FoM como métrica primaria, la IoU y el Kappa como
complemento y la descomposición del desacuerdo en cantidad y asignación como
diagnóstico, lo que permite evaluar fuera de muestra la simulación histórica y}
\textcolor{blue}{examinar la estabilidad temporal del desempeño}
\textcolor{red}{(R1). Como el FoM depende de la magnitud del cambio, el desempeño se lee como
estabilidad entre ventanas de un mismo sistema urbano y se reporta el rango
completo. La comparación con otros estudios se limita a situar ese rango en el
espectro documentado en la literatura.}

\textcolor{red}{La tercera concierne a la función de transición. Se construye
con Pesos de Evidencia porque cada peso corresponde a un intervalo concreto de
una variable y admite lectura espacial, que es lo que exige el requerimiento
R4. La suma de los pesos, ponderada por el Information Value, pasa por una
función logística; al resultado se le suma la fracción de vecinos urbanos
multiplicada por $\alpha$, y la celda transita si el total supera $\theta$ y un
sorteo lo confirma. La regla integra en un solo paso la evidencia espacial y la
vecindad. Su costo es que la cantidad de cambio queda determinada por $\theta$
y $\alpha$, sin una matriz markoviana, de modo que el desacuerdo de cantidad
depende de la calibración; el Capítulo~\ref{ch:resultados-analisis} lo
cuantifica.}

\textcolor{red}{La cuarta concierne a la reproducibilidad. El flujo completo,
desde la clasificación de las imágenes hasta la simulación y la evaluación, se
implementa con herramientas de código abierto (R3),}
\textcolor{blue}{distinguiendo la dependencia externa de Google Earth para la
obtención de las imágenes del carácter abierto del procesamiento y de la
implementación del modelo}\textcolor{red}{. Las imágenes son capturas de Google
Earth en escritorio, sin recurrir a Google Earth Engine.}

\textcolor{red}{Las limitaciones del WoE se atienden solo en parte. La
discretización se hace por cuantiles, de modo que los cortes dependen de la
distribución de cada variable y son los mismos en todas las ventanas de
evaluación. La ponderación por Information Value atenúa la redundancia entre
variables correlacionadas, sin eliminarla. La independencia condicional y la
autocorrelación espacial quedan sin corrección formal: el trabajo no aplica la
prueba ómnibus ni la prueba exacta, y la validación en varias ventanas examina
la estabilidad temporal sin sustituir esas pruebas. Aplicar la prueba ómnibus a
las siete variables de evidencia queda como trabajo futuro
(Sección~\ref{sec:trabajo-futuro}).}

\textcolor{red}{Frente al antecedente de Querétaro, la diferencia está en el
protocolo: los parámetros se fijan antes de observar las ventanas de
evaluación y el desempeño se reporta en cinco ventanas. La lectura de los
mapas simulados como escenarios de expansión (R2) se discute en el
Capítulo~\ref{ch:conclusiones}.}

% CAMBIO RM: Por indicación de la Dra. Lárraga, las secciones del capítulo
% terminan en el hueco que deja la literatura y las decisiones de la tesis se
% reúnen aquí. La síntesis se divide en dos partes: los huecos identificados y
% cómo los atiende el trabajo, incluidas las limitaciones que no resuelve. El
% párrafo anterior de tres decisiones se sustituyó por esta versión; sus dos
% fragmentos en azul se conservan literales en la segunda y la cuarta decisión.
```

### Cap. 3 Estado del arte, cambio 2

**Después de la línea 780**, pega:

```latex
\subsection{\textcolor{red}{Huecos identificados}}

```

### Cap. 3 Estado del arte, cambio 3

Empieza con: «ajuste puntual a las características de ese lustro. Conviene, entonces, un…»

**Reemplaza las líneas 605–609** por:

```latex
ajuste puntual a las características de ese lustro. \textcolor{red}{La revisión
sugiere, entonces, que un protocolo de validación informativo combine el FoM
como métrica primaria, la IoU y el Kappa como complemento, la descomposición de
Pontius como diagnóstico y varias ventanas temporales desplazadas del mismo
sistema urbano.}

% CAMBIO RM: Se retiró la remisión a la implementación de la tesis; la decisión
% se recoge en la síntesis del capítulo.
```

### Cap. 3 Estado del arte, cambio 4

Empieza con: «entrenamiento. Este trabajo opta por discretizar en cuantiles uniformes:…»

**Reemplaza las líneas 518–522** por:

```latex
entrenamiento. \textcolor{red}{La discretización por cuantiles renuncia a la
partición óptima en términos de IV o de contraste, a cambio de fijar los cortes
según la distribución de cada variable, con independencia del ajuste al período
de calibración.}

% CAMBIO RM: La decisión de la tesis de discretizar por cuantiles se trasladó a
% la síntesis del capítulo y al Capítulo 4; aquí queda la alternativa general.
```

### Cap. 3 Estado del arte, cambio 5

Empieza con: «aunque no sustituye una prueba formal de independencia condicional}. Aplicar…»

**Reemplaza las líneas 509–511** por:

```latex
aunque no sustituye una prueba formal de independencia condicional}.

% CAMBIO RM: Lo que la tesis hace frente a este supuesto (no aplica las
% pruebas y deja la ómnibus como trabajo pendiente) pasó a la síntesis del
% capítulo. El texto en azul de la Dra. Lárraga se conserva intacto.
```

### Cap. 3 Estado del arte, cambio 6

Empieza con: «la prueba exacta de \textcite{AgterbergCheng2002}. El presente trabajo no las…»

**Reemplaza las líneas 505–507** por:

```latex
la prueba exacta de \textcite{AgterbergCheng2002}. \textcolor{red}{Cuando
estas pruebas no se aplican, queda el contraste empírico con el cambio
observado;} una validación en varios períodos
```

### Cap. 3 Estado del arte, cambio 7

Empieza con: «% CAMBIO RM: Se nombra la regla que se adopta y se hace explícita su…»

**Reemplaza las líneas 488–491** por:

```latex
% CAMBIO RM: La descripción de la regla que adopta la tesis y la frase «El
% capítulo siguiente adopta esta formulación» pasaron a la síntesis del
% capítulo. Aquí queda solo la contrapartida general de esa alternativa.
```

### Cap. 3 Estado del arte, cambio 8

Empieza con: «transición.} \textcolor{red}{En la formulación que se adopta, la probabilidad…»

**Reemplaza las líneas 478–486** por:

```latex
transición.} \textcolor{red}{Sin un módulo de cantidad, el total de cambio
queda determinado por los parámetros de la propia regla, y el desacuerdo de
cantidad pasa a depender de su calibración.}
```

### Cap. 3 Estado del arte, cambio 9

Empieza con: «ruleta. Conviene tomar postura sobre la diferencia con el presente trabajo,…»

**Reemplaza las líneas 464–465** por:

```latex
ruleta. \textcolor{red}{Estos} marcos descomponen el problema en un módulo de
```

### Cap. 3 Estado del arte, cambio 10

**Después de la línea 438**, pega:

```latex
% CAMBIO RM: Se retiró la frase que indicaba qué suma usa la tesis; está en el
% Capítulo 4 y en la síntesis de este capítulo.

```

### Cap. 3 Estado del arte, cambio 11

Empieza con: «entre las distribuciones condicionales del evento. \textcolor{red}{En este…»

**Reemplaza las líneas 433–435** por:

```latex
entre las distribuciones condicionales del evento. La presente
```

### Cap. 3 Estado del arte, cambio 12

Empieza con: «(ecuación~\ref{eq:woe_sigmoid}). Para ponderar la contribución de cada variable se…»

**Reemplaza las líneas 429–430** por:

```latex
(ecuación~\ref{eq:woe_sigmoid}). \textcolor{red}{La contribución de cada
variable puede ponderarse con} el Information Value, un estadístico ajeno a la formulación
```

### Cap. 3 Estado del arte, cambio 13

Empieza con: «la asociación entre dicha clase y la ocurrencia del evento de interés, en este…»

**Reemplaza las líneas 425–426** por:

```latex
la asociación entre dicha clase y la ocurrencia del evento de interés,
\textcolor{red}{en el modelado urbano}, la transición de suelo no urbano a urbano. El peso total por celda es
```

### Cap. 3 Estado del arte, cambio 14

Empieza con: «el estudio de las ciudades. En este caso se usan capturas de Google Earth en…»

**Reemplaza las líneas 400–402** por:

```latex
el estudio de las ciudades.

% CAMBIO RM: Las dos frases sobre la fuente de imágenes de la tesis y su
% implementación pasaron a la síntesis del capítulo. «De acceso libre» se
% cambió por «de consulta gratuita», como en el resto de la tesis.
```

### Cap. 3 Estado del arte, cambio 15

Empieza con: «Conviene, entonces, una implementación en código abierto sobre imágenes de…»

**Reemplaza las líneas 389–390** por:

```latex
\textcolor{red}{La revisión deja abierta, entonces, la necesidad de
implementaciones en código abierto sobre imágenes de archivo de consulta
gratuita.} \textcolor{blue}{En este punto debe distinguirse entre
```

### Cap. 3 Estado del arte, cambio 16

Empieza con: «El método de Pesos de Evidencia (WoE) provee esa posibilidad de descomposición.…»

**Reemplaza las líneas 344–348** por:

```latex
El método de Pesos de Evidencia (WoE) provee esa posibilidad de
descomposición\textcolor{red}{, como se revisa en la
Sección~\ref{sec:woe-literatura}}.

% CAMBIO RM: La justificación de la elección de WoE en la tesis (requerimiento
% R4) se trasladó a la síntesis del capítulo.
```

### Cap. 3 Estado del arte, cambio 17

Empieza con: «puede transmitir una unicidad que los datos no necesariamente respaldan. El…»

**Reemplaza las líneas 298–302** por:

```latex
puede transmitir una unicidad que los datos no necesariamente respaldan.
\textcolor{red}{De ello se sigue que la comparación más informativa es la del
desempeño de un mismo modelo entre varias ventanas de un mismo sistema urbano,
con el rango completo de valores a la vista.}

% CAMBIO RM: La frase sobre lo que hace el capítulo siguiente con el FoM se
% sustituyó por el criterio que se desprende de la revisión; la decisión de la
% tesis se recoge en la síntesis del capítulo.
```

### Cap. 3 Estado del arte, cambio 18

Empieza con: «% JimenezLopez2018.…»

**Reemplaza la línea 277** por:

```latex
% JimenezLopez2018. La remisión al protocolo de la tesis y la frase sobre cómo
% se eligen aquí los parámetros pasaron a la síntesis del capítulo.
```

### Cap. 3 Estado del arte, cambio 19

Empieza con: «reportado mezcla ajuste y prueba. Por ello, en este trabajo los parámetros del…»

**Reemplaza las líneas 271–272** por:

```latex
reportado mezcla ajuste y prueba.}
```

### Cap. 3 Estado del arte, cambio 20

Empieza con: «examinar la estabilidad del desempeño desde diferentes perspectivas}. El…»

**Reemplaza las líneas 267–268** por:

```latex
examinar la estabilidad del desempeño desde diferentes perspectivas}.
```

### Cap. 3 Estado del arte, cambio 21

Empieza con: «% CAMBIO RM: El final original decía que "cada configuración candidata produce…»

**Reemplaza las líneas 215–218** por:

```latex
% CAMBIO RM: Por indicación de la Dra. Lárraga, esta sección termina en el
% hueco que deja la literatura. La descripción de cómo la tesis reduce la
% búsqueda a theta y alpha sobre ventanas internas se trasladó a la síntesis
% del capítulo (Sección sec:sintesis-estado-arte).
```

### Cap. 3 Estado del arte, cambio 22

Empieza con: «Esa literatura deja abierto un recorte distinto: calibrar pocos parámetros que…»

**Reemplaza las líneas 202–213** por:

```latex
\textcolor{red}{Esa literatura deja abierta una vía con menos parámetros
libres: estimar la evidencia espacial directamente de los datos históricos,
calibrar solo los parámetros que la combinan con la vecindad y elegirlos con
datos distintos de los que sirven para evaluar el modelo.}
```

### Cap. 3 Estado del arte, cambio 23

Empieza con: «de evaluación independientes del ajuste.} Esta combinación de costo,…»

**Reemplaza las líneas 195–197** por:

```latex
de evaluación independientes del ajuste.} \textcolor{red}{Esta combinación de
costo, dependencia de trayectoria y equifinalidad hace recomendable evaluar el
desempeño en más de una ventana temporal.} Una sola
```

## Cap. 4 Modelo — `cap05_modelo_crecimiento_urbano/cap05_modelo_crecimiento_urbano.tex`

3 cambios.

### Cap. 4 Modelo, cambio 1

Empieza con: «donde $\sigma(\cdot)$ es la función logística; $\mathrm{WoE}_k(i,j)$ es el peso de evidenc…»

**Reemplaza la línea 731** por:

```latex
donde $\sigma(\cdot)$ es la función logística; $\mathrm{WoE}_k(i,j)$ es el peso de evidencia de la $k$-ésima variable espacial, ponderado por su \textit{Information Value} normalizado $\omega_k$ (de modo que las variables más discriminativas pesan más); y $N_{i,j} \in [0,1]$ es la influencia del vecindario, definida como la fracción de vecinos urbanos en la vecindad de Moore (las ocho celdas adyacentes), con peso vecinal $\alpha$. Nótese que la idoneidad WoE se transforma primero con la sigmoide y la vecindad se suma después, acotando el resultado a $[0,1]$. La proximidad al frente urbano no aparece como un término separado: la introduce la propia variable de distancia, la de mayor \textit{Information Value}, dentro del campo WoE. En la implementación validada no se incluye un término adicional de decaimiento espacial (por ejemplo, una ponderación por distancia de Chebyshev sobre el vecindario); la distancia al frente urbano actúa únicamente como una variable más dentro del campo WoE, y la vecindad se incorpora como la fracción de vecinos urbanos en la vecindad de Moore. La celda transita a urbana en el paso siguiente solo si $P_{i,j}$ supera el umbral $\theta$ y si, además, un sorteo aleatorio $u \sim \mathcal{U}(0,1)$ lo confirma ($u < P_{i,j}$). La configuración validada en este trabajo fija \textcolor{red}{$\theta = 0{,}85$ y $\alpha = 0{,}60$ (Sección~\ref{ssec:calibracion-parametros})}. \textcolor{red}{La Figura~\ref{fig:concepto-woe} sigue el cálculo completo para una celda candidata, desde los pesos de un intervalo hasta la comparación con el umbral.}

\begin{figure}[htbp]
  \centering
  \shorthandoff{<>}%
  \resizebox{\linewidth}{!}{%
  \begin{tikzpicture}[x=1cm,y=1cm,
    caja/.style={draw=black, line width=0.5pt, fill=white, rounded corners=1.5pt,
      font=\scriptsize, align=left, text width=6.0cm, inner sep=3pt, anchor=north west},
    flecha/.style={-{Stealth[length=1.8mm]}, line width=0.6pt}]
  \node[font=\small\bfseries, anchor=south] at (7.0,0.45) {Pesos de evidencia y regla de transición: ejemplo para una celda};
  % ---- panel izquierdo: tabla de conteos ----
  \node[font=\footnotesize\bfseries, anchor=north west] at (0,0) {1. Peso de cada intervalo (distancia al frente)};
  \node[font=\scriptsize, anchor=north west, inner sep=0pt] at (0,-0.55) {%
    \renewcommand{\arraystretch}{1.25}%
    \begin{tabular}{@{}lrrr@{}}
      \textbf{Intervalo} & \textbf{transita} & \textbf{no transita} & \textbf{$W^{+}$}\\\hline
      $0$--$100$~m & 847 & 312 & \textcolor{Acerto}{$+2{,}39$}\\
      $100$--$300$~m & 423 & 621 & \textcolor{Acerto}{$+1{,}01$}\\
      $>300$~m & 89 & 4\,521 & \textcolor{Transicion}{$-2{,}54$}\\\hline
      total & 1\,359 & 5\,454 & \\
    \end{tabular}};
  \node[font=\scriptsize, anchor=north west, text width=6.4cm, align=left] at (0,-2.85)
    {$W^{+}=\ln\dfrac{\text{fracción de las que transitan}}{\text{fracción de las que no transitan}}$;\\[2pt]
     cerca del frente el peso es positivo (favorece la transición) y lejos es negativo.};
  % ---- panel derecho: una celda ----
  \node[font=\footnotesize\bfseries, anchor=north west] at (7.4,0) {2. Una celda candidata};
  \node[caja] (c1) at (7.4,-0.55) {%
    distancia: $\omega=0{,}28$, $W=+2{,}39$\\
    densidad $3{\times}3$: $\omega=0{,}25$, $W=+0{,}91$\\
    otras cinco variables: $\sum\omega_k W_k=+0{,}10$\\[2pt]
    $\sum_k\omega_k W_k = 0{,}67+0{,}23+0{,}10\approx +1{,}00$};
  \node[caja, below=0.22cm of c1.south west, anchor=north west] (c2) {%
    aptitud: $\sigma(1{,}00)=0{,}73$\\
    vecindad: 3 de 8 vecinos urbanos, $\alpha N=0{,}60\cdot\tfrac{3}{8}=0{,}23$};
  \node[caja, below=0.22cm of c2.south west, anchor=north west, draw=Transicion, line width=0.8pt] (c3) {%
    $P=0{,}73+0{,}23=0{,}96>\theta=0{,}85$: la celda es candidata\\
    y transita si el sorteo da $u<0{,}96$.\\[1pt]
    \textit{Solo con la aptitud ($0{,}73$) no superaría el umbral.}};
  \draw[flecha] (c1.south) ++(-2.2,0) -- ++(0,-0.22);
  \draw[flecha] (c2.south) ++(-2.2,0) -- ++(0,-0.22);
  \end{tikzpicture}}
  \caption{Ejemplo del cálculo de WoE: la variable \textit{distancia al frente urbano} acumula pesos positivos en bins próximos y negativos en bins lejanos. La suma ponderada pasa por la sigmoide \textcolor{red}{y, sumada a la influencia de la vecindad,} se compara con un umbral de transición $\theta$. \textit{\textcolor{red}{Elaboración propia}, con base en la formulación de \textcite{BonhamCarter1994}\textcolor{red}{; los conteos son ilustrativos y $\omega$, $\alpha$ y $\theta$ son los del modelo calibrado}.}}
  \label{fig:concepto-woe}
\end{figure}
% CAMBIO RM: Figura trasladada desde el Capítulo 2 por indicación de la
% Dra. Lárraga: ilustra la regla del modelo propio con sus valores calibrados.

```

### Cap. 4 Modelo, cambio 2

**Después de la línea 635**, pega:

```latex
\textcolor{red}{La escala convencional del \textit{Information Value} (Sección~\ref{ssec:information-value}) califica de sospechoso un valor mayor que $0{,}50$, porque en el crédito el predictor y el evento provienen de registros distintos y un valor tan alto suele delatar que la variable ya contiene la respuesta. En este modelo los predictores se derivan del mismo mapa binario que el objetivo, y que la urbanización ocurra junto a lo ya urbano es el mecanismo que se quiere capturar, no un artefacto del muestreo. El alcance de esta lectura en los pesos estimados se precisa en la Sección~\ref{ssec:alcance-iv}.}

% CAMBIO RM: Texto trasladado desde el Capítulo 2 (Information Value), donde
% describía los predictores del modelo propio.

```

### Cap. 4 Modelo, cambio 3

Empieza con: «La primera decisión del modelo es cómo cuantificar la evidencia que vincula la configuraci…»

**Reemplaza la línea 616** por:

```latex
La primera decisión del modelo es cómo cuantificar la evidencia que vincula la configuración espacial de una celda con su probabilidad de transición. La vía más común en la literatura es la regresión logística, pero conlleva un supuesto fuerte: que el logit de la probabilidad es una función lineal de las variables explicativas. Las relaciones espaciales del crecimiento urbano no son lineales; la probabilidad de urbanización cae de forma abrupta en los primeros cientos de metros desde el frente urbano y luego se aplana. Por eso se elige el método de Pesos de Evidencia (WoE, ecuación~\ref{eq:woe} del Capítulo~\ref{ch:marco-teorico}), que no impone ninguna forma funcional: discretiza cada variable en intervalos y estima un peso bayesiano independiente para cada uno, capturando la curvatura real de la relación. \textcolor{red}{Los intervalos se fijan por cuantiles de cada variable, con un máximo de diez por variable; así, los cortes dependen de la distribución de la variable, con independencia del ajuste, y son los mismos en todas las ventanas de evaluación.} Como ventaja adicional, cada peso es interpretable en términos geográficos, lo que satisface el requisito de transparencia que motivó este trabajo y que los enfoques de caja negra no ofrecen.
```

## Cap. 5 Resultados — `cap06_resultados_analisis/cap06_resultados_analisis.tex`

3 cambios.

### Cap. 5 Resultados, cambio 1

**Después de la línea 565**, pega:

```latex
\textcolor{red}{En conjunto, tres de los cuatro criterios se satisfacen y C2 se satisface de forma parcial. La hipótesis general queda apoyada con esa reserva: el modelo reproduce el patrón espacial del crecimiento en las cinco ventanas, con una función de transición interpretable y un flujo reproducible, aunque su desempeño numérico varía con la dinámica de cada período.}

% CAMBIO RM: Se añade el veredicto global de la hipótesis, que esta sección
% evaluaba solo criterio por criterio, y se deja explícita la reserva de C2
% para que coincida con el Capítulo 6.

```

### Cap. 5 Resultados, cambio 2

Empieza con: «El promedio, sin embargo, oculta que la componente dominante no es la misma en todas las v…»

**Reemplaza la línea 482** por:

```latex
El promedio, sin embargo, oculta que la componente dominante no es la misma en todas las ventanas, y la división coincide con la \textcolor{red}{magnitud} del crecimiento observado. En las dos ventanas de menor crecimiento, la cantidad domina con holgura: 2011--2016 concentra en cantidad el $86{,}2\%$ de su desacuerdo, algo esperable porque es la única ventana con crecimiento neto observado negativo, de modo que cualquier crecimiento simulado es exceso por definición, y en 2014--2019 la proporción llega al $77{,}5\%$. En las tres ventanas de \textcolor{red}{mayor} crecimiento observado, en cambio, la cantidad baja al rango $42{,}7$--$48{,}6\%$ y el desacuerdo de asignación pasa a ser el mayoritario. La implicación metodológica se desplaza según el régimen: cuando la señal de cambio es débil, el margen de mejora está en estimar la magnitud del crecimiento; cuando es clara, el límite lo marca la regla que decide dónde ocurre.
```

### Cap. 5 Resultados, cambio 3

Empieza con: «La variación entre períodos refleja diferencias en la dinámica observada. El período 2011-…»

**Reemplaza la línea 242** por:

```latex
La variación entre períodos refleja diferencias en la dinámica observada. El período 2011--2016 muestra el FoM más bajo ($0{,}240$): el área urbana clasificada en 2016 \textit{decreció} levemente respecto a 2011, una anomalía atribuible a la variabilidad de la clasificación binaria RGB año a año, más que a una contracción urbana real. Los períodos 2012--2017, 2013--2018 y 2015--2020, \textcolor{red}{los de mayor crecimiento neto observado (entre $27{,}9$ y $33{,}5\%$, frente a $9{,}5\%$ en 2014--2019)}, exhiben FoM entre $0{,}345$ y $0{,}384$, con mayor concordancia entre el modelo y los mapas observados. La Tabla~\ref{tab:literature_comparison} sitúa estos valores frente al espectro documentado en la literatura.
```

## Cap. 6 Conclusiones — `cap07_conclusiones/cap07_conclusiones.tex`

6 cambios.

### Cap. 6 Conclusiones, cambio 1

Empieza con: «Dos ajustes del procedimiento de estimación son de implementación inmediata. \textcolor{re…»

**Reemplaza la línea 161** por:

```latex
Dos ajustes del procedimiento de estimación son de implementación inmediata. \textcolor{red}{Ambos depuran la interpretación de los \textit{Information Value}, pero ninguno es neutro para los resultados: los dos modifican los pesos y la ponderación entre variables, de modo que exigirían repetir la calibración y la evaluación.} El primero es restringir el muestreo a las celdas elegibles, es decir, a las no urbanas en el año inicial de cada par: con ello desaparecen los \textit{bins} sin transiciones observadas, los IV vuelven a la escala convencional de interpretación y la jerarquía entre variables queda libre del efecto trivial que hoy la condiciona (Sección~\ref{ssec:alcance-iv}). El segundo es emparejar cada configuración espacial anual con su transición correspondiente, en lugar de promediar las variables sobre el período de entrenamiento. \textcolor{red}{Con ello cada transición se explicaría solo con la configuración del año en que ocurre, mientras que el promedio actual incluye también años posteriores a ella.} Ese cambio mostraría además si los pesos son estables entre décadas o si el régimen espacial de la zona cambió a lo largo de los 26 períodos. \textcolor{red}{A estos dos ajustes se suma un diagnóstico que el modelo no incorpora: aplicar a las siete variables de evidencia la prueba ómnibus de independencia condicional de \textcite{BonhamCarter1994}, que mediría cuánto de la evidencia se cuenta más de una vez (Capítulo~\ref{ch:estado-del-arte}).}
```

### Cap. 6 Conclusiones, cambio 2

Empieza con: «El uso del Weight of Evidence y del autómata celular no constituye, por sí mismo, una nove…»

**Reemplaza la línea 84** por:

```latex
El uso del Weight of Evidence y del autómata celular no constituye, por sí mismo, una novedad metodológica, pues ambos están ampliamente documentados en la literatura. La aportación central de esta investigación reside en el \textbf{protocolo de evaluación}: el acoplamiento WoE-AC \textbf{calibrado y validado de forma reproducible sobre cinco ventanas temporales desplazadas}, \textcolor{red}{con código abierto}, aplicado a una ciudad intermedia mexicana. El WoE se incorpora como un campo probabilístico continuo ponderado por el \textit{Information Value}, pero el valor diferenciador está en poder distinguir un desempeño estable de un ajuste a un único período. De esa aportación principal se derivan varias contribuciones específicas. La primera es el \textbf{protocolo de validación quinquenal múltiple}, de carácter \textit{metodológico}: la evaluación sobre cinco períodos desplazados (2011--2020) con métricas píxel a píxel (FoM, Kappa, IoU), la descomposición del error de Pontius entre desacuerdo de cantidad y de asignación y un análisis visual comparativo de patrones morfológicos evalúa la estabilidad temporal del modelo con mayor rigor que la validación en un único período, y es la aportación que sostiene las demás. La segunda, de orden \textit{práctico}, es la \textbf{reproducibilidad abierta del pipeline}: un sistema modular en Python que transforma capturas RGB en mapas binarios y en proyecciones de crecimiento urbano (clasificación híbrida, WoE, AC y evaluación), ejecutable con herramientas de código abierto, documentado y publicado. \textcolor{red}{El protocolo se apoya, además, en una pieza instrumental que no se cuenta como aportación propia:} el \textbf{campo WoE continuo ponderado por Information Value}, la implementación concreta que hace auditable la regla de transición, donde la idoneidad WoE no es un mapa estático, sino un campo probabilístico ponderado por el \textit{Information Value} normalizado de cada variable, que se recalcula en cada paso sobre el estado actualizado del mapa y se acopla a la vecindad de Moore. Ese recálculo acarrea una propiedad que conviene consignar como \textcolor{red}{condición de validez} y no como consecuencia de la escasez de datos: al derivarse las siete variables del propio mapa urbano del año simulado, ninguna incorpora\textcolor{red}{, durante la simulación,} información posterior al momento que se predice, y el modelo no puede apoyarse en capas cuya versión disponible sea en parte resultado del crecimiento que se busca explicar. La \textcolor{red}{tercera}, de naturaleza \textit{aplicada}, es la \textbf{aplicación a una ciudad intermedia mexicana}: la adaptación del enfoque WoE-AC al contexto de la Zona Metropolitana de Querétaro. Conviene acotarla con precisión, porque existe un antecedente de simulación con autómatas celulares para esta misma ciudad en el período 2003--2017 \parencite{JimenezLopez2018}; lo que aquí se añade no es el caso de estudio sino el esquema de evaluación. En ese antecedente la regla de transición se elige entre las 256 posibles por su ajuste al mapa observado de 2017 y es sobre ese mismo mapa que se reporta su Kappa e índice de Jaccard, de modo que las cifras publicadas corresponden al máximo de una búsqueda y no a una prueba independiente. En este trabajo los pesos WoE, su ponderación por \textit{Information Value}\textcolor{red}{,} el umbral de transición \textcolor{red}{y el peso de vecindad} quedan fijados con datos de 1984 a 2010 y se evalúan después, sin reajuste, sobre cinco ventanas quinquenales de 2011 a 2020, reportando el rango completo de resultados y no el más favorable. La \textcolor{red}{cuarta}, de carácter \textit{empírico}, es la \textbf{serie temporal de Querétaro}: un conjunto de datos procesado de 37 años (1984--2020) de mapas binarios urbano/no-urbano, que puede utilizarse como base para investigaciones futuras en la región. La \textcolor{red}{quinta}, igualmente \textit{empírica}, es el \textbf{posicionamiento frente a la literatura}: el contraste del modelo WoE-AC con estudios internacionales mediante métricas estandarizadas (FoM, IoU), que contextualiza los resultados en el espectro empírico multi-sitio de \textcite{pontius2008comparing}.

% CAMBIO RM: La ponderación por Information Value deja de contarse entre las
% aportaciones, en línea con el Capítulo 1, que la presenta como decisión
% instrumental; se describe como la pieza que hace auditable la regla y se
% renumeran las aportaciones restantes.
```

### Cap. 6 Conclusiones, cambio 3

**Después de la línea 80**, pega:

```latex
% CAMBIO RM: La columna «Enunciado (Cap. 1)» reproduce ahora literalmente los
% criterios C1 a C4 del Capítulo 1; antes aparecían parafraseados y C2 y C4
% decían algo distinto de lo fijado por anticipado.
```

### Cap. 6 Conclusiones, cambio 4

Empieza con: «C1 & \textit{Interpretabilidad.} La regla de transición WoE ponderada por IV produce un mo…»

**Reemplaza las líneas 76–79** por:

```latex
C1 & \textit{Interpretabilidad.} \textcolor{red}{La regla de transición, con los pesos de evidencia ponderados por su Information Value, produce un modelo auditable variable por variable, en el que la distancia al frente urbano y la densidad de vecindad concentran el mayor peso, por encima de la fragmentación y el gradiente.} & Satisfecho & IV concentrado en distancia al área urbana y densidad de vecindad ($0{,}782$ del peso normalizado), por encima de fragmentación y gradiente. El orden entre distancia y densidad depende del tratamiento de las celdas no elegibles (Cap.~\ref{ch:resultados-analisis}). \\
C2 & \textit{Comportamiento temporal.} \textcolor{red}{El desempeño se mantiene dentro de una misma banda en las cinco ventanas quinquenales desplazadas, y las diferencias entre ellas se explican por la dinámica observada en cada período y no por un fallo de calibración.} & Parcial & FoM entre $0{,}240$ y $0{,}384$; el mínimo (2011--2016) coincide con crecimiento neto observado negativo. El patrón morfológico se sostiene; las métricas píxel a píxel varían. \\
C3 & \textit{Reproducibilidad.} \textcolor{red}{El procesamiento posterior a la adquisición de las capturas se ejecuta íntegramente con herramientas de código abierto en una CPU estándar, sin GPU ni software propietario.} & Satisfecho & Implementación íntegra en Python (NumPy, SciPy, scikit-learn); la adquisición de las capturas depende del visor de Google Earth; cómputo del orden de tres horas en CPU (Sección~\ref{sec:accesibilidad}). \\
C4 & \textit{Desempeño comparable.} \textcolor{red}{El Figure of Merit obtenido se sitúa dentro del espectro empírico multi-sitio que documenta la literatura, acompañado de valores de Kappa e Intersection over Union que admitan lectura frente a sus respectivas escalas de referencia.} & Satisfecho & FoM promedio $0{,}325$, Kappa $0{,}526$ e IoU $0{,}661$. El contraste con caja negra es cualitativo (sitios y horizontes distintos). \\
```

### Cap. 6 Conclusiones, cambio 5

Empieza con: «\textbf{Hipótesis general.} Un modelo de crecimiento urbano basado en autómatas celulares …»

**Reemplaza la línea 62** por:

```latex
\textbf{Hipótesis general.} Un modelo de crecimiento urbano basado en autómatas celulares y Pesos de Evidencia, calibrado y evaluado mediante un protocolo de validación multitemporal, permite reproducir el patrón espacial del crecimiento urbano observado en la Zona Metropolitana de Querétaro durante el período 1984--2020, manteniendo una función de transición interpretable a partir de las variables espaciales utilizadas y un flujo de procesamiento reproducible a partir de las imágenes históricas disponibles. \textbf{Estado: apoyada por la evidencia}\textcolor{red}{\textbf{, con el criterio C2 satisfecho de forma parcial.} El patrón morfológico se reproduce en las cinco ventanas, aunque el FoM varía entre $0{,}240$ y $0{,}384$ según la dinámica de cada período.} El modelo alcanzó en las cinco ventanas quinquenales desplazadas un FoM promedio de $0{,}325$, dentro de la banda intermedia documentada por \textcite{pontius2008comparing}, y un Kappa promedio de $0{,}526$, acuerdo moderado en la escala de \textcite{LandisKoch1977}, con implementación íntegra en Python \textcolor{red}{y herramientas de código abierto}. La Tabla~\ref{tab:hypothesis_validation} recoge el veredicto de los cuatro criterios.
```

### Cap. 6 Conclusiones, cambio 6

Empieza con: «Los cinco objetivos específicos planteados en el Capítulo~\ref{ch:introduccion} se cumplie…»

**Reemplaza la línea 42** por:

```latex
Los cinco objetivos específicos planteados en el Capítulo~\ref{ch:introduccion} se cumplieron del modo que se describe a continuación. El primero (OE1) condujo a la \textbf{serie histórica binaria}: se construyó la serie anual 1984--2020 (37 mapas urbano/no-urbano) a partir de imágenes de archivo de Google Earth, con una cadena de clasificación LBP+K-Means+SVM que no requiere verdad terreno etiquetada y que se implementó íntegramente sobre bibliotecas de código abierto, de modo que el procesamiento es reejecutable por un tercero (Sección~\ref{sec:preprocesamiento}). El segundo (OE2) abordó las \textbf{variables espaciales y el WoE}: se calcularon las siete variables derivadas del mapa binario y se adoptó el WoE como función de transición por su descomposición auditable mediante el \textit{Information Value} (Capítulos~\ref{ch:modelo-crecimiento-urbano} y \ref{ch:resultados-analisis}). El tercero (OE3) corresponde al \textbf{diseño e implementación del WoE-AC}: se implementó el autómata celular que simula la transición no urbano $\rightarrow$ urbano sobre la ZMQ (Capítulo~\ref{ch:modelo-crecimiento-urbano}). El cuarto (OE4) cubrió la \textbf{calibración y el protocolo de validación}: se calibraron umbral y peso de vecindad por búsqueda en cuadrícula y se validó el modelo en cinco ventanas quinquenales desplazadas con FoM, Kappa e IoU (Capítulo~\ref{ch:resultados-analisis}). El quinto (OE5) atendió \textcolor{red}{el \textbf{comportamiento temporal} y el \textbf{posicionamiento}: se analizó el comportamiento del desempeño a lo largo de los cinco períodos y se contrastó el FoM con el espectro multi-sitio de \textcite{pontius2008comparing} y con \textcite{Tang2024}. El contraste con los modelos de caja negra fue cualitativo, porque se refieren a sitios, clases y horizontes distintos, y en la literatura revisada no se dispuso de valores de IoU comparables} (Capítulos~\ref{ch:resultados-analisis} y \ref{ch:estado-del-arte}).

% CAMBIO RM: OE5 se reformula con el verbo del Capítulo 1 («analiza el
% comportamiento») y se precisa qué comparación se hizo realmente: cuantitativa
% con Pontius y Tang, cualitativa con caja negra, sin IoU de la literatura.
```
