# Resultado rev2 — Lote 04 de 6 (11 referencias)

Revalidación literal del texto **tal como está hoy** en los `.tex`. No se arrastran hallazgos de rondas previas.

Fuentes consultadas por referencia (se indica en cada caso): API de Crossref, API de OpenAlex, API de Semantic Scholar, páginas de editor (ScienceDirect, MIT Press, SciELO, Redalyc), y **PDF completo abierto** cuando fue posible.

**Metadatos: las 11 referencias existen y sus metadatos son correctos.** No hay ninguna REFERENCIA INEXISTENTE en este lote.

---

## Almeida2003

**Metadatos: CORRECTOS.** Crossref (DOI `10.1016/S0198-9715(02)00042-X`) devuelve exactamente: de Almeida, Batty, Vieira Monteiro, Câmara, Soares-Filho, Cerqueira, Pennachin; *Computers, Environment and Urban Systems* **27**(5):481–509, 2003. Los siete autores, el volumen, el número y las páginas del `.bib` coinciden uno a uno.

**Contenido verificado con PDF completo abierto** (copia de autor en complexcity.info, primera página y resumen idénticos al registro de Crossref) y con el resumen depositado en Zenodo y en el repositorio del INPE.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03:275` | En las aplicaciones urbanas de Almeida2003 la ruta WoE «se desarrolló y estimó empíricamente» | **CORRECTA** |

Evidencia textual del artículo: «we propose a structure for simulating urban change based on estimating land use transitions using elementary probabilistic methods which draw their inspiration from Bayes' theory and the related *'weights of evidence'* approach»; palabras clave declaradas: «Transition probabilities; Bayesian methods; 'Weights of evidence'». El cuerpo confirma que esas probabilidades se insertan en «a more general CA framework called DINAMICA developed at the Center for [Remote Sensing, UFMG]», con vecindad de Moore de ocho celdas, aplicado a Bauru (oeste del estado de São Paulo) para 1979–1988. El propio subtítulo del artículo es «empirical development and estimation», de modo que el verbo «se desarrolló y estimó empíricamente» es literalmente lo que el artículo dice de sí mismo. Sin problemas.

---

## Batty2005

**Metadatos: CORRECTOS.** Página oficial de MIT Press para el ISBN 9780262025836: *Cities and Complexity — Understanding Cities with Cellular Automata, Agent-Based Models, and Fractals*, Michael Batty, The MIT Press, fecha de publicación 9 de septiembre de 2005, tapa dura, ISBN-10 0262025833. El catálogo bibliotecario consultado confirma pie de imprenta «Cambridge; London: MIT, c2005», coherente con el campo `address = {Cambridge, MA}`.

**Contenido verificado con la descripción editorial de MIT Press y con el índice de capítulos publicado por el propio autor** (sitio del libro en CASA-UCL). No se consultó el interior del libro.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03:31` | El libro «examina en un mismo tratamiento los autómatas celulares, los modelos basados en agentes y las descripciones fractales» | **CORRECTA** |

Evidencia: la descripción editorial dice «Batty begins with models based on cellular automata (CA) […] He then introduces agent-based models (ABM) […] can combine with new forms of geometry associated with fractal patterns and chaotic dynamics». El índice publicado por Batty confirma los tres bloques en un mismo volumen (cap. 2–4 CA; cap. 5–6 agentes; cap. 11 «The Fractal City»). La atribución de que el marco «ciudad como sistema complejo cuyo patrón macroscópico emerge de interacciones locales» quedó sistematizado ahí es también coherente con el cap. 1 «Urban Change: Complexity and Emergence» y con «a comprehensive view of urban dynamics in the context of complexity theory».

---

## ClarkeHoppen1997

**Metadatos: CORRECTOS.** Crossref (DOI `10.1068/b240247`): Clarke, Hoppen, Gaydos; *Environment and Planning B: Planning and Design* **24**(2):247–261, abril de 1997. Confirmado además por la ficha del USGS Publications Warehouse (Index ID 70019926). El `.bib` coincide en todo.

**Contenido verificado con el PDF completo abierto** alojado por el propio Keith Clarke (`people.geog.ucsb.edu/~kclarke/Papers/clarkehoppengaydos.pdf`), leído en su totalidad.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03:97` | El resultado «no solo depende de los valores iniciales, sino también de la trayectoria que se sigue, lo que da lugar a un sistema muy sensible a cambios» | **SOBREEXTENDIDA** |

Qué sí dice el artículo, textualmente:
- «In self-modifying cellular automata, the rules are allowed to change as the system grows or changes (that is, by a feedback mechanism).»
- «when the absolute amount of growth in any year exceeds a critical value, the DIFFUSION, SPREAD, and BREED factors are increased by a multiplier greater than one […] when the system growth rate falls below another critical value, [esos factores] are decreased by a multiplier less than one.»
- «Some of the factors are more system sensitive than others and, in a system this complicated, a set of […]»

Es decir: la **primera** mitad de la oración de la tesis (la automodificación de los coeficientes durante la corrida en respuesta a la tasa de crecimiento acumulada, descrita en la oración inmediatamente anterior del mismo párrafo) está sostenida sin reservas por la fuente. Lo que la fuente **no** enuncia es la consecuencia interpretativa: no aparece el término «path dependence» ni equivalente, no hay discusión de dependencia de la trayectoria frente a las condiciones iniciales, y la única mención de sensibilidad es la frase citada arriba, que habla de que *algunos factores son más sensibles que otros* — una afirmación más débil y de otro objeto. La inferencia es razonable, pero hoy el `\parencite` cuelga de la oración interpretativa y la presenta como algo que el artículo afirma.

**Redacción corregida propuesta** (reatribuye el mecanismo a la fuente y deja la consecuencia como lectura propia):

> Aunado a ello, SLEUTH es un autómata celular que se automodifica: cuando el crecimiento anual supera un valor crítico alto, los coeficientes de difusión, *breed* y *spread* se multiplican por un factor mayor que uno, y cuando la tasa cae por debajo de un valor crítico bajo se multiplican por un factor menor que uno; la resistencia a la pendiente y la gravedad de camino se ajustan, a su vez, conforme se agota el suelo desarrollable y se extiende la red vial \parencite{ClarkeHoppen1997}. Ese mecanismo de retroalimentación implica que el resultado de una corrida no depende solo de los valores iniciales de los coeficientes, sino de la secuencia de tasas de crecimiento que la propia simulación genera.

**Advertencia adicional derivada de esta misma fuente** (no es un hallazgo contra `ClarkeHoppen1997`, porque el texto no la cita ahí, pero afecta a `cap03:87`): el artículo de 1997 dice explícitamente «The values for DIFFUSION, BREED, SPREAD, and SLOPE-RESISTANCE range from 0–100, and ROAD-GRAVITY ranges from 0–20», y que en la primera fase de calibración «Most parameters varied from 0 to 100, necessitating 101 separate runs per variable». Véase la nota bajo Silva2002.

---

## HernandezGuerrero2015

**Metadatos: CORRECTOS.** Crossref (DOI `10.4067/S0718-34022015000200004`): Hernández-Guerrero, Juan; *Revista de Geografía Norte Grande*, número 61, pp. 45–64, septiembre de 2015. La revista no usa volumen, sólo número, tal como está en el `.bib`. La URL de Redalyc del `.bib` resuelve al artículo correcto.

**Contenido verificado con el PDF completo abierto** descargado de Redalyc (21 páginas, texto extraído y leído). Este es el punto delicado 4 del encargo.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap05:40` | «valoración visual sistemática de la calidad ambiental del área urbana municipal, a cargo de observadores capacitados no residentes en las zonas evaluadas» | **CORRECTA** |

Evidencia textual, sección metodológica del artículo (p. 51 del PDF):

> «El levantamiento de los cuestionarios se llevó a cabo con recorridos circulares (dirección de las manecillas del reloj) a fin de valorar visualmente toda el AGEB; el punto de salida y llegada fue el centro de la propia unidad de análisis. En el transcurso del recorrido se dispusieron de puntos de observación. **Cabe señalar que los recorridos se efectuaron por personas no residentes a las AGEB para evitar sesgos en la valoración. Los participantes, mayores de 18 años, recibieron capacitación previa y fueron seleccionados de manera individual debido a sus conocimientos básicos sobre análisis visual del paisaje.**»

Los tres elementos que la tesis afirma quedan confirmados uno a uno: (a) observadores **no residentes**, (b) **capacitados** previamente y seleccionados por conocimientos de análisis visual del paisaje, (c) el motivo declarado es **evitar sesgos**. Y el diseño **no** es percepción de los habitantes: el instrumento es un cuestionario de valoración visual aplicado por observadores externos sobre 27 variables ambientales (agua, suelo, aire) en unidades censales (AGEB). El ámbito también coincide: «área urbana del municipio de Querétaro (AUMQ)», 122,44 km² y 703 699 habitantes al 2010. Sin problemas.

---

## Jantz2003

**Metadatos: CORRECTOS en los campos, con una observación sobre la clave.** Crossref (DOI `10.1068/b2983`): Jantz, Goetz, Shelley; *Environment and Planning B: Planning and Design* **31**(2):251–271, **abril de 2004**. El campo `year = {2004}` del `.bib` es el correcto; la **clave** `Jantz2003` es engañosa (el artículo es de 2004 y así se renderiza la cita), y varios autores lo citan erróneamente como «2003». No hay que cambiar el año; conviene renombrar la clave a `Jantz2004` si se hace una limpieza del `.bib`, o dejarla y saber que no es un error de dato.

**Contenido verificado con el PDF completo abierto** (copia institucional en woodwellclimate.org, idéntica a la paginación 251–271 de *EPB*). Este es el punto delicado 2 del encargo.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03:87` | «una calibración completa puede tomar más de una semana en un clúster de la época» | **CORRECTA** |

Evidencia textual, p. 258 del artículo:

> «Calibration was performed on a Beowulf PC Cluster at the USGS's Rocky Mountain Mapping Center in Denver, CO. **The cluster is a 16-node system (1 master node and 15 computing nodes)**, with each node containing an AMD Athlon/Duron processor, an AMD 750-MHz Thunderbird CPU, and 1.5-GB RAM […] **Over a week of processing time was required to complete the calibration.**»

La cifra es exacta y la formulación «clúster de la época» es una descripción fiel (16 nodos, 750 MHz, 1,5 GB por nodo). Vale la pena notar, por si el autor quiere reforzar la oración, que el dato admite ser citado con todo el detalle sin riesgo: *más de una semana de procesamiento en un clúster Beowulf de 16 nodos a 750 MHz*. También conviene saber que ese estudio calibró a 45 m porque 30 m «exceeded the available computational resources», lo que redondea el argumento del costo.

Observación menor de coherencia interna, no un problema de fuente: Jantz et al. nombran sus fases de calibración «coarse, medium, and fine», mientras la tesis (misma línea 87) atribuye a Silva2002 los nombres «gruesa, fina y final». Ambas cosas son ciertas en sus respectivas fuentes; no hay contradicción, pero si el párrafo llegara a sugerir que todos usan los mismos tres nombres, sería inexacto.

---

## LandisKoch1977

**Metadatos: CORRECTOS.** Crossref (DOI `10.2307/2529310`): Landis, J. Richard; Koch, Gary G.; *Biometrics* **33**(1), 1977. Crossref registra la página inicial 159; PubMed (PMID 843571) confirma el rango completo 159–174, tal como está en el `.bib`.

**Contenido verificado con el PDF completo abierto** del artículo original (copia académica del facsímil JSTOR), no con fuentes secundarias.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap06:200` | Kappa 0,445 = acuerdo moderado, «en la parte baja de ese rango (0,41–0,60)» | **CORRECTA** |
| 2 | `cap06:520` | El Kappa de 0,445 «se interpreta frente a la escala de LandisKoch1977» | **CORRECTA** |
| 3 | `cap07:26` | Kappa 0,445 = «acuerdo moderado en la escala de LandisKoch1977» | **CORRECTA** |
| 4 | `cap07:62` | Kappa promedio 0,445 = «acuerdo moderado en la escala de LandisKoch1977» | **CORRECTA** |

Evidencia textual (p. 165 del original): «the following labels will be assigned to the corresponding ranges of kappa: […] < 0.00 Poor | 0.00–0.20 Slight | 0.21–0.40 Fair | **0.41–0.60 Moderate** | 0.61–0.80 Substantial | 0.81–1.00 Almost Perfect». El intervalo citado por la tesis y la etiqueta «moderado» son exactos, y 0,445 cae en efecto en el tercio inferior de la banda.

Sugerencia de prudencia, opcional y no un error: los propios autores escriben en la misma página «Although these divisions are clearly arbitrary, they do provide useful *benchmarks*». Si el capítulo quiere blindarse frente a la crítica estándar a esta escala, basta escribir «según los intervalos de referencia —expresamente calificados de arbitrarios por sus autores— de \textcite{LandisKoch1977}». Nada obliga a cambiarlo.

En las afirmaciones 2 y 3 hay una segunda atribución, «ese estudio no reporta Kappa ni IoU» / «ninguna de las dos aparece en aquel estudio», que se refiere a `pontius2008comparing` y **no** a `LandisKoch1977`; queda fuera de este lote y debe validarla quien tenga esa clave. Se señala aquí sólo para que no se pierda.

---

## Liu2017

**Metadatos: CORRECTOS.** Crossref (DOI `10.1016/j.landurbplan.2017.09.019`): los nueve autores en el mismo orden del `.bib`, *Landscape and Urban Planning* **168**:94–116, 2017, Elsevier. La revista no numera *issue* en ese volumen, coherente con la ausencia de `number`.

**Contenido verificado con el PDF completo abierto** del artículo (copia de los autores en geosimulation.cn), más el manual oficial de GeoSOS-FLUS.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03:41` | FLUS «mantuvo esa arquitectura pero sustituyó el componente estadístico por una red neuronal artificial» | **CORRECTA** |
| 2 | `cap03:293` | FLUS «sustituye esa idoneidad por la probabilidad de una red neuronal y la asigna mediante un autómata celular con inercia auto-adaptativa y competencia por ruleta» | **CORRECTA** |

Evidencia textual: el artículo describe dos módulos, «1) [an] artificial neural network is used to train and estimate the probability-of-occurrence of each land use type on a specific grid cell, and 2) an elaborate self-adaptive inertia and competition mechanism is designed to address the competition and interactions among the different land use types»; y en la asignación, «a specific land grid either retains the current land use type or transforms into another type depending on their combined probabilities and **the roulette selection**». La contraposición con la familia de idoneidad estadística también es del propio artículo, que sitúa a CLUE-S (Verburg et al., 2002) entre los modelos que «simply estimate the probabilities of individual land use types separately and assign the highest value to the land grid». Los tres componentes que la tesis nombra —probabilidad por red neuronal, inercia auto-adaptativa, competencia por ruleta— están los tres en la fuente, con esos nombres. Sin problemas.

---

## Sante2010

**Metadatos: CORRECTOS.** Crossref (DOI `10.1016/j.landurbplan.2010.03.001`): Santé, García, Miranda, Crecente; *Landscape and Urban Planning* **96**(2):108–122, 2010. Coincide con el `.bib` en todo.

**Contenido: sólo pude acceder al resumen, la introducción completa y los encabezados de sección** (página de ScienceDirect; el artículo es de pago y no hay copia abierta en OpenAlex ni en Unpaywall). No abrí el cuerpo del artículo.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03:526` | La revisión «documenta la heterogeneidad de los métodos de calibración y validación entre los modelos que examina» | **NO VERIFICABLE** |

Qué sí pude confirmar textualmente: «an analysis of **33 urban CA models** has been performed. The differences among the different approaches are highlighted and a classification of the models is proposed […] the general characteristics of urban CA are described and **the different techniques used in the design of such models are characterized and classified. Such techniques have been summarized in several tables.** In addition, the strengths and weaknesses of the different models have been identified». Los encabezados de sección visibles son «Relaxations of CA for urban simulation», «Analysis of urban CA models», «Strengths and weaknesses of urban CA models» y «Conclusions».

Qué **no** pude confirmar: que entre esas técnicas clasificadas figuren específicamente las de **calibración y validación**, que es lo que la oración de la tesis afirma. Es muy probable que sí (la subsección de calibración de este artículo es de uso corriente en la literatura), pero no lo vi con mis ojos y por regla no lo doy por bueno. No es un error detectado: es una verificación pendiente.

**Dos salidas, a elección del autor.** (a) Conseguir el PDF por el acceso institucional de la UNAM y confirmar la subsección de calibración; entonces la oración se queda tal cual. (b) Si no se quiere depender de eso, reformular a lo que está probado en el resumen:

> Las revisiones sistemáticas de \textcite{Sante2010} y \textcite{Aburas2016} clasifican los modelos de autómatas celulares aplicados a casos urbanos reales —treinta y tres modelos en el primer caso— y documentan la heterogeneidad de las técnicas empleadas en su diseño, así como sus fortalezas y debilidades.

---

## Silva2002

**Metadatos: CORRECTOS.** Crossref (DOI `10.1016/S0198-9715(01)00014-X`): Silva, E. A.; Clarke, K. C.; *Computers, Environment and Urban Systems* **26**(6):525–552, noviembre de 2002. Coincide con el `.bib`.

**Contenido verificado con la página de artículo de ScienceDirect (que expone fragmentos literales del cuerpo) y con la primera página del PDF reproducida íntegra en un repositorio de documentos.** No tuve el PDF completo. Este es el punto delicado 1 del encargo.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03:18` | «el acrónimo se consolidó con la aplicación a Lisboa y Oporto» | **CORRECTA** |
| 2 | `cap03:87` | Cinco coeficientes «cada uno acotado al intervalo entero $[0,100]$, lo que da del orden de $10^{10}$ combinaciones» | **NO VERIFICABLE** |
| 3 | `cap03:87` | Las tres fases «gruesa, fina y final» se presentan «con esos nombres» en la calibración de Lisboa y Oporto | **CORRECTA** |

**Afirmación 1.** Verificada textualmente: «This paper focuses on calibrating the SLEUTH model, **formerly the Clarke Cellular Automaton Urban Growth Model** (Clarke & Gaydos, 1998; Clarke, Hoppen, & Gaydos, 1997) for two Portuguese metropolitan areas. **SLEUTH is an acronym for the input layers** that the model uses in gridded map form: Slope, Land Use, Exclusion, Urban Extent, Transportation and Hillshade». El artículo es, en efecto, donde el acrónimo se presenta y desplaza al nombre anterior. El reparto que hace la tesis en esa misma oración también queda respaldado por la fuente, que dice que «the calibration […] followed the techniques developed for the model **as applied to the San Francisco and Washington/Baltimore areas** (Clarke & Gaydos, 1998; Clarke et al., 1996; Clarke et al., 1997)»: la prioridad de esos dos casos es anterior a Silva2002 y corresponde a la clave `Clarke1998`, tal como la tesis la coloca.

**Afirmación 3.** Verificada textualmente: «**Results from the three phases of the calibration mode (Coarse, Fine, and Final calibrations)** are presented in Table 1, Table 2, Table 3. Each table presents the sorted top five highest scoring results from thousands of model runs». Los tres nombres, en ese orden, son los del artículo. La afirmación de la tesis es exacta, incluida la cautela «con esos nombres».

**Afirmación 2: qué falta.** En lo que pude leer de Silva2002 no aparece ni el intervalo `[0,100]` ni ningún recuento de combinaciones; lo que sí aparece es que los cinco coeficientes (*diffusion, breed, spread, slope, road*) controlan el comportamiento del autómata. La aritmética de la tesis es correcta ($101^5 \approx 1{,}05\times10^{10}$; $100^5 = 10^{10}$), y el orden de magnitud es el que se maneja en la literatura, pero **no lo pude atribuir a esta fuente**. La afirmación está co-citada con `Clarke2008`, que no pertenece a este lote: es ahí donde hay que buscar el respaldo.

Además, y esto es lo que conviene mirar con cuidado antes de reescribir: **el artículo original de 1997, que la propia tesis cita catorce líneas más abajo, dice lo contrario para uno de los cinco coeficientes** — «The values for DIFFUSION, BREED, SPREAD, and SLOPE-RESISTANCE range from 0–100, and **ROAD-GRAVITY ranges from 0–20**». En versiones posteriores de SLEUTH los cinco se manejan en 0–100, y la tabla de calibración de Jantz et al. (2004) usa rango «1–100» para los cinco; es decir, el `[0,100]` es defendible para el SLEUTH que calibran Silva y Clarke, pero no es universal a lo largo de la historia del modelo. El orden de magnitud aguanta en cualquier caso ($101^4\times21\approx2{,}2\times10^{9}$ en la parametrización de 1997).

**Redacción corregida propuesta** (mueve el dato numérico a la fuente que lo sostiene y no compromete a Silva2002 con algo que no dice):

> …se resuelve mediante búsqueda por fuerza bruta sobre el espacio de sus cinco coeficientes de crecimiento \parencite{Silva2002}; con los rangos enteros que documenta la implementación del modelo, del orden de $10^{10}$ combinaciones posibles \parencite{Clarke2008}, cada una evaluada con simulaciones Monte Carlo de múltiples iteraciones.

Si se prefiere conservar el `[0,100]` explícito, hay que verificarlo en `Clarke2008` o en la guía de implementación de Project Gigalopolis y citar esa fuente, no Silva2002.

---

## SuarezDelgado2007

**Metadatos: CORRECTOS.** Crossref (DOI `10.24201/edu.v22i1.1295`): Suárez, Manuel; Delgado, Javier; *Estudios Demográficos y Urbanos* **22**(1), 2007, El Colegio de México; página inicial 101 (el `.bib` da 101–142, coherente). Título exacto, incluida la versión bilingüe del registro.

**Contenido verificado con el texto completo abierto en SciELO México** (`S0186-72102007000100101`), leído íntegro. Este es el punto delicado 3 del encargo, y las tres cosas que había que verificar **se confirman las tres**.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03:395` | «uno de los primeros ejercicios de pronóstico espacialmente explícito para la ZMCM, con antecedentes en los escenarios de poblamiento de CONAPO y una metodología adaptada de la literatura estadounidense sobre asignación de suelo» | **CORRECTA** |
| 2 | `cap03:401-403` | Calibra regresión logística binomial sobre rejilla de una hectárea, variable dependiente = cambio no urbano → urbano entre 1990 y 2000, muestra de **15 670** celdas | **CORRECTA** |
| 3 | `cap03:405-406` | Ajuste dentro de muestra con **82,9 %** de celdas clasificadas correctamente | **CORRECTA** |
| 4 | `cap03:406-407` | «publica los coeficientes de todas las variables, de modo que la función de asignación es auditable» | **CORRECTA** |
| 5 | `cap03:407-411` | No es un autómata celular (sin regla de vecindad ni iteración temporal); la asignación se ordena por probabilidad descendente dentro de cada municipio; varias covariables están agregadas a escala municipal; no hay validación fuera de muestra | **CORRECTA** |
| 6 | `cap03:477` (tabla) | Fila R1 = no, R2 = ✓, R3 = no, R4 = ✓ | **CORRECTA** |

Evidencia textual, una por una:

- **15 670 celdas y rejilla de una hectárea.** «Se crearon mapas raster con resolución de una hectárea y se asignó un valor de 1 a los sitios que cambiaron de "no urbanos" a "urbanos" y un valor de 0 a los sitios que permanecieron como "no urbanos". El área urbana que ya existía en 1990 fue excluida del análisis. **De estos mapas se extrajo una muestra de n = 15 670 puntos.**» Y sobre el método: «Debido al carácter categórico de la variable dependiente (Y), **se utilizó una regresión logística binomial** y no la regresión lineal común». (El artículo tiene una errata interna: dos párrafos después escribe «los 15 760 puntos de la muestra». El valor correcto, el que aparece en el pie del cuadro de resultados, es **15 670**, que es el que usa la tesis. Bien elegido.)
- **82,9 %.** Pie del cuadro de resultados del modelo: «N = 15 670 | % casos clasificados correctamente (valor de corte = 0.15) **0 = 83 % | 1 = 79.9 % | Total = 82.9 %**». La tesis cita el total y lo califica correctamente de ajuste **dentro de muestra**.
- **Coeficientes publicados.** El cuadro da B y error estándar de las ocho covariables más la constante: DISTRAKM −0,005; INGRESO 0,841; TAMLOC 0,003; PROPRIEG 1,12; MANUFAC 0,086; SERVS −0,389; RELIEVE −0,069; DISTLOC −0,001; CONSTANTE −1,141, con −2 log likelihood 6748,227, R² Cox-Snell 0,213 y R² Nagelkerke 0,434. La auditabilidad que afirma la tesis es literal.
- **No es AC.** Búsqueda exhaustiva en el texto completo: cero ocurrencias de «autómata», «vecindad» e «iteración». El mecanismo es el que describe la tesis: «El tercer paso consiste en asignar el crecimiento proyectado en cada escenario a los sitios que permiten el desarrollo urbano […] **en orden de probabilidad descendente por municipio**»; y en el resumen metodológico, «se seleccionaron, **en orden descendente**, los sitios con mayor probabilidad de urbanización **en cada municipio** hasta cubrir la superficie total requerida».
- **Covariables agregadas a escala municipal.** El artículo lo reconoce él mismo: «En este caso, **al estar agregada por municipio**, es probable que el coeficiente positivo [de INGRESO] exprese más el efecto de la infraestructura…», y discute la autocorrelación espacial que eso genera: «sólo tres variables disminuyen la generación de autocorrelación espacial: la distancia a la localidad, la distancia al transporte, y el relieve […] El resto de las variables utilizadas aumenta la generación de autocorrela[ción]».
- **Sin validación fuera de muestra.** Cero ocurrencias de «valida-» en el texto completo. El artículo sólo reporta el ajuste del modelo calibrado 1990–2000 y proyecta a 2020; no compara contra un mapa observado posterior. La afirmación de la tesis es correcta y está bien acotada.
- **Antecedentes CONAPO y metodología estadounidense.** «En un estudio previo sobre el tema **publicado por Conapo (1998)** se examinan las tendencias de urbanización de la ZMCM y se hace un pronóstico original del crecimiento por tipo de poblamiento; sin embargo tal pronóstico se limita a presentar mediante flechas de diverso tamaño la expansión». Y: «La metodología aquí propuesta para predecir la urbanización futura **modifica la que utilizaron Landis y Reilly (2003)** en su pronóstico del crecimiento urbano en el estado de California hacia el año 2100». El cauteloso «uno de los primeros» está además respaldado por el propio contraste que el artículo traza frente al CONAPO 1998.
- **Fila de la tabla R1–R4.** Con las definiciones que el capítulo da (R1 validación fuera de muestra del crecimiento histórico; R2 soporte a planeación estratégica; R3 apertura y replicabilidad por repositorio público; R4 interpretabilidad de la función de transición), las cuatro celdas son consistentes con lo verificado: R1 no (no hay validación), R2 sí (tres escenarios a 2020 y recomendaciones de administración metropolitana), R3 no (no hay código ni repositorio), R4 sí (coeficientes completos publicados).

Sin problemas en esta referencia. Es, de hecho, la mejor sostenida del lote.

---

## Waddell2002

**Metadatos: CORRECTOS.** Crossref (DOI `10.1080/01944360208976274`): Waddell, Paul; *Journal of the American Planning Association* **68**(3):297–314, 2002. Coincide con el `.bib`.

**Contenido verificado con el resumen indexado en OpenAlex** (texto del resumen del editor). El artículo es de pago y no hay copia abierta; no leí el cuerpo.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03:49-51` | UrbanSim «acopla usos de suelo, transporte y ambiente en un mismo sistema» | **CORRECTA** |

Evidencia textual del resumen: «Metropolitan areas have come under intense pressure to respond to federal mandates to **link planning of land use, transportation, and environmental quality** […] **UrbanSim is a new model system** that was developed to respond to these emerging requirements and has now been applied in three metropolitan areas». El acoplamiento de los tres dominios en un sistema de modelos es exactamente lo que la fuente declara.

Dos matices que no cambian la etiqueta. Primero, «es la referencia de esa vertiente» es un juicio de posicionamiento del autor de la tesis, no un dato atribuido a Waddell; como tal es defendible y no requiere respaldo en la fuente. Segundo, la oración siguiente del mismo párrafo («requiere datos socioeconómicos desagregados de hogares, empleo y transporte que rara vez existen con esa resolución para ciudades intermedias») es una valoración propia y **no** lleva cita, que es lo correcto: el resumen confirma que el sistema se aplicó a tres áreas metropolitanas, pero no dice nada sobre disponibilidad de datos en ciudades intermedias. Conviene que siga sin cita para no atribuirle a Waddell una limitación que él no enuncia.

---

# Resumen contable

| Etiqueta | Conteo |
|---|---|
| CORRECTA | **19** |
| SOBREEXTENDIDA | **1** |
| INCORRECTA | **0** |
| NO VERIFICABLE | **2** |
| **Total de afirmaciones evaluadas** | **22** |

Desglose por referencia (afirmaciones):

| Clave | Metadatos | C | S | I | NV |
|---|---|---|---|---|---|
| Almeida2003 | correctos | 1 | – | – | – |
| Batty2005 | correctos | 1 | – | – | – |
| ClarkeHoppen1997 | correctos | – | 1 | – | – |
| HernandezGuerrero2015 | correctos | 1 | – | – | – |
| Jantz2003 | correctos (clave engañosa: el artículo es 2004) | 1 | – | – | – |
| LandisKoch1977 | correctos | 4 | – | – | – |
| Liu2017 | correctos | 2 | – | – | – |
| Sante2010 | correctos | – | – | – | 1 |
| Silva2002 | correctos | 2 | – | – | 1 |
| SuarezDelgado2007 | correctos | 6 | – | – | – |
| Waddell2002 | correctos | 1 | – | – | – |

**Referencias inexistentes: ninguna.** Las 11 entradas existen y todos los campos verificables (autores, título, revista o editorial, año, volumen, número, páginas, DOI, ISBN) son correctos.

**Lo único que exige tocar el texto** es `cap03:97` (ClarkeHoppen1997): reatribuir el mecanismo de automodificación a la fuente y presentar la dependencia de la trayectoria como lectura propia.

**Lo único que exige una consulta más antes de reescribir** son `cap03:526` (Sante2010, confirmar que la revisión compara métodos de calibración y validación) y la mitad numérica de `cap03:87` (el `[0,100]` y las $10^{10}$ combinaciones, que hay que apoyar en `Clarke2008`, no en Silva2002 — teniendo en cuenta que ClarkeHoppen1997 acota *road-gravity* a 0–20).

**Dos pendientes que quedan fuera de este lote** y que no hay que perder: la afirmación «ese estudio no reporta Kappa ni IoU» sobre `pontius2008comparing` (`cap06:520`, `cap07:26`), y el respaldo numérico de `Clarke2008`.
