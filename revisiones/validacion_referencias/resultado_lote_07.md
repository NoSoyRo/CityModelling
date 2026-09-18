# Validación LOTE 7 de 8 — 11 referencias

Fecha: 2026-09-18. Métodos usados, en orden: (1) resolución de DOI; (2) API de Crossref
`https://api.crossref.org/works/{DOI}`; (3) OpenAlex `https://api.openalex.org/works/doi:{DOI}`;
(4) descarga y extracción de texto del PDF de acceso abierto; (5) búsqueda del título exacto.
Todas las citas textuales de abajo provienen de páginas o PDF que se abrieron y cuya URL se
indica. Cuando no fue posible acceder al contenido, se marca NO VERIFICABLE y se detalla el intento.

---

## Beven2006

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.1016/j.jhydrol.2005.07.007` devuelve
  type `journal-article`; title `A manifesto for the equifinality thesis`; author `Keith Beven`;
  container `Journal of Hydrology`; volume `320`; issue `1-2`; page `18-36`; issued `2006-03`.
  Coincidencia exacta con los diez campos de la entrada.
- correcciones: ninguna.

**Afirmaciones:**

1. `cap03_estado_del_arte.tex:98` — equifinalidad procede del modelado
   ambiental; aceptar múltiples parametrizaciones aceptables obliga a reportar el conjunto de
   soluciones y no un óptimo único. **CORRECTA.**
   Respaldo textual, del preprint del autor (misma versión que el artículo):
   «The argument is made that the potential for multiple acceptable models as representations of
   hydrological and other environmental systems (the equifinality thesis) should be given more
   serious consideration than hitherto.»
   Y sobre reportar el conjunto: «The weights then control the form of a cumulative density
   (possibility) function for any predicted variable over the complete set of behavioural models,
   from which any desired prediction limits can be obtained.»
   Sobre el riesgo de generalizar desde un óptimo: «computer intensive studies of responses across
   the model space have shown that these mappings are too simplistic, since they arbitrarily
   exclude many models that are very nearly as good as the "optima".»
   URL: https://eprints.lancs.ac.uk/id/eprint/4419/1/Manifesto12.pdf

2. `cap03_estado_del_arte.tex:155` — dos conjuntos de coeficientes que
   empatan en ajuste pueden divergir en la proyección; reportar uno solo transmite una unicidad
   que los datos no respaldan. **CORRECTA para la parte que sostiene Beven.**
   Respaldo textual: «It is necessary to assume that the behavioural models in calibration will
   also be behavioural in prediction; this procedure only (at best) gives the tolerance limits
   (in the calibration period) or the prediction limits of the weighted simulations of any
   variable.» Es decir, el desempeño en calibración no garantiza el desempeño en predicción, que
   es exactamente lo que la tesis afirma.
   URL: https://eprints.lancs.ac.uk/id/eprint/4419/1/Manifesto12.pdf
   Matiz: Beven trabaja en hidrología, no en autómatas celulares; la cita conjunta con
   `Dietzel2007` es la que traslada el argumento a la calibración por fuerza bruta. Ver
   observación 3 de `Dietzel2007`.

**Riesgo en examen:** bajo. Es la fuente canónica de equifinalidad en modelado ambiental y dice
literalmente lo que se le atribuye.

---

## BonhamCarter1989

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.4095/128059` (registro del propio editor,
  Natural Resources Canada) devuelve autores `G F Bonham-Carter`, `F P Agterberg`, `D F Wright`,
  en ese orden, y el título `Weights of evidence modelling: a new approach to mapping mineral
  potential`. El registro es de tipo `report`, consistente con un GSC Paper.
  Serie, editor del volumen y paginación se confirman de forma concordante en cuatro fuentes
  independientes que citan el capítulo: `https://www.isprs.org/proceedings/xxxiv/part4/pdfpapers/268.pdf`
  («Statistical Applications in the Earth Sciences, Geological Survey of Canada, Paper 89-9,
  pp. 171-183»); `https://pubs.usgs.gov/of/2004/1245/of2004-1245.pdf`; y los listados de Springer.
- correcciones: ninguna obligatoria. Aviso: el catálogo del propio editor fecha el volumen
  **1990**, no 1989. Ejemplo verificable, mismo volumen, otro capítulo:
  «Geological Survey of Canada, Paper 89-9, 1990; pages 157-169»
  (https://geochem.nrcan.gc.ca/cdogs/content/pub/pub04001_e.htm). La cita canónica en la
  literatura de WoE es 1989 y es la que usan Bonham-Carter (1994), Agterberg y Cheng (2002) y el
  resto del campo; se recomienda **conservar 1989** y no abrir el tema.

**Afirmaciones:**

1. `cap02_marco_teorico.tex:138` — el WoE se formuló para diagnóstico médico
   y se adaptó a finales de los ochenta al mapeo de potencial mineral con SIG. **CORRECTA.**
   Respaldo textual, documentación oficial del método (ArcWofE, equipo de Bonham-Carter):
   «The method was originally developed for a nonspatial application in medical diagnosis, in
   which the evidence consisted of a set of symptoms and the hypothesis was of the type "this
   patient has disease x". […] Weights of evidence was adapted in the late 1980s for mineral
   potential mapping with GIS.»
   URL: https://www.ige.unicamp.br/wofe/documentation/wofeintr.htm
   Confirmación independiente, de Agterberg y Cheng: «Weights-of-evidence modeling was developed
   originally for medical diagnosis, but was applied subsequently to mineral-potential mapping.»
   URL: https://www.ige.unicamp.br/sdm/ArcSDM31/documentation/CI_Agterberg.pdf
   Matiz honesto: no se pudo leer el capítulo de 1989 en sí, de modo que no puede afirmarse que la
   frase sobre el origen médico esté *en ese capítulo*; lo que está verificado es el contenido
   de la afirmación y que 1989 es la referencia estándar de la adaptación al dominio mineral.

2. `cap03_estado_del_arte.tex:253` — origen médico, adaptación al mapeo
   mineral, consolidación como procedimiento estándar en el capítulo 9 de Bonham-Carter (1994).
   **CORRECTA.** El mismo respaldo que arriba para la primera mitad. Para «capítulo 9»:
   «WofE allows the user to carry out a pair-wise conditional independence test, and an "omnibus"
   test as described in Bonham-Carter (1994, ch. 9).»
   URL: https://www.ige.unicamp.br/wofe/documentation/wofeintr.htm

3. `cap03_estado_del_arte.tex:310` — tres pruebas de independencia
   condicional: por pares de Bonham-Carter (1989), ómnibus del capítulo 9 de Bonham-Carter (1994)
   con tolerancia del 15 %, y prueba exacta de Agterberg y Cheng (2002). **PARCIAL.**
   - Regla del 15 %: **VERIFICADA.** «For example, in Bonham-Carter (1994, p. 316), it is argued
     that T should not exceed n by more than 15%.» La p. 316 corresponde al capítulo 9.
     URL: https://www.ige.unicamp.br/sdm/ArcSDM31/documentation/CI_Agterberg.pdf
   - «Prueba exacta» de Agterberg y Cheng: **VERIFICADA con la palabra de los autores.**
     «This new test is exact and simpler to use than other tests including the
     Kolmogorov-Smirnov test and various chi-squared tests adapted from discrete multivariate
     statistics.» Misma URL. Añadido útil: los autores llaman «informal» a la regla del 15 %
     («in accordance with the informal 15% rule of the earlier version of the omnibus test»);
     conviene calcarles el adjetivo.
   - «Pruebas por pares de Bonham-Carter (1989)»: **NO VERIFICABLE.** Vías intentadas, sin éxito:
     (a) DOI 10.4095/128059, que redirige a
     `https://ostrnrcan-dostrncan.canada.ca/handle/1845/128652`, ficha sin PDF y renderizada por
     JavaScript; (b) la API DSpace de ese repositorio, que no devolvió JSON; (c) el servidor
     GEOSCAN clásico, HTTP 500; (d) cuatro rutas plausibles de PDF en IG-Unicamp y
     publications.gc.ca, todas 404; (e) OpenAlex, que reporta `best_oa_location: null`.
     Lo que sí está verificado es que la prueba χ² por pares existe y forma parte del método,
     pero la documentación oficial la atribuye al **capítulo 9 de Bonham-Carter (1994)**, junto
     con la ómnibus, no al capítulo de 1989 (misma cita del punto 2).
     Corrección sugerida, conservadora: «entre ellas la prueba χ² por pares y la prueba ómnibus
     del capítulo 9 de \textcite{BonhamCarter1994}, que considera aceptable un exceso […] no
     mayor al 15 %, y la prueba exacta de \textcite{AgterbergCheng2002}». Con esto se elimina la
     única atribución no verificable de la frase sin perder nada del argumento.

**Riesgo en examen:** medio-bajo. El origen médico y la regla del 15 % son defendibles con cita
textual. La atribución de la prueba por pares al capítulo de 1989 es el punto flojo: si un
sinodal pide la página exacta, no hay forma de darla.

---

## Chen2022

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.1016/j.envsoft.2022.105362` devuelve
  autores `Changjie Chen`, `Jasmeet Judge`, `David Hulse`, en ese orden; título
  `PyLUSAT: An open-source Python toolkit for GIS-based land use suitability analysis`;
  container `Environmental Modelling & Software`; volume `151`; article-number y page `105362`;
  issued `2022-05`. Coincidencia exacta.
- correcciones: ninguna.

**Afirmaciones:**

1. `cap03_estado_del_arte.tex:233` — «el conjunto de herramientas en Python
   para análisis de aptitud del suelo de \textcite{Chen2022}». **CORRECTA.** La afirmación es una
   paráfrasis literal del título registrado en Crossref: *open-source Python toolkit for
   GIS-based land use suitability analysis*. No se le atribuye ninguna cifra ni hallazgo.
   URL: https://api.crossref.org/works/10.1016/j.envsoft.2022.105362
   Nota: no se consiguió el texto completo (ScienceDirect devolvió bloqueo Cloudflare y
   OpenAlex no registra copia en abierto), pero la afirmación no requiere más que el título.

**Riesgo en examen:** nulo.

---

## Dietzel2007

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.1111/j.1467-9671.2007.01031.x` devuelve
  autores `Charles Dietzel`, `Keith C Clarke`; título
  `Toward Optimal Calibration of the SLEUTH Land Use Change Model`; container
  `Transactions in GIS`; volume `11`; issue `1`; page `29-45`; issued `2007-01-12`.
  La cabecera del propio artículo lo confirma: «Transactions in GIS, 2007, 11(1): 29–45 © 2007
  The Authors», con los autores adscritos al Department of Geography, University of California –
  Santa Barbara.
  URL: https://doi.org/10.1111/j.1467-9671.2007.01031.x
- correcciones: ninguna.

**Afirmaciones:**

1. `cap03_estado_del_arte.tex:86` — el procedimiento estrecha la búsqueda
   en tres fases (gruesa, fina, final), «esquema planteado ya en la formulación original del
   autómata auto-modificable \parencite{ClarkeHoppen1997} y discutido después de forma sistemática
   por \textcite{Dietzel2007}». **PARCIAL.**
   Las tres fases están verificadas, y el tratamiento sistemático también: «By running the model
   in calibration mode, a set of control parameters is refined in the sequential "brute-force"
   calibration phases: coarse, fine and final calibrations (Silva and Clarke 2002) […] The
   calibration process is done in three stages: coarse, fine, and final.» Y el cierre de la fase
   final: «an even narrower range of parameters is selected ideally with unit increments. The best
   fit of the parameters from this third calibration are then the parameters that are used in
   forecasting.»
   URL: https://doi.org/10.1111/j.1467-9671.2007.01031.x
   El matiz es de atribución: **Dietzel y Clarke acreditan el esquema de tres fases a Silva y
   Clarke (2002), no a Clarke y Hoppen (1997).** Del artículo original de 1997 dicen únicamente
   que la calibración empezó siendo jerárquica por resolución espacial: «Initially the model was
   calibrated using hierarchical spatial resolutions, beginning with data of coarser
   resolution…». Como la tesis cita a `Dietzel2007` en la misma frase, un sinodal puede abrir el
   artículo y ver que contradice la atribución.
   Corrección sugerida: «esquema derivado de la calibración jerárquica por resolución de
   \parencite{ClarkeHoppen1997}, nombrado en tres fases por Silva y Clarke (2002) y discutido
   después de forma sistemática por \textcite{Dietzel2007}».

2. `cap03_estado_del_arte.tex:98` — «la discusión sobre qué constituye una
   calibración óptima permanece abierta \parencite{Dietzel2007}, y sin un criterio adicional la
   elección entre combinaciones equifinales termina dependiendo de quien realiza la calibración».
   **CORRECTA.**
   Respaldo textual: «Narrowing of the parameter set can be based on a variety of different
   goodness of fit measures or their combinations. Despite numerous applications of SLEUTH all
   over the world, there is no clear consensus as to which metrics are the appropriate ones to use
   during the calibration process.» Y el artículo documenta esa dependencia del analista con
   ejemplos: «Jantz et al. (2004) used the compare, population, and Lee-Sallee statistics, while
   in Atlanta, Yang and Lo (2003) used a weighted sum of all the metrics, and Silva and Clarke
   (2002) used only the Lee-Sallee metric».
   URL: https://doi.org/10.1111/j.1467-9671.2007.01031.x
   Dos matices menores, ninguno invalidante: (a) el artículo **no usa la palabra
   «equifinality»** en ningún punto, de modo que el término es de la tesis, sostenido por
   `Beven2006`, lo cual es legítimo; (b) el artículo no se limita a constatar el problema, propone
   una solución, el OSM, «the product of the compare, population, edges, clusters, slope, X-mean,
   and Y-mean metrics». Decir «permanece abierta» es fiel al diagnóstico del artículo pero omite
   que ofrece un candidato a criterio; si se quiere blindar la frase, basta añadir «pese a la
   propuesta del Optimal SLEUTH Metric».

3. `cap03_estado_del_arte.tex:155` — dos conjuntos de coeficientes que
   empatan en la métrica de ajuste «pueden divergir en la proyección a largo plazo».
   **PARCIAL.**
   Lo que Dietzel y Clarke sí sostienen: que la calibración por fuerza bruta genera un espacio
   enorme de combinaciones entre las que hay que elegir. «SLEUTH's calibration uses the brute
   force method: every possible combination and permutation of its control parameters is tried
   […] we instead attempt to create the complete set of possible outcomes with the goal of
   examining them to select the optimum from among the millions of possibilities», con conjuntos
   derivados de «1 million to 4.1 million entries for 18 dimensions».
   Lo que **no** dicen en ningún punto del artículo: que conjuntos de coeficientes con ajuste
   equivalente divergen en la proyección a largo plazo. Se verificó el texto completo buscando
   `diverg`, `long-term`, `long term`, `forecast`, `different parameter sets`, `multiple
   parameter`: las únicas apariciones de `forecast` describen que los parámetros de la tercera
   calibración son los que se usan para pronosticar, y la única aparición de `long-term` está en
   el título de una referencia (Clarke y Gaydos 1998).
   URL: https://doi.org/10.1111/j.1467-9671.2007.01031.x
   Dado que la cita es conjunta `\parencite{Beven2006,Dietzel2007}`, la frase es defendible si se
   entiende que Dietzel aporta la multiplicidad de combinaciones y Beven la divergencia
   predictiva. Corrección sugerida, para que el reparto quede explícito: «la calibración por
   fuerza bruta deja millones de combinaciones entre las que elegir \parencite{Dietzel2007}, y
   conjuntos con ajuste equivalente en calibración no tienen por qué serlo en predicción
   \parencite{Beven2006}».

**Riesgo en examen:** medio. Las tres fases y la falta de consenso son citables textualmente. Los
dos puntos a corregir son de atribución interna, del tipo que un lector del artículo detecta.

---

## Hagenauer2022

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.1080/13658816.2021.1871618` devuelve
  autores `Julian Hagenauer`, `Marco Helbich`; título
  `A geographically weighted artificial neural network`; container
  `International Journal of Geographical Information Science`; volume `36`; issue `2`;
  page `215-235`. La cita oficial del repositorio del primer autor coincide: «Julian Hagenauer &
  Marco Helbich (2022) A geographically weighted artificial neural network, International Journal
  of Geographical Information Science, 36:2, 215-235».
  URL: https://github.com/jhagenauer/gwann
  Nota sobre el año: Crossref registra `issued 2021-02-08`, la fecha de publicación en línea; el
  número impreso es 2022. `year = {2022}` es correcto y es la forma en que los propios autores
  citan el trabajo.
- correcciones: ninguna.

**Afirmaciones:**

1. `cap03_estado_del_arte.tex:55` — «\textcite{Hagenauer2022} introdujo una
   red neuronal geográficamente ponderada para relaciones espaciales no lineales en vivienda y
   calidad del aire, no como modelo de crecimiento urbano». **CORRECTA.**
   - Novedad y motivo, del resumen: «it is usually assumed that the relationships between the
     dependent and the independent variables are linear. In practice, however, it is often the
     case that variables are nonlinearly associated. To address this issue, we propose a
     geographically weighted artificial neural network (GWANN).»
     URL: https://doi.org/10.1080/13658816.2021.1871618
   - Vivienda: «3.2. Experiment 2: house prices in Austria […] Data on 3,887 geocoded
     single-family houses in Austria were provided by UniCredit Bank Austria AG […] Individual
     transaction prices of house purchases recorded in euros were collected from 1998 to 2009».
     URL: https://www.tandfonline.com/doi/full/10.1080/13658816.2021.1871618
   - Calidad del aire: «We developed a GWANN and GWR model to estimate nitrogen dioxide
     concentrations across Austria», y el índice de materiales suplementarios lista
     «0.4 Experiment 4: Land use regression model for nitrogen dioxide in Austria».
     URL: https://doi.org/10.1080/13658816.2021.1871618
   - Que no es un modelo de crecimiento urbano: correcto, los cuatro experimentos son datos
     sintéticos, precios de vivienda en Austria, Boston housing y NO₂ en Austria.
   Matiz, por precisión: el experimento de calidad del aire está en los **materiales
   suplementarios**, no en el cuerpo del artículo. Nota 1 del artículo: «Two additional
   experiments are given in the supplemental materials. The first one uses housing benchmark data
   to predict house prices and the second one traffic and land-use data to predict nitrogen
   dioxide concentrations.» Si se quiere ser exacto: «en precios de vivienda y, en material
   suplementario, en concentraciones de dióxido de nitrógeno».

**Riesgo en examen:** bajo.

---

## JimenezLopez2021

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.24201/edu.v36i3.1997` devuelve autores
  `Eduardo Jiménez López`, `Carlos Garrocho Rangel`, `Tania Chávez Soto`, en ese orden; título
  `Autómatas Celulares en Cascada para modelar la expansión urbana con áreas restringidas`;
  container `Estudios Demográficos y Urbanos`; volume `36`; issue `3`; page `779-823`;
  publisher `El Colegio de Mexico, A.C.`; issued `2021-09-17`. La ficha de la revista repite la
  cita exacta: «Jiménez López, E., Garrocho Rangel, C., & Chávez Soto, T. (2021). […] Estudios
  Demográficos Y Urbanos, 36(3), 779–823.»
  URL: https://estudiosdemograficosyurbanos.colmex.mx/index.php/edu/article/view/1997
- correcciones: ninguna.

**Afirmaciones:**

1. `cap03_estado_del_arte.tex:438` — generalización del método como
   autómatas celulares en cascada, con áreas restringidas, filtro de bondad de ajuste de cuatro
   indicadores (entropía de Shannon, dimensión fractal, Kappa de Cohen, Jaccard), prueba empírica
   en Tijuana, Acapulco y Toluca, y disponibilidad en código abierto. **CORRECTA en sus cinco
   componentes.** Texto completo leído en SciELO México:
   URL: https://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S0186-72102021000300779
   - Los cuatro indicadores, literal: «Cada indicador que utilizamos en el filtro mide aspectos
     claves del proceso de expansión urbana […] i. entropía de Shannon, que estima lo compacto o
     disperso de la mancha urbana […]; ii. dimensión fractal, que sintetiza el crecimiento y la
     forma de la mancha urbana […]; iii. índice de Kappa de Cohen, que mide la similitud entre dos
     mapas descontando la coincidencia esperada por el azar […]; y iv. índice de Jaccard».
   - Las tres ciudades: «valorar la bondad de ajuste entre las simulaciones de los modelos
     construidos con nuestro método y la expansión urbana observada en tres ciudades seleccionadas
     por sus características contrastantes: las áreas metropolitanas de Toluca, Tijuana y
     Acapulco».
   - Áreas restringidas: «Incluimos en el método ACC zonas restringidas a la expansión de la
     ciudad: calles, parques, áreas naturales protegidas, límites internacionales […] o límites
     naturales».
   - Código abierto: «Estos modelos están disponibles en código abierto y sin costo en la Estación
     de Inteligencia Territorial de El Colegio Mexiquense (http://www.christaller.org.mx/).»
   - Filtro en cascada de dos niveles, global y local: «Si los índices de Kappa y Jaccard
     registran valores aceptables de similitud […] Éste es el segundo y último nivel del filtro en
     cascada.»
   Matiz relevante para la afirmación 2: la nota al pie 1 precisa que el acceso es **bajo
   demanda**: «Los módulos que se han implementado en la Estación de Inteligencia Territorial
   Christaller son de código abierto y están disponibles, sin costo, bajo demanda. Los interesados
   pueden consultar el sitio http://www.christaller.org.mx/ y contactar a tchavez@cmq.edu.mx».
   La tesis dice «declarando la disponibilidad», que es exactamente el verbo correcto: se declara,
   no se verifica un repositorio público.

2. `cap03_estado_del_arte.tex:472` — ubicación de `JimenezLopez2021` en la
   tabla R1–R4 como parcial / ✓ / ✓ / parcial. **CORRECTA como juicio evaluativo, con una
   salvedad.** No es una afirmación *sobre* la fuente sino una clasificación de la tesis según sus
   propios criterios (R1 simulación histórica con precisión, R2 soporte a planeación estratégica,
   R3 replicabilidad y apertura, R4 interpretabilidad de la función de transición). Cada celda es
   sostenible con el texto leído:
   - R1 parcial: la regla ganadora se elige contra el mismo mapa de 2017 con el que después se
     reporta el ajuste, tal como la propia tesis argumenta en las líneas 450-468.
   - R2 ✓: «Ilustramos cómo nuestro método puede impulsar el codiseño de políticas urbanas»
     (resumen del artículo, misma URL).
   - R3 ✓: la declaración de código abierto citada arriba.
   - R4 parcial: las reglas de transición se identifican por número (91, 206, 211, 223, 236, 47
     según ciudad y filtro), lo que da trazabilidad pero no una descomposición interpretable de la
     contribución de cada variable.
   Salvedad defensiva: el ✓ pleno en R3 es el más atacable, porque la disponibilidad es «bajo
   demanda» y no un repositorio público versionado. Si el criterio R3 exige apertura y no solo
   gratuidad, como dice la propia tesis en la línea 233, conviene o bajar la celda a «parcial» o
   añadir una nota al pie que explicite el «bajo demanda». Es el tipo de detalle que la Dra.
   Lárraga o un sinodal con lectura fina puede señalar, y la defensa es de una línea si está
   anticipado.

**Riesgo en examen:** bajo, y es la referencia mejor sostenida del lote. Recomendación: añadir la
nota al pie sobre «bajo demanda» para que el ✓ en R3 quede blindado.

---

## Jolliffe2002

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.1007/b98835` devuelve type `book`; título
  `Principal Component Analysis`; container `Springer Series in Statistics`;
  publisher `Springer-Verlag`; issued `2002`; ISBN `0387954422`. El DOI corresponde a la segunda
  edición, de 2002; la primera es de 1986, de modo que `edition = {2}` es correcto.
  `address = {New York}` corresponde al sello Springer New York que publica esta serie.
- correcciones: ninguna. Mejora opcional, no obligatoria:
  `series = {Springer Series in Statistics}` e `isbn = {978-0-387-95442-4}`.
  No se pudo abrir la ficha de Springer, `https://link.springer.com/book/10.1007/b98835` agotó el
  tiempo de espera, pero el registro de Crossref proviene del propio editor y es suficiente.

**Afirmaciones:**

1. `cap02_marco_teorico.tex:48` — «El análisis de componentes principales
   \parencite{Jolliffe2002} atiende el segundo problema», es decir, la redundancia entre
   descriptores correlacionados. **CORRECTA.** Es una atribución genérica al texto canónico de la
   técnica, sin cifras, sin hallazgos y sin afirmaciones sobre lo que el libro hizo o no hizo.
   La definición que la tesis da a continuación, direcciones ortogonales de máxima varianza y
   proyección ordenada por varianza explicada, es la definición estándar del libro.
   URL: https://api.crossref.org/works/10.1007/b98835

**Riesgo en examen:** nulo.

---

## SEDATU2024NOM005

**Metadatos:** CORRECTO

- evidencia: la ficha oficial del Sistema de Información del DOF confirma título, dependencia y
  fecha: «Norma Oficial Mexicana NOM-005-SEDATU-2024, Contenidos generales para planes o programas
  municipales de ordenamiento territorial y/o desarrollo urbano. poder ejecutivo / SECRETARIA DE
  DESARROLLO AGRARIO, TERRITORIAL Y URBANO. Publicado el: 09-07-2024. Sección: unica. Edición:
  Matutina», con la clave 5732738 en la propia URL.
  URL: https://sidof.segob.gob.mx/notas/5732738
  El texto íntegro de la norma está en `https://dof.gob.mx/normasOficiales/9450/sedatu/sedatu.html`.
  La ficha de la Plataforma Tecnológica Integral de Infraestructura de la Calidad añade:
  «Estado de la Norma: vigente. Fecha de publicación en el DOF: 9/7/2024. Fecha de entrada en
  vigor: 5/1/2025.»
  URL: https://platiica.economia.gob.mx/normalizacion/nom-005-sedatu-2024/
- El campo `note` también está verificado, palabra por palabra: «El tema "Homologación de
  Contenidos Generales para Programas Municipales de Ordenamiento Territorial y/o Desarrollo
  Urbano", se inscribió en el Suplemento del Programa Nacional de Infraestructura de la Calidad,
  publicado el 01 de junio de 2023, en el Diario Oficial de la Federación (DOF).»
  URL: https://dof.gob.mx/normasOficiales/9450/sedatu/sedatu.html
- correcciones: ninguna. Detalle cosmético: el título oficial usa coma después de la clave, no dos
  puntos. Aviso operativo: el `url` con `nota_detalle.php` agotó el tiempo de espera
  al consultarlo; `https://sidof.segob.gob.mx/notas/5732738` responde y es igualmente oficial, por
  si conviene sustituirlo antes de la entrega.

**Afirmaciones:**

1. `cap01_introduccion.tex:24` — «Su cobertura efectiva es incompleta, y una
   parte de los planes publicados no se ha actualizado». **CORRECTA, y la fuente dice bastante
   más de lo que la tesis aprovecha.** Cita textual de la norma:
   «a partir de la revisión de los instrumentos que conforman el SGPT, la SEDATU identificó en
   2023 que, en el ámbito particular de la planeación a escala municipal, solo 567 municipios
   (22.94%) cuentan con un instrumento en materia de ordenamiento territorial y/o desarrollo
   urbano, de los cuales casi 45%, fueron publicados hace más de 15 años, por lo que han perdido
   su vigencia y se encuentran desactualizados (SEDATU, 2023).»
   URL: https://dof.gob.mx/normasOficiales/9450/sedatu/sedatu.html
   Refuerzo adicional en la misma norma: «Las entidades federativas no cuentan con un criterio
   oficial de contenidos para la elaboración de sus instrumentos de planeación en la escala
   municipal».
   Recomendación, no corrección: sustituir «incompleta» y «una parte» por las cifras oficiales,
   567 municipios, 22,94 % del total, y casi 45 % de ellos con más de quince años. Convierte una
   frase impresionista en un dato citable de fuente primaria.

**Riesgo en examen:** nulo, y con margen de mejora al alza.

---

## Siddiqi2006

**Metadatos:** CORRECTO

- evidencia: el clúster de OpenLibrary para «Credit Risk Scorecards» de Naeem Siddiqi incluye los
  ISBN `047175451X` y `9780471754510`, y lista entre sus editoriales `Wiley` y `SAS Publishing`,
  lo que corresponde a la coedición Wiley y SAS Business Series.
  URL: https://openlibrary.org/search.json?q=Credit+Risk+Scorecards+Siddiqi
  Un paper de SAS cita el libro con la paginación exacta del pasaje relevante:
  «Siddiqi, Naeem (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit
  Scoring. SAS Institute, pp 79-83.»
  URL: https://support.sas.com/resources/papers/proceedings13/095-2013.pdf
- correcciones: ninguna. Dos avisos: (a) OpenLibrary registra `first_publish_year 2005`, el libro
  salió en diciembre de 2005 con copyright 2006, y la literatura lo cita universalmente como
  Siddiqi (2006), de modo que conviene **conservar 2006**; (b) el campo `pages` no aplica a un
  libro, pero si se quiere blindar la afirmación 2 se puede añadir la localización del pasaje,
  pp. 79-83, en el `\parencite`.
  No se consiguió abrir la ficha de Wiley, HTTP 404 en la URL de catálogo probada, ni el texto del
  libro; todas las verificaciones de contenido de abajo son por tanto **secundarias** y se
  declaran como tales.

**Afirmaciones:**

1. `cap02_marco_teorico.tex:179` — el Information Value no forma parte de la
   formulación del WoE de Bonham-Carter (1994) sino de la práctica de tarjetas de puntuación en
   riesgo de crédito; acompaña habitualmente al peso W⁺; equivale a la divergencia de
   Kullback-Leibler simetrizada. **PARCIAL.** Tres piezas, con tres estados distintos.
   - Que pertenece a la práctica de credit scoring: **verificado** de forma secundaria. El IV es
     el estadístico estándar de selección de variables en esa práctica y Siddiqi es su manual de
     referencia: «Siddiqi suggests that variables with extremely high IVs invite suspicion. He
     provides the following rule of thumb for approaching variables based on their Information
     Value».
     URL: https://support.sas.com/resources/papers/proceedings13/095-2013.pdf
   - La equivalencia con la divergencia de Kullback-Leibler simetrizada: **verificado** como
     hecho matemático. «This is the symmetrised Kullback–Leibler divergence (Jeffreys divergence)
     between the event and non-event distributions across bins.»
     URL: https://evandeilton.github.io/OptimalBinningWoE/articles/introduction.html
     La tesis lo enuncia como propiedad formal, no como cita de Siddiqi, lo cual es correcto.
   - Que **no** forma parte de la formulación de Bonham-Carter (1994): **NO VERIFICABLE
     directamente.** Es una afirmación negativa y no hubo acceso al libro de 1994. Evidencia
     indirecta favorable: la documentación oficial del método enumera los estadísticos que el
     WoE geocientífico calcula, W⁺, W⁻, contraste y su desviación estándar, y el IV no aparece en
     ningún punto, ni en la introducción del método ni en el artículo de Agterberg y Cheng.
     URLs: https://www.ige.unicamp.br/wofe/documentation/wofeintr.htm y
     https://www.ige.unicamp.br/sdm/ArcSDM31/documentation/CI_Agterberg.pdf
     Sugerencia de redacción a prueba de objeciones, que evita la negación absoluta: «el
     estadístico no pertenece al núcleo de la formulación geocientífica del WoE, centrada en W⁺,
     W⁻ y el contraste \parencite{BonhamCarter1994}, sino a la práctica de construcción de
     tarjetas de puntuación \parencite{Siddiqi2006}».
   - Precisión adicional: Siddiqi no inventa el IV, lo atribuye a la teoría de la información.
     Según una lectura directa del libro reportada públicamente, «In Naeem Siddiqi's well-known
     book Credit Risk Scorecards, he writes "Information Value, […] comes from information
     theory" and references Kulback's 1959 book Information Theory and Statistics».
     URL: https://sqlpete.wordpress.com/2017/02/12/upper-bound-of-the-information-value-statistic/
     La tesis dice «tomado de la práctica de construcción de tarjetas de puntuación», que es
     compatible: atribuye la práctica, no la invención. No requiere cambio.

2. `cap02_marco_teorico.tex:186` — escala de cinco tramos: <0,02 sin valor
   predictivo; 0,02-0,10 débil; 0,10-0,30 medio; 0,30-0,50 fuerte; **>0,50 muy fuerte**, con la
   advertencia de que valores muy altos indican sobreajuste o un bin trivial. **INCORRECTA en el
   quinto tramo.**
   Los cuatro primeros tramos coinciden exactamente con Siddiqi. El quinto no: Siddiqi no lo
   califica de «muy fuerte» sino de **sospechoso**, y la recomendación práctica es investigarlo y
   posiblemente excluir la variable, no celebrarla.
   Cuatro fuentes secundarias independientes, coincidentes y todas abiertas:
   - «< 0.02: unpredictive / 0.02 to 0.1: weak / 0.1 to 0.3: medium / 0.3 to 0.5: strong /
     > 0.5: **suspicious**», citando Siddiqi (2006) pp. 79-83.
     URL: https://support.sas.com/resources/papers/proceedings13/095-2013.pdf
   - «| >0.5 | **Suspicious or too good to be true** |», en una tabla presentada explícitamente
     como la regla de Naeem Siddiqi.
     URL: https://ucanalytics.com/blogs/information-value-and-weight-of-evidencebanking-case/
   - «≥0.50 Suspicious — almost always leakage», con la atribución «the package grades it with the
     bands from Siddiqi (2006)».
     URL: https://evandeilton.github.io/OptimalBinningWoE/articles/introduction.html
   - Lectura directa del libro, reportada públicamente: «In his book, Siddiqi gives the following
     rule of thumb regarding the value of IV: […] 0.5+ | "should be checked for over-predicting"».
     URL: https://sqlpete.wordpress.com/2017/02/12/upper-bound-of-the-information-value-statistic/
   Corrección propuesta, texto exacto: «La escala convencional de interpretación, tomada de esa
   misma práctica \parencite[pp.~79--83]{Siddiqi2006}, lo divide en cinco tramos: por debajo de
   $0{,}02$ la variable no tiene valor predictivo; entre $0{,}02$ y $0{,}10$ es débil; entre
   $0{,}10$ y $0{,}30$, medio; entre $0{,}30$ y $0{,}50$, fuerte; y por encima de $0{,}50$ el
   propio Siddiqi lo califica de sospechoso, porque suele indicar fuga de información hacia la
   variable respuesta o un bin que separa el resultado de forma trivial.»
   Advertencia sobre por qué esto importa: si alguna variable del modelo
   tiene IV superior a 0,50, con la escala corregida la tesis
   estaría obligada a discutirla como sospechosa, no a presentarla como la más informativa. Vale
   la pena revisar los IV antes de cerrar el capítulo 5.

3. `cap03_estado_del_arte.tex:253` — el IV es «un estadístico ajeno a la
   formulación original del WoE y tomado de la práctica de construcción de tarjetas de puntuación
   \parencite{Siddiqi2006}, que agrega sobre los bins la separación entre las distribuciones
   condicionales del evento». **CORRECTA.** La descripción operativa del estadístico es exacta y
   coincide con la definición verificada: suma sobre bins de la diferencia de proporciones
   multiplicada por el logaritmo de su razón, esto es, la divergencia de Jeffreys entre las dos
   distribuciones condicionales.
   URL: https://evandeilton.github.io/OptimalBinningWoE/articles/introduction.html
   Aquí la formulación «ajeno a la formulación original» es más cuidadosa que en `cap02:179` y no
   necesita cambio.

**Riesgo en examen:** el más alto del lote junto con `Wolfram1984`. La escala de IV es un
elemento que un sinodal de cómputo puede verificar en treinta segundos, y el tramo mal rotulado
invierte la recomendación del autor citado.

---

## UNHabitat2020

**Metadatos:** CORRECTO

- evidencia: se descargó el PDF desde la URL de la entrada, HTTP 200, `application/pdf`,
  26 161 648 bytes, 418 páginas. Portada y página de créditos: «World Cities Report 2020 — The
  Value of Sustainable Urbanization. First published 2020 by United Nations Human Settlements
  Programme (UN-Habitat). Copyright © United Nations Human Settlements Programme, 2020»,
  con «HS/045/20E», «ISBN: 978-92-1-132872-1», «eISBN: 978-92-1-0054386»,
  «Print ISSN: 2518-6515».
  URL: https://unhabitat.org/sites/default/files/2020/10/wcr_2020_report.pdf
- correcciones: no hay errores, pero la entrada está **incompleta** para un `@misc` institucional.
  Añadir: `publisher = {United Nations Human Settlements Programme (UN-Habitat)}`,
  `address = {Nairobi}`, `isbn = {978-92-1-132872-1}` y `urldate`.

**Afirmaciones:**

1. `cap01_introduccion.tex:16` — proceso social de urbanización; fracción
   significativa y creciente de la población mundial vive en ciudades; la superficie urbana se
   expande más rápido que la población; «en particular esto se observa en México».
   **PARCIAL.** Dos de las tres piezas están verificadas en el informe; la tercera no.
   - Fracción significativa y creciente: **verificado.** «Urban areas are already home to 55 per
     cent of the world's population, and that figure is expected to grow to 68 per cent by 2050.»
   - La superficie urbana crece más rápido que la población: **verificado, y con cifras.**
     Encabezado de sección: «1.4.2. Urban footprints growing faster than urban population». Y en
     el resumen ejecutivo: «Cities are consuming land faster than they grow in population […] the
     physical extent of urban areas is growing much faster than their population, thereby
     consuming more land for urban development.» Con el dato: «Findings from a global sample of
     200 cities with over 100,000 inhabitants show that between 1990 and 2015, cities in developed
     countries increased their urban land area by 1.8-fold while the urban population increased by
     1.2-fold; thus, implying that the expansion of urban areas in relation to urban population
     growth increased by a ratio of 1.5.»
     URL: https://unhabitat.org/sites/default/files/2020/10/wcr_2020_report.pdf
   - «En particular esto se observa en México»: **no sostenido por esta fuente.** Búsqueda de
     «Mexico» en el informe: 89 apariciones, todas en otros contextos, migración en la frontera
     con Estados Unidos, la constitución de la Ciudad de México y el derecho a la ciudad,
     industrias creativas en la Ciudad de México. El informe no formula un enunciado sobre la
     relación entre expansión de superficie urbana y crecimiento poblacional específico de
     México. Tampoco lo hacen las otras dos referencias de la misma cita, `Seto2012` y
     `Angel2012`, que son de alcance global.
     Corrección sugerida: o se suprime el remate sobre México, o se respalda con una fuente
     mexicana, y la natural es la que la tesis ya usa en la línea 24,
     `ONUHabitat2016TendenciasMexico`, o datos del propio INEGI.
   - Observación aparte, de redacción, no de fuentes: la frase tiene erratas que la hacen
     ilegible, «La urbanizacipon en un proceso social», «una fracción significativa y que aparte
     es creciente de el total de población mundial», y el `\parencite` va después del punto. Es la
     primera página del capítulo 1. Conviene reescribirla completa.

**Riesgo en examen:** bajo en cuanto a fuentes, alto en cuanto a forma. La afirmación sobre México
es un añadido sin respaldo en ninguna de las tres referencias citadas.

---

## Wolfram1984

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.1038/311419a0` devuelve autor `Stephen Wolfram`;
  título `Cellular automata as models of complexity`; container `Nature`; volume `311`;
  issue `5985`; page `419-424`; issued `1984-10`. Confirmado en el PDF del propio artículo,
  distribuido por Wolfram: portada «nature […] Vol 311 No 5985 4-10 October 1984» y cabecera de
  la página 419 «NATURE VOL. 311 4 OCTOBER 1984 — REVIEW ARTICLE — Cellular automata as models of
  complexity — Stephen Wolfram — The Institute for Advanced Study, Princeton, New Jersey 08510,
  USA».
  URL: https://content.wolfram.com/sw-publications/2020/07/cellular-automata-models-complexity.pdf
- correcciones: ninguna obligatoria. Dos notas: el título original está en minúsculas, «Cellular
  automata as models of complexity», y el artículo es un *Review Article*, no un artículo de
  investigación. `@article` es adecuado. Importante para no confundirse: **esta entrada no es** el
  Wolfram de Physica D 10:1-35, *Universality and complexity in cellular automata*, que es donde
  están formalmente las cuatro clases; el artículo de *Nature* las resume.

**Afirmaciones:**

1. `cap02_marco_teorico.tex:74` — «Un Autómata Celular (AC), en el sentido
   formalizado por \textcite{Wolfram1984}, es un sistema dinámico discreto definido sobre una
   cuadrícula regular de $N$ celdas», con la cuádrupla $(Q, \mathcal{N}, \delta, q_0)$.
   **PARCIAL.**
   Lo que el artículo sí dice, literal, y respalda «sistema dinámico discreto»:
   «In the first approach, cellular automata are viewed as discrete dynamical systems […] or
   discrete idealizations of partial differential equations.»
   Y la única definición formal que ofrece es unidimensional:
   «A one-dimensional cellular automaton consists of a line of sites, with each site carrying a
   value 0 or 1 (or in general 0,..., k − 1). The value a_i of the site at each position i is
   updated in discrete time steps according to an identical deterministic rule depending on a
   neighbourhood of sites around it», seguida de la ecuación (1).
   URL: https://content.wolfram.com/sw-publications/2020/07/cellular-automata-models-complexity.pdf
   Lo que **no** contiene: la cuádrupla $(Q,\mathcal{N},\delta,q_0)$, la noción de rejilla finita
   de $N$ celdas, ni una definición bidimensional general. Búsqueda de `grid`: 0 apariciones.
   `lattice`: 1 aparición, y es sobre vórtices en fluidos turbulentos, no sobre la definición.
   `two-dimensional`: 2 apariciones, ambas de pasada, crecimiento dendrítico y el «Game of Life».
   El artículo trata además configuraciones infinitas, «The set of possible (infinite)
   configurations of a cellular automaton forms a Cantor set», que es lo contrario de un dominio
   de $N$ celdas.
   Nota favorable: la frase posterior de la tesis, línea 77, «Los autómatas unidimensionales, que
   Wolfram empleó para clasificar los comportamientos posibles del formalismo», es **exacta** y
   coincide con la fuente.
   Corrección sugerida: «Un Autómata Celular es un sistema dinámico discreto cuyas celdas se
   actualizan en paralelo según una regla local idéntica \parencite{Wolfram1984}; siguiendo la
   formalización habitual en modelado urbano, queda definido por la cuádrupla
   $(Q,\mathcal{N},\delta,q_0)$ sobre una cuadrícula de $N=R\times C$ celdas», y citar para la
   cuádrupla una fuente que la use, no a Wolfram.

2. `cap02_marco_teorico.tex:99` — pie de la Figura `fig:vecindades`: «Vecindad
   de Moore ($8$ celdas adyacentes) frente a Von Neumann ($4$ ortogonales). […] Imagen elaborada
   con IA generativa, con base en la definición de \textcite{Wolfram1984}.»
   **INCORRECTA. Es el problema más grave del lote.**
   Se extrajo el texto completo de las seis páginas del artículo, de la 419 a la 424, y se contaron
   apariciones: **`Moore` → 0 ocurrencias. `Neumann` → 0 ocurrencias.** El artículo no define, no
   nombra y no ilustra ninguna de las dos vecindades. Su única noción de vecindad es el radio $r$
   en una línea de sitios, ecuación (1), y los ejemplos son «k = 2, r = 1 rules with rule numbers
   128, 4 and 126» y «a k = 2, r = 2 rule with totalistic code 52».
   URL: https://content.wolfram.com/sw-publications/2020/07/cellular-automata-models-complexity.pdf
   Es una atribución de contenido a una fuente que no lo contiene, y además está en un pie de
   figura, donde es trivialmente comprobable.
   Corrección propuesta: cambiar la atribución del pie por una fuente que sí defina vecindades
   bidimensionales. La opción más cercana y del mismo autor es Packard, N. H. y Wolfram, S.
   (1985), *Two-dimensional cellular automata*, Journal of Statistical Physics 38:901-946,
   DOI `10.1007/BF01010423`, cuyos registros se verificaron en
   `https://link.springer.com/article/10.1007/BF01010423`, y que trata explícitamente reglas
   bidimensionales de 5 y 9 vecinos, es decir Von Neumann y Moore. Alternativa, si se prefiere
   mantener el marco urbano: `White1997`. Texto sugerido para el pie: «Imagen elaborada con IA
   generativa, con base en las vecindades bidimensionales de \textcite{PackardWolfram1985}».
   Nota: esto exige dar de alta una entrada nueva en `referencias.bib`; si se prefiere no tocar la
   bibliografía, la salida mínima es suprimir la atribución y dejar solo «Imagen elaborada con IA
   generativa».

**Riesgo en examen:** alto. Un sinodal que abra el artículo de *Nature* buscando Moore o Von
Neumann no encuentra nada, y el hallazgo desacredita por contagio el resto de las citas del
capítulo 2.

---

# Cierre

| Clave | Metadatos | Afirmaciones |
|---|---|---|
| Beven2006 | CORRECTO | 2 correctas |
| BonhamCarter1989 | CORRECTO (aviso 1989 vs 1990) | 2 correctas, 1 parcial |
| Chen2022 | CORRECTO | 1 correcta |
| Dietzel2007 | CORRECTO | 1 correcta, 2 parciales |
| Hagenauer2022 | CORRECTO | 1 correcta |
| JimenezLopez2021 | CORRECTO | 2 correctas |
| Jolliffe2002 | CORRECTO | 1 correcta |
| SEDATU2024NOM005 | CORRECTO | 1 correcta |
| Siddiqi2006 | CORRECTO (aviso 2005 vs 2006) | 1 correcta, 1 parcial, 1 **incorrecta** |
| UNHabitat2020 | CORRECTO (incompleto) | 1 parcial |
| Wolfram1984 | CORRECTO | 1 parcial, 1 **incorrecta** |

**Totales:** 11 referencias, 20 afirmaciones. 11 correctas, 7 parciales, 2 incorrectas,
0 metadatos incorrectos.

**VEREDICTO DEL LOTE: REQUIERE CORRECCIONES.**

Orden de prioridad para el implementador:
1. `cap02_marco_teorico.tex:99` — quitar la atribución de Moore y Von Neumann a Wolfram (1984).
2. `cap02_marco_teorico.tex:186` — corregir el quinto tramo del IV, de «muy fuerte» a
   «sospechoso», y comprobar si algún IV del modelo cae en ese tramo.
3. `cap01_introduccion.tex:16` — reescribir la frase, con sus erratas, y suprimir o respaldar el
   remate sobre México.
4. `cap03_estado_del_arte.tex:86` y `:155` — repartir explícitamente qué sostiene Dietzel y qué
   sostiene Beven, y no atribuir las tres fases a Clarke y Hoppen (1997).
5. `cap03_estado_del_arte.tex:310` — mover la prueba por pares de Bonham-Carter (1989) a
   Bonham-Carter (1994, cap. 9), que es la atribución documentada.
6. `cap02_marco_teorico.tex:74` — no presentar la cuádrupla como formalización de Wolfram (1984).
7. `referencias.bib` — completar `UNHabitat2020` con `publisher`, `address` e `isbn`.
8. Opcional, al alza: `cap01_introduccion.tex:24`, incorporar las cifras oficiales de la NOM,
   567 municipios, 22,94 %, casi 45 % con más de quince años.
