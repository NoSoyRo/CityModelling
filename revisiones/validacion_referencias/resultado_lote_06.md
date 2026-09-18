# VALIDACIÓN DE FUENTES — LOTE 6 de 8 (12 referencias)

## aguilar2003urbanization
**Metadatos:** INCORRECTO (referencia fantasma)
- evidencia: búsqueda por título en Crossref `https://api.crossref.org/works?query.bibliographic=Urbanization,%20population%20growth,%20and%20employment%20in%20Mexico` → ningún registro con ese título. Búsqueda dirigida en el ISSN de *Cities* (0264-2751), autor Aguilar, 2002–2004: `https://api.crossref.org/journals/0264-2751/works?query.author=Aguilar` devuelve **un solo** artículo en el vol. 20 núm. 1: Aguilar, Adrián G.; Ward, Peter M.; Smith Sr, C. B. (2003), «Globalization, regional development, and mega-city expansion in Latin America: Analyzing Mexico City's peri-urban hinterland», *Cities* 20(1):3–21, DOI `10.1016/S0264-2751(02)00092-6`. Búsqueda web del título entre comillas: no existe tal publicación; lo más cercano es el libro coordinado Aguilar (2003), *Urbanización, cambio tecnológico y costo social. El caso de la región centro de México*, UNAM-IG/CONACYT/M. Á. Porrúa, 336 pp., ISBN 970-701-361-3 (`https://maporrua.com.mx/product/urbanizacion-cambio-tecnologico-y-costo-social/`).
- correcciones: la entrada mezcla los metadatos reales de Aguilar/Ward/Smith (revista, volumen, número, año) con un título y una autoría únicos inventados.
  - `title = {Urbanization, population growth, and employment in México}` -> `{Globalization, regional development, and mega-city expansion in Latin America: Analyzing Mexico City's peri-urban hinterland}`
  - `author = {Aguilar, Adrian Guillermo}` -> `{Aguilar, Adri{\'a}n Guillermo and Ward, Peter M. and Smith Sr, Chris B.}`
  - `pages = {3--19}` -> `{3--21}`
  - falta `doi = {10.1016/S0264-2751(02)00092-6}`

**Afirmaciones:**
1. `cap01_introduccion/cap01_introduccion.tex:22` — «las ciudades intermedias […] mexicanas, poco representadas en la literatura de modelado urbano pese a concentrar buena parte del crecimiento reciente del país». **INCORRECTA.** No hay fuente: la entrada no corresponde a ninguna publicación. Además, el artículo real cuyos metadatos se usaron trata de la **megaciudad** de la Ciudad de México y su periferia periurbana, no de ciudades intermedias, y no dice nada sobre representación en la literatura de **modelado** urbano. URL: `https://api.crossref.org/journals/0264-2751/works?query.author=Aguilar&filter=from-pub-date:2002-01-01,until-pub-date:2004-12-31`

**Riesgo en examen:** ALTO. Es una referencia inexistente sosteniendo una de las premisas de la introducción. Si un sinodal la busca, no la encuentra. Sustituir por una fuente que efectivamente documente el peso de las ciudades intermedias mexicanas (p. ej. el SUN de CONAPO/SEDATU) y separar la afirmación sobre la literatura de modelado, que debe respaldarse con el propio estado del arte del Cap. 3.

---

## almeida2008stochastic
**Metadatos:** CORRECTO
- evidencia: `https://api.crossref.org/works/10.1080/13658810701731168` → *International Journal of Geographical Information Science* 22(9):943–963, 2008; autores C. M. Almeida, J. M. Gleriani, E. F. Castejon, B. S. Soares-Filho; Informa UK (Taylor & Francis). Coincide con dblp (`https://dblp.uni-trier.de/rec/journals/gis/AlmeidaGCS08.html`).
- correcciones: ninguna. (Nota menor: el título en el registro del editor usa guiones de división «intra-urban land-use»; el `.bib` está bien.)

**Afirmaciones:**
1. `cap01:117` — «Los modelos de caja negra basados en aprendizaje automático […] alcanzan concordancia espacial alta en los casos publicados, pero sus pesos internos no son legibles como evidencia espacial». **PARCIAL.** El resumen confirma que la validación se hizo «based on fuzzy similarity measures», pero no pude acceder al texto completo (Unpaywall: `is_oa=False, oa_status=closed`) para verificar que los valores sean «altos», y **los autores no caracterizan su modelo como caja negra ni declaran esa limitación**: es una lectura del tesista. Recomendación: atribuir la crítica a la tesis («no permiten leer…») y no a los autores. URL del resumen depositado por la autora: `http://urlib.net/sid.inpe.br/mtc-m18@80/2008/07.15.23.03`
2. `cap03:55` — «acopló una red neuronal feedforward a un autómata celular estocástico para Piracicaba, Brasil». **CORRECTA.** Cita textual del resumen: «a supervised back-propagation neural network has been employed in the parameterization of several biophysical and infrastructure variables […] The spatial land-use transition probabilities estimated thereof feed a cellular automaton (CA) simulation model, based on stochastic transition rules. The model has been tested in a medium-sized town in the Midwest of São Paulo State, Piracicaba.» URL: `http://urlib.net/sid.inpe.br/mtc-m18@80/2008/07.15.23.03`
3. `cap03:155` — «los trabajos de aprendizaje automático y profundo reportan mejoras de concordancia espacial en sus propios casos de estudio». **PARCIAL.** El paper reporta validación por similitud difusa sobre «the best results», pero no pude leer el texto completo para confirmar que reporte una **mejora** frente a una línea base. Misma URL.
4. `cap03:179` — «entrenó una red neuronal feedforward con backpropagation sobre variables biofísicas e infraestructurales de Piracicaba (Brasil, 1985--1999); la red aprende pesos sinápticos que alimentan a un CA estocástico». **CORRECTA** en todos sus elementos verificables: backpropagation ✓, variables biofísicas e infraestructurales ✓, Piracicaba ✓, 1985–1999 ✓ («A series of simulation outputs for the case study town in the period 1985-1999»), alimenta un CA estocástico ✓. Misma URL.

**Riesgo en examen:** BAJO para las afirmaciones 2 y 4 (descripción exacta del caso y del método). MEDIO para 1 y 3: si un sinodal pregunta «¿qué concordancia reportaron y contra qué?», no hay cifra. Reformular como «reportan validación por medidas de similitud difusa» en lugar de «concordancia espacial alta».

---

## Brenning2012
**Metadatos:** CORRECTO
- evidencia: `https://api.crossref.org/works/10.1109/igarss.2012.6352393` → proceedings-article, *2012 IEEE International Geoscience and Remote Sensing Symposium*, pp. 5372–5375, Alexander Brenning, IEEE, 2012-07. Coincide con dblp `conf/igarss/Brenning12` y con la cita canónica del paquete (`https://github.com/giscience-fsu/sperrorest/`).
- correcciones: ninguna.

**Afirmaciones:**
1. `cap03:344` — «la sobreestimación del desempeño que resulta de tratar observaciones espacialmente dependientes como independientes está documentada en la literatura de validación de modelos espaciales \parencite{Legendre1993,Brenning2012}». **CORRECTA con matiz.** El resumen sostiene la necesidad de corregir por dependencia espacial: «the accuracy assessment of such predictive models in a spatial context needs to account for the presence of spatial autocorrelation in geospatial data by using spatial cross-validation and bootstrap strategies instead of their now more widely used non-spatial equivalent» (URL: `https://www.researchgate.net/publication/261228271_Spatial_cross-validation_and_bootstrap_for_the_assessment_of_prediction_rules_in_remote_sensing_The_R_package_sperrorest`). La palabra **«sobreestimación» no aparece en el resumen**; la frase «overly optimistic error estimates» atribuida a Brenning (2012) proviene de literatura que lo cita, no de una página del artículo que yo haya abierto. El texto completo es cerrado (Semantic Scholar: `openAccessPdf.status = CLOSED`; DOI `10.1109/IGARSS.2012.6352393`), y `ieeexplore.ieee.org/document/6352393` exige JavaScript. La afirmación está redactada con prudencia («documentada en la literatura»), lo que la hace defendible.

**Riesgo en examen:** BAJO. La cita es plural y genérica, y el resumen del propio artículo la respalda. No afirmar que Brenning «cuantificó» la sobreestimación.

---

## ClarkeHoppen1997
**Metadatos:** CORRECTO
- evidencia: `https://api.crossref.org/works/10.1068/b240247` → Clarke, K. C.; Hoppen, S.; Gaydos, L., *Environment and Planning B: Planning and Design* 24(2):247–261, abril 1997, SAGE. PDF abierto verificado: `https://www.macs.hw.ac.uk/~dwcorne/Teaching/sfbay.pdf` («Environment and Planning B: Planning and Design 1997, volume 24, pages 247 - 261»).
- correcciones: ninguna. **Advertencia:** la regla del repositorio `.cursor/rules/proyecto-tesis.mdc` fecha este artículo en **1998**; el año correcto es **1997** y el `.bib` está bien. Corregir la regla, no la bibliografía.

**Afirmaciones:**
1. `cap03:86` — «El procedimiento habitual estrecha la búsqueda en tres fases sucesivas (gruesa, fina y final), esquema planteado ya en la formulación original del autómata auto-modificable \parencite{ClarkeHoppen1997}». **INCORRECTA (error de atribución).** Busqué en el texto completo del PDF de 1997 los términos `coarse`, `fine`, `brute force`: **ninguno aparece**. Lo que el artículo describe es otra cosa: «Calibration consisted of four steps. The first of these was validation. In this step, the model was allowed to run to completion for a single iteration with unit increments in the control parameters and with self-modification disabled»; el «second phase of calibration involved writing two versions of the program with a full set of graphical user-interface tools» y «a third-phase batch version of the model, without graphics». Es decir, **cuatro pasos de desarrollo y prueba de software**, no el estrechamiento gruesa→fina→final del espacio de parámetros. El esquema de tres fases se atribuye en la literatura a Silva y Clarke (2002) / Project Gigalopolis-USGS, y lo describe Clarke (2008): «Three phases are used in calibration, with user choices between them. At first, large increments of the parameters are used» (URL: `http://www.ncgia.ucsb.edu/projects/gig/Pub/SLEUTHPapers_Nov24/Clarke_Lincoln2008.pdf`). Corroborado además por Jantz y Goetz (2005), que atribuyen el procedimiento a «Candau 2002, Silva and Clarke 2002, US Geological Survey 2003» y **no** a Clarke et al. 1997 (URL: `https://www.woodwellclimate.org/wp-content/uploads/2015/09/JantzetalJGIS.05.pdf`).
   - Corrección propuesta: «…esquema formalizado en la práctica de calibración por fuerza bruta de SLEUTH \parencite{Silva2002,Clarke2008} y discutido después de forma sistemática por \textcite{Dietzel2007}». (La tesis ya cita `Silva2002,Clarke2008` dos oraciones antes, así que el arreglo es local.)
   - URL del PDF de 1997 consultado: `https://www.macs.hw.ac.uk/~dwcorne/Teaching/sfbay.pdf`
2. `cap03:96` — «los coeficientes de crecimiento se reescriben durante la propia corrida […] el resultado no solo depende de los valores iniciales, sino también de la trayectoria que se sigue, lo que da lugar a un sistema muy sensible a cambios». **CORRECTA.** Resumen: «the control parameters of the model are allowed to self-modify: that is, the CA adapts itself to the circumstances it generates, in particular, during periods of rapid growth or stagnation». Y en el cuerpo: «The self-modification rules, summarized in figure 6, allow much control of the system from only two factors: a 'critical high' growth rate and a 'critical low' growth rate»; además «Such a system, when it involves randomization, always results in chaotic behavior […] the cities should show extreme variation between multiple iterations in the Monte Carlo sense […] The statistics derived in this calibration indeed bear this out». URL: `https://www.macs.hw.ac.uk/~dwcorne/Teaching/sfbay.pdf`

**Riesgo en examen:** ALTO para la afirmación 1. Es exactamente el tipo de atribución que un sinodal que conoce SLEUTH puede refutar abriendo el paper de 1997. La afirmación 2, en cambio, es sólida y citable casi literalmente.

---

## Gong2013
**Metadatos:** CORRECTO con un error de nombre
- evidencia: `https://api.crossref.org/works/10.1080/01431161.2012.748992` → *International Journal of Remote Sensing* 34(7):2607–2654; primer autor Peng Gong; Informa UK. Publicación en línea 2012-12-21, número de revista 2013 (el año 2013 del `.bib` es el correcto para el fascículo).
- correcciones:
  - `Huang, Xiaomei` -> `Huang, Xiaomeng` (Crossref lista **Xiaomeng** Huang, no Xiaomei).
  - El registro de Crossref trae 43 autores; el `and others` del `.bib` es admisible, pero conviene verificar que el orden de los diez primeros coincida: en Crossref el 8.º es Xiaomeng Huang y el 9.º Haohuan Fu, igual que en la entrada.

**Afirmaciones:**
1. `cap02_marco_teorico:24` — «Los programas de observación de larga duración, en particular la familia Landsat, hicieron posible construir **series históricas** de cobertura del suelo **con décadas de profundidad** y cartografiarlas a escala global \parencite{Gong2013}». **PARCIAL.** Gong et al. (2013) respalda la segunda mitad (cartografía global con Landsat TM/ETM+ a 30 m) pero **no** la primera: es un mapa de **una sola época**, no una serie multidecadal. Cita textual: «A total of 8929 Landsat TM/ETM+ scenes were collected […] About 74% of the imagery was acquired after 2006 […] approximately three quarters of the imagery is circa 2010 and one quarter is circa 2000. **Only 18 scenes acquired before 1998 were used**». Y el propio título dice «**first mapping results**». URL: `https://www.tandfonline.com/doi/full/10.1080/01431161.2012.748992`
   - Corrección propuesta: dividir la afirmación. Para la cartografía global a 30 m con Landsat, mantener `Gong2013`. Para «series históricas con décadas de profundidad», citar una fuente de series temporales Landsat (p. ej. Hansen et al. 2013, o el propio archivo Landsat) o remitir a `Weng2002`, que ya se cita en la misma oración.

**Riesgo en examen:** MEDIO. La cifra no está mal, pero la fuente está sobreextendida: quien conozca FROM-GLC sabe que es un producto de época única.

---

## INEGI2020Censo
**Metadatos:** CORRECTO
- evidencia: la URL `https://www.inegi.org.mx/programas/ccpv/2020/` resuelve y corresponde al programa «Censo de Población y Vivienda (CPV) 2020», INEGI, cobertura temporal 2020 (metadatos schema.org en la propia página).
- correcciones: ninguna obligatoria. Sugerencia: el título oficial de la página es «Censo de Población y Vivienda (CPV) 2020»; el `.bib` usa la forma sin siglas, que también es la que INEGI emplea en sus publicaciones. Para las dos afirmaciones numéricas conviene citar el tabulado o el comunicado concreto y no la página del programa.

**Afirmaciones:**
1. `cap01:24` — «El INEGI documenta que la mayor parte de la población reside en localidades urbanas». **CORRECTA.** Cita textual del INEGI: «En 2020, 79% de la población reside en localidades de 2 500 habitantes o más, mientras que 21% vive en localidades de menor tamaño (menos de 2 500 habitantes)». URL: `https://www.inegi.org.mx/contenidos/saladeprensa/aproposito/2021/EAP_POBLAC21.pdf`
2. `cap01:66` — «una zona metropolitana que rebasó el millón y medio de habitantes tras décadas de expansión sostenida». **CORRECTA.** Con la delimitación oficial de 2018 (cinco municipios, incluido Apaseo el Alto, Gto.) la ZM de Querétaro registró **1 594 212** habitantes en 2020; con los cuatro municipios queretanos, **1 530 820**. Ambas cifras rebasan 1,5 millones. La serie 579 597 (1990) → 816 481 (2000) → 1 097 025 (2010) → 1 530 820 (2020) respalda «décadas de expansión sostenida». URLs: `https://es.wikipedia.org/wiki/Zona_metropolitana_de_Quer%C3%A9taro` (tabla censal por municipio) y `https://ru.iiec.unam.mx/7584/1/243-Bustamante-Final.pdf` (Tabla 2, misma serie, elaborada con censos INEGI 2000–2020).
3. `cap05:40` — «agrupa varios municipios en torno a la capital del estado y […] rebasó el millón y medio de habitantes, tras décadas de expansión sostenida». **CORRECTA**, mismas evidencias. Los municipios son Querétaro, Corregidora, El Marqués y Huimilpan (más Apaseo el Alto, Gto., en la delimitación de 2018).

**Riesgo en examen:** BAJO, pero conviene blindarlo: decir explícitamente cuántos municipios y con qué delimitación, porque 1 530 820 y 1 594 212 son ambas «correctas» según el criterio, y un sinodal puede pedir la cifra exacta. Citar además el comunicado `EAP_POBLAC21.pdf` para el 79 %.

---

## JantzGoetz2005
**Metadatos:** CORRECTO
- evidencia: `https://api.crossref.org/works/10.1080/13658810410001713425` → Claire A. Jantz y Scott J. Goetz, *International Journal of Geographical Information Science* 19(2):217–241, febrero 2005. Confirmado en el catálogo ENSG/IGN: «vol 19 n° 2 (february 2005), pp 217 - 241» (`https://documentation.ensg.eu/index.php?id=27187&lvl=notice_display`). PDF abierto: `https://www.woodwellclimate.org/wp-content/uploads/2015/09/JantzetalJGIS.05.pdf`
- correcciones: ninguna.

**Afirmaciones:**
1. `cap03:96` — «\textcite{JantzGoetz2005} **demuestran** que un conjunto muy ajustado a una escala **no se transfiere** a otra». **PARCIAL.** Lo que el artículo demuestra es la **inestabilidad de los coeficientes según el tamaño de celda**, no un experimento de transferencia de un conjunto calibrado de una escala a otra: los autores corrieron **cuatro calibraciones gruesas independientes**, una por resolución («We performed four separate coarse calibration procedures, using seven Monte Carlo iterations for each cell size»). Sus conclusiones: «The fact that some parameters exhibit stability at certain cell sizes and not at others indicates that scale influences the parameters differently […] **Parameter instability poses an issue for calibration**, since it makes the choice of appropriate parameter value ranges» y «We also observed significant differences in the sensitivity of the growth rules across cell sizes, indicating that SLEUTH may perform better at certain cell sizes than at others». Matiz que la tesis omite y que un sinodal puede usar: «**While the model was able to capture the rate of growth reliably across all cell sizes**, differences in its ability to simulate growth patterns across scales were substantial» — es decir, la tasa sí se reproduce a todas las escalas; lo que no se transfiere es el **patrón** y la importancia relativa de las reglas.
   - Corrección propuesta: «\textcite{JantzGoetz2005} muestran que la sensibilidad de las reglas de crecimiento y la estabilidad de los coeficientes dependen del tamaño de celda: el modelo reproduce la tasa de crecimiento a todas las escalas, pero no el patrón, lo que compromete la transferibilidad de un conjunto calibrado entre resoluciones.»
   - URL: `https://www.woodwellclimate.org/wp-content/uploads/2015/09/JantzetalJGIS.05.pdf`

**Riesgo en examen:** MEDIO. El verbo «demuestran» y el sujeto «un conjunto muy ajustado» exceden lo que el diseño experimental del artículo permite concluir. Es un ajuste de redacción, no un cambio de argumento: la conclusión de fondo (los coeficientes de SLEUTH son sensibles a la resolución) sí es del artículo.

---

## Liu2017
**Metadatos:** CORRECTO
- evidencia: `https://api.crossref.org/works/10.1016/j.landurbplan.2017.09.019` → *Landscape and Urban Planning* 168:94–116, diciembre 2017; nueve autores en el orden exacto del `.bib` (Liu, Liang, Li, Xu, Ou, Chen, Li, Wang, Pei); Elsevier BV. PDF de los autores: `http://www.geosimulation.cn/Papers/2017LUP-FLUS.pdf`
- correcciones: ninguna.

**Afirmaciones:**
1. `cap03:41` — «sistemas como CLUE-S, Dinamica EGO, CA-Markov y, más recientemente, FLUS \parencite{Liu2017} brindaron arquitecturas alternativas al autómata celular puro **en las que la función de transición emerge de regresiones logísticas o pesos bayesianos** sobre datos históricos». **PARCIAL / INCORRECTA en lo que toca a FLUS.** La función de transición de FLUS **no** proviene de una regresión logística ni de pesos bayesianos, sino de una **red neuronal artificial**, y el propio artículo contrapone explícitamente ambas cosas: «We chose to base FLUS on an ANN (artificial neural network) because the ANN algorithm was proven to be an effective way to map the complex, nonlinear relationship between historical land use and various ancillary data sources, and **it is stronger than other simple methods such as the logistic regression**». Además la tesis se contradice: en `cap03:288` describe correctamente que FLUS «sustituye esa idoneidad por la probabilidad de una red neuronal». Corrección propuesta: excluir FLUS de la cláusula «regresiones logísticas o pesos bayesianos», p. ej. «…y, más recientemente, FLUS \parencite{Liu2017}, cuya idoneidad proviene de una red neuronal». URLs: `http://www.geosimulation.cn/Papers/2017LUP-FLUS.pdf` y `http://www.geosimulation.cn/Papers/UGB-FLUS.pdf`
2. `cap03:288` — «FLUS \parencite{Liu2017} sustituye esa idoneidad por la probabilidad de una red neuronal y la asigna mediante un autómata celular con **inercia auto-adaptativa** y **competencia por ruleta**». **CORRECTA**, en los tres elementos. Cita textual: «1) an artificial neural network is used to train and estimate the probability-of-occurrence of each land use type on a specific grid cell, and 2) an elaborate **self-adaptive inertia and competition mechanism** is designed to address the competition and interactions among the different land use types […] In the allocation process, a specific land grid either retains the current land use type or transforms into another type depending on their combined probabilities and the **roulette selection**». URL: `http://www.geosimulation.cn/Papers/2017LUP-FLUS.pdf`

**Riesgo en examen:** MEDIO. La afirmación 2 es impecable y casi literal. La 1 es un error de descripción de método que, además, choca con lo que la propia tesis dice 250 líneas después; un sinodal atento lo notará como inconsistencia interna.

---

## ONUHabitat2016TendenciasMexico
**Metadatos:** PARCIALMENTE INCORRECTO (año)
- evidencia: la URL `https://onu-habitat.org/index.php/tendencias-del-desarrollo-urbano-en-mexico` resuelve, el título coincide exactamente («Tendencias del desarrollo urbano en México») y la página está fechada **«20 junio 2017»**. Al pie cita como fuentes SEDATU (2013), *Programa Nacional de Desarrollo Urbano y Ordenamiento del Territorio*, y ONU-Habitat México (2016), *Índice de prosperidad urbana en la República Mexicana. Reporte Nacional de Tendencias de la Prosperidad Urbana en México*, 78 pp. — que es justo lo que declara la `note` del `.bib`.
- correcciones: `year = {2016}` -> `{2017}` (fecha de la página web citada). Alternativa preferible: citar directamente el reporte ONU-Habitat México (2016) como `@report`, ya que es la fuente sustantiva; el `note` actual reconoce esto pero el `url`/`title` apuntan a la nota de divulgación.

**Afirmaciones:**
1. `cap01:24` — «Su cobertura efectiva es incompleta, y una parte de los planes publicados no se ha actualizado \parencite{SEDATU2024NOM005,ONUHabitat2016TendenciasMexico}». **PARCIAL** respecto de esta fuente. Leí la página completa y **no dice nada sobre cobertura de planes de ordenamiento ni sobre planes sin actualizar**. Lo que sí sostiene, y es lo más cercano, es la debilidad institucional para planear: «las tendencias de la urbanización mexicanas, además de ser muy dinámicas, enfrentan riesgos derivados de una **limitada institucionalidad para la gobernanza**» y «las tendencias de los gobiernos locales se orientan a un **debilitamiento de la gobernanza urbana por falta de capacidades técnicas, organizativas, de información y sobre todo de recursos** para enfrentar la acelerada urbanización».
   - Corrección propuesta: si `SEDATU2024NOM005` no documenta la desactualización de los planes, reformular a «…y las capacidades técnicas y financieras de los gobiernos locales para ejercerlos son limitadas \parencite{ONUHabitat2016TendenciasMexico}», que es literalmente lo que la fuente afirma.
   - URL: `https://onu-habitat.org/index.php/tendencias-del-desarrollo-urbano-en-mexico`

**Riesgo en examen:** MEDIO. La cifra central de la página (83,2 % de población urbana en 961 ciudades hacia 2030, proyección del SUN-CONAPO) no es la que se usa, y la afirmación que se le cuelga no está en el texto.

---

## SuarezDelgado2007
**Metadatos:** CORRECTO
- evidencia: `https://api.crossref.org/works/10.24201/edu.v22i1.1295` → Manuel Suárez y Javier Delgado, *Estudios Demográficos y Urbanos* 22(1), 2007, El Colegio de México (Crossref registra sólo la página inicial, 101). SciELO México confirma el rango completo: «Estud. demogr. urbanos [online]. 2007, vol.22, n.1, pp.**101-142**» (`https://www.scielo.org.mx/scielo.php?lng=es&pid=S0186-72102007000100101&script=sci_isoref`); Redalyc y Dialnet coinciden (`https://www.redalyc.org/pdf/312/31222105.pdf`, «pp. 101-142»).
- correcciones: ninguna. El rango `101--142` del `.bib` es correcto aunque Crossref muestre sólo `101`.

**Afirmaciones:**
1. `cap01:119` — «Los trabajos representativos para ciudades mexicanas \parencite{SuarezDelgado2007,...} satisfacen parcialmente R1 y R2, pero **no R3 ni R4**». **INCORRECTA para SuarezDelgado2007, y además contradice la propia tesis.** (a) R4 es «interpretabilidad de la función de transición: sus pesos deben ser legibles como evidencia espacial cuantificable, no como parámetros opacos de una red neuronal» (`cap01:103-106`). Suárez y Delgado usan una **regresión logística binomial** cuyos coeficientes reportan e **interpretan uno por uno**: «La variable ingreso muestra un coeficiente positivo, ya que a mayor ingreso, mayor probabilidad de urbanización». Eso **satisface** R4. (b) La tabla `tab:cuadrante-mexico` de `cap03:472` asigna a este mismo trabajo **R1 = no y R4 = ✓**, mientras `cap01:119` dice «parcialmente R1 […] no R4». Las dos afirmaciones no pueden ser ciertas a la vez. URL: `https://www.redalyc.org/pdf/312/31222105.pdf`
2. `cap03:390` — «constituyen uno de los primeros ejercicios de pronóstico espacialmente explícito para la ZMCM, con antecedentes en los escenarios de poblamiento de CONAPO y una metodología adaptada de la literatura estadounidense sobre asignación de suelo». **CORRECTA en sus dos cláusulas verificables; la prioridad («uno de los primeros») es NO VERIFICABLE.** CONAPO: «Un primer escenario parte de las proyecciones de Conapo, por municipio […] los siguientes dos escenarios alternos de poblamiento parten también de la proyección de población total de Conapo para la Zona Metropolitana». Literatura estadounidense de asignación de suelo: el artículo declara explícitamente su fuente metodológica — «[el método que] utilizaron **Landis y Reilly (2003) en su pronóstico del crecimiento urbano en el estado de California hacia el año 2100** […] y sigue tres pasos fundamentales: Primero, se crea un modelo estadístico espacial calibrado para predecir el crecimiento urbano en un periodo observado a partir de los datos socioeconómicos del mismo». Sobre «uno de los primeros» no encontré base en el artículo ni forma de verificar la prioridad histórica; es un juicio del tesista. URL: `https://www.redalyc.org/pdf/312/31222105.pdf`
3. `cap03:472` (tabla, fila `SuarezDelgado2007` = no / ✓ / no / ✓). **CORRECTA y mejor fundada que la de `cap01:119`.** R1 = no: búsqueda en el texto completo de `kappa`, `bondad de ajuste`, `pseudo R`, `aciertos`, `correctamente clasificad` y **no aparece ninguna métrica de validación espacial**; el artículo calibra sobre 1989–1990 y sustituye valores para los años base de predicción (1999–2000), sin reportar concordancia. R2 = ✓: produce mapas de probabilidad de urbanización y tres escenarios (E1, E2 pesimista, E3), con expansión estimada «de entre 38 mil y 56 mil hectáreas». R3 = no: usa SIG propietario y datos INEGI no reproducibles según se describen. R4 = ✓: coeficientes de regresión logística reportados e interpretados. URL: `https://www.redalyc.org/pdf/312/31222105.pdf`

**Riesgo en examen:** ALTO por la **contradicción interna** entre `cap01:119` y la tabla de `cap03:472`. Un sinodal que compare ambas páginas encontrará que la tesis se califica a sí misma con dos varas distintas sobre el mismo trabajo. Hay que armonizarlas, y la versión correcta es la de la tabla.

---

## Tucker1979
**Metadatos:** CORRECTO
- evidencia: `https://api.crossref.org/works/10.1016/0034-4257(79)90013-0` → Compton J. Tucker, «Red and photographic infrared linear combinations for monitoring vegetation», *Remote Sensing of Environment* 8(2):127–150, mayo 1979, Elsevier BV.
- correcciones: ninguna.

**Afirmaciones:**
1. `cap02_marco_teorico:36` — «pueden construirse índices que realzan la presencia de vegetación, como el índice verde--rojo normalizado (NGRDI) **o los índices de exceso de verde y de exceso de rojo** \parencite{Tucker1979}». **PARCIAL: correcta para NGRDI, INCORRECTA para ExG y ExR.**
   - NGRDI → Tucker (1979): **correcto**. La base canónica *Awesome Spectral Indices* mapea `NGRDI, (G - R) / (G + R)` al DOI `10.1016/0034-4257(79)90013-0` (URL: `https://zenodo.org/records/7728129/files/spectral-indices-table.csv?download=1`), y la literatura lo confirma: «Na sua pesquisa nota-se a primeira aparição do índice denominado Normalized Green Red Difference Index - NGRDI» (URL: `https://periodicos.ufpe.br/revistas/index.php/jhrs/article/download/242924/34089/158332`).
   - **Exceso de verde (ExG = 2G − R − B) → Woebbecke, Meyer, Von Bargen y Mortensen (1995)**, «Color Indices for Weed Identification Under Various Soil, Residue, and Lighting Conditions», *Transactions of the ASAE* 38(1):259–269, DOI `10.13031/2013.27838`. La misma base lo registra con ese DOI, no con el de Tucker (`https://zenodo.org/records/7728129/files/spectral-indices-table.csv?download=1`).
   - **Exceso de rojo (ExR = 1.3R − G) → Meyer et al. (1998/1999)**: «Meyer et al. (1998) también desarrolló un índice […] el cual fue intitulado Excess Red Vegetative Index – ExR» (URL: `https://periodicos.ufpe.br/revistas/index.php/jhrs/article/download/242924/34089/158332`).
   - Corrección propuesta: «…como el índice verde–rojo normalizado (NGRDI) \parencite{Tucker1979} o los índices de exceso de verde \parencite{Woebbecke1995} y de exceso de rojo \parencite{Meyer1998}». Requiere dos entradas nuevas en `referencias.bib`.

**Riesgo en examen:** MEDIO-ALTO. Atribuir ExG y ExR a un artículo de 1979 sobre combinaciones rojo/infrarrojo es verificable en un minuto y es del tipo de detalle que un sinodal de teledetección detecta de inmediato. Además el capítulo describe features que el código sí calcula, así que la corrección es de cita, no de método.

---

## YinYan1988
**Metadatos:** CORRECTO (verificado por vía indirecta; el original no está digitalizado)
- evidencia: no existe DOI ni versión electrónica de las actas de 1988. La cita aparece de forma consistente en tres fuentes independientes que la referencian de primera mano: «Yin KL, Yan TZ (1988) Statistical prediction model for slope instability of metamorphosed rocks. In: **Bonnard C (ed)** Proc 5th Int Symp Landslides, Lausanne. **Balkema, Rotterdam**, vol 2, pp **1269–1272**» (URL: `https://link.springer.com/article/10.1007/s00254-003-0917-8`); idéntico en `https://www.issmge.org/uploads/publications/105/106/ISL2020-70.pdf` y en `http://www.geosocindia.org/index.php/jgsi/article/view/57472`.
- correcciones: opcionales, para completar la entrada. `editor = {Bonnard, C.}`, `publisher = {Balkema}`, `address = {Rotterdam}` (el `address = {Lausanne}` actual es la **sede del simposio**, no el lugar de edición; la convención BibTeX espera el lugar del editor). Nota: parte de la literatura cita el título en plural, «Statistical prediction model**s**…»; la forma singular del `.bib` es la mayoritaria.

**Afirmaciones:**
1. `cap02_marco_teorico:179` — «en la literatura de susceptibilidad geoespacial el término \textit{information value} designa un método bivariado distinto \parencite{YinYan1988}, que aquí no se emplea». **CORRECTA.** La taxonomía estándar del campo separa ambos métodos: «Information value method (Kobashi and Suzuki, 1988; **Yin and Yan, 1988**), **weight of evidence modeling method** (Spiegelhalter, 1986; Bonham-Carter, 1996)» — dos entradas distintas de la lista (URL: `https://users.metu.edu.tr/suzen/phd/ch2.pdf`). Y es efectivamente bivariado: «The bivariate statistical method applied is the Information Value (**Yin and Yan, 1988**) […] IVij = ln[(Si/Ni)/(S/N)]» (URL: `https://www.issmge.org/uploads/publications/105/106/ISL2020-70.pdf`).
   - Matiz que conviene anticipar: la misma fuente señala que IV y WoE son **formalmente próximos** — «in the information value method the log value of the quotient of class density over map density is entered, whereas in the susceptibility method the difference in densities was used» (`https://users.metu.edu.tr/suzen/phd/ch2.pdf`). Es decir, «distinto» es correcto como **método nombrado y como fórmula**, pero no conviene presentarlos como conceptualmente ajenos: ambos son razones de densidad log-transformadas. La aclaración de la tesis (que su IV es el de la tradición de scoring, no el de Yin y Yan) es pertinente y está bien planteada.

**Riesgo en examen:** BAJO. La distinción terminológica es correcta y además prudente: previene precisamente la confusión que un sinodal podría plantear. Sólo conviene no negar el parentesco formal con WoE.

---

## Resumen del lote

**Referencias revisadas: 12.** Metadatos verificados contra Crossref, editor, SciELO/Redalyc, INEGI y PDFs de acceso abierto; afirmaciones verificadas leyendo el texto de la fuente cuando fue accesible.

**Metadatos incorrectos: 3** (1 grave, 2 menores). **Afirmaciones: 21 revisadas** → 9 correctas, 8 parciales, 3 incorrectas, 1 no verificable.

Problemas graves, en orden de gravedad:

1. **`aguilar2003urbanization` — referencia fantasma.** `cap01:22`. El título «Urbanization, population growth, and employment in México» con autoría única de Aguilar **no existe**; los metadatos (Cities 20(1), 2003) son los de Aguilar, Ward y Smith, «Globalization, regional development, and mega-city expansion in Latin America», pp. **3–21**, DOI 10.1016/S0264-2751(02)00092-6 — un artículo sobre la **megaciudad** de México, que no respalda la afirmación sobre ciudades intermedias.
2. **`ClarkeHoppen1997` — atribución falsa.** `cap03:86`. Las tres fases «gruesa, fina y final» **no están** en el paper de 1997 (no contiene «coarse», «fine» ni «brute force»; describe cuatro pasos de desarrollo de software). El esquema es de **Silva y Clarke (2002)/USGS**, como confirma Clarke (2008) y como atribuye el propio Jantz y Goetz (2005).
3. **`SuarezDelgado2007` — contradicción interna.** `cap01:119` («no R3 ni R4») choca con la tabla de `cap03:472` (R1 = no, **R4 = ✓**). La tabla es la correcta: el trabajo usa regresión logística con coeficientes reportados e interpretados, luego **sí** satisface R4.
4. **`Tucker1979` — atribución parcialmente falsa.** `cap02:36`. NGRDI sí es de Tucker (1979); **exceso de verde es de Woebbecke et al. (1995)** (DOI 10.13031/2013.27838) y **exceso de rojo de Meyer et al. (1998)**.
5. **`Liu2017` — descripción de método incorrecta.** `cap03:41` mete a FLUS en «regresiones logísticas o pesos bayesianos»; FLUS usa **red neuronal** y el paper contrapone explícitamente ANN a regresión logística. La propia tesis lo dice bien en `cap03:288`.
6. **`Gong2013` — fuente sobreextendida.** `cap02:24`. FROM-GLC es un mapa global de **época única** (¾ circa 2010, ¼ circa 2000, sólo 18 escenas anteriores a 1998); no respalda «series históricas con décadas de profundidad». Además `Huang, Xiaomei` → **`Huang, Xiaomeng`**.
7. **`ONUHabitat2016TendenciasMexico`** — `cap01:24`. La página **no menciona** cobertura ni desactualización de planes; sí documenta «limitada institucionalidad para la gobernanza» y falta de capacidades locales. Y está fechada **2017**, no 2016.
8. **`JantzGoetz2005` — verbo excesivo.** `cap03:96`. No «demuestran» la no transferencia entre escalas (calibraron cada escala por separado); demuestran inestabilidad de coeficientes, y aclaran que **la tasa de crecimiento sí se reproduce a todas las escalas**.

Sin problemas: `Brenning2012`, `INEGI2020Censo` y `YinYan1988` (metadatos y afirmaciones correctos; `YinYan1988` verificado por vía indirecta porque las actas de 1988 no están digitalizadas, y admite completar editor/publisher).

**Veredicto del lote: REQUIERE CORRECCIONES.**
