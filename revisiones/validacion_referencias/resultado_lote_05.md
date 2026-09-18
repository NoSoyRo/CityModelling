# VALIDACIÓN DE FUENTES — Lote 5 de 8 (11 referencias, 21 afirmaciones)

## ArribasBel2014
**Metadatos:** CORRECTO
- evidencia: https://api.crossref.org/works/10.1016/j.apgeog.2013.09.012 → autor único Daniel Arribas-Bel; *Applied Geography* 49:45-53; impresa mayo 2014. Confirmado en https://research.birmingham.ac.uk/en/publications/accidental-open-and-everywhere-emerging-data-sources-for-the-unde/ ("Pages (from-to) 45-53 | Applied Geography | Published - May 2014").
- correcciones: ninguna.

**Afirmaciones:**
1. cap03:233 (datos gubernamentales abiertos + sensores móviles + actividad empresarial en línea) — **CORRECTA**. Resumen del autor: «These are data collected from mobile sensors carried by individuals, data derived from businesses moving their activity online and government data released in an open format». URL: https://research.birmingham.ac.uk/en/publications/accidental-open-and-everywhere-emerging-data-sources-for-the-unde/ (también https://pure-oai.bham.ac.uk/ws/portalfiles/portal/20957037/arribas2012data_aoe.pdf). Las tres fuentes de la tesis corresponden una a una con las tres del artículo.

**Riesgo en examen:** nulo. Cita exacta y bien atribuida.

---

## DelgadoLopez2018
**Metadatos:** INCORRECTO — la referencia no corresponde a ninguna publicación localizable. Muy probablemente es una **referencia fantasma** propagada en la literatura.
- evidencia:
  - https://api.crossref.org/journals/0186-7210/works (ISSN de *Estudios Demográficos y Urbanos*, también consultado 2448-6515), filtrado a 2018: el vol. 33 núm. 1 está completo y contiguo — pp. 9-42, 43-78, 79-110, 111-147, 149-186, 187-224, 225-252, y las notas/reseñas 253-257, 259-261, 263-266, 267-272. **No hay hueco para pp. 173-205**, que además se solaparían con dos artículos existentes (149-186 y 187-224).
  - Búsqueda de autor: los únicos trabajos de Javier Delgado en esa revista son «Querétaro: hacia la ciudad-región» (vol. 8, núm. 3, p. 655), «De los anillos a la segregación» (vol. 5, núm. 2) y «La expansión urbana probable de la Ciudad de México» (vol. 22, núm. 1, con Manuel Suárez). Ninguno es el citado.
  - Búsqueda bibliográfica por título en Crossref: sin coincidencias.
  - La cita sí circula, pero solo como referencia de segunda mano: https://www.revistas.unam.mx/index.php/aca/article/view/95031 la lista como «Delgado, J. y A. López, 2018 […] vol. 33, núm. 1, pp. 173-205». Esa misma lista incluye otra entrada de EDU («Zamorano y Velázquez, 2019, vol. 34, núm. 1») que tampoco aparece en el índice de la revista, lo que refuerza la sospecha de una cadena de citas no verificadas.
  - El portal del editor (estudiosdemograficosyurbanos.colmex.mx) y SciELO México devolvieron muro anti-bot y 404 respectivamente; la evidencia del índice proviene del depósito Crossref del propio editor.
- correcciones: **eliminar la entrada o sustituirla por una fuente real**. Alternativas documentadas y verificables sobre la ZMQ: Obregón-Biosca, «Patrones de viajes […] en la Zona Metropolitana de Querétaro», *EDU* 38(1):207-245; o la fuente del Observatorio de Ciudades ya presente en el `.bib`.

**Afirmaciones:**
1. cap05:44 (dinámica de crecimiento «bien documentada para la zona metropolitana») — **NO VERIFICABLE**. No existe la fuente que la respalda. La afirmación de fondo es defendible, pero el soporte citado no.

**Riesgo en examen:** **ALTO**. Un sinodal que intente localizar la referencia no la encontrará. Es el problema más serio del lote.

---

## Herold2003
**Metadatos:** INCORRECTO (nombre de autor)
- evidencia: https://api.crossref.org/works/10.1016/S0034-4257(03)00075-0 → autores «Martin Herold, **Noah C. Goldstein**, Keith C. Clarke»; *Remote Sensing of Environment* 86(3):286-302, agosto 2003. Confirmado de forma independiente en https://www.wikidata.org/wiki/Q57199382 («Noah C. Goldstein»).
- correcciones: `Nicholas C. Goldstein` → `Noah C. Goldstein`. Volumen, número, páginas, año y DOI correctos.

**Afirmaciones:**
1. cap02:22 (de las cuatro resoluciones, el estudio de la forma de la mancha urbana «exige sobre todo continuidad temporal y consistencia geométrica, más que una gran finura espectral») — **NO VERIFICABLE**. El artículo es de pago y no se localizó una copia íntegra legítima. El resumen que sí se leyó (https://www.academia.edu/1882981/The_spatiotemporal_form_of_urban_growth_measurement_analysis_and_modeling) describe una serie de 72 años a partir de fotografía aérea histórica e imágenes IKONOS con métricas espaciales y SLEUTH, pero **no enuncia la tipología de cuatro resoluciones ni el compromiso temporal/espectral** que la tesis le atribuye. Intentos: DOI de Elsevier (cerrado), OpenAlex (`oa_status: closed`), rutas directas en geog.ucsb.edu (404), academia.edu (solo resumen).
   - Sugerencia: reformular como cita de apoyo («en la línea de…») o trasladar el respaldo a una fuente que sí defina las cuatro resoluciones.
2. cap05:50 (serie homogénea para estudiar la forma espaciotemporal del crecimiento urbano desde percepción remota, «en la línea de trabajos que miden esa dinámica desde imágenes») — **CORRECTA**. El resumen dice: «This study explores the combined application of remote sensing, spatial metrics and spatial modeling to the analysis and modeling of urban growth in Santa Barbara, California […] based on a 72-year time series data set». URL: https://www.academia.edu/1882981/The_spatiotemporal_form_of_urban_growth_measurement_analysis_and_modeling

**Riesgo en examen:** medio. El nombre mal escrito es corregible en un minuto; la afirmación 1 conviene suavizarla porque no está anclada en texto leído.

---

## Jantz2003
**Metadatos:** CORRECTO
- evidencia: https://api.crossref.org/works/10.1068/b2983 → Claire A Jantz, Scott J Goetz, Mary K Shelley; *Environment and Planning B: Planning and Design* 31(2):251-271, 2004.
- nota importante: el PDF de autor alojado en Woodwell (https://www.woodwellclimate.org/wp-content/uploads/2015/09/JantzEnvPlanB.03.pdf) lleva un encabezado de prueba que dice «2003, volume 30, pages 251-271». Ese encabezado es erróneo: el índice de EPB vol. 30 núm. 2 (2003) va de la p. 163 a la 324 con artículos en 239-254, 255-270 y 271-296 — sin hueco para 251-271 — mientras que el vol. 31 núm. 2 (2004) encaja exactamente (163-165, 167-194, 195-212, 213-233, 235-250, **251-271**, 273-296, 297-309, 311-324). Verificado en https://api.crossref.org/journals/0265-8135/works. La entrada `.bib` tiene el año y volumen correctos.
- correcciones: ninguna en los campos. La **clave** `Jantz2003` no coincide con el año 2004 (cosmético, sin efecto en la salida).

**Afirmaciones:**
1. cap03:86 («una calibración completa puede tomar más de una semana en un clúster de la época») — **CORRECTA**, y con una precisión notable. Cita textual: «Calibration was performed on a Beowulf PC Cluster at the USGS's Rocky Mountain Mapping Center in Denver, CO. The cluster is a 16-node system (1 master node and 15 computing nodes), with each node containing an AMD Athlon/Duron processor, an AMD 750-MHz Thunderbird CPU, and 1.5-GB RAM […] **Over a week of processing time was required to complete the calibration.**» URL: https://www.woodwellclimate.org/wp-content/uploads/2015/09/JantzEnvPlanB.03.pdf

**Riesgo en examen:** bajo. Es la cita mejor sostenida del lote.

---

## Ma2019
**Metadatos:** CORRECTO
- evidencia: https://api.crossref.org/works/10.1016/j.isprsjprs.2019.04.015 → Lei Ma, Yu Liu, Xueliang Zhang, Yuanxin Ye, Gaofei Yin, Brian Alan Johnson; *ISPRS Journal of Photogrammetry and Remote Sensing* 152:166-177, junio 2019.
- correcciones: ninguna.

**Afirmaciones:** (texto íntegro leído en https://www.iges.or.jp/en/publication_documents/pub/peer/en/6898/Ma+et+al+2019.pdf)
1. cap02:28 (el aprendizaje profundo es supervisado y «su desempeño depende de la disponibilidad de grandes volúmenes anotados») — **CORRECTA**. Conclusiones: «For scene classification, object detection, semantic segmentation, and LULC classification, a supervised DL model (e.g., CNN) must be based on **large quantities of training samples**. In practice, the acquisition cost of training samples is relatively high».
2. cap03:189 (la revisión «documenta que la disponibilidad de datos de entrenamiento etiquetados es una de las dificultades recurrentes del área») — **CORRECTA**. Además de lo anterior: «Unsupervised DL models are also attractive for overcoming the **training data limitations**» y «The important limitation of DL in image registration is the lack of available public training datasets». El artículo se describe a sí mismo como análisis sistemático vía meta-análisis, por lo que «revisión sistemática» es admisible.
3. cap07:95 («suelen exigir conjuntos anotados grandes **y, en la práctica, hardware gráfico**») — **PARCIAL**. La primera mitad está plenamente sostenida. La segunda no: el artículo menciona GPU **una sola vez** en todo el texto, y de forma histórica y ajena a la teledetección — «[AlexNet] won the popular ImageNet contest by a wide margin in 2012 […] This was primarily because of its efficient use of graphics processing units (GPU), rectified linear units (ReLUs), and many training examples». Las palabras *hardware*, *computing power*, *computational cost* y *computational resource* no aparecen (0 ocurrencias).
   - Corrección propuesta: dejar `\parencite{Ma2019}` únicamente sobre los conjuntos anotados y retirar o respaldar por separado la mención al hardware gráfico.

**Riesgo en examen:** bajo-medio. Solo el inciso del hardware es atacable.

---

## ObservatorioCiudades2022Queretaro
**Metadatos:** CORRECTO
- evidencia: https://observatoriodeciudades.mx/homepage/blog/ presenta la entrada «Vivienda y expansión urbana: el caso de Querétaro» con el texto inicial que cita el Censo de Población y Vivienda más reciente. La entrada está viva en la URL registrada en el `.bib`; el acceso directo devolvió errores intermitentes (500 y timeout), pero el contenido es recuperable.
- correcciones: ninguna. El tipo `@online` y la `note` sobre el Censo 2020 (INEGI) son adecuados. Observación menor: una fuente terciaria (Academia XXII) la atribuye a una «Revista de Estudios Territoriales»; conviene **no** adoptar esa atribución, que no consta en el sitio.

**Afirmaciones:**
1. cap01:71 (el patrón de expansión dispersa «se ha documentado para Querétaro en relación con la producción de vivienda») — **CORRECTA**. Citas de la entrada: «se pasó de un parque de 304 mil 700 unidades en 2010 a uno de 498 mil 035 en 2020 […] Este aumento del 63.45% en el número de viviendas contra el crecimiento de 47.9% en el número de habitantes»; «La ciudad de Querétaro crece rápidamente hacia el horizonte, y la mancha urbana se esparce: el **45.7% de la vivienda nueva construida entre 2010 y 2020 se erigió en zonas que no tenían ninguna vivienda en 2010**». URL: https://observatoriodeciudades.mx/blog/vivienda-y-expansion-urbana-el-caso-de-queretaro/

**Riesgo en examen:** bajo. Es fuente gris, pero la tesis la usa con el alcance adecuado (documentación de un patrón, no de una cifra técnica).

---

## PontiusMillones2011
**Metadatos:** CORRECTO
- evidencia: https://api.crossref.org/works/10.1080/01431161.2011.552923 → Robert Gilmore Pontius, Marco Millones; *International Journal of Remote Sensing* 32(15):4407-4429, 2011, Taylor & Francis.
- correcciones: ninguna.

**Afirmaciones:** (manuscrito íntegro leído en https://commons.clarku.edu/cgi/viewcontent.cgi?article=1759&context=faculty_geography)
1. cap02:315 (Kappa «mezcla en un solo número dos fuentes de error metodológicamente distintas») — **CORRECTA**. §4.1: «Kappa's ratio is unnecessarily complicated because usually the most relevant ingredient to Kappa is only one part of the numerator, i.e., the total disagreement […] **This total disagreement can be expressed as the sum of two components of quantity disagreement and allocation** [disagreement]». Y más adelante: «κstandard fails to reveal that the Wundram and Löffler (2008) application […] has more quantity disagreement than the Ruelland et al. (2008) application […] κstandard is designed neither to penalize substantially for large quantity disagreement nor to reward substantially for small quantity disagreement».
2. cap02:349 (formalizan la descomposición y recomiendan reportarla por separado en lugar de un índice Kappa) — **CORRECTA**. Resumen: «we recommend that the profession abandon[] the use of Kappa indices for purposes of accuracy assessment and map comparison, and instead summarize the crosstabulation matrix with two much simpler summary parameters: quantity disagreement and allocation disagreement».
3. cap03:367 (confunden dos fuentes conceptualmente distintas, dificultan diagnosticar el tipo de fallo, recomiendan abandonarlos) — **CORRECTA**. §4.1 se titula literalmente «Reasons to abandon Kappa»; y «The separation of the overall disagreement into components of quantity and allocation reveals that their maps are actually much more accurate for their particular purpose than implied by the reported overall errors […] The κstandard indices do not offer this type of insight».
4. cap06:417 (definiciones operativas: extensión equivocada = cantidad; total correcto mal localizado = asignación) — **CORRECTA**. Definiciones originales: «We define **quantity disagreement** as the amount of difference between the reference map and a comparison map that is due to the less than perfect match in the proportions of the categories» y «We define **allocation disagreement** as the amount of difference between the reference map and a comparison map that is due to the less than optimal match in the spatial allocation of the categories, given the proportions of the categories in the reference and comparison maps».
   - Matiz interno (no es error de atribución): las ecuaciones de la tesis, $Q=|FP-FN|/N$ y $A=2\min(FP,FN)/N$, se calculan sobre las 5 419 008 celdas urbano/no-urbano, no sobre el subconjunto de cambio. La prosa las glosa en términos de «extensión de cambio». Conviene homogeneizar la redacción; el respaldo formal es correcto.
5. cap06:419 (pie de tabla atribuyendo la descomposición) — **CORRECTA**, incluido el factor 2 de la asignación: «Allocation disagreement is always an even number of pixels, because allocation disagreement always occurs in pairs of misallocated pixels. Each pair consists of one pixel of omission for a particular category and one pixel of commission for the same category».

**Riesgo en examen:** nulo. Las cinco afirmaciones son fieles al original.

---

## Torrens2000
**Metadatos:** CORRECTO
- evidencia: https://discovery.ucl.ac.uk/1371 → «Torrens, P.M.; (2000) How cellular models of urban systems work (1. theory). (CASA Working Papers 28). Centre for Advanced Spatial Analysis (UCL): London, UK. Type: Working / discussion paper». La ficha de CASA (https://www.ucl.ac.uk/bartlett/publications/2000/nov/casa-working-paper-28) da «Publication Date: 1/11/2000», consistente con `month = {11}`.
- correcciones: ninguna. `@techreport` con `type = {CASA Working Paper}`, `number = {28}`, `address = {London}` es el asiento adecuado.

**Afirmaciones:**
1. cap03:31 (describe en el plano teórico de qué piezas se compone un modelo celular urbano y cómo debe modificarse el formalismo del autómata) — **CORRECTA**, y de forma casi literal. Del PDF (https://discovery.ucl.ac.uk/id/eprint/1371/1/paper28.pdf): «Section 4.2 describes how CA must be modified for urban applications»; y en §4.2: «the basic CA, as defined by Ulam, von Neumann, Conway, and Wolfram […] is not well suited to urban applications […] it is necessary that CA be heavily modified from the formal parameterizations outlined in section 3. Indeed, **quite radical modification is necessary** before CA can approximate even a crude representation of an urban system. This often necessitates the introduction of additional components […] The next sections discuss urban CA modifications in detail, referring to the adaptation of **cell-states, lattices, neighborhoods, time, and transition rules**». Las cinco piezas (§4.2.1 a §4.2.5) son exactamente «de qué piezas se compone un modelo celular urbano».

**Riesgo en examen:** nulo.

---

## Wang2021
**Metadatos:** CORRECTO
- evidencia: https://api.crossref.org/works/10.3390/ijerph182111013 → Renyang Wang, Qingsong He, Lu Zhang, Huiying Wang; *IJERPH* 18(21):11013, MDPI, 20 oct 2021. Texto íntegro en PMC8583206.
- correcciones: ninguna.

**Afirmaciones:** (texto íntegro leído vía https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=8583206&retmode=xml — https://pmc.ncbi.nlm.nih.gov/articles/PMC8583206/)
1. cap01:117 (los metaheurísticos «persiguen la configuración urbana que optimiza una función objetivo definida por el analista […] y su convergencia demanda tiempo y recursos computacionales considerables») — **PARCIAL**. La primera mitad está sostenida: «The genetic algorithm fitness function was defined as follows: UV = argmax ∑∑ w_ik · x_ik […] The goal of optimization is to continuously pursue a higher UV value», y el propósito es normativo («optimally allocates cells with different UGPs, creating a city form that promotes urban vitality»). La segunda mitad **no aparece en el artículo**: no hay ninguna mención de tiempo de ejecución, coste computacional ni hardware; el único dato relacionado es «After about **650 iterations**, the value of the evaluation function (total vitality) stabilized», con población 100, sin juicio sobre su coste. El propio apartado de limitaciones no menciona el cómputo, solo la antigüedad y el carácter estático de los datos.
   - Corrección propuesta: acotar la cita a la parte normativa, o respaldar el coste computacional con otra fuente (p. ej. `Jantz2003`, que sí lo documenta).
2. cap03:55 (combinó AC con algoritmo genético para optimizar la forma urbana de Wuhan) — **CORRECTA**. Título y resumen: «Coupling Cellular Automata and a Genetic Algorithm to Generate a Vibrant Urban Form—A Case Study of Wuhan, China […] this paper applies a coupling model called the "promoting urban vitality model," based on cellular automata (CA) and genetic algorithm (GA) (abbreviated as UV-CAGA)».
3. cap03:189 (integra AC y AG «para generar formas urbanas que maximizan un índice de vitalidad en Wuhan») — **CORRECTA**. El índice de vitalidad se construye con «POI density (POID), land function mix (MIX), and sign-in density (CIQD) […] A straightforward method, POID × MIX × CIQD, is used to evaluate urban vitality». Resultado: «The urban vitality of the optimized urban form scheme was **4.8% higher** than the simulated natural expansion scheme». La tesis no cita esa cifra, así que no hay riesgo numérico.

**Riesgo en examen:** bajo-medio, solo por el inciso del coste computacional.

---

## White1993
**Metadatos:** CORRECTO
- evidencia: https://api.crossref.org/works/10.1068/a251175 y https://ideas.repec.org/a/sae/envira/v25y1993i8p1175-1199.html → R White & G Engelen, *Environment and Planning A*, vol. 25(8):1175-1199, agosto 1993.
- correcciones: ninguna.

**Afirmaciones:**
1. cap03:18 (reglas locales basadas en distancia y vecindad reproducen estructuras de uso de suelo con propiedades fractales comparables a las de ciudades reales) — **CORRECTA**. Resumen del autor: «a cellular automaton is developed to model the spatial structure of urban land use over time. For realistic parameter values, the model produces **fractal or bifractal land-use structures** for the urbanized area and for each individual land-use type. **Data for a set of US cities show that they have very similar fractal dimensions**» (https://journals.sagepub.com/doi/10.1068/a251175). Sobre «distancia y vecindad», fuente secundaria revisada por pares: «White and Engelen (1993) firstly proposed this kind of configuration of neighborhood [extended neighborhood, con ponderación por distancia] for exploring the relationship of CA-based model of urban form evolution» (http://giswin.geo.tsukuba.ac.jp/sis/staff/yaolongzhao/pdf/ICCS2007paper.pdf). El artículo es de pago; el texto íntegro no fue accesible, pero el resumen del propio autor cubre el núcleo de la afirmación.
   - Nota: «lo cual fue sorprendente para el año en el que se publica» es un juicio del tesista, no una atribución; no requiere respaldo.

**Riesgo en examen:** bajo.

---

## White1997
**Metadatos:** CORRECTO con una omisión menor
- evidencia: https://api.crossref.org/works/10.1068/b240235 y https://ideas.repec.org/a/sae/envirb/v24y1997i2p235-246.html → R White & G Engelen, *Environment and Planning B: Planning and Design*, vol. 24, **núm. 2**, pp. 235-246, abril 1997.
- correcciones: falta `number = {2}`. Sin efecto visible con la mayoría de estilos, pero conviene añadirlo.

**Afirmaciones:**
1. cap02:91 («Esta extensión [el AC probabilístico], base de los modelos urbanos basados en AC **desde \textcite{White1997}**…») — **INCORRECTA en la atribución de prioridad**. El término de perturbación estocástica del potencial de transición en la familia White-Engelen se atribuye en la literatura a **White y Engelen 1993**, no a 1997: «stochastic perturbation was introduced into the calculation of transition potential (**White & Engelen, 1993**). This factor is computed using Eq. 2: v = 1 + (−ln(rand))^α […] As White and Engelen (1993) points out, a low α value implies simpler and more compact growth forms, while a high α value reports a more random structure of the city» (https://doi.org/10.1016/j.compenvurbsys.2022.101895). El resumen de White1997 no menciona transiciones probabilísticas: describe el acoplamiento CA + SIG + modelos regionales aplicado a Santa Lucía. No se pudo leer el texto íntegro de 1997 (de pago, SAGE bloquea el acceso automatizado), así que no se afirma que 1997 carezca del término, solo que **la prioridad no le corresponde** y que el resumen no lo respalda.
   - Corrección propuesta: `desde \textcite{White1997}` → `desde \textcite{White1993}` (o `\textcite{White1993,White1997}`). `White1993` ya existe en el `.bib`, así que el arreglo no añade entradas.
2. cap03:18 («extendió después ese esquema acoplando el autómata a un sistema de información geográfica y a modelos regionales») — **CORRECTA**, casi textual. Resumen: «We present an integrated model of regional spatial dynamics consisting of a **cellular automaton-based model of land use linked both to a geographic information system (GIS) and to standard nonspatial models of regional economics and demographics**, as well as to a simple model of environmental change» (https://ideas.repec.org/a/sae/envirb/v24y1997i2p235-246.html). El «después» respecto de 1993 es correcto.
   - Precaución para el resto de la tesis: el caso de aplicación de este artículo es **la isla de Santa Lucía**, no una ciudad europea ni Cincinnati. No atribuirle otras ciudades.

**Riesgo en examen:** medio. La afirmación 1 es un error de atribución de prioridad entre dos artículos de los mismos autores, ambos en la bibliografía; es exactamente el tipo de detalle que un sinodal familiarizado con la literatura de AC urbanos detecta.

---

## Resumen del lote

Revisadas las 11 referencias del lote 5 y sus 21 afirmaciones. Resultado: 2 metadatos incorrectos, 1 afirmación incorrecta, 2 parciales y 2 no verificables; las 16 restantes correctas.

Problemas graves:

1. **`DelgadoLopez2018` — referencia inexistente.** `cap05_modelo_crecimiento_urbano.tex:44`. No hay ningún artículo de Delgado y López en *Estudios Demográficos y Urbanos* 33(1); el índice del número es contiguo y las pp. 173-205 se solaparían con dos artículos reales. La cita solo circula de segunda mano en una lista de referencias que contiene otra entrada igualmente inexistente. Hay que eliminarla o sustituirla por una fuente real sobre la ZMQ.
2. **`White1997` — atribución de prioridad equivocada.** `cap02_marco_teorico.tex:91`. La perturbación estocástica del potencial de transición la introdujeron White y Engelen en **1993**, no en 1997. Corrección: `\textcite{White1997}` → `\textcite{White1993}`. La entrada ya está en el `.bib`.
3. **`Herold2003` — nombre de autor mal escrito.** `Nicholas C. Goldstein` → **`Noah C. Goldstein`** (Crossref y Wikidata coinciden).

Afirmaciones parciales que conviene acotar:

4. **`Ma2019`**, `cap07_conclusiones.tex:95`: «en la práctica, hardware gráfico». El artículo menciona GPU una sola vez, y solo al narrar el triunfo de AlexNet en ImageNet 2012; no dice nada sobre requisitos de hardware en teledetección. La parte de los conjuntos anotados sí está bien sostenida.
5. **`Wang2021`**, `cap01_introduccion.tex:117`: «su convergencia demanda tiempo y recursos computacionales considerables». El artículo no reporta tiempos, coste ni hardware; solo que el ajuste se estabilizó tras unas 650 iteraciones.

No verificable por acceso cerrado: la afirmación de `Herold2003` en `cap02_marco_teorico.tex:22` sobre el compromiso entre las cuatro resoluciones. El resumen del artículo no la enuncia; conviene suavizarla o reanclarla.

Aciertos que vale la pena señalar: la cita de `Jantz2003` sobre la calibración de SLEUTH es textualmente exacta («Over a week of processing time was required to complete the calibration», sobre un clúster Beowulf de 16 nodos a 750 MHz), y las cinco afirmaciones atribuidas a `PontiusMillones2011` son fieles al original, incluidas las fórmulas.
