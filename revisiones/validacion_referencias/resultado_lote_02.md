# VALIDACIÓN DE FUENTES — LOTE 2 (10 referencias, 21 afirmaciones)

**Nota previa sobre líneas:** los números de línea del archivo de entrada están desfasados respecto al estado actual de los `.tex`. Ubicaciones reales verificadas con `grep`: cap01 → 35, 38, 69, 146 (no 24, 66, 143); cap02 → 138, **168**, 179, 201; cap03 → **256, 317, 327, 350, 371**; cap06 → **236**, 242; cap05 → 17, 40, 242 (sin cambio).

---

## alonso1964location
**Metadatos:** CORRECTO
- evidencia: https://www.degruyterbrill.com/document/doi/10.4159/harvard.9780674730854/html — «Location and Land Use — Toward a General Theory of Land Rent, William Alonso, Published/Copyright: 1964», Harvard University Press. Confirmado en https://books.google.com/books/about/Location_and_Land_Use.html?id=-E9ZmQEACAAJ (204 pp., *Harvard Economic Studies* vol. 124).
- correcciones: ninguna obligatoria. Opcional: añadir `address={Cambridge, MA}` y la serie.

**Afirmaciones:**
1. `cap05:17` — decaimiento por distancia a centros de empleo como mecanismo central de la teoría clásica de la renta del suelo → **CORRECTA**. Texto del propio libro: «Price of land, P, varies with distance from the center of the city, t. […] It will be assumed here that the price of land decreases with increasing distance from the center of the city»; y la restricción presupuestal «Individual's income = land costs + commuting costs + all other expenditures». URL: https://dokumen.pub/location-and-land-use-toward-a-general-theory-of-land-rent-reprint-2013.html
   - *Matiz:* el modelo de Alonso es **monocéntrico** (un solo centro). «Centros de empleo» en plural es glosa moderna (policéntrica); no invalida la afirmación pero conviene el singular.

**Riesgo en examen:** bajo. Atribución canónica y correctamente formulada.

---

## BonhamCarter1994
**Metadatos:** CORRECTO
- evidencia: https://www.sciencedirect.com/bookseries/computer-methods-in-the-geosciences/vol/13/suppl/C — «Volume 13: Geographic Information Systems for Geoscientists – Modelling with GIS […] Pages 1-398 (1994)». Portada y página legal: https://api.pageplace.de/preview/DT0400.9781483144948_A23873095/preview-9781483144948_A23873095.pdf («PERGAMON […] First edition 1994 […] Computer methods in the geosciences»). Serie n.º 13 confirmada en https://findit.library.nd.edu/Record/001291990 («Computer methods in the geosciences ; 13»).
- correcciones: ninguna. `volume=13`, `pagetotal=398`, Pergamon, Oxford: todos correctos.
- **Índice verificado:** CHAPTER 9 – *Tools for Map Analysis: Multiple Maps*, pp. 267–337; dentro de él «Bayesian Methods 302», «Conditional Independence 312», «Discussion of Weights of Evidence 328» (https://vdoc.pub/documents/geographic-information-systems-for-geoscientists-modelling-with-gis-lhpfnr3afo40).

**Afirmaciones:**
1. `cap02:138` — WoE como técnica bayesiana que cuantifica asociación variable espacial–evento → **CORRECTA**. La documentación derivada del capítulo lo formula así: «For the derivation of weights, see Bonham-Carter (1994, ch. 9). […] the weights for binary themes are given by the ratio of the following conditional probabilities». URL: https://www.ige.unicamp.br/wofe/documentation/wofeintr.htm
2. `cap02:168` — pie de figura «con base en la formulación de Bonham-Carter (1994)» → **CORRECTA**. La formulación de $W^+$, $W^-$ y $C=W^+-W^-$ es del cap. 9: «The difference between the weights is know as the contrast, C. Thus C=W+ - W-». Misma URL.
3. `cap02:179` — el Information Value **no** forma parte de la formulación WoE de Bonham-Carter → **CORRECTA**. Verificada por ausencia: ni el índice del cap. 9 (pp. 267–337) ni la documentación WofE derivada de él contienen «information value»; el aparato estadístico es $W^+$, $W^-$, $C$, $s(C)$ y contraste estudentizado. URLs: https://vdoc.pub/documents/geographic-information-systems-for-geoscientists-modelling-with-gis-lhpfnr3afo40 y https://www.ige.unicamp.br/wofe/documentation/wofeintr.htm
   - *Nota:* es una afirmación negativa; queda sustentada por índice + documentación derivada, no por lectura de las 70 páginas completas. Aun así, el matiz es seguro y además la tesis atribuye correctamente el IV a `Siddiqi2006`.
4. `cap02:201` — «el método no corrige por sí solo la **autocorrelación espacial**, de modo que las celdas vecinas aportan información que no es independiente \parencite{BonhamCarter1994}» → **NO VERIFICABLE, y con alta probabilidad mal atribuida**.
   - Vías intentadas: (a) índice completo del cap. 9; (b) documentación WofE derivada explícitamente del cap. 9 (búsqueda de `autocorrel` → **cero coincidencias**); (c) búsqueda web dirigida. Lo que el cap. 9 sí documenta es **independencia condicional entre capas** (p. 312) y el doble conteo de evidencia correlacionada — que son las limitaciones 1 y 3 de la lista, esas sí correctas.
   - La única mención verificada de autocorrelación espacial en esta línea de trabajo está en **otro** texto: Bonham-Carter, Agterberg y Wright (1988), *Integration of Geological Datasets for Gold Exploration in Nova Scotia*: «a very large number of small sampling cells must be created, and this is undesirable because of the resulting large attribute file and degree of spatial autocorrelation present in such a dataset». URL: https://www.ige.unicamp.br/sdm/ArcSDM2/documentation/WofE3.pdf — y ahí se refiere al tamaño de celda como desventaja de la regresión, no a una limitación declarada del WoE.
   - corrección sugerida: quitar `\parencite{BonhamCarter1994}` de la cuarta limitación y citar `Legendre1993,Brenning2012` (ya usados en `cap03:350` para exactamente ese punto), o bien `BonhamCarter1988/1989` si se quiere conservar la autoría del grupo.
5. `cap03:256` — el método «se consolidó como procedimiento estándar en el capítulo 9» → **CORRECTA**. Cap. 9 = *Tools for Map Analysis: Multiple Maps*, pp. 267–337, con «Bayesian Methods 302 […] Application of Weights of Evidence to Mineral Potential Mapping […] Discussion of Weights of Evidence 328». URL: https://vdoc.pub/documents/geographic-information-systems-for-geoscientists-modelling-with-gis-lhpfnr3afo40
6. `cap03:317` — prueba ómnibus del cap. 9 con umbral de exceso **≤ 15 %** → **CORRECTA**, y con página exacta. Agterberg (autor de la prueba exacta y revisor del manuscrito del libro) escribe: «This is the rationale of the overall or so-called "omnibus test" […] **For example, in Bonham-Carter (1994, p. 316), it is argued that T should not exceed n by more than 15%**», donde T = suma de probabilidades posteriores y n = número de eventos. URL: https://www.ige.unicamp.br/sdm/ArcSDM31/documentation/CI_Agterberg.pdf
   - *Matiz de precisión:* el **nombre** «prueba ómnibus» se acuña después (Kemp, Bonham-Carter y Wright, 1999) — el mismo texto lo dice y llama a la versión de 1994 «the informal 15% rule of the earlier version of the omnibus test». La tesis puede decir «la regla del 15 % del cap. 9 […], conocida después como prueba ómnibus».
7. `cap03:327` — «propone elegir los cortes **maximizando el contraste estudentizado**» → **PARCIAL**. El cap. 9 elige el corte donde **el contraste C alcanza su máximo**, y presenta el contraste estudentizado como columna auxiliar: «The contrast, C, reaches a maximum value at class 5 (1.25 km). […] Addition of an extra column (not shown here) containing the **Studentized contrast (C/s(C)) is also helpful for choosing the cutoff distance**, because it shows the contrast relative to the uncertainty due to the weights. **(From Bonham-Carter, 1994, ch. 9)**». URL: https://www.ige.unicamp.br/wofe/documentation/wofeintr.htm
   - corrección sugerida: «propone elegir los cortes maximizando el contraste, apoyándose en el contraste estudentizado $C/s(C)$ como medida de su significancia».
8. `cap03:350` — la formulación original estima la varianza de los pesos a partir del conteo de eventos de entrenamiento → **CORRECTA**. La tabla del cap. 9 incluye las columnas «W+ | s(W+) | W- | s(W-) | C», con «s(W+) is the standard deviation of W+», calculadas sobre «The point column is the number of cells containing a point». URL: https://www.ige.unicamp.br/wofe/documentation/wofeintr.htm — La segunda mitad («no incorpora esta estructura espacial») está redactada como descripción de la formulación, no como aserción del autor, y la sobreestimación se atribuye correctamente a `Legendre1993,Brenning2012`.

**Riesgo en examen:** medio, concentrado en un solo punto. Siete de ocho afirmaciones son defendibles y la del 15 % incluso con página. La cuarta limitación (`cap02:201`) es el único flanco: un sinodal que conozca el libro puede objetar que la autocorrelación espacial entre celdas no es un tema del cap. 9.

---

## CONABIO2024Porque
**Metadatos:** INCORRECTO (año)
- evidencia: https://www.biodiversidad.gob.mx/biodiversidad/porque — la página cierra con «**Actualizado en: 08/10/2022 - 16:33hrs.**». No hay indicio de contenido de 2024.
- correcciones: `year = {2024}` → `year = {2022}` (fecha de última actualización declarada). Conservar `urldate`. Autor institucional CONABIO: correcto (sitio *Biodiversidad Mexicana*, operado por CONABIO).

**Afirmaciones:**
1. `cap01:35` — la pérdida de hábitat es la principal causa de pérdida de biodiversidad, y la urbanización contribuye a ella → **CORRECTA**. Cita textual de la página: «**La pérdida y deterioro de los hábitats es la principal causa de pérdida de biodiversidad.** Al transformar selvas, bosques, matorrales, pastizales, manglares, lagunas, y arrecifes en campos agrícolas, ganaderos, granjas camaroneras, presas, carreteras **y zonas urbanas** destruimos el hábitat de miles de especies». URL: https://www.biodiversidad.gob.mx/biodiversidad/porque
   - *Matiz menor:* la frase de la fuente es general, no explícitamente «en el país»; el encuadre nacional se sostiene por el contexto inmediato («en México se ha perdido alrededor del 50% de los ecosistemas naturales») y por tratarse de la comisión mexicana. Defendible.

**Riesgo en examen:** bajo, una vez corregido el año.

---

## CONAPO2018Delimitacion
**Metadatos:** INCORRECTO (campo de autor mal construido, orden institucional y tipo de entrada)
- evidencia: portada y página legal del documento oficial, descargado y leído directamente (285 pp.): «Secretaría de Desarrollo Agrario, Territorial y Urbano / Consejo Nacional de Población / Instituto Nacional de Estadística y Geografía — *Delimitación de las zonas metropolitanas de México 2015* — **Primera edición: febrero 2018**». URL: https://inegi.org.mx/contenido/productos/prod_serv/contenidos/espanol/bvinegi/productos/nueva_estruc/702825006792.pdf
- correcciones:
  - `author = {{Consejo Nacional de Población (CONAPO) and SEDATU and INEGI}}` → **defecto real de sintaxis**: al estar todo dentro de un único grupo de llaves, biblatex lo trata como **un solo nombre literal** e imprime el texto «and SEDATU and INEGI» tal cual. Debe ser: `author = {{Secretaría de Desarrollo Agrario, Territorial y Urbano} and {Consejo Nacional de Población} and {Instituto Nacional de Estadística y Geografía}}`.
  - orden de autoría: CONAPO primero → **SEDATU primero**, como en la portada.
  - tipo: `@online` → `@report` (o `@book`), con `publisher/institution = {SEDATU, CONAPO e INEGI}`, `address = {Ciudad de México}`, `edition = {Primera}`. Es un libro paginado, no un recurso web; el enlace de gob.mx además está caído (404 en el CMS y bloqueo de challenge en la ficha).

**Afirmaciones:**
1. `cap01:69` — «una zona metropolitana que **rebasó el millón y medio de habitantes** tras décadas de expansión sostenida \parencite{CONAPO2018Delimitacion,INEGI2020Censo}» → **PARCIAL**. El millón y medio **no** proviene de esta fuente: el documento reporta para la ZM de Querétaro **1 323 640** habitantes en 2015 (Cuadros 5a/5b y Cuadro 6.22.01, leídos en el PDF), y la ubica entre las zonas que «rebasó el millón de habitantes», no millón y medio. La cifra de 1,5 millones corresponde al Censo 2020 (`INEGI2020Censo`).
   - corrección sugerida: reordenar para que CONAPO respalde la delimitación y el crecimiento sostenido (el propio documento registra una tasa de 2,8 % anual 2010–2015, «Dentro de este grupo de metrópolis destaca el crecimiento de Querétaro (2.8)»), y dejar la población de 1,5 M exclusivamente a cargo de `INEGI2020Censo`. URL: misma que arriba.
2. `cap05:40` — «agrupa varios municipios en torno a la capital del estado» → **CORRECTA**. Cuadro 6.22.01 del documento lista los cinco integrantes: Querétaro (878 931, municipio central, sede de Santiago de Querétaro), Corregidora (181 684), El Marqués (156 275), Huimilpan (38 295) y Apaseo el Alto, Guanajuato (68 455). El texto del informe confirma el carácter interestatal: «la adición del municipio Apaseo el Alto a la zm de Querétaro la convirtió en una metrópoli interestatal». URL: misma que arriba.
   - En esta oración la atribución está bien repartida: delimitación → CONAPO; población → INEGI 2020.

**Riesgo en examen:** medio-bajo en el fondo, pero el campo `author` roto se verá impreso en la bibliografía; eso sí lo nota un revisor.

---

## HernandezGuerreroOsorno2018
**Metadatos:** CORRECTO (con una salvedad en los apellidos)
- evidencia: Crossref `https://api.crossref.org/works/10.4067/S0718-34022018000300147` devuelve título idéntico, autores «Juan Hernández Guerrero» y «Tamara Osorno Sánchez», *Revista de geografía Norte Grande*, issue 71, pp. 147–166, 2018-12. Artículo leído en https://scielo.conicyt.cl/scielo.php?script=sci_arttext&pid=S0718-34022018000300147 («Rev. geogr. Norte Gd. no.71 Santiago dic. 2018», ambos autores: Universidad Autónoma de Querétaro).
- correcciones: `Hern{\'a}ndez-Guerrero, Juan` → `Hern{\'a}ndez Guerrero, Juan`; `Osorno-S{\'a}nchez, Tamara` → `Osorno S{\'a}nchez, Tamara` (la publicación no usa guion). Sin guion hay que proteger el apellido compuesto: `{Hern{\'a}ndez Guerrero}, Juan`. Volumen: correctamente ausente (la revista numera solo por número).

**Afirmaciones:**
1. `cap05:40` — paisaje urbano heterogéneo estudiado «desde las diferencias entre zonas de la propia ciudad» → **CORRECTA**. Resumen del artículo: «El presente estudio analiza las **diferencias ambientales en el paisaje urbano de Juriquilla y Santa Rosa Jáuregui, al norte de la ciudad de Querétaro**, México. Se aplicó una metodología basada en un índice de calidad ambiental de paisaje urbano». URL: https://scielo.conicyt.cl/scielo.php?script=sci_arttext&pid=S0718-34022018000300147
   - *Matiz:* el objeto del artículo es la **calidad ambiental percibida/valorada visualmente** de dos zonas, no el crecimiento urbano. La inferencia de la tesis («da cuenta de un crecimiento desigual») es interpretación propia razonable, pero conviene no presentarla como hallazgo del artículo.

**Riesgo en examen:** bajo.

---

## Li2007
**Metadatos:** CORRECTO
- evidencia: `https://api.crossref.org/works/10.1016/j.jenvman.2006.11.006` → «Defining agents' behaviors to simulate complex residential development using multicriteria evaluation», Xia Li y Xiaoping Liu, *Journal of Environmental Management*, **85(4): 1063–1075**, 2007. Confirmado en Europe PMC (`pageInfo: 1063-1075`, `doi: 10.1016/j.jenvman.2006.11.006`): https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%2210.1016/j.jenvman.2006.11.006%22&resultType=core&format=json
- correcciones: ninguna. **El DOI del `.bib` es el correcto**; no es el `…2006.10.010` que apunta a otro artículo del mismo volumen.

**Afirmaciones:**
1. `cap01:146` — los ABM «exigen declarar el comportamiento de cada agente» → **CORRECTA**. Resumen del artículo: «Agent-based modeling can solve some of the problems in addressing individuals' influences in urban systems. **However, there is a general lack of methodology on how to define agents' properties.** This paper uses multicriteria evaluation techniques to determine some of the parameters for the agent-based model». El problema de especificar el comportamiento de los agentes es literalmente el objeto del paper (y de su título). URL: https://europepmc.org/article/MED/17275169
   - Caso de estudio: Guangzhou, sur de China («the simulation of the residential development in a fast growing city, Guangzhou, in south China»), consistente con el uso que hace la tesis.

**Riesgo en examen:** bajo. Es una de las atribuciones mejor elegidas del lote.

---

## Ojala2002
**Metadatos:** CORRECTO
- evidencia: `https://api.crossref.org/works/10.1109/TPAMI.2002.1017623` → T. Ojala, M. Pietikainen, T. Maenpaa, *IEEE Transactions on Pattern Analysis and Machine Intelligence*, **24(7): 971–987**, julio 2002. Confirmado en https://doi.org/10.1109/TPAMI.2002.1017623 y en Scholarpedia (http://scholarpedia.org/article/Local_Binary_Patterns): «IEEE Trans. Pattern Analysis and Machine Intelligence 24(7): 971-987».
- correcciones: ninguna.

**Afirmaciones:**
1. `cap02:40` — «Los patrones binarios locales, **propuestos por** \textcite{Ojala2002}, describen esa textura comparando la intensidad de cada píxel con la de sus vecinos sobre un círculo de radio $r$ con $P$ puntos de muestreo» → **PARCIAL**. La **formulación circular $(P,R)$ sí es de este artículo**: «We derive the operator for a general case based on a circularly symmetric neighbor set of P members on a circle of radius R […] gray values of P equally spaced pixels on a circle of radius R (R>0) that form a circularly symmetric neighbor set». Pero el LBP original **no** se propone aquí: el propio artículo lo dice, «If we set (P=8,R=1), we obtain LBP8,1 which is **similar to the LBP operator we proposed in [28]**», donde [28] = «Ojala T, Pietikäinen M and Harwood D (1996) A comparative study of texture measures…». URL: http://vision.stanford.edu/teaching/cs231b_spring1415/papers/lbp.pdf
   - corrección sugerida: «en la generalización multirresolución de \textcite{Ojala2002}» o «propuestos por Ojala et al. (1996) y generalizados a vecindades circulares por \textcite{Ojala2002}».
   - El resto de la oración y la frase siguiente («invariante a cambios monótonos de iluminación») son exactas: «the operator is by definition invariant against any monotonic transformation of the gray scale».
2. `cap05:242` — LBP «robustos a cambios de iluminación y útiles para distinguir la rugosidad del tejido construido» → **CORRECTA** en la primera mitad, con cita literal del resumen: «The proposed approach is **very robust in terms of gray scale variations**, since the operator is by definition invariant against any monotonic transformation of the gray scale». El artículo además evalúa explícitamente clasificación «illumination and rotation invariant» (test suite Outex_TC_00012). La utilidad para «tejido construido» es aplicación propia de la tesis, no del artículo, y así está redactada. URL: http://vision.stanford.edu/teaching/cs231b_spring1415/papers/lbp.pdf

**Riesgo en examen:** bajo-medio. El único punto atacable es el «propuestos por» de `cap02:40`, y es un desliz de un solo verbo, fácil de blindar.

---

## Seto2012teleconnections
**Metadatos:** INCORRECTO (lista de autores truncada y DOI ausente)
- evidencia: Europe PMC, registro completo: «Seto KC, Reenberg A, Boone CG, Fragkias M, Haase D, Langanke T, Marcotullio P, Munroe DK, Olah B, Simon D.», *Proc Natl Acad Sci U S A*, **109(20): 7687–7692**, mayo 2012, DOI `10.1073/pnas.1117622109`, PMCID PMC3356653. URL: https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%2210.1073/pnas.1117622109%22&resultType=core&format=json
- correcciones:
  - `author = {Seto, Karen C. and others}` → listar los **diez** autores (Seto, Reenberg, Boone, Fragkias, Haase, Langanke, Marcotullio, Munroe, Olah, Simon). El `and others` funciona en salida (`et al.`) pero deja la entrada incompleta y es mala práctica en una bibliografía de tesis.
  - añadir `doi = {10.1073/pnas.1117622109}` (ausente).
  - volumen, número y páginas: **correctos**.

**Afirmaciones:**
1. `cap01:38` — «los efectos del cambio de suelo urbano […] alcanzan a los territorios con los que la ciudad se conecta mediante flujos de recursos \parencite{Seto2012teleconnections}, por ejemplo rios que desembocan en lagos, por ejemlplo el rio Mixcoac que desemboca en lagos cercanos a la ciudad de México…» → **PARCIAL, con un problema grave en el ejemplo**.
   - **La primera cláusula es CORRECTA.** Cita textual del artículo: «**We introduce the concept of urban land teleconnections to refer to the distal flows and connections of people, economic goods and services, and land use change processes that drive and respond to urbanization.**» Y también: «In an increasingly urban world, characterized by **global flows of commodities, capital, and people** […] teleconnections captures links between distant processes and places». URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC3356653/
   - *Matiz léxico:* los flujos del artículo son de personas, bienes y servicios económicos, capital y procesos de cambio de uso de suelo. «Flujos de recursos» es paráfrasis aceptable pero inexacta; mejor «flujos de personas, bienes y capital».
   - **El ejemplo del río Mixcoac NO está en Seto et al. (2012)** — el artículo es conceptual, sobre China, India y desafíos globales, y no menciona México ni ese río (verificado sobre el texto completo en PMC). Además **es factualmente falso**: el Mixcoac fue entubado en 1955 y su cauce lo ocupa la avenida homónima («En 1955 se realizan las obras de entubamiento del Rio Mixcoac, cuyo cauce es ocupado por la avenida del mismo nombre», https://doczz.net/doc/1090369/diario-de-los-debates---asamblea-legislativa-del-distrito...); desembocaba en el Magdalena para formar el Churubusco, y el Churubusco **fue desviado en 1952 justamente para que dejara de alimentar los lagos** de Xochimilco, Mixquic y Tláhuac («a partir de 1952 se desvió el curso del río Churubusco, con la finalidad de que ya no alimentara los lagos de Xochimilco, Mixquic y Tláhuac», https://relatosehistorias.mx/nuestras-historias/que-avenidas-importantes-de-la-ciudad-de-mexico-eran-caudalosos-rios-de-agua).
   - La oración además arrastra tres erratas: «rios» → «ríos», «ejemlplo» → «ejemplo», y «por ejemplo… por ejemplo» repetido.
   - corrección sugerida: **eliminar el ejemplo completo** y cerrar en la cláusula respaldada, o sustituirlo por un ejemplo verificable y con fuente propia.

**Riesgo en examen:** **ALTO**, no por la cita sino por el ejemplo. Una frase sin revisar, con dos erratas y un hecho geográfico falso, en el capítulo 1, es exactamente lo que un sinodal señala primero.

---

## Tang2024
**Metadatos:** CORRECTO
- evidencia: `https://api.crossref.org/works/10.1038/s41598-024-71709-4` → Xiaoyan Tang, Funan Liu, Xinling Hu, *Scientific Reports* **14**(1), 2024-09-10. Metadatos del editor: `prism.startingPage = 21106`; cita sugerida por Nature: «Tang, X., Liu, F. & Hu, X. […] *Sci Rep* **14**, 21106 (2024)». URL: https://www.nature.com/articles/s41598-024-71709-4
- correcciones: ninguna. `pages={21106}` es el número de artículo, y coincide con la primera página declarada por el editor.

**Afirmaciones:**
1. `cap03:371` — «acoplan un autómata celular con un algoritmo de búsqueda gravitacional para simular la expansión de Urumqi, en el noroeste de China, y reportan FoM de $0{,}430$ en la calibración (2000–2010) y de $0{,}376$ en la validación (2010–2020)» → **CORRECTA, con exactitud numérica**. Cita textual: «…the GSA-CA model for the arid city of **Urumqi in Northwest China from 2000 to 2010, and validated the model from 2010 to 2020** […] GSA-CA achieved an overall accuracy of 98.42% and a **figure of merit (FOM) of 43.03% for the year 2010**, and an overall accuracy of 98.52% with **FOM of 37.64% for 2020**». Y en resultados: «**During the calibration phase**, the highest overall accuracy (98.42%) and **FOM (43.03%)** were observed at kbest value of 0.5. A similar pattern was observed in **the validation phase**, with peaks in overall accuracy (98.52%) and **FOM (37.64%)**». 43,03 % → 0,430 ✓; 37,64 % → 0,376 ✓. Períodos y fases correctamente asignados. URL: https://www.nature.com/articles/s41598-024-71709-4
2. `cap06:236` — fila de tabla «AC con búsqueda gravitacional — FoM = 0,430 en calibración y 0,376 en validación; contexto árido con drivers físicos explícitos» → **CORRECTA**. GSA = Gravitational Search Algorithm; contexto árido explícito en el propio artículo («Urumqi is a typical arid city in Northwest China»). Misma URL.
3. `cap06:242` — «reporta un FoM de $0{,}376$ en **validación independiente**» → **CORRECTA** en la cifra y la fase. *Matiz para defensa:* la validación es un hold-out temporal (2010–2020, distinto del período de ajuste), pero el 37,64 % reportado es el **máximo sobre el barrido de kbest en la propia fase de validación** («with peaks in overall accuracy (98.52%) and FOM (37.64%)»), de modo que «independiente» es correcto en cuanto al período y algo generoso en cuanto a la selección del hiperparámetro. Misma URL.

**Riesgo en examen:** bajo. Es la referencia mejor sustentada del lote; las tres cifras coinciden al tercer decimal con el artículo. Si un sinodal aprieta sobre «independiente», el matiz anterior basta como respuesta.

---

## Waddell2002
**Metadatos:** INCORRECTO (título)
- evidencia: `https://api.crossref.org/works/10.1080/01944360208976274` → «UrbanSim: Modeling Urban Development for Land Use, **Transportation, and** Environmental Planning», Paul Waddell, *Journal of the American Planning Association* **68(3): 297–314**, 2002. Confirmado en https://www.tandfonline.com/doi/abs/10.1080/01944360208976274 y en la ficha TRID (https://trid.trb.org/View/721649: «Pagination: p. 297-314; Volume 68; Issue Number 3»).
- correcciones: `Land Use, Transportation and Environmental Planning` → `Land Use, Transportation, and Environmental Planning` (falta la coma serial antes de «and»). Volumen, número, páginas y DOI: correctos.

**Afirmaciones:**
1. `cap03:49` — «UrbanSim es la referencia de esa vertiente y acopla usos de suelo, transporte y ambiente en un mismo sistema» → **CORRECTA**. Resumen: «Metropolitan areas have come under intense pressure to respond to federal mandates to **link planning of land use, transportation, and environmental quality** […] **UrbanSim is a new model system** that was developed to respond to these emerging requirements and has now been applied in three metropolitan areas. This article describes the model system and its application to Eugene-Springfield, Oregon». URL: https://www.tandfonline.com/doi/abs/10.1080/01944360208976274
   - «Es la referencia de esa vertiente» es valoración de la tesis, no del artículo; queda respaldada por su recepción (≈1 350 citas según SciSpace) y es formulación prudente.

**Riesgo en examen:** bajo.

---

## Resumen de correcciones al `.bib`

| clave | campo | valor actual → valor correcto |
|---|---|---|
| `CONABIO2024Porque` | `year` | `2024` → `2022` |
| `CONAPO2018Delimitacion` | `author` | un solo grupo con «and» dentro → tres grupos: `{Secretaría de Desarrollo Agrario, Territorial y Urbano} and {Consejo Nacional de Población} and {Instituto Nacional de Estadística y Geografía}` |
| `CONAPO2018Delimitacion` | tipo | `@online` → `@report`, con `edition={Primera}` y `address={Ciudad de México}` |
| `Seto2012teleconnections` | `author` | `Seto, Karen C. and others` → los 10 autores |
| `Seto2012teleconnections` | `doi` | ausente → `10.1073/pnas.1117622109` |
| `Waddell2002` | `title` | `…Transportation and Environmental…` → `…Transportation, and Environmental…` |
| `HernandezGuerreroOsorno2018` | `author` | `Hern{\'a}ndez-Guerrero` / `Osorno-S{\'a}nchez` → sin guion, apellidos compuestos protegidos |

---

## Resumen del lote

**Revisadas 10 referencias y 21 afirmaciones.** Todas las cifras atribuidas a terceros resultaron exactas; los problemas están en metadatos y en una frase de redacción propia.

- **Metadatos incorrectos: 4** (+1 salvedad menor).
- **Afirmaciones: 16 correctas, 4 parciales, 1 no verificable, 0 con cifra errónea.**

**Problemas graves:**

1. **`Seto2012teleconnections` — `cap01_introduccion.tex:38`.** El ejemplo «el rio Mixcoac que desemboca en lagos cercanos a la ciudad de México» no está en Seto et al. (2012) —artículo conceptual sobre China, India y retos globales— **y es falso**: el Mixcoac fue entubado en 1955 (hoy es la avenida homónima) y el Churubusco, al que alimentaba, fue desviado en 1952 precisamente para dejar de abastecer los lagos de Xochimilco, Mixquic y Tláhuac. Más dos erratas: «rios», «ejemlplo». Corrección: eliminar el ejemplo; la cláusula citada («flujos…») sí la sostiene el artículo, aunque su término exacto es «flows of people, economic goods and services».
2. **`BonhamCarter1994` — `cap02_marco_teorico.tex:201`.** La cuarta limitación («no corrige la autocorrelación espacial») no se pudo verificar en el capítulo 9; lo que el libro documenta es independencia condicional entre capas. La mención verificada de autocorrelación espacial está en Bonham-Carter, Agterberg y Wright (1988), y sobre tamaño de celda. Corrección: citar ahí `Legendre1993,Brenning2012` (ya usados en `cap03:350` para ese mismo punto).
3. **`CONAPO2018Delimitacion` — `cap01_introduccion.tex:69`.** El «millón y medio de habitantes» no sale de esta fuente: el documento reporta **1 323 640** para la ZM de Querétaro en 2015 y la ubica entre las que rebasaron **el millón**. La cifra de 1,5 M es del Censo 2020. Corrección: dejar la población a cargo solo de `INEGI2020Censo`.
4. **`CONAPO2018Delimitacion`, campo `author`.** Todo el campo va dentro de un solo grupo de llaves, así que biblatex imprimirá literalmente «and SEDATU and INEGI» como si fuera un nombre. Además el orden oficial es SEDATU, CONAPO, INEGI, y el tipo debería ser `@report`, no `@online`.
5. **`Ojala2002` — `cap02_marco_teorico.tex:40`.** «Los patrones binarios locales, propuestos por Ojala (2002)»: el propio artículo remite el LBP original a Ojala, Pietikäinen y Harwood (1996); de 2002 es la generalización circular $(P,R)$. Corrección de un verbo.

**Correcciones menores de metadatos:** año de CONABIO 2024→2022; `Seto2012` con lista de autores truncada y sin DOI (`10.1073/pnas.1117622109`); título de `Waddell2002` sin la coma serial («Transportation, and»); apellidos de Hernández Guerrero y Osorno Sánchez sin guion en el original.

**Lo que salió impecable:** `Tang2024` (FoM 0,430 calibración y 0,376 validación coinciden con 43,03 % y 37,64 % del artículo, con fases y períodos bien asignados), la regla del 15 % de la prueba ómnibus (verificada con página: Bonham-Carter 1994, p. 316), `Li2007`, `Waddell2002` y `alonso1964location` en el fondo de sus afirmaciones.

**VEREDICTO DEL LOTE: REQUIERE CORRECCIONES.**
