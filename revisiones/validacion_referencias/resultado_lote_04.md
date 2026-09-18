# Resultado — Lote 4 de 8 (11 referencias)

Fecha de validación: 18 de septiembre de 2026.

Método: (1) API de Crossref para metadatos de los 8 DOIs del lote; (2) API de PubMed,
Europe PMC y OpenAlex para resúmenes; (3) OpenLibrary para el libro; (4) la página oficial
del editor para el informe; (5) un único PDF de acceso abierto localizado por búsqueda web
para la referencia sin DOI (`JimenezLopez2018`). Las referencias R1–R4 que aparecen en las
afirmaciones son, según `cap03_estado_del_arte.tex:10`: R1 simulación histórica con
precisión, R2 soporte a planeación estratégica, R3 replicabilidad y apertura, R4
interpretabilidad de la función de transición.

---

## Arfiansyah2024

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.1007/s41324-024-00599-5` devuelve
  `journal-article`, título «Cellular automata modelling to simulate patterns of urban
  growth for Nusantara: Indonesia's new capital», *Spatial Information Research*,
  vol. 32, núm. 6, pp. 829–849, autores en el orden Arfiansyah, Dody; Hawken, Scott;
  Zlatanova, Sisi; Han, Hoon; publicado en papel 12/2024.
- correcciones: ninguna. Los cuatro autores, el orden, año, volumen, número, páginas y DOI
  coinciden exactamente con la entrada del `.bib`.

**Afirmaciones:**

1. `cap03_estado_del_arte.tex:31` — «lo aplicó a Nusantara, la capital planificada de
   Indonesia, un caso de ciudad de nueva creación» — **CORRECTA**.
   Resumen literal en Crossref: «This paper uses cellular automata (CA) modelling to
   simulate possible patterns of urban growth for Nusantara–Indonesia's new capital. […]
   The study's cellular automata approach and methodology can be employed for urban
   planning and biodiversity impact assessment in similar contexts of new city
   development.»
   URL: `https://api.crossref.org/works/10.1007/s41324-024-00599-5`

**Riesgo en examen:** nulo. Metadatos exactos y la afirmación es prácticamente el título
del artículo.

---

## Batty2005

**Metadatos:** CORRECTO

- evidencia: `https://openlibrary.org/books/OL3309559M.json` devuelve
  `title: "Cities and complexity"`, `subtitle: "understanding cities with cellular
  automata, agent-based models, and fractals"`, `publishers: ["MIT Press"]`,
  `publish_date: 2005`, `number_of_pages: 565`. La búsqueda
  `https://openlibrary.org/search.json?q=title:"Cities and complexity" author:Batty`
  confirma autor único Michael Batty, ediciones 2005 y 2007, MIT Press,
  ISBN 0262025833 / 9780262025836 (tapa dura 2005) y 0262524791 / 9780262524797
  (rústica 2007).
- correcciones: ninguna incorrecta. Incompletitud (no error): faltan `address = {Cambridge, MA}`
  e `isbn`. Si la Dra. Lárraga exige ISBN en los libros, conviene añadir
  `9780262025836` (la edición de 2005, que es la que corresponde al año citado).
- no verificable: el sitio de MIT Press devolvió HTTP 403 y la API de Google Books
  devolvió respuesta vacía; el subtítulo se confirmó por OpenLibrary, no por el editor.

**Afirmaciones:**

1. `cap01_introduccion.tex:143` — «los AC capturan la dinámica de difusión espacial
   emergente sin requerir la especificación de reglas de comportamiento individual, y
   mantienen la interpretabilidad del modelo al nivel de la rejilla espacial» —
   **PARCIAL**.
   El subtítulo verificado acredita que el libro trata AC, ABM y fractales en un mismo
   volumen, de modo que la cita es pertinente para el contraste AC/ABM. No se pudo leer el
   cuerpo del libro (565 pp., sin copia de acceso abierto), así que no se puede certificar
   que Batty enuncie esa comparación en esos términos. La afirmación va además respaldada
   por `Clarke1998` y `Sante2010` en la misma llamada, lo que reparte la carga.
   URL: `https://openlibrary.org/books/OL3309559M.json`
2. `cap03_estado_del_arte.tex:31` — «quedó sistematizado en Batty (2005), que examina en un
   mismo tratamiento los autómatas celulares, los modelos basados en agentes y las
   descripciones fractales» — **CORRECTA**.
   Subtítulo verificado: «understanding cities with cellular automata, agent-based models,
   and fractals».
   URL: `https://openlibrary.org/books/OL3309559M.json`

**Riesgo en examen:** bajo. La afirmación 2 es literalmente el subtítulo. La 1 es una
lectura estándar y compartida con otras dos citas; sostenible, aunque no se verificó el
texto interior.

---

## Couclelis1985

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.1068/a170585` devuelve `journal-article`,
  «Cellular Worlds: A Framework for Modeling Micro—Macro Dynamics», *Environment and
  Planning A: Economy and Space*, SAGE, vol. 17, núm. 5, pp. 585–596, 05/1985, autora
  Couclelis, H.
- correcciones: ninguna sustantiva. Dos puntos cosméticos:
  - título: `Micro-Macro` -> `Micro—Macro` (Crossref usa raya, el `.bib` guion; irrelevante
    para la cita).
  - nombre de la revista: en 1985 la revista se llamaba **Environment and Planning A**; el
    sufijo «: Economy and Space» es del retitulado de 2018. El `.bib` reproduce el nombre
    actual, que es lo que devuelve Crossref, así que es defendible; solo mencionarlo si un
    sinodal es purista con el nombre de época.
  - «Helen» es correcto: Crossref abrevia a «H».

**Afirmaciones:**

1. `cap03_estado_del_arte.tex:18` — «formalizado en términos teóricos por
   \textcite{Couclelis1985}» — **CORRECTA**.
   Resumen literal: «Despite their obvious spatial interpretation, standard cell-space
   models are too constrained by their background conventions to be useful in realistic
   geographic applications. In this paper, a generalization of the cell-space principle is
   presented, based on discrete model theory, and then applied to a hypothetical but fairly
   complex problem of individual decision and large-scale urban change.»
   «Generalización del principio de espacio celular basada en teoría de modelos discretos»
   es exactamente una formalización teórica.
   URL: `https://api.crossref.org/works/10.1068/a170585`

**Riesgo en examen:** nulo.

---

## Gorelick2017

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.1016/j.rse.2017.06.031` devuelve
  `journal-article`, «Google Earth Engine: Planetary-scale geospatial analysis for
  everyone», *Remote Sensing of Environment*, Elsevier, vol. 202, pp. 18–27, 12/2017,
  autores Gorelick, Noel; Hancher, Matt; Dixon, Mike; Ilyushchenko, Simon; Thau, David;
  Moore, Rebecca (mismo orden que el `.bib`).
- correcciones: ninguna. La revista no asigna número a ese volumen, por lo que la ausencia
  de `number` es correcta.

**Afirmaciones:**

1. `cap02_marco_teorico.tex:24` — «plataformas de procesamiento en la nube como Google
   Earth Engine, que permiten consultar y procesar grandes volúmenes sin descargarlos» —
   **CORRECTA**.
   Resumen literal (OpenAlex): «Google Earth Engine is a cloud-based platform for
   planetary-scale geospatial analysis that brings Google's massive computational
   capabilities to bear on a variety of high-impact societal issues […] It is unique in the
   field as an integrated platform designed to empower not only traditional remote sensing
   scientists, but also a much wider audience that lacks the technical capacity needed to
   utilize traditional supercomputers or large-scale commodity cloud computing resources.»
   El núcleo (plataforma en la nube para análisis geoespacial a gran escala) está
   literalmente en el resumen. El matiz «sin descargarlos» no aparece con esas palabras,
   pero es la propiedad definitoria de una plataforma en la nube y no es una cifra ni una
   atribución de hallazgo.
   URL: `https://api.openalex.org/works/https://doi.org/10.1016/j.rse.2017.06.031`
2. `cap05_modelo_crecimiento_urbano.tex:44` — «en este trabajo **no** se empleó Google Earth
   Engine \parencite{Gorelick2017} ni descarga masiva por API» — **CORRECTA**.
   Uso puramente referencial: la cita solo identifica la plataforma que la tesis declara no
   haber usado. No atribuye nada al artículo.
   URL: `https://api.crossref.org/works/10.1016/j.rse.2017.06.031`

**Riesgo en examen:** nulo.

---

## ITDP2019Externalidades

**Metadatos:** CORRECTO

- evidencia: la URL del `.bib` responde HTTP 200 y la página se titula «Externalidades
  negativas asociadas al transporte terrestre (2019) – ITDP México». Texto literal de la
  página: «Iniciativa Climática de México (ICM) y el Instituto de Políticas para el
  Transporte y el Desarrollo (ITDP, por sus siglas en inglés) presentan un estudio que
  estima el costo social de las externalidades negativas del transporte terrestre para
  México y sus 20 zonas metropolitanas más pobladas.»
  URL: `https://mexico.itdp.org/download/externalidades-negativas-asociadas-al-transporte-terrestre-2019/`
- correcciones: ninguna. Los dos autores institucionales del `.bib` coinciden con los del
  estudio, `institution = {ITDP México}` coincide con el portal que lo publica, y el año
  2019 aparece en el título de la publicación. Advertencia menor: la página muestra «Fecha
  de creación 26 enero, 2023», que es la fecha de carga del archivo al sitio, no la del
  informe; si un sinodal lo señala, la respuesta es que el título oficial lleva «(2019)».

**Afirmaciones:**

1. `cap01_introduccion.tex:24` — «El transporte terrestre que la dispersión convierte en
   necesario tiene externalidades negativas documentadas para el caso mexicano» —
   **CORRECTA**.
   Texto literal de la página: «se analizan los costos sociales del transporte terrestre
   en cinco dimensiones principales: gases de efecto invernadero (GEI), contaminación del
   aire por material particulado (PM10 y PM2.5), congestión, siniestros de tránsito y
   ruido. El ITDP estima que el costo social de estas externalidades negativas suma entre
   3 y 5% del producto interno del bruto (PIB) del país.»
   La tesis no cita ninguna cifra del informe, solo la existencia de la documentación, que
   queda acreditada. Si se quisiera reforzar el argumento, la cifra 3–5 % del PIB está
   disponible y es citable.
   URL: `https://mexico.itdp.org/download/externalidades-negativas-asociadas-al-transporte-terrestre-2019/`

**Riesgo en examen:** bajo. Se verificó la página de descarga del editor, no el PDF; pero la
afirmación de la tesis es tan genérica que la página basta.

---

## JimenezLopez2018

**Metadatos:** INCORRECTO

- evidencia: se localizó el PDF completo del artículo en el repositorio institucional de la
  UAEMex. Encabezado literal de la primera página: «Geografía y Sistemas de Información
  Geográfica (GEOSIG). Revista digital del Grupo de Estudios sobre Geografía y Análisis
  Espacial con Sistemas de Información Geográfica (GESIG). Programa de Docencia e
  Investigación en Sistemas de Información Geográfica (PRODISIG). Universidad Nacional de
  Luján, Argentina. / http://www.revistageosig.wixsite.com/geosig (ISSN 1852-8031) /
  Luján, Año 10, Número 12, 2018, Sección II: Metodología. pp. 1-26». La cita sugerida por
  la propia revista, al pie del artículo, es: «Jiménez López, E.; Chávez Soto, T.;
  Garrocho, C. 2018. Modelando la expansión urbana con autómatas celulares: aplicación de
  la estación de inteligencia territorial (CHRISTALLER). Geografía y Sistemas de
  Información Geográfica (GeoSIG). 10(12) Sección II: 1-26».
  URL: `http://ri.uaemex.mx/bitstream/handle/20.500.11799/112650/79758e_5db4574cbd884d7b89e96df748dca7cc.pdf?sequence=1`
- correcciones:
  - `volume = {12}` -> `volume = {10}`
  - (campo ausente) -> `number = {12}`
  - opcionalmente añadir `issn = {1852-8031}` y `note = {Sección II: Metodología}`
  - autores, orden (Jiménez López, Chávez Soto, Garrocho), año, título y páginas 1–26 son
    **correctos** y coinciden con el encabezado y la cita sugerida.

**Afirmaciones:**

1. `cap03_estado_del_arte.tex:438` — «instrumentan autómatas celulares deterministas sobre
   rejilla raster y, mediante un barrido exhaustivo de las 256 reglas de transición
   posibles, identifican la que mejor replica la expansión observada de Querétaro, San Luis
   Potosí y Toluca entre 2003 y 2017; para Querétaro la regla 192 alcanza un Kappa de
   Cohen de 0,53 y un índice de Jaccard de 0,76» — **CORRECTA en todos sus elementos**.
   Citas textuales del PDF:
   - período y ciudades: «usamos software y herramientas de simulación para instrumentar
     modelos de AC que repliquen la expansión de la mancha urbana entre 2003 y 2017, de
     tres grandes ciudades mexicanas: Querétaro, San Luis Potosí y Toluca».
   - 256 reglas y barrido exhaustivo: «CHRISTALLER hace un barrido completo de las 256
     Reglas de Transición, estima los indicadores de bondad de ajuste e identifica la Regla
     de Transición que mejor replica la expansión de la mancha urbana de cada ciudad en el
     periodo de experimentación».
   - regla 192: «Para Querétaro la Regla de Transición que mostró el mejor ajuste entre la
     imagen satelital de 2017 y la proyección del modelo de AC fue la 192; para San Luis
     Potosí fue la 218 y para Toluca la Regla 222».
   - cifras: el Cuadro 3 del artículo contiene la fila «Querétaro | 0.53 | 0.76», y el
     texto añade «Querétaro registró un Jaccard de 0.76, Toluca 0.68, y San Luis Potosí un
     sorprendente 0.91».
   - rejilla raster: «el espacio geográfico se representa en una malla regular, que es la
     base de los AC. La malla tiene (n x m) células, lo que determina su resolución
     espacial».
   URL: `http://ri.uaemex.mx/bitstream/handle/20.500.11799/112650/79758e_5db4574cbd884d7b89e96df748dca7cc.pdf?sequence=1`
2. `cap03_estado_del_arte.tex:472` (tabla, fila `JimenezLopez2018`: R1 parcial, R2 ✓,
   R3 parcial, R4 parcial) — **PARCIAL**, con una advertencia.
   R1 y R4: defendibles. El artículo sí simula 2003→2017 y reporta bondad de ajuste, pero
   sin prueba fuera de muestra (véase afirmación 5); y las reglas de Wolfram de tres bits
   son explícitas pero no expresan impulsores geográficos.
   R2: verificado. Objetivo declarado en el resumen: «devela si los AC deterministas son
   capaces de simular la expansión de la mancha urbana […] con el fin de realizar
   proyecciones».
   **R3 es el punto débil.** El resumen del artículo se autodescribe como «software
   especializado de código abierto desarrollado en El Colegio Mexiquense». Calificar R3
   («replicabilidad y apertura») como «parcial» es una valoración del tesista, no un dato
   del artículo, y contradice en apariencia lo que la fuente dice de sí misma. La tesis ya
   la argumenta en `cap03_estado_del_arte.tex:233` («que exige apertura y no solo
   gratuidad») y el artículo de 2021 del mismo grupo dice que los módulos «están
   disponibles, sin costo, bajo demanda», lo que apoya la distinción. Recomendación: dejar
   explícito en la nota al pie de la tabla que «parcial» en R3 significa ausencia de
   repositorio público, no ausencia de declaración de código abierto.
   URL: misma que arriba.
3. `cap03_estado_del_arte.tex:489` — «Querétaro ya ha sido modelada con autómatas
   celulares» — **CORRECTA**. Verificado en la afirmación 1.
4. `cap05_modelo_crecimiento_urbano.tex:358` — «existe un antecedente de simulación con
   autómatas celulares para Querétaro entre 2003 y 2017» — **CORRECTA**. Verificado en la
   afirmación 1.
5. `cap06_resultados_analisis.tex:244` — la exclusión del antecedente de la tabla
   comparativa por dos razones declaradas — **CORRECTA, y es el uso más sólido de esta
   fuente en toda la tesis**. Se verificaron los cuatro elementos del argumento:
   - «no reporta FoM, sino Kappa de Cohen, índice de Jaccard y dimensión fractal»:
     correcto. Palabras clave del artículo: «Expansión urbana, Autómatas Celulares,
     Dimensión Fractal, Índice Kappa de Cohen, Índice de Jaccard». No aparece FoM en ningún
     lugar del artículo.
   - «horizonte de catorce años»: correcto, 2003–2017.
   - «sobre la clase urbana completa»: correcto. El artículo compara «mapas de categorías
     binarias» de urbano/no urbano y declara «casi 7.0 millones de pixeles por ciudad».
   - «el mapa observado de 2017 se emplea simultáneamente para seleccionar la regla de
     transición entre las 256 posibles y para medir el ajuste de la regla seleccionada»:
     correcto y textualmente respaldado por el pasaje ya citado en la afirmación 1
     («barrido completo de las 256 Reglas […] e identifica la Regla de Transición que mejor
     replica la expansión […] en el periodo de experimentación»). La crítica metodológica
     es legítima y está bien fundada.
   URL: misma que arriba.
6. `cap07_conclusiones.tex:84` — «existe un antecedente de simulación con autómatas
   celulares para esta misma ciudad en el período 2003--2017» — **CORRECTA**. Verificado en
   la afirmación 1.

**Riesgo en examen:** el contenido es sólido y verificado al detalle, incluidas las cuatro
cifras. El único riesgo real es el metadato: `volume = {12}` en lugar de `volume = {10},
number = {12}`. Es un error de ficha, no de fondo, pero es el tipo de cosa que un sinodal
mexicano familiarizado con El Colegio Mexiquense puede detectar. Corregirlo antes de
entregar.

---

## Montejano2023

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.22198/rys2023/35/1734` devuelve
  `journal-article`, «Expansión y crecimiento urbanos en México, 1975-2020», *región y
  sociedad*, El Colegio de Sonora, vol. 35, `page: e1734`, publicado 11/09/2023, autores
  Montejano-Escamilla, Jorge Alberto; Caudillo-Cos, Camilo Alberto; Ávila-Jiménez, Felipe
  Gerardo; Tapia-McClung, Rodrigo; Barrera-Alarcón, Itzia Gabriela (mismo orden que el
  `.bib`).
- correcciones: ninguna sustantiva. Nota: Crossref registra `e1734` como **página**, no
  como número de fascículo (que viene vacío). El `.bib` lo pone en `number = {e1734}`. Es
  una convención aceptable para artículos con número de artículo, pero si se quiere
  coincidencia exacta con el registro, sería `pages = {e1734}`.

**Afirmaciones:**

1. `cap05_modelo_crecimiento_urbano.tex:44` — «dentro de la expansión urbana que el país
   arrastra desde los años setenta» — **CORRECTA**.
   Resumen literal: «se presenta una estimación del aumento de la población y de la
   expansión urbana en México […] entre 1975 y 2020 […] se concluye que mientras que el
   suelo edificado total creció tres veces en cuarenta años, la población tan solo se
   duplicó. […] se confirma que hay una tendencia generalizada a la expansión urbana
   dispersa.»
   La tesis no atribuye ninguna cifra al artículo, solo el encuadre temporal, que coincide
   con la ventana 1975–2020 del estudio.
   URL: `https://api.crossref.org/works/10.22198/rys2023/35/1734`

**Riesgo en examen:** nulo. Si se quisiera reforzar, el dato «el suelo edificado creció
tres veces mientras la población se duplicó» está verificado y es citable.

---

## Sante2010

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.1016/j.landurbplan.2010.03.001` devuelve
  `journal-article`, «Cellular automata models for the simulation of real-world urban
  processes: A review and analysis», *Landscape and Urban Planning*, Elsevier, vol. 96,
  núm. 2, pp. 108–122, 05/2010, autores Santé, Inés; García, Andrés M.; Miranda, David;
  Crecente, Rafael (mismo orden que el `.bib`). Confirmado de forma independiente por
  `https://api.openalex.org/works/https://doi.org/10.1016/j.landurbplan.2010.03.001`
  (volumen 96, fascículo 2, pp. 108–122).
- correcciones: ninguna. Todos los campos coinciden.

**Afirmaciones:**

> Advertencia de acceso: **no se pudo leer el resumen ni el cuerpo de este artículo.**
> Intentos: Crossref (sin resumen); API de Semantic Scholar en lote (el editor suprimió el
> campo `abstract`, respuesta explícita «The following paper fields have been elided by the
> publisher: {'abstract'}»); OpenAlex (`abstract_inverted_index` nulo); Europe PMC (sin
> registro para este DOI); ScienceDirect `article/abs/pii/S0169204610000708` (HTTP 403);
> `ouci.dntb.gov.ua` (página que requiere JavaScript); `colab.ws` (HTTP 403, DDoS-Guard);
> `scilit.com` (HTTP 403); API de CORE (`totalHits: 0`); una búsqueda web, que solo
> devolvió artículos que lo citan, no su texto. OpenAlex confirma que **no existe copia de
> acceso abierto**. Por eso las tres afirmaciones se apoyan únicamente en el título
> verificado y en fuentes secundarias.

1. `cap01_introduccion.tex:117` — «Los modelos de caja blanca más citados
   \parencite{Clarke1998,Sante2010,Aburas2016} atienden la interpretabilidad, pero en la
   mayoría de los casos se validan sobre un único período quinquenal» — **NO VERIFICABLE, y
   es la afirmación de mayor riesgo del lote**.
   Dos problemas distintos:
   - **Cuantitativo**: «en la mayoría de los casos se validan sobre un único período
     quinquenal» es una afirmación empírica específica sobre la distribución de prácticas de
     validación en la literatura. No se pudo acceder al artículo para comprobar si Santé et
     al. la sostienen, y no se encontró ninguna fuente accesible que se la atribuya. Un
     sinodal puede pedir la página exacta.
   - **De categoría**: `Sante2010` y `Aburas2016` son **revisiones**, no modelos. La frase
     los enumera entre «los modelos de caja blanca más citados» junto a `Clarke1998`, que sí
     es un modelo (SLEUTH). Esto es un desliz de redacción que conviene corregir aun sin
     tocar el contenido: bastaría «los modelos de caja blanca más citados
     \parencite{Clarke1998} y las revisiones que los sistematizan
     \parencite{Sante2010,Aburas2016}».
   URL consultadas sin éxito: `https://www.sciencedirect.com/science/article/abs/pii/S0169204610000708`
   (403), `https://api.openalex.org/works/https://doi.org/10.1016/j.landurbplan.2010.03.001`
   (sin resumen).
2. `cap01_introduccion.tex:143` — «los AC capturan la dinámica de difusión espacial
   emergente sin requerir la especificación de reglas de comportamiento individual, y
   mantienen la interpretabilidad del modelo al nivel de la rejilla espacial» —
   **PARCIAL**.
   Compatible con el objeto declarado del artículo (revisión de modelos de AC para procesos
   urbanos reales), pero no verificado en el texto. Es una caracterización general de los AC
   que la afirmación reparte entre tres fuentes, no una cifra ni un hallazgo exclusivo, así
   que el riesgo es bajo.
3. `cap03_estado_del_arte.tex:511` — «Las revisiones sistemáticas de \textcite{Sante2010} y
   \textcite{Aburas2016} documentan la heterogeneidad de los métodos de calibración y
   validación entre las decenas de modelos que examinan» — **PARCIAL**.
   El carácter de revisión está verificado por el título («A review and analysis»). Lo que
   **no** se pudo verificar es el cuantificador «decenas de modelos» ni que el artículo
   documente específicamente la heterogeneidad de calibración y validación. Evidencia
   secundaria, no del propio artículo: el trabajo posterior del mismo grupo (García, Santé,
   Boullón, Crecente; DOI `10.1080/13658816.2012.762454`) afirma en su resumen «CA models
   still have to overcome some shortcomings related to their flexibility and difficult
   calibration» y cita como apoyo la revisión de 2010, lo que es congruente pero no es la
   fuente citada.
   URL de la evidencia secundaria: `https://doi.org/10.1080/13658816.2012.762454`
   Recomendación: sustituir «las decenas de modelos que examinan» por una formulación sin
   cifra, o bien conseguir el PDF por la biblioteca digital de la UNAM (`pbidi`, acceso
   Elsevier) y anotar el número exacto de modelos revisados y la página de la sección de
   calibración.

**Riesgo en examen:** MEDIO-ALTO, y es el punto más expuesto del lote. Los metadatos son
impecables, pero tres afirmaciones se apoyan en un artículo de pago que no se pudo abrir, y
una de ellas (afirmación 1) contiene un cuantificador fuerte, «en la mayoría de los casos […]
un único período quinquenal», que además sostiene la justificación central de la tesis, las
cinco ventanas desplazadas. Si un sinodal pide la fuente de ese «la mayoría», hay que poder
mostrarla. Es imprescindible descargar el PDF por el proxy de la UNAM y verificar, o
reformular para que el peso lo lleve una fuente que sí se pueda exhibir.

> **Nota del orquestador (verificada de forma independiente).** El cuantificador sí tiene
> fuente exhibible, pero no es Santé et al. Es van Vliet et al. (2016), ya presente en el
> `.bib` y ya citada en `cap03:373` para exactamente este punto: documenta que el 31 % de
> las aplicaciones no reporta validación alguna y que el resto usa típicamente una sola
> comparación. La corrección consiste en trasladar el peso del cuantificador a `vanVliet2016`
> y dejar a `Sante2010` y `Aburas2016` como lo que son, revisiones.

---

## Tobler1979

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.1007/978-94-009-9394-5_18` devuelve
  `book-chapter`, «Cellular Geography», contenedor *Philosophy in Geography*, Springer
  Netherlands, pp. 379–386, 1979, autor Tobler, W. R. Se consultó además el registro del
  libro contenedor `https://api.crossref.org/works/10.1007/978-94-009-9394-5`, que devuelve
  `type: book`, título «Philosophy in Geography» y editores **Gale, Stephen; Olsson,
  Gunnar**, exactamente los del `.bib`.
- correcciones: ninguna. `publisher = {Reidel}` y `address = {Dordrecht}` corresponden al
  pie editorial original de 1979 (D. Reidel, Dordrecht); Crossref muestra «Springer
  Netherlands» porque es el sello sucesor. Ambas formas son correctas; la del `.bib` es la
  histórica y es la preferible.

**Afirmaciones:**

1. `cap03_estado_del_arte.tex:18` — «el uso de autómatas celulares para representar
   fenómenos geográficos fue propuesto por \textcite{Tobler1979}» — **PARCIAL**.
   No se pudo leer el capítulo: Crossref y OpenAlex no registran resumen y no existe copia
   de acceso abierto. Lo verificado es el metadato completo, incluido el título «Cellular
   Geography», que es congruente con la atribución.
   Evidencia secundaria, esta sí de una fuente abierta y leída: el propio artículo de
   `JimenezLopez2018` sitúa a Tobler como origen de la línea, «La literatura reporta
   ejemplos que muestran logros cuando se articulan AC con SIG para simular fenómenos
   urbanos (Tobler 1979, White y Engelen 1993, Clarke y Gaydos 1998, …)», y lo registra en
   su bibliografía como «Tobler, W. R. (1979). Cellular geography. In Philosophy in
   geography. Springer Netherlands».
   URL: `http://ri.uaemex.mx/bitstream/handle/20.500.11799/112650/79758e_5db4574cbd884d7b89e96df748dca7cc.pdf?sequence=1`

**Riesgo en examen:** bajo. La paternidad de Tobler sobre la geografía celular es un lugar
común del campo y el título del capítulo la respalda, pero no se leyó el texto original. Si
se quiere blindar, la revisión de Santé et al. (2010) o el propio `JimenezLopez2018` sirven
como aval secundario explícito.

---

## Verburg2002

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.1007/s00267-002-2630-x` devuelve
  `journal-article`, «Modeling the Spatial Dynamics of Regional Land Use: The CLUE-S
  Model», *Environmental Management*, vol. 30, núm. 3, pp. 391–405, 09/2002, autores
  Verburg, Peter H.; Soepboer, Welmoed; Veldkamp, A.; Limpiada, Ramil; Espaldon, Victoria;
  Mastura, Sharifah S. A. (mismo orden que el `.bib`). Confirmado de forma independiente en
  PubMed (PMID 12148073): «Environ Manage. 2002 Sep;30(3):391-405».
- correcciones: ninguna. `publisher = {Springer}` es redundante en un `@article` pero no es
  un error.

**Afirmaciones:**

> Acceso: se obtuvo el resumen completo por dos vías independientes (PubMed y el repositorio
> de Wageningen), pero **no el cuerpo del artículo**. OpenAlex lista seis ubicaciones y
> ninguna ofrece PDF (`pdf_url: None` en todas); Unpaywall devolvió lista vacía de
> `oa_locations` con `url_for_pdf`; la página de Wageningen no expone enlace a PDF.

1. `cap03_estado_del_arte.tex:41` — «la metodología LUCC introdujo modelos basados en mapas
   de idoneidad de uso de suelo de base estadística: sistemas como CLUE-S […] brindaron
   arquitecturas alternativas al autómata celular puro en las que la función de transición
   emerge de regresiones logísticas o pesos bayesianos sobre datos históricos» —
   **PARCIAL**.
   Verificado por el resumen: CLUE-S es «a dynamic, spatially explicit, land-use change
   model […] for the regional scale», «developed for the analysis of land use in small
   regions (e.g., a watershed or province) at a fine spatial resolution», con estructura
   «based on systems theory» y factores «socio-economic and biophysical driving factors».
   Eso acredita que es un modelo LUCC y no un AC puro. Lo que el resumen **no** dice es
   «regresión logística»; el resumen no nombra ninguna técnica estadística. La afirmación de
   la tesis es además disyuntiva y colectiva («regresiones logísticas o pesos bayesianos»,
   sobre cuatro sistemas), así que el peso sobre Verburg es menor.
   URL: `https://research.wur.nl/en/publications/modeling-the-spatial-dynamics-of-regional-land-use-the-clue-s-mod`
2. `cap03_estado_del_arte.tex:288` — «CLUE-S acopla una idoneidad derivada de regresión
   logística con una asignación iterativa gobernada por la demanda sectorial de suelo» —
   **PARCIAL**.
   Es una descripción técnica precisa de dos componentes concretos (regresión logística para
   la idoneidad; módulo de demanda que gobierna la asignación iterativa) y **ninguno de los
   dos aparece en el resumen verificado**. El resumen sí respalda la parte de asignación
   condicionada por restricciones: «Stability is incorporated by a set of variables that
   define the relative elasticity of the actual land-use type to conversion», y «The model
   explicitly addresses the hierarchical organization of land use systems, spatial
   connectivity between locations and stability». No es una cifra ni una atribución de
   hallazgo ajeno, y la descripción es la canónica de CLUE-S, pero no se pudo confirmar en
   la fuente.
   URL: `https://research.wur.nl/en/publications/modeling-the-spatial-dynamics-of-regional-land-use-the-clue-s-mod`

**Riesgo en examen:** BAJO-MEDIO. Metadatos impecables y el encuadre general verificado. La
descripción «regresión logística + demanda sectorial» es correcta según el consenso del
campo, pero conviene poder señalar la sección del artículo. Recomendación: descargar el PDF
por el proxy de la UNAM y anotar la página de la ecuación de la regresión logística y la del
módulo de demanda, o bien apoyar la afirmación 2 en el manual de CLUE-S, que sí es de acceso
libre.

---

## Weng2002

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.1006/jema.2001.0509` devuelve
  `journal-article`, «Land use change analysis in the Zhujiang Delta of China using
  satellite remote sensing, GIS and stochastic modelling», *Journal of Environmental
  Management*, Elsevier, vol. 64, núm. 3, pp. 273–284, 03/2002, autor único Weng, Qihao.
  Confirmado en Europe PMC, que además entrega el resumen completo.
- correcciones: ninguna sustantiva. El `.bib` capitaliza el título en estilo título y el
  original va en minúsculas de oración; irrelevante si el estilo bibliográfico normaliza.

**Afirmaciones:**

1. `cap02_marco_teorico.tex:24` — «Los programas de observación de larga duración, en
   particular la familia Landsat, hicieron posible construir series históricas de cobertura
   del suelo […] y sobre ellas se apoya buena parte de los estudios de cambio de uso de
   suelo \parencite{Weng2002}» — **PARCIAL**.
   Verificado: Weng (2002) es en efecto un estudio de cambio de uso de suelo basado en
   percepción remota satelital. Resumen literal en Europe PMC: «land use change dynamics
   were investigated by the combined use of satellite remote sensing, geographic information
   systems (GIS), and stochastic modelling technologies. The results indicated that there
   has been a notable and uneven urban growth and a tremendous loss in cropland between 1989
   and 1997. […] The further integration of these two technologies with Markov modelling was
   found to be beneficial in describing and analyzing land use change process.»
   El matiz: el resumen **no nombra Landsat** ni ningún sensor. La frase de la tesis cita a
   Weng como ejemplo de estudio de cambio de uso de suelo, lo cual está plenamente
   acreditado, pero encadena la cita a la mención de «la familia Landsat», de modo que un
   lector puede entender que Weng usó Landsat. No se pudo verificar el sensor porque no se
   accedió al cuerpo del artículo. Si se quiere eliminar la ambigüedad, basta con separar las
   dos ideas o citar a Weng explícitamente como «estudio de cambio de uso de suelo en el
   delta del Zhujiang entre 1989 y 1997».
   URL: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%2210.1006/jema.2001.0509%22&resultType=core&format=json`

**Riesgo en examen:** bajo. El uso como ejemplo es legítimo y el resumen lo respalda. Solo
la adyacencia con «Landsat» podría dar pie a una pregunta, y se resuelve con una coma.

---

# Resumen

Referencias revisadas: **11 de 11**. Metadatos verificados contra la publicación real en
todas: 8 por DOI en Crossref, 1 libro en OpenLibrary, 1 informe en la página del editor, 1
artículo sin DOI mediante el PDF completo del repositorio de la UAEMex.

- Metadatos **INCORRECTOS: 1** (`JimenezLopez2018`).
- Metadatos **CORRECTOS: 10**, de los cuales 3 con observaciones cosméticas o de
  completitud (`Batty2005` sin ISBN ni `address`; `Couclelis1985` con el nombre actual de la
  revista en lugar del de 1985; `Montejano2023` con `e1734` en `number` en vez de `pages`).
- Afirmaciones evaluadas: **21**. Correctas **13**, parciales **7**, no verificables **1**,
  incorrectas **0**.

Problemas por orden de gravedad:

1. **`Sante2010` — `cap01_introduccion/cap01_introduccion.tex:117`.** «en la mayoría de los
   casos se validan sobre un único período quinquenal» es una afirmación empírica fuerte que
   **no se pudo verificar** (artículo de pago sin copia abierta; nueve vías de acceso
   agotadas). Sostiene la justificación central de la tesis. Además la frase clasifica dos
   revisiones (`Sante2010`, `Aburas2016`) como «modelos de caja blanca». Acción acordada:
   trasladar el cuantificador a `vanVliet2016`, que sí lo documenta y ya está en el `.bib`, y
   reformular como «los modelos de caja blanca más citados \parencite{Clarke1998} y las
   revisiones que los sistematizan \parencite{Sante2010,Aburas2016}».
2. **`JimenezLopez2018` — `report/tesis/tesis_indice_nuevo/back/referencias.bib`.** Metadato
   incorrecto: `volume = {12}` -> `volume = {10}` y añadir `number = {12}`. La ficha
   correcta, según la cita que sugiere la propia revista, es GeoSIG 10(12), Sección II,
   pp. 1–26, ISSN 1852-8031.
3. **`Sante2010` — `cap03_estado_del_arte/cap03_estado_del_arte.tex:511`.** El cuantificador
   «las decenas de modelos que examinan» no está verificado. Acción: quitar la cifra.
4. **`Verburg2002` — `cap03_estado_del_arte/cap03_estado_del_arte.tex:288`.** «regresión
   logística» y «demanda sectorial de suelo» no figuran en el resumen verificado y no se
   accedió al cuerpo. Descripción canónica y casi con certeza correcta, pero sin folio que
   exhibir.
5. **`JimenezLopez2018` — `cap03_estado_del_arte/cap03_estado_del_arte.tex:472`.** La tabla
   marca R3 como «parcial», mientras el artículo se autodescribe como «software especializado
   de código abierto». La distinción que hace la tesis es legítima y ya está argumentada en la
   línea 233, pero conviene explicitarla en la nota de la tabla.
6. **`Weng2002` — `cap02_marco_teorico/cap02_marco_teorico.tex:24`.** La cita queda adyacente
   a «la familia Landsat» y el resumen de Weng no nombra ningún sensor. Riesgo menor de
   lectura equivocada.

Lo más sólido del lote: las cuatro cifras de `JimenezLopez2018` (256 reglas, regla 192,
Kappa 0,53, Jaccard 0,76, período 2003–2017, tres ciudades) están verificadas una por una
contra el PDF original, y la crítica metodológica de `cap06:244` —que el mapa de 2017 se usa
a la vez para seleccionar la regla y para medir el ajuste— está respaldada por una cita
textual del propio artículo. Es el pasaje mejor blindado del lote.
