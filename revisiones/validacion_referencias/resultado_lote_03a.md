# Validación de fuentes — Lote 3a

Fecha: 18 de septiembre de 2026. Seis referencias, trece afirmaciones.

Aviso previo: varias líneas del archivo de entrada estaban desactualizadas porque el `.tex`
se editó mientras corría la validación. En el texto vigente, `cap01:143` → `cap01:150`,
`cap03:18` → `cap03:27`, `cap03:217` → `cap03:227`, `cap03:511` → `cap03:515`. La cita de
`cap01:16` que aparecía en el lote (con las erratas «urbanizacipon» y la coletilla «en
particular esto se observa en México») ya no existe: esa línea fue reescrita antes de que el
validador terminara. La validación se hizo contra el texto vigente.

---

## Aburas2016

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.1016/j.jag.2016.07.007` devuelve
  `journal-article`; título «The simulation and prediction of spatio-temporal urban growth
  trends using cellular automata models: A review»; autores Maher Milad Aburas, Yuek Ming Ho,
  Mohammad Firuz Ramli, Zulfa Hanan Ash'aari (en ese orden); *International Journal of
  Applied Earth Observation and Geoinformation*; vol. 52; pp. 380-389; octubre de 2016;
  Elsevier BV.
- correcciones: ninguna. Los ocho campos del `.bib` coinciden campo por campo.
- nota de acceso: el artículo es de acceso cerrado. Unpaywall reporta `"oa_status":
  "closed"`, `"has_repository_copy": false`, `"oa_locations": []`. No existe copia legítima
  de texto completo. Sólo se pudo leer el resumen íntegro en ADS.

**Afirmaciones:**

1. `cap01_introduccion.tex:117` — **PARCIAL**, con un problema de categoría.

   Aburas et al. (2016) **no es un modelo**: es una revisión de literatura. El propio resumen
   lo dice: «The aim of this paper is to provide a basis for a literature review of urban
   Cellular Automata (CA) models to find the most suitable approach for a realistic
   simulation of land use changes»
   (`http://ui.adsabs.harvard.edu/abs/2016IJAEO..52..380A/abstract`). Agruparlo bajo «los
   modelos de caja blanca más citados» junto con Clarke1998 mezcla un modelo con dos
   revisiones (Sante2010 también lo es). Además, nada en el resumen respalda que los modelos
   revisados «se validen sobre un único período quinquenal»; las debilidades que menciona son
   otras: «weaknesses in the quantitative aspect, and the inability to include the driving
   forces of urban growth in the simulation process».

2. `cap03_estado_del_arte.tex:515` — **NO VERIFICABLE**

   Se intentó: (a) Crossref, sin resumen; (b) Semantic Scholar, que devuelve `"abstract":
   null` con el aviso de que el editor elidió el campo; (c) Unpaywall, `closed`, sin
   repositorio; (d) el resumen completo en ADS. El resumen confirma que es una revisión y que
   clasifica técnicas de diseño de modelos, pero **no menciona calibración ni validación**
   como ejes del análisis, ni da un conteo de modelos examinados.

**Riesgo en examen:** medio-alto en `cap01:117`. Un sinodal que conozca el artículo objetará
que se cita una revisión como si fuera un modelo.

---

## Angel2012

**Metadatos:** CORRECTO

- evidencia: `https://www.lincolninst.edu/publications/books/planet-cities/` — «Planet of
  Cities. Shlomo Angel. September 2012, English. Lincoln Institute of Land Policy».
  OpenLibrary confirma autor único Shlomo Angel, `first_publish_year: 2012`, editorial
  Lincoln Institute of Land Policy, ISBN 9781558442450 / 1558442456, 343 páginas.
- correcciones: ninguna. Sugerencia opcional: añadir `isbn = {9781558442450}` y
  `address = {Cambridge, MA}`.

**Afirmaciones:**

1. `cap01_introduccion.tex:16` — **PARCIAL** por el lugar donde se pudo verificar, no por el
   contenido.

   La afirmación es exactamente el hallazgo central del programa de investigación de Angel, y
   se verificó textualmente en el informe compañero del mismo autor y la misma editorial: «On
   average, the annual growth rate of urban land cover was twice that of the urban population
   between 1990 and 2000» — Shlomo Angel, con Jason Parent, Daniel L. Civco y Alejandro M.
   Blei, *Making Room for a Planet of Cities*, Policy Focus Report, Lincoln Institute of Land
   Policy, 2011, ISBN 978-1-55844-212-2, resumen ejecutivo
   (`https://smartnet.niua.org/sites/default/files/resources/Making%20Room%20for%20a%20Planet%20of%20Cities.pdf`).
   El mismo documento cuantifica el mecanismo: las densidades del área construida cayeron a
   una tasa media anual de 2,01 % entre 1990 y 2000, de 144 a 112 personas por hectárea, y
   descendieron en 75 de 88 ciudades de países en desarrollo y en las 32 de países
   desarrollados.

   El matiz: ese texto es *Making Room for a Planet of Cities* (2011, cuatro autores), no
   *Planet of Cities* (2012, autor único). Son publicaciones distintas de la misma
   institución y del mismo programa. No se consiguió texto completo del libro de 2012.

   Nota adicional: la coletilla «en particular esto se observa en México» que aparecía en el
   archivo de lote **ya no está** en el `.tex`. Bien, porque Angel trabaja con una muestra
   global de 120 ciudades y no singulariza a México.

**Riesgo en examen:** bajo. La atribución es correcta en sustancia.

---

## Clarke1998

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.1080/136588198241617` — `journal-article`;
  «Loose-coupling a cellular automaton model and GIS: long-term urban growth prediction for
  San Francisco and Washington/Baltimore»; Keith C. Clarke y Leonard J. Gaydos, en ese orden;
  *International Journal of Geographical Information Science*; vol. 12, núm. 7, pp. 699-714;
  noviembre de 1998. Confirmado de forma independiente por PubMed (PMID 12294536).
- correcciones: ninguna. Sólo capitalización del título; es cosmético.
- nota de acceso: el texto completo está tras el muro de Taylor & Francis. Se verificó contra
  el resumen íntegro en PubMed y contra dos fuentes secundarias fiables.

**Afirmaciones:**

1. `cap01_introduccion.tex:117` — **INCORRECTA**. Es el hallazgo más grave del lote.

   Clarke y Gaydos calibran sobre una serie histórica larga con múltiples fechas de control,
   no sobre un quinquenio. El resumen lo dice: «Prior research developed a cellular automaton
   model, that was calibrated by using historical digital maps of urban areas and can be used
   to predict the future extent of an urban area… This paper presents the calibration and
   prediction results for both regions» (`https://pubmed.ncbi.nlm.nih.gov/12294536/`). Y el
   propio Clarke describe después el procedimiento: «the performance of a parameter set
   creating feasible system behavior may be controlled by having actual data on urban areas
   for 1965, 1980, 1990 and 2006»
   (`http://www.ncgia.ucsb.edu/projects/gig/Pub/SLEUTHPapers_Nov24/Clarke_Lincoln2008.pdf`,
   p. 4). Son cuatro fechas de control repartidas en cuatro décadas. Lo contrario de un único
   período quinquenal.

2. `cap01_introduccion.tex:150` — **NO VERIFICABLE** en lo que toca a Clarke1998.

   El resumen del artículo no aborda ni el contraste con modelos basados en agentes ni la
   interpretabilidad. La afirmación es razonable y probablemente la sostengan los co-citados
   Batty2005 y Sante2010.

3. `cap02_marco_teorico.tex:91` — **CORRECTA**, con respaldo secundario.

   SLEUTH es efectivamente estocástico. Clarke (2008) describe parámetros «that control the
   random likelihood of any pixel turning urban (dispersion)» y una calibración basada en
   «Monte Carlo iterations». La revisión independiente del modelo lo confirma: «The outer
   loop executes Monte Carlo iterations and the inner loop executes the growth rules»
   (`https://ijerr.gau.ac.ir/article_1688_7eecf8813c80d3ca017280879c5a6360.pdf`).

4. `cap02_marco_teorico.tex:93` — **PARCIAL**. Tres matices, ninguno fatal.

   (a) **El acrónimo SLEUTH no aparece en Clarke y Gaydos (1998)**; se consolidó con Silva y
   Clarke (2002), como la propia tesis reconoce en `cap03:27`. Desplegar el acrónimo con
   `\parencite{Clarke1998}` es anacrónico y contradice, a dos capítulos de distancia, lo que
   la tesis afirma en el estado del arte.

   (b) **Los cinco coeficientes se atribuyen habitualmente a Clarke, Hoppen y Gaydos (1997)**,
   no a 1998. La revisión del modelo los lista y los referencia a «(Clarke et al., 1997;
   Sietchiping, 2004; Gazulis and Clarke, 2006)», mientras que a Clarke y Gaydos (1998) le
   atribuye los cuatro *tipos* de crecimiento: «The four growth types that determine the
   probability of a cell becoming urbanized are termed: diffusive growth, new spreading
   center, organic growth and road influenced growth (Clarke and Gaydos, 1998)».

   (c) «Es el más difundido» no lo sostiene el artículo de 1998 sino la literatura posterior;
   la tesis ya lo respalda bien en `cap03:28` con `\textcite{Clarke2008}`.

5. `cap03_estado_del_arte.tex:27` — **CORRECTA**

   El título del artículo nombra las dos regiones, y el resumen lo detalla: «The model has now
   been applied to two rapidly growing, but remarkably different urban areas: the San
   Francisco Bay region in California and the Washington/Baltimore corridor in the Eastern
   United States». La cautela de la tesis al escribir «el modelo que hoy se conoce como
   SLEUTH» es exactamente la correcta, y es la que falta en `cap02:93`.

6. `cap03_estado_del_arte.tex:227` — **PARCIAL**: el hecho es cierto y verificado, pero la
   cita señala el lugar equivocado.

   El hecho está confirmado en la fuente primaria del código, Project Gigalopolis
   (NCGIA/UCSB): «SLEUTH is a tightly coupled, modified cellular automaton model of urban and
   other land class change… All C language code and libraries, and a sample data set,
   demo_city, is contained within the compressed SLEUTH3.0beta file»
   (`http://www.ncgia.ucsb.edu/projects/gig/Dnload/download.htm`). Clarke (2008) lo dice de
   forma citable: «the model has been open source since its outset, and a complete set of
   source code and test data can be downloaded from the website. Recent notable contributions
   have allowed the model to run under Linux and cygwin, a Windows-based UNIX emulator»
   (p. 2).

   El problema: **un artículo de 1998 en IJGIS no publica código fuente**. El código se
   distribuye por Project Gigalopolis, financiado por USGS/NSF. Citar `Clarke1998` para
   respaldar la disponibilidad del código es un error de locus.

**Riesgo en examen:** alto en `cap01:117`, medio en `cap02:93` y `cap03:227`, bajo en el resto.

---

## CortesVapnik1995

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.1007/BF00994018` — `journal-article`;
  «Support-vector networks»; Corinna Cortes y Vladimir Vapnik, en ese orden; *Machine
  Learning*; vol. 20, núm. 3, pp. 273-297; septiembre de 1995; Springer.
- correcciones: ninguna. Coinciden los nueve campos.

**Afirmaciones:**

1. `cap02_marco_teorico.tex:65` — **CORRECTA**

   Se leyó el PDF del artículo en `https://homepages.math.uic.edu/~lreyzin/papers/cortes95.pdf`.
   Las tres piezas de la afirmación están ahí:

   — Clasificador supervisado binario: «The support-vector network is a new learning machine
   for **two-group classification problems**» (resumen).

   — Hiperplano de margen máximo: «An **optimal hyperplane** is here defined as the linear
   decision function with **maximal margin** between the vectors of the two classes» (§1).

   — Distancia a las observaciones más próximas de cada lado: es literalmente la ecuación (12)
   del artículo, cuyo óptimo es 2/|w₀|.

   La frase siguiente de la tesis, sobre el margen blando con parámetro de regularización,
   también está respaldada: «We here extend this result to **non-separable** training data»
   (resumen) y §3, «Soft margin hyperplane».

**Riesgo en examen:** nulo. Es la afirmación mejor sustentada del lote.

---

## GonzalezWoods2018

**Metadatos:** CORRECTO

- evidencia: página de créditos del propio libro: «Authorized adaptation from the United
  States edition, entitled *Digital Image Processing, Fourth Edition*, ISBN
  978-0-13-335672-4, by Rafael C. Gonzalez and Richard E. Woods, published by Pearson
  Education © 2018». Corroborado por la ficha del editor
  (`https://www.pearson.com/en-us/subject-catalog/p/digital-image-processing/P200000003224/9780133356724`).
- correcciones: ninguna sustantiva. Un detalle menor y opcional: `address = {New York}` no se
  pudo confirmar en ningún colofón; el sello estadounidense ha usado Upper Saddle River y
  Hoboken (NJ) según la época.

**Afirmaciones:**

1. `cap02_marco_teorico.tex:34`, sobre RGB — **CORRECTA**

   Sección 6.2, «The RGB Color Model»: «In the RGB model, each color appears in its primary
   spectral components of red, green, and blue… Images represented in the RGB color model
   consist of three component images, one for each primary color… each RGB color pixel [that
   is, a **triplet of values (R, G, B)**] has a depth of 24 bits». La palabra «terna»
   corresponde exactamente a «triplet».

2. `cap02_marco_teorico.tex:34`, sobre CIE L\*a\*b\* y HSV — **PARCIAL**. La primera mitad es
   correcta; la segunda atribuye a Gonzalez y Woods algo que el libro no contiene.

   **CIE L\*a\*b\*: verificado.** Está en §6.2, bajo el epígrafe «A Device Independent Color
   Model»: «The model of choice for many color management systems (CMS) is the **CIE
   $L^*a^*b^*$** model, also called CIELAB (CIE [1978], Robertson [1977])». Y la descripción
   coincide con la de la tesis: «the $L^*a^*b^*$ system is an excellent decoupler of intensity
   (represented by lightness $L^*$) and color (represented by $a^*$ for red minus green and
   $b^*$ for green minus blue)». Matiz terminológico menor: el libro dice *lightness*
   (luminosidad), no *luminance* (luminancia). Son magnitudes distintas: $L^*$ es una función
   no lineal de la luminancia $Y/Y_W$, ecuación (6-31) del libro.

   **HSV: no está en el libro.** Se buscó la cadena «HSV» en el texto completo de los
   capítulos 1 a 6 y aparece **cero veces**. Gonzalez y Woods desarrollan el modelo **HSI
   (hue, saturation, intensity)**, no HSV, con subsecciones «The HSI Color Model», «Converting
   Colors from RGB to HSI», «Converting Colors from HSI to RGB» y «Manipulating HSI Component
   Images». El índice detallado publicado por los autores lo confirma
   (`https://www.imageprocessingplace.com/downloads_V3/dip4e_downloads/dip4e_sample_book_material/dip4e_detailed_TOC.pdf`):
   los modelos que el capítulo cubre son RGB, CMY/CMYK, HSI y CIELAB. No hay HSV.

   HSV y HSI comparten matiz y saturación pero difieren en el tercer canal ($V = \max(R,G,B)$
   frente a $I = (R+G+B)/3$), así que no son intercambiables. Y aquí importa: el pipeline de
   la tesis usa HSV de verdad (`FeatureExtractor` extrae LAB y HSV), de modo que **no** se
   debe «corregir» el texto cambiando HSV por HSI, porque entonces el marco teórico dejaría de
   describir el código. La corrección correcta es dejar HSV y darle su fuente propia: Alvy Ray
   Smith, «Color gamut transform pairs», *ACM SIGGRAPH Computer Graphics* 12(3):12-19, 1978,
   DOI `10.1145/965139.807361`, que es el origen de HSV.

**Riesgo en examen:** medio-bajo, pero es un error concreto y comprobable en treinta segundos
por cualquiera con el libro a mano.

---

## Huacuz2018Metropolizacion

**Metadatos:** CORRECTO

- evidencia: `https://api.crossref.org/works/10.29105/contexto12.16-6` — `journal-article`;
  título bilingüe «El proceso de metropolización en Querétaro 1990-2010» / «The
  Metropolization processes in Queretaro 1990-2010»; autores Rafael de Jesús Huacuz Elías y
  Rubí del Rocío Vázquez Cruz, en ese orden; *CONTEXTO. Revista de la Facultad de Arquitectura
  de la Universidad Autónoma de Nuevo León*; vol. 12, núm. 16; pp. 79-91; 25 de abril de 2018.
- correcciones: ninguna en los campos sustantivos.

**Afirmaciones:**

1. `cap05_modelo_crecimiento_urbano.tex:44` — **CORRECTA**

   El resumen depositado por el propio editor en Crossref lo sostiene de forma directa:
   «Aplicando el enfoque del análisis territorial el estudio se enfoca al proceso de
   metropolización en la Zona Metropolitana de Querétaro, la cual ha destacado por lo acelerado
   que se ha expandido en su superficie de urbanización a través del tiempo. […] la
   investigación muestra la manera en que un territorio de escala metropolitana se va
   cubriendo de urbanizaciones y como puede explorarse comparativamente entre las décadas
   1990-2010».

   Coinciden los tres elementos que la tesis necesita. Único matiz: la ventana del artículo es
   1990-2010, así que respalda «bien documentada» pero no cubre la década 2010-2020 del
   período de la tesis.

**Riesgo en examen:** bajo. La cita es pertinente, local y bien colocada.

---

# Resumen

Revisadas **6 referencias** y **13 afirmaciones**.

**Metadatos: 6 de 6 correctos.** Ningún campo bibliográfico requiere corrección.

**Afirmaciones: 5 correctas, 5 parciales, 1 incorrecta, 2 no verificables.**

Problemas graves, por orden de gravedad:

1. **`Clarke1998` — `cap01:117`.** La tesis afirmaba que estos modelos «en la mayoría de los
   casos se validan sobre un único período quinquenal». Falso para Clarke y Gaydos (1998):
   calibran con mapas históricos sobre fechas de control repartidas en décadas. Es la única
   afirmación *incorrecta* del lote y toca el argumento central de la tesis.
2. **`Aburas2016` — mismo `cap01:117`.** Se le llama «modelo de caja blanca» a una revisión de
   literatura; lo mismo ocurre con `Sante2010`. Error de categoría.
3. **`Clarke1998` — `cap03:227`.** El código C de SLEUTH existe y es público, pero lo publica
   Project Gigalopolis (NCGIA/UCSB), no el artículo de IJGIS de 1998.
4. **`Clarke1998` — `cap02:93`.** El acrónimo SLEUTH no aparece en el artículo de 1998, y los
   cinco coeficientes se atribuyen en la literatura a Clarke, Hoppen y Gaydos (1997).
5. **`GonzalezWoods2018` — `cap02:34`.** HSV no aparece ni una vez en el libro; Gonzalez y
   Woods desarrollan HSI. Como el pipeline sí usa HSV, no cambiar HSV por HSI: darle su cita
   propia (Smith, 1978). Además, «luminancia $L^*$» → «luminosidad $L^*$».
6. **`Aburas2016` — `cap03:515`.** Artículo de acceso cerrado sin copia en repositorio; no se
   pudo confirmar que documente heterogeneidad de calibración y validación.
