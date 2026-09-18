# Resultado — Lote 3b (5 referencias)

Validado el 18 de septiembre de 2026. Nota: los números de línea del archivo de entrada
ya no coinciden con el estado actual de los `.tex`; se indican los vigentes.

---

## MacQueen1967

**Metadatos:** CORRECTO

- evidencia: https://projecteuclid.org/euclid.bsmsp/1200512992 — «Some methods for
  classification and analysis of multivariate observations. Chapter Author(s) J. MacQueen.
  Editor(s) Lucien M. Le Cam, Jerzy Neyman. Berkeley Symp. on Math. Statist. and Prob.,
  1967: 281-297 (1967) … Proceedings of the Fifth Berkeley Symposium on Mathematical
  Statistics and Probability, Volume 1: Statistics … University of California Press».
  Texto completo verificado en
  https://digitalassets.lib.berkeley.edu/math/ucb/text/math_s5_v1_article-17.pdf
- correcciones: ninguna obligatoria. La firma del artículo es «J. MacQueen»; el `.bib` usa
  «MacQueen, James B.», que es su nombre completo y es la forma habitual en la literatura.
  El volumen 1 lleva subtítulo «Statistics», omitido en el `.bib`; es opcional.

**Afirmaciones:**

1. `cap02_marco_teorico/cap02_marco_teorico.tex:61` — **PARCIAL** (es el problema grave del lote).

   Lo que sí sostiene la fuente:
   - Que MacQueen propone y bautiza el proceso: «The process, which is called 'k-means',
     appears to give partitions which are reasonably efficient in the sense of within-class
     variance» (p. 281).
   - Que el criterio es la suma de distancias cuadradas al centroide del grupo: define
     `W²(S)` como la varianza intraclase respecto de la media condicional `uᵢ` de cada
     conjunto `Sᵢ`, y «Thus the problem is to locate a partition minimizing W²(S)» (p. 282).
   - Que el resultado no es óptimo garantizado: «In general, the k-means procedure will not
     converge to an optimal partition, although there are special cases where it will» (p. 282).

   Lo que **no** sostiene la fuente: el procedimiento que la tesis describe.
   La tesis dice «Procede de forma iterativa: asigna cada observación al centroide más
   próximo, recalcula los centroides como la media de los puntos asignados y repite hasta
   que las asignaciones se estabilizan». Eso es el algoritmo por lotes (Lloyd/Forgy), y
   MacQueen lo acredita explícitamente a otras personas, no a sí mismo. En su §3.6,
   titulada «A two-step improvement procedure» (p. 294):

   > «The method of obtaining partitions with low within-class variance which was suggested
   > by Forgy and Jennrich (see section 1.1) works as follows. Starting with an arbitrary
   > partition into k sets, the means of the points in each set are first computed. Then a
   > new partition of the points is formed by the rule of putting the points into groups on
   > the basis of nearness to the first set of means. […] Computationally, the two steps of
   > the method are (1) compute the means of the points in each set in the initial partition
   > and (2) reclassify the points on the basis of nearness to these means, thus forming a
   > new partition. This can be iterated…»

   Y en la introducción (p. 282):

   > «Also, a simple and elegant method which would appear to yield partitions with low
   > within-class variance, was noticed by Edward Forgy and Robert Jennrich, independently
   > of one another, and communicated to the writer sometime in 1963.»

   El procedimiento propio de MacQueen es secuencial, de una sola pasada, y en línea (p. 283):

   > «Stated informally, the k-means procedure consists of simply starting with k groups each
   > of which consists of a single random point, and thereafter adding each new point to the
   > group whose mean the new point is nearest. After a point is added to a group, the mean
   > of that group is adjusted in order to take account of the new point.»

   URL: https://digitalassets.lib.berkeley.edu/math/ucb/text/math_s5_v1_article-17.pdf

   El desajuste no es sólo de la prosa: el `Algoritmo~\ref{alg:kmeans}` del
   `apendice_arquitectura/apendice_arquitectura.tex:88--106` es literalmente el esquema de
   dos pasos de Forgy/Jennrich con `n_init` reinicios, y la tesis lo presenta como «el
   procedimiento completo» de lo propuesto por MacQueen.

   Requiere añadir `Lloyd1982` (o `Forgy1965`) a `referencias.bib`.

   Matiz adicional, menor: «minimizando» es más fuerte de lo que MacQueen afirma de su
   propio proceso («tends to be low»). Describe bien el objetivo del método por lotes, así
   que con la corrección anterior deja de ser un problema.

**Riesgo en examen:** alto si el sinodal es de ciencias de la computación. La confusión
MacQueen/Lloyd es de manual y está en un capítulo teórico, junto a un algoritmo formal en
el apéndice que hace visible la discrepancia. Es barato de arreglar y caro de defender.

---

## RamirezHernandez2021

**Metadatos:** CORRECTO

- evidencia: Crossref `10.22201/iiec.9786073044349e.2021` devuelve título, autor Roberto
  Ramírez Hernández, año 2021, editor «Universidad Nacional Autónoma de México, Instituto de
  Investigaciones Económicas, CoHu, PUEC», ISBN 9786073044349. Ficha editorial en
  https://libros.iiec.unam.mx/roberto-ramirez_zona-metropolitana-ciudad-mexico_crecimiento-expansion-2040
  — «Año de edición (versión impresa) 2020 / Año de edición (versión digital) 2021 /
  ISBN (impreso) 978-607-30-3533-0 / ISBN (pdf) 978-607-30-4434-9 / Páginas(impreso) 312».
- correcciones: ninguna necesaria. El año 2021 corresponde a la edición digital (la impresa
  es 2020) y es el correcto para el DOI citado. Opcionalmente podría añadirse
  `isbn = {978-607-30-4434-9}` y `pages = {312}`.

**Afirmaciones:**

1. `cap01_introduccion/cap01_introduccion.tex:119` — **PARCIAL / INCONSISTENTE con la propia
   tesis**, en lo que toca a esta referencia.

   - R2 (mapas espacialmente explícitos para planeación): CORRECTO. El libro produce
     prospectiva territorial cartografiada al 2040.
   - R3 (sin software propietario): CORRECTO que no lo cumple, y el libro lo dice sin
     ambigüedad: «Los parámetros del modelo […] se estimaron mediante ejercicios de regresión
     logística binomial y multinomial en el programa spss v. 18», y el epígrafe
     «Programación del modelo en lenguaje sas». Ambos son software propietario.
   - R1 (simulación histórica evaluada con FoM, Kappa e IoU en ventanas desplazadas):
     **el libro no la cumple ni parcialmente**. Buscando en el texto completo de las 310
     páginas no aparece «kappa», ni figura de mérito, ni IoU, ni ninguna comparación
     simulado-observado; la única mención de validación es genérica y metodológica
     («h) Validación de resultados y calibración de parámetros»).

   Es decir: la tabla del cap. 3 marca `R1 = no` para esta referencia, lo cual está bien
   sustentado, pero la frase del cap. 1 dice «satisfacen parcialmente R1». Las dos
   afirmaciones no pueden ser ciertas a la vez. La frase del cap. 1 sólo es defendible para
   `Chihuahua2023`, que la tabla marca `R1 = parcial`.

   URL de respaldo: https://libros.iiec.unam.mx/media/76/download?attachment (PDF completo
   del libro).

2. `cap03_estado_del_arte/cap03_estado_del_arte.tex:408` — **CORRECTA**, y casi literal.

   El libro se autoclasifica exactamente así (cap. IV, p. ~308 del PDF):

   > «De acuerdo con el diseño planteado, consiste en un modelo econométrico con simulación
   > espacial basado en una simulación de Montecarlo, aplicación de autómatas celulares y
   > mecanismos de transición probabilística mediante modelos Logit».

   Cada elemento que la tesis enumera está respaldado:
   - «celdas territoriales»: «cada unidad territorial de observación se denominará celda
     territorial o ct».
   - «vecindad de Moore»: «El tipo de vecindad es conocida como vecindad de Moore, misma que
     involucra las ocho celdas contiguas a ctᵢⱼᵗ y sus respectivos usos de suelo».
   - «regresiones logísticas binomial y multinomial»: «se estimaron mediante ejercicios de
     regresión logística binomial y multinomial en el programa spss v. 18».
   - «rutina de Monte Carlo»: «es necesario incorporar un mecanismo de aleatoriedad para
     evaluar los parámetros estimados […] se generan para cada ciclo poblaciones en celdas al
     azar mediante números aleatorios con distribución de Poisson».

   Queda confirmado también el punto que planteaba el encargo: **no es un modelo inspirado
   en SLEUTH**. SLEUTH aparece en el libro sólo en la revisión de literatura, y siempre como
   trabajo ajeno: «Tal es el caso de modelos muy consolidados como el SLEUTH, desarrollado
   por Keith C. Clarke…» y «Posiblemente el único antecedente directo de un modelo de
   expansión urbana basado en autómatas celulares para una ciudad mexicana sea el de
   Márquez [2008], quien propone el uso del conocido modelo SLEUTH para Ciudad Juárez».

   URL: https://libros.iiec.unam.mx/media/76/download?attachment

3. `cap03_estado_del_arte/cap03_estado_del_arte.tex:481` (fila `R1=no, R2=✓, R3=no,
   R4=parcial`) — **CORRECTA con una reserva en R4.**

   R1, R2 y R3 quedan verificados arriba. La reserva es sobre `R4 = parcial`: el criterio R4
   de la tesis exige que los pesos sean «legibles como evidencia espacial cuantificable, no
   parámetros opacos de una red neuronal». El libro publica las tablas completas de
   coeficientes logit estimados, que son plenamente auditables e interpretables como razones
   de momios, aunque no como pesos de evidencia espacial. «Parcial» es defendible bajo la
   lectura estricta del criterio, pero conviene tener el argumento preparado: un sinodal
   puede objetar que una regresión logística es tan interpretable como el WoE.

**Riesgo en examen:** medio. La caracterización del modelo (afirmación 2) es impecable y
citable palabra por palabra. El riesgo está en la contradicción entre el cap. 1 y la tabla
del cap. 3 sobre R1, que es exactamente el tipo de detalle que un sinodal detecta al leer
los dos capítulos seguidos.

---

## Silva2002

**Metadatos:** CORRECTO

- evidencia: Crossref `10.1016/S0198-9715(01)00014-X` — «Calibration of the SLEUTH urban
  growth model for Lisbon and Porto, Portugal», E.A. Silva y K.C. Clarke, *Computers,
  Environment and Urban Systems*, vol. 26, núm. 6, pp. 525-552, noviembre 2002. Confirmado en
  https://www.sciencedirect.com/science/article/abs/pii/S019897150100014X
- correcciones: ninguna.

**Afirmaciones:**

1. `cap03_estado_del_arte/cap03_estado_del_arte.tex:27--28` («el acrónimo se consolidó con la
   aplicación a Lisboa y Oporto») — **CORRECTA**.

   El artículo introduce el nombre SLEUTH señalando expresamente que sustituye al anterior:

   > «The SLEUTH model (slope, landuse, exclusion, urban extent, transportation and
   > hillshade), formerly called the Clarke Cellular Automaton Urban Growth Model, was
   > developed for and tested on various cities in North America…»

   y

   > «This paper focuses on calibrating the SLEUTH model, formerly the Clarke Cellular
   > Automaton Urban Growth Model (Clarke & Gaydos, 1998; Clarke, Hoppen, & Gaydos, 1997) for
   > two Portuguese metropolitan areas. SLEUTH is an acronym for the input layers that the
   > model uses in gridded map form: Slope, Land Use, Exclusion, Urban Extent, Transportation
   > and Hillshade.»

   El artículo refuerza además el carácter de hito de la aplicación: «Application of the
   model to Lisbon and Porto in Portugal is the first application to European cities, and
   indeed the first major application outside of the United States.»

2. `cap03_estado_del_arte/cap03_estado_del_arte.tex:86--90` (fuerza bruta sobre los cinco
   coeficientes, cada uno en el intervalo entero [0,100], del orden de 10¹⁰ combinaciones,
   cada una con simulaciones Monte Carlo) — **PARCIAL**, correcta en sustancia.

   Lo que se verifica directamente en Silva y Clarke (2002):
   - Los cinco coeficientes: «Diffusion, breed, spread, slope and road coefficients control
     the behavior of the cellular automaton».
   - La exploración exhaustiva y multietapa del espacio de parámetros: el resumen enuncia
     «sequential multistage optimization by automated exploration of model parameter space,
     the problem of equifinality, and parameter sensitivity to local conditions».
   - El volumen de corridas: «Results from the three phases of the calibration mode (Coarse,
     Fine, and Final calibrations) are presented in Table 1, Table 2, Table 3. Each table
     presents the sorted top five highest scoring results from thousands of model runs.»

   Lo que **no** se pudo leer en el propio artículo: la acotación explícita al intervalo
   entero [0,100] ni la palabra «fuerza bruta». Ambos datos sí están documentados en la
   literatura primaria de SLEUTH que la tesis cocita:
   - Jantz, Goetz y Shelley (2003), *Environment and Planning B*: «this was achieved in the
     SLEUTH modeling environment through a brute-force Monte Carlo method […] calibration was
     performed in three phases: coarse, medium, and fine. For the coarse calibration, the
     maximum parameter value range (1–100) was used».
     URL: https://www.woodwellclimate.org/wp-content/uploads/2015/09/JantzEnvPlanB.03.pdf
   - Guan y Clarke, *Getting started with pSLEUTH*: «In a comprehensive calibration, every
     possible combination of the five parameter values has to be evaluated with multiple
     Monte Carlo seeds. If 100 Monte Carlo iterations were applied, a comprehensive
     calibration over a 20-year period would consist of 101⁵ × 100 × 20 growth cycles. […]
     The current SLEUTH model uses Brute Force method to seek the best-fit combination».
     URL: https://digitalcommons.unl.edu/natrespapers/218

   101⁵ ≈ 1,05 × 10¹⁰, así que el orden de magnitud 10¹⁰ que declara la tesis es correcto.
   Recomendación: como la cifra concreta proviene de la documentación del modelo y no del
   artículo de Lisboa y Oporto, conviene que el `\parencite` incluya también una fuente que
   enuncie el rango.

**Hallazgo colateral, relevante para este lote.** En `cap03:90--92` la tesis escribe: «El
procedimiento habitual estrecha la búsqueda en tres fases sucesivas (gruesa, fina y final),
esquema planteado ya en la formulación original del autómata auto-modificable
\parencite{ClarkeHoppen1997}». La atribución es probablemente incorrecta: el trabajo donde
las tres fases aparecen nombradas como *Coarse, Fine y Final* es precisamente Silva y Clarke
(2002), y es a ese artículo al que Dietzel y Clarke (2007) acreditan el esquema. Sugerencia:
`Silva2002` como origen del esquema en tres fases, reservando `ClarkeHoppen1997` para la
automodificación, que es lo que ese artículo sí formula.

**Riesgo en examen:** bajo para las dos afirmaciones numeradas. Medio para la atribución
colateral de las tres fases a Clarke y Hoppen (1997), que es fácil de rebatir con Dietzel y
Clarke (2007) en la mano.

---

## SoaresFilho2004

**Metadatos:** CORRECTO

- evidencia: Crossref `10.1111/j.1529-8817.2003.00769.x` — «Simulating the response of
  land-cover changes to road paving and governance along a major Amazon highway: the
  Santarém–Cuiabá corridor», *Global Change Biology*, vol. 10, núm. 5, pp. 745-764, 2004,
  Wiley. Los ocho autores y su orden coinciden exactamente con el `.bib`.
- correcciones: ninguna sustantiva. Crossref registra «Sérgio Rivero» con acento, frente a
  «Sergio Rivero» en el `.bib`; irrelevante para la compilación. Conviene saber que algunas
  fuentes secundarias citan «v. 10, n. 7»; el número correcto según el editor es el 5.

**Afirmaciones:**

1. `cap03_estado_del_arte/cap03_estado_del_arte.tex:270--276` (en SoaresFilho2004 «quedó
   incorporada al uso corriente del marco» la ruta de Pesos de Evidencia) — **CORRECTA**, con
   la salvedad de que la verificación es indirecta.

   No se pudo abrir el texto completo: Wiley devuelve 403 y Unpaywall no ofrece otra
   localización. La confirmación viene de fuentes del propio grupo autor:

   - Soares-Filho y colaboradores, en un artículo de congreso del mismo año (SBSR 2004,
     INPE), describiendo su propio trabajo de 2004: «Soares-Filho et al. (2004) adaptaram o
     método de Pesos de Evidência (Goodacre et al., 1993) para calcular relações empíricas
     entre variáveis espaciais e mudanças de uso do solo. […] Sua única suposição refere-se à
     independência entre as variáveis espaciais, o que pode ser testado usando-se o teste do
     coeficiente de contingência (Bonham-Carter, 1994)».
     URL: http://marte.sid.inpe.br/col/ltid.inpe.br/sbsr/2004/10.27.15.15/doc/3357.pdf
   - La documentación oficial de Dinamica EGO, del mismo laboratorio (CSR-UFMG), toma este
     artículo como referencia canónica del método: «The Weights of Evidence method consists
     of a Bayesian approach that calculates the influence of explanatory variables on the
     spatial prediction of a response variable (Bonham-Carter 1994, Soares-Filho et al. 2004)».
     URL: https://csr.ufmg.br/dinamica/dokuwiki/doku.php?id=habitat_suitability_modeling_with_small_sample_size

   Queda además verificada la premisa que la tesis enuncia en la misma frase, la de que la
   arquitectura admite indistintamente regresión logística o Pesos de Evidencia. Lo dice la
   descripción de DINAMICA publicada por el propio grupo en 2002: «DINAMICA can employ two
   methods for calculating the spatial transition probabilities: weights of evidence
   (Goodacre et al., 1993; Bonham-Carter, 1994) and logistic regression (Hosmer and Lemeshow,
   1989)», y el mismo texto asigna cada ruta a su linaje: «the application of either logistic
   regression (Soares-Filho et al., 2001; Soares-Filho et al. 2002a) or weights of evidence
   (Almeida et al., 2002a)».
   URL: http://marte.sid.inpe.br/col/ltid.inpe.br/sbsr/2002/11.16.12.49/doc/06_257.pdf

   Esa última cita respalda de paso, y de forma independiente, los otros dos eslabones de la
   genealogía que la tesis reconstruye.

**Riesgo en examen:** bajo. La afirmación es correcta y la genealogía que traza la tesis está
respaldada por documentos del propio laboratorio que creó DINAMICA.

---

## Wahyudi2016

**Metadatos:** CORRECTO

- evidencia: Crossref `10.14246/irspsd.4.2_60` — Agung Wahyudi y Yan Liu, «Cellular Automata
  for Urban Growth Modelling:» con subtítulo «A Review on Factors defining Transition Rules»,
  *International Review for Spatial Planning and Sustainable Development*, vol. 4, núm. 2,
  pp. 60-75, 2016. PDF de acceso abierto verificado en
  https://www.jstage.jst.go.jp/article/irspsd/4/2/4_60/_pdf
- correcciones: ninguna. Sólo una diferencia de mayúsculas frente al original.

**Afirmaciones:**

1. `cap03_estado_del_arte/cap03_estado_del_arte.tex:511--517` — **CORRECTA** en sus tres
   componentes, y con un grado de coincidencia literal poco común.

   - Las 88 aplicaciones: el diagrama de selección en tres etapas va de 470 artículos a 165 y
     de ahí a 88, y el texto confirma: «The complete list of the 88 selected articles that
     were reviewed in this paper can be obtained from the correspondent author» (p. 63).
   - El período 1993-2012: con cuatro ventanas quinquenales 1993-1997, 1998-2002, 2003-2007
     y 2008-2012 (p. 64).
   - El desequilibrio geográfico, que es prácticamente la misma frase que usa la tesis (p. 67):

     > «Figure 4 visually suggests the imbalanced distribution of CA applications around the
     > world with at least 50 percent of articles having mainly been implemented in cities of
     > USA (North America) and China whilst the other 50 percent was spread across other
     > regions. The high number of CA applications in USA and China was in strike contrast
     > with the implementation of CA in cities of Africa, South America, and the rest of Asia,
     > where it has been scarce.»

   La Tabla 2 del artículo cuantifica los totales: Norteamérica 24 (27 %), China 31 (35 %),
   Europa Occidental 14 (16 %), resto de Asia 12 (14 %), otros 7 (8 %).

   Dos matices menores, ninguno exige corrección. Primero, la tesis dice «América Latina» y
   el artículo dice «South America». Segundo, la categoría «North America» del artículo
   incluye Australia según su propia nota al pie.

   Nota: la distribución que documenta este trabajo es **Estados Unidos y China**, no «China,
   Europa y Estados Unidos». La tesis lo dice bien; era la formulación del encargo la que
   estaba desplazada.

**Riesgo en examen:** nulo. Es la referencia mejor sustentada del lote.

---

# Resumen

Revisadas 5 referencias y 8 afirmaciones.

- Metadatos incorrectos: **0 de 5**.
- Afirmaciones: 5 CORRECTAS, 3 PARCIALES, 0 INCORRECTAS de forma absoluta, 0 no verificables.
- Fuentes no accesibles en texto completo: 2 (Silva2002 y SoaresFilho2004), ambas verificadas
  por fuentes del propio grupo autor. Wahyudi2016, MacQueen1967 y RamirezHernandez2021 se
  leyeron íntegros.

Problemas graves, por orden de prioridad:

1. **MacQueen1967 — `cap02:61` y `apendice_arquitectura.tex:88`.** La tesis atribuye a
   MacQueen el algoritmo iterativo por lotes. MacQueen acredita ese procedimiento a Edward
   Forgy y Robert Jennrich en su §3.6, p. 294; su propio k-means es secuencial y de una sola
   pasada. Corregir la frase y añadir `Lloyd1982`.
2. **RamirezHernandez2021 — `cap01:119` frente a la tabla del cap. 3.** El libro no reporta
   FoM, Kappa, IoU ni comparación simulado-observado alguna, de modo que la tabla es la que
   está bien y la frase del cap. 1 debe restringir el «parcialmente R1» a `Chihuahua2023`.
3. **Silva2002 (colateral) — `cap03:91`.** El esquema de calibración en tres fases se
   atribuye a `ClarkeHoppen1997`. Quien lo presenta con esos nombres es Silva y Clarke
   (2002), y Dietzel y Clarke (2007) se lo acreditan a ellos.
4. **Silva2002 — `cap03:86-89`, menor.** El rango entero [0,100] y las ~10¹⁰ combinaciones
   son correctos, pero proceden de la documentación de SLEUTH, no del artículo de Lisboa y
   Oporto.
