# Resultado de revalidación — lote `rev2_06.md` (11 referencias, 18 afirmaciones)

Fecha de la revisión: 18 de septiembre de 2026.
Alcance: metadatos bibliográficos y afirmaciones atribuidas a terceros en `cap03`, `cap05` y `cap07`.

**Ninguna de las 11 referencias es inexistente.** Las 11 se localizaron en Crossref con metadatos que coinciden
exactamente con la entrada BibTeX (incluido el capítulo de libro sin DOI, `Clarke2008`, verificado contra el PDF
original del propio autor).

Nivel de verificación declarado por referencia:

| Clave | Metadatos | Nivel de acceso al contenido |
|---|---|---|
| `Beven2006` | Crossref, coincidencia exacta | Resumen del editor + texto del preprint del autor (Lancaster eprints) |
| `Brenning2012` | Crossref, coincidencia exacta | Resumen del editor (IEEE, vía índice) |
| `Clarke2008` | Verificado contra el PDF del capítulo | **PDF completo abierto** (sitio Gigalopolis/NCGIA-UCSB) |
| `Couclelis1985` | Crossref, coincidencia exacta | Resumen del editor (CiteSeerX) |
| `Herold2003` | Crossref, coincidencia exacta | Resumen del editor |
| `JimenezLopez2021` | Crossref, coincidencia exacta | **PDF completo abierto** (Redalyc, 46 pp.) |
| `Legendre1993` | Crossref, coincidencia exacta | Resumen del editor (Wiley/ESA) |
| `Ma2019` | Crossref, coincidencia exacta | **Texto completo** (copia abierta IGES) |
| `SoaresFilho2004` | Crossref, coincidencia exacta | Resumen del editor + **lista de referencias del propio artículo** (Crossref). Texto completo NO accesible (Wiley bloqueado) |
| `Wang2021` | Crossref, coincidencia exacta | **Texto completo** (Europe PMC, PMC8583206) |
| `White1993` | Crossref, coincidencia exacta | Resumen del editor |

---

## `Beven2006`

**Metadatos: CORRECTOS.** Crossref (`10.1016/j.jhydrol.2005.07.007`) devuelve: Beven, Keith;
*A manifesto for the equifinality thesis*; Journal of Hydrology; 2006; vol. 320; núm. 1-2; pp. 18–36.
Coincide en los ocho campos con la entrada BibTeX.

| # | Archivo:línea | Etiqueta |
|---|---|---|
| 1 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:99` | CORRECTA |
| 2 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:156` (línea real 170) | CORRECTA |

**Afirmación 1.** «…argumenta que aceptar la existencia de múltiples parametrizaciones aceptables debe llevar a
reportar el conjunto de soluciones y no un óptimo único».

Sostenida literalmente. Resumen del artículo: *«The argument is made that the potential for multiple acceptable
models as representations of hydrological and other environmental systems (the equifinality thesis) should be
given more serious consideration than hitherto.»* Y en el cuerpo: *«The equifinality thesis is intended to focus
attention on the fact that there are many acceptable representations that cannot be easily rejected and that
should be considered in assessing the uncertainty associated with predictions… Implicit in this usage is the
rejection of the assumption that a single correct representation of the system can be found.»*

Matiz sin consecuencia para el texto: Beven habla de *modelos* aceptables en sentido amplio (estructuras,
entradas, conjuntos de parámetros y errores), no solo de «parametrizaciones». La lectura de la tesis es un
subconjunto legítimo de lo que dice la fuente, no una extensión.

**Afirmación 2.** «Algo análogo ocurre con la calibración por fuerza bruta \parencite{Beven2006,Dietzel2007}: dos
conjuntos de coeficientes que empatan en la métrica de ajuste pueden divergir en la proyección a largo plazo…»

Correcta. Lo que se atribuye tras los dos puntos —conjuntos equifinales que empatan en la métrica y divergen en la
predicción, de donde se sigue el deber de declarar el conjunto y no un único óptimo— es exactamente la tesis de
equifinalidad de Beven. El encuadre «calibración por fuerza bruta» es contexto propio de la tesis y queda cubierto
por la cita conjunta a `Dietzel2007`, que sí trata la calibración de SLEUTH. No hay sobreextensión.

---

## `Brenning2012`

**Metadatos: CORRECTOS.** Crossref (`10.1109/igarss.2012.6352393`) devuelve: Brenning, Alexander; título íntegro
coincidente; *2012 IEEE International Geoscience and Remote Sensing Symposium*; 2012; pp. 5372–5375; IEEE.
Tipo `proceedings-article`, consistente con el `@inproceedings` de la entrada.

| # | Archivo:línea | Etiqueta |
|---|---|---|
| 1 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:349` (línea real 355–358) | CORRECTA |

**Afirmación.** «…la sobreestimación del desempeño que resulta de tratar observaciones espacialmente dependientes
como independientes está documentada en la literatura de validación de modelos espaciales
\parencite{Legendre1993,Brenning2012}».

En lo que toca a `Brenning2012`, sostenida. Resumen del editor: *«the accuracy assessment of such predictive
models in a spatial context needs to account for the presence of spatial autocorrelation in geospatial data by
using spatial cross-validation and bootstrap strategies instead of their now more widely used non-spatial
equivalent.»* Es decir: la fuente trata precisamente de *evaluación de exactitud de modelos predictivos*
espaciales y sostiene que la variante no espacial es inadecuada por la autocorrelación.

Declaración de alcance: verifiqué el resumen del editor, no el texto completo (IEEE cerrado, sin copia abierta).
El resumen no emplea literalmente la palabra «sobreestimación»; la dirección del sesgo (optimista) es la lectura
estándar del trabajo y aparece explícita en la literatura que lo cita, pero no la leí en el propio artículo. Como
la tesis atribuye la afirmación de forma colectiva («está documentada en la literatura…») y no pone la palabra en
boca de Brenning, no hay problema de redacción.

---

## `Clarke2008`

**Metadatos: CORRECTOS Y VERIFICADOS EN LA FUENTE PRIMARIA.** El PDF del propio capítulo, alojado en el sitio del
proyecto Gigalopolis (NCGIA, UC Santa Barbara), abre con la línea de referencia del autor:

> *«Clarke, K.C. (2008) A Decade of Cellular Urban Modeling with SLEUTH: Unresolved Issues and Problems, Ch. 3 in
> Planning Support Systems for Cities and Regions (Ed. Brail, R. K., Lincoln Institute of Land Policy, Cambridge,
> MA, pp 47-60.»*

Coincide campo por campo con la entrada BibTeX: autor, título, `booktitle`, editor, `publisher`, `address`, año,
`chapter = {3}` y `pages = {47--60}`. No hay nada que corregir.

| # | Archivo:línea | Etiqueta |
|---|---|---|
| 1 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:18` (líneas 28–29) | CORRECTA |
| 2 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:87` (líneas 87–90) | CORRECTA |
| 3 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:219` (líneas 231–233) | CORRECTA |

**Afirmación 1 (el conteo de aplicaciones y el periodo).** «Una década más tarde \textcite{Clarke2008}
contabilizaba más de cien aplicaciones documentadas en la década transcurrida.»

Verificada con cita textual, dos veces en el capítulo:

> *«SLEUTH is a mature cellular automaton urban model, now applied to over 100 different cities and regions.»*
> (resumen)

> *«At last count, the number of applications of the model to cities and regions was over one hundred. By any
> measure, this makes SLEUTH a highly successful urban and land use change model.»* (introducción)

El periodo también es correcto: el título del capítulo es literalmente «A Decade of Cellular Urban Modeling with
SLEUTH», y la primera aplicación que el propio capítulo data es la de la bahía de San Francisco publicada en 1998,
de modo que «una década más tarde» y «en la década transcurrida» son exactos. Sin objeción.

**Afirmación 2 (los cinco coeficientes, el intervalo y el orden de magnitud).** «…cinco coeficientes de
crecimiento, cada uno acotado al intervalo entero $[0, 100]$, lo que da del orden de $10^{10}$ combinaciones
posibles \parencite{Silva2002,Clarke2008}, cada una evaluada con simulaciones Monte Carlo de múltiples
iteraciones…»

Verificada. El capítulo dice: *«Five parameters control SLEUTH's behavior entirely, each with a possible integer
value between 0 and 100.»* La aritmética de la tesis es correcta: $101^5 \approx 1{,}05\times10^{10}$, o sea del
orden de $10^{10}$. La cita conjunta a `Silva2002` también está bien puesta, porque el propio capítulo remite a
ese trabajo para la fuerza bruta: *«Successive search using brute force methods (Silva and Clarke, 2002) or
genetic algorithms (Goldstein, 2005) then reveals the "best" parameter set.»* Y el capítulo confirma las
iteraciones Monte Carlo como parte de la calibración.

**Afirmación 3 (Gigalopolis y no el artículo).** «SLEUTH es la excepción: su código en C y los datos de prueba se
distribuyen desde el inicio por el sitio del proyecto Gigalopolis, no a través del artículo que describe el modelo
\parencite{Clarke2008}.»

Verificada en lo sustantivo, con cita textual:

> *«The reformulated model was released on the World-wide Web under project Gigalopolis (Clarke et al. 1997),
> first at New York's Hunter College and then at the University of California, Santa Barbara… With federal
> support, the model has been open source since its outset, and a complete set of source code and test data can be
> downloaded from the website.»*

Esto respalda los tres elementos de la afirmación: (a) código fuente y datos de prueba, (b) desde el inicio
(*«open source since its outset»*), (c) por el sitio del proyecto Gigalopolis y no por el artículo. Nótese además
que el capítulo respalda la matización que sigue en la tesis («su compilación en sistemas actuales exige
ajustes»), al mencionar que las contribuciones recientes permitieron correr el modelo bajo Linux y cygwin.

Observación menor, sin etiqueta de error: el detalle «en C» no aparece de forma literal en el capítulo (que sí
menciona la reescritura del código con llamadas a MPI y la versión 3). Es un dato correcto y documentado en el
sitio del proyecto, pero si se quiere que **todo** lo que va antes del `\parencite` provenga del capítulo, basta
escribir «su código fuente y los datos de prueba». Es opcional.

---

## `Couclelis1985`

**Metadatos: CORRECTOS.** Crossref (`10.1068/a170585`): Couclelis, H.; *Cellular Worlds: A Framework for Modeling
Micro—Macro Dynamics*; Environment and Planning A: Economy and Space; 1985; vol. 17; núm. 5; pp. 585–596.
Coincidencia exacta.

| # | Archivo:línea | Etiqueta |
|---|---|---|
| 1 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:18` (líneas 19–21) | CORRECTA |

**Afirmación.** «…el uso de autómatas celulares para representar fenómenos geográficos fue propuesto por
\textcite{Tobler1979} y formalizado en términos teóricos por \textcite{Couclelis1985}».

La parte atribuida a Couclelis está sostenida por el resumen del editor, que es justamente una declaración de
formalización teórica para uso geográfico:

> *«Despite their obvious spatial interpretation, standard cell-space models are too constrained by their
> background conventions to be useful in realistic geographic applications. In this paper, a generalization of the
> cell-space principle is presented, based on discrete model theory, and then applied to a hypothetical but fairly
> complex problem of individual decision and large-scale urban change.»*

«Generalización del principio de espacio celular basada en teoría de modelos discretos» es exactamente
«formalizado en términos teóricos», y el propio resumen enmarca el problema como la inadecuación de los modelos
estándar para aplicaciones geográficas realistas. El verbo de la tesis es además el prudente: no dice que
Couclelis «propuso» (la prioridad se le asigna a Tobler), sino que «formalizó», que es lo que el artículo hace.

Fuera de alcance de este lote: la atribución a `Tobler1979` en la misma oración corresponde a otro lote.

---

## `Herold2003`

**Metadatos: CORRECTOS.** Crossref (`10.1016/S0034-4257(03)00075-0`): Herold, Martin; Goldstein, Noah C.;
Clarke, Keith C.; *The spatiotemporal form of urban growth: measurement, analysis and modeling*; Remote Sensing of
Environment; 2003; vol. 86; núm. 3; pp. 286–302. Coincidencia exacta.

| # | Archivo:línea | Etiqueta |
|---|---|---|
| 1 | `cap05_modelo_crecimiento_urbano/cap05_modelo_crecimiento_urbano.tex:50` | CORRECTA |

**Afirmación.** «…orientado a obtener una serie temporal homogénea para estudiar la forma espaciotemporal del
crecimiento urbano a partir de percepción remota, en la línea de trabajos que miden esa dinámica desde
imágenes~\parencite{Herold2003}».

Sostenida. Resumen del editor: *«This study explores the combined application of remote sensing, spatial metrics
and spatial modeling to the analysis and modeling of urban growth in Santa Barbara, California. The investigation
is based on a 72-year time series data set compiled from interpreted historical aerial photography and from IKONOS
satellite imagery.»* La atribución es deliberadamente débil («en la línea de trabajos que miden esa dinámica desde
imágenes») y la fuente la sostiene con holgura: serie temporal larga, percepción remota, medición de la forma
espaciotemporal del crecimiento urbano. Confirmación cruzada adicional: el capítulo de `Clarke2008` cita este
trabajo como el que traza la forma urbana con métricas de paisaje.

Observación sin etiqueta de error: la serie de Herold es discontinua y heterogénea (fotografía aérea histórica más
IKONOS), mientras que el adjetivo «homogénea» describe la serie de *esta* tesis, no la de Herold. Tal como está
redactada la oración, el `\parencite` cae sobre «trabajos que miden esa dinámica desde imágenes», así que no hay
atribución cruzada. No requiere cambio.

---

## `JimenezLopez2021`

**Metadatos: CORRECTOS.** Crossref (`10.24201/edu.v36i3.1997`): Jiménez López, Eduardo; Garrocho Rangel, Carlos;
Chávez Soto, Tania; *Autómatas Celulares en Cascada para modelar la expansión urbana con áreas restringidas*;
Estudios Demográficos y Urbanos; 2021; vol. 36; núm. 3; pp. 779–823; El Colegio de México, A.C.
Coincidencia exacta. Texto completo consultado (46 pp., copia abierta en Redalyc).

| # | Archivo:línea | Etiqueta |
|---|---|---|
| 1 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:443` (líneas 448–453) | CORRECTA |
| 2 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:477` (fila de la Tabla `tab:cuadrante-mexico`) | **INCORRECTA** |
| 3 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:493` (nota al pie de la tabla) | CORRECTA |

**Afirmación 1.** «El mismo grupo generalizó después el método bajo el nombre de autómatas celulares en cascada,
incorporando áreas restringidas a la expansión y un filtro de bondad de ajuste con cuatro indicadores (entropía de
Shannon, dimensión fractal, Kappa de Cohen y Jaccard), con prueba empírica en Tijuana, Acapulco y Toluca, y
declarando la disponibilidad de los modelos en código abierto.»

Los cuatro elementos verificados con cita textual del artículo:

- *Cuatro indicadores, en el mismo orden*: «*i. entropía de Shannon, que estima lo compacto o disperso de la
  mancha urbana…; ii. dimensión fractal, que sintetiza el crecimiento y la forma de la mancha urbana…; iii. índice
  de Kappa de Cohen, que mide la similitud entre dos mapas descontando la coincidencia esperada por el azar…; y
  iv. índice de Jaccard…*»
- *Filtro en cascada*: «*Valoramos los resultados del modelo con indicadores de bondad de ajuste global y local
  entre mapas…, articulados en un filtro en cascada diseñado para este trabajo.*»
- *Las tres ciudades*: «*…tres ciudades seleccionadas por sus características contrastantes: las áreas
  metropolitanas de Toluca, Tijuana y Acapulco.*»
- *Áreas restringidas*: «*Los modelos consideran restricciones a la expansión urbana (vialidades, parques, etc.).*»
- *Código abierto*: «*Los módulos que se han implementado en la Estación de Inteligencia Territorial Christaller
  son de código abierto y están disponibles, sin costo, bajo demanda.*»

Sin objeción. El verbo «declarando la disponibilidad» está bien elegido: la tesis reporta lo que los autores
declaran, sin adoptarlo como hecho comprobado.

**Afirmación 2 — INCORRECTA.** La fila de la tabla asigna a `JimenezLopez2021` una palomita (`\checkmark`) en la
columna R3, mientras que a `JimenezLopez2018` le asigna «parcial».

Esto contradice dos cosas a la vez. Primero, la fuente: el artículo declara que los módulos están disponibles
«**bajo demanda**», previo contacto por correo electrónico (`tchavez@cmq.edu.mx`) y consulta del sitio
`christaller.org.mx`; no hay repositorio público. Segundo, y más grave para la coherencia interna del capítulo, la
propia nota al pie de esa misma tabla (líneas 495–499) explica que **la valoración es «parcial»** y por qué:

> «Nota sobre R3. Los trabajos de \textcite{JimenezLopez2018,JimenezLopez2021} describen CHRISTALLER como software
> de código abierto. La valoración «parcial» no cuestiona esa descripción: registra que la distribución se hace
> bajo solicitud al grupo desarrollador y no mediante un repositorio público con historial de versiones, que es lo
> que R3 exige.»

La nota habla en plural de ambos trabajos y justifica «parcial», pero la tabla marca ✓ para el de 2021. Como los
dos artículos describen el mismo software (CHRISTALLER) y el mismo régimen de distribución bajo demanda, no hay
base en la fuente para diferenciarlos en R3.

*Propuesta de corrección (una sola celda de la Tabla `tab:cuadrante-mexico`):*

```latex
\textcite{JimenezLopez2021}     & parcial & \checkmark & parcial    & parcial   \\
```

Es decir, sustituir `\checkmark` por `parcial` en la columna R3 de la fila de `JimenezLopez2021`. Con eso la tabla
queda alineada con su propia nota al pie y con lo que el artículo declara.

**Afirmación 3.** «Los trabajos de \textcite{JimenezLopez2018,JimenezLopez2021} describen CHRISTALLER como
software de código abierto.»

Correcta y textual para el trabajo de 2021 (véase la cita de arriba). La verificación de `JimenezLopez2018`
corresponde a otro lote.

---

## `Legendre1993`

**Metadatos: CORRECTOS.** Crossref (`10.2307/1939924`): Legendre, Pierre; *Spatial Autocorrelation: Trouble or New
Paradigm?*; Ecology; 1993; vol. 74; núm. 6; pp. 1659–1673; Wiley. Coincidencia exacta.

| # | Archivo:línea | Etiqueta |
|---|---|---|
| 1 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:349` (líneas 355–358) | **SOBREEXTENDIDA** |

**Afirmación.** «La formulación original estima la varianza de los pesos a partir del conteo de eventos de
entrenamiento y no incorpora esta estructura espacial \parencite{BonhamCarter1994}; **la sobreestimación del
desempeño** que resulta de tratar observaciones espacialmente dependientes como independientes está documentada en
**la literatura de validación de modelos espaciales** \parencite{Legendre1993,Brenning2012}.»

El problema no es la existencia de la fuente ni su tema, sino el alcance. El resumen del editor dice:

> *«Spatial autocorrelation… presents a problem for statistical testing because autocorrelated data violate the
> assumption of independence of most standard statistical procedures. The paper discusses first how autocorrelation
> in ecological variables can be described and measured… Then, proper statistical testing in the presence of
> autocorrelation is briefly discussed. Finally, ways are presented of explicitly introducing spatial structures
> into ecological models.»*

Legendre (1993) documenta que la dependencia espacial invalida el **supuesto de independencia de las pruebas
estadísticas estándar**, es decir, afecta la **inferencia** (errores estándar subestimados, tamaño muestral
efectivo menor, significancia inflada). No es un trabajo sobre **validación de modelos predictivos** ni documenta
la **sobreestimación del desempeño** en ese sentido; ese contenido lo aporta `Brenning2012`, no Legendre. Es un
artículo de ecología de 1993 anterior al debate sobre validación cruzada espacial.

Lo llamativo es que la propia tesis **ya dice bien la parte que sí corresponde a Legendre**, dos líneas antes:
«los pesos pueden estimarse con una precisión aparente mayor que la real». Esa frase es exactamente Legendre.
Basta repartir las dos citas según lo que cada una sostiene.

*Propuesta de corrección (líneas 353–358):*

```latex
La formulación original estima la varianza de los pesos a partir del
conteo de eventos de entrenamiento y no incorpora esta estructura espacial
\parencite{BonhamCarter1994}: tratar observaciones espacialmente dependientes como
independientes vulnera el supuesto sobre el que descansan los procedimientos
estadísticos estándar y reduce el tamaño muestral efectivo \parencite{Legendre1993},
de modo que el desempeño estimado con un esquema de validación no espacial resulta
optimista frente al que se obtiene con validación cruzada espacial \parencite{Brenning2012}.
```

Así Legendre queda asociado a la violación del supuesto de independencia (lo que dice) y Brenning a la
sobreestimación del desempeño en la validación (lo que dice).

**Observación adicional fuera del lote, misma pareja de citas.** En
`cap02_marco_teorico/cap02_marco_teorico.tex:201` la tesis escribe: «*el método no corrige por sí solo la
autocorrelación espacial, de modo que las celdas vecinas aportan información que no es independiente
\parencite{Legendre1993,Brenning2012}*». Esa formulación **sí** es correcta para Legendre y no requiere cambio. La
menciono solo para que no se «corrija» por arrastre al aplicar el ajuste de cap03.

---

## `Ma2019`

**Metadatos: CORRECTOS.** Crossref (`10.1016/j.isprsjprs.2019.04.015`): Ma, Lei; Liu, Yu; Zhang, Xueliang;
Ye, Yuanxin; Yin, Gaofei; Johnson, Brian Alan; *Deep learning in remote sensing applications: A meta-analysis and
review*; ISPRS Journal of Photogrammetry and Remote Sensing; 2019; vol. 152; pp. 166–177. Coincidencia exacta.
Texto completo consultado (copia abierta del IGES).

| # | Archivo:línea | Etiqueta |
|---|---|---|
| 1 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:191` (líneas 199–202) | CORRECTA |
| 2 | `cap07_conclusiones/cap07_conclusiones.tex:95` | CORRECTA |

**Afirmación 1.** «…la segunda es el aprendizaje profundo aplicado a teledetección, cuya revisión sistemática
\parencite{Ma2019} documenta que la disponibilidad de datos de entrenamiento etiquetados es una de las
dificultades recurrentes del área».

Verificada con cita textual del artículo, en tres pasajes independientes:

> *«A supervised DL model usually requires a large number of training samples. In the field of remote-sensing,
> labeling the observed data to prepare training samples for each LULC class is highly time- and/or
> cost-intensive.»*

> *«The important limitation of DL in image registration is the lack of available public training datasets, which
> should be a future endeavor of the remote sensing community.»*

> *«Unsupervised DL models are also attractive for overcoming the training data limitations… For medium-resolution
> and low-resolution satellite image data, in particular, there is a lack of benchmark datasets… This lack of
> benchmark datasets for some types of images and applications may be restricting the development of DL in these
> areas.»*

El calificativo «recurrente» está bien puesto: el problema reaparece en varias de las tareas que la revisión
examina (registro, clasificación LULC, segmentación), no en una sola. También es correcto llamarla «revisión
sistemática»: el propio resumen la describe como metaanálisis y revisión de más de 200 publicaciones.

**Afirmación 2.** «Los enfoques de aprendizaje profundo en teledetección suelen exigir conjuntos anotados grandes
\parencite{Ma2019} y, en la práctica, hardware gráfico; esta tesis no midió horas de GPU de terceros.»

Correcta, y la redacción está bien construida en el punto delicado. El `\parencite{Ma2019}` está colocado
**inmediatamente después** de «conjuntos anotados grandes», que es lo que la fuente sostiene literalmente
(«*usually requires a large number of training samples*»). La mención al hardware gráfico queda fuera del alcance
de la cita, va marcada como observación práctica («en la práctica») y se acompaña de un descargo explícito («esta
tesis no midió horas de GPU de terceros»). No se le atribuye a Ma et al. ninguna afirmación sobre costo
computacional ni sobre GPU: revisé el texto completo y el artículo no reporta requisitos de hardware gráfico.

Sugerencia opcional, no un error: si se quiere blindar la oración frente a una lectura rápida que extienda la cita
a toda la frase, puede moverse el descargo antes del punto, por ejemplo «…conjuntos anotados grandes
\parencite{Ma2019}; el requerimiento de hardware gráfico es una observación de la práctica común y no un dato
tomado de esa revisión, y esta tesis no midió horas de GPU de terceros».

---

## `SoaresFilho2004`

**Metadatos: CORRECTOS.** Crossref (`10.1111/j.1529-8817.2003.00769.x`): Soares-Filho, Britaldo; Alencar, Ane;
Nepstad, Daniel; Cerqueira, Gustavo; Vera Diaz, Maria del Carmen; Rivero, Sérgio; Solórzano, Luis; Voll, Eliane;
título íntegro coincidente; Global Change Biology; 2004; vol. 10; núm. 5; pp. 745–764. Los ocho autores, en ese
orden, coinciden con la entrada BibTeX.

| # | Archivo:línea | Etiqueta |
|---|---|---|
| 1 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:275` (línea 281) | CORRECTA (verificación indirecta) |

**Afirmación.** «…y en \textcite{SoaresFilho2004} donde quedó incorporada [la ruta WoE] al uso corriente del marco
para escenarios de cambio de cobertura».

**Declaración de alcance, importante:** no pude abrir el texto completo. Wiley responde con una comprobación
anti-bot (Cloudflare) y OpenAlex reporta la copia como *bronze OA* sin réplica en repositorio. Lo que sí verifiqué
es primario y suficiente para sostener la afirmación:

1. **La lista de referencias del propio artículo**, depositada por el editor en Crossref, incluye:
   `{"volume-title": "Geographic Information Systems for Geoscientists: Modelling with GIS", "author":
   "Bonham-Carter G", "year": "1994"}`. Es decir, el artículo de 2004 cita el texto canónico del método de Pesos de
   Evidencia. Es difícil explicar esa referencia si el trabajo no empleara WoE.
2. **El resumen del editor** confirma la arquitectura que la tesis describe: *«A scenario-generating submodel is
   coupled to a landscape dynamics simulator, "DINAMICA", which spatially allocates the land-cover transitions
   using a GIS database»*, con corrida a 30 años en pasos anuales y escenarios de política. Eso respalda tanto
   «escenarios de cambio de cobertura» como la separación cantidad/localización que la tesis explica a
   continuación.
3. **Literatura independiente revisada por pares** que usa DINAMICA cita a Soares-Filho et al. 2002 y 2004
   conjuntamente como la referencia del método de pesos de evidencia dentro de la plataforma.

Con eso, la atribución es sólida. El verbo elegido («quedó incorporada al uso corriente») es además el prudente:
no reclama prioridad ni invención para el trabajo de 2004, que es lo que la genealogía del párrafo asigna
correctamente a `Almeida2003`. Si se quiere una verificación de primera mano antes de la entrega final, basta
abrir el PDF desde la red universitaria y confirmar la sección de métodos.

---

## `Wang2021`

**Metadatos: CORRECTOS.** Crossref (`10.3390/ijerph182111013`): Wang, Renyang; He, Qingsong; Zhang, Lu;
Wang, Huiying; *Coupling Cellular Automata and a Genetic Algorithm to Generate a Vibrant Urban Form—A Case Study of
Wuhan, China*; International Journal of Environmental Research and Public Health; 2021; vol. 18; núm. 21;
art. 11013. Coincidencia exacta. Texto completo consultado vía Europe PMC (PMC8583206).

| # | Archivo:línea | Etiqueta |
|---|---|---|
| 1 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:56` (líneas 60–62) | CORRECTA |
| 2 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:191` (líneas 192–198) | CORRECTA |

**Afirmación 1.** «Años más tarde, \textcite{Wang2021} combinó un autómata celular con un algoritmo genético para
optimizar la forma urbana de Wuhan.»

Verificada. Resumen: *«this paper applies a coupling model called the "promoting urban vitality model," based on
cellular automata (CA) and genetic algorithm (GA) (abbreviated as UV-CAGA)… Wuhan, the largest city in Central
China, was selected as a case study to simulate and optimize its urban morphology for 2025.»* (La parte de la
oración relativa a `Hagenauer2022` corresponde a otro lote.)

**Afirmación 2 y las frases que la desarrollan.** «\textcite{Wang2021} integra un autómata celular a un algoritmo
genético para generar formas urbanas que maximizan un índice de vitalidad en Wuhan. Su codificación sí es legible,
pues cada gen corresponde a una celda del territorio, pero el ejercicio es normativo antes que predictivo: busca la
configuración urbana deseable bajo una función objetivo definida por el analista y no contrasta su resultado
contra un mapa observado posterior…»

Las tres afirmaciones sustantivas quedan verificadas en el texto completo, incluidas las dos que no llevan marca
de cita propia y que por eso merecían comprobación:

- *«cada gen corresponde a una celda del territorio»* — literal: «*we mapped the cell space to the chromosome in
  the genetic algorithm; a single cell was expressed as the gene in the genetic algorithm. The three UGPs of the
  cell, defined as infilling, edge, and outlying, were encoded as 1, 2, and 3 in the gene, respectively.*»
- *«función objetivo… índice de vitalidad»* — la función de aptitud del AG es literalmente
  $UV = \arg\max \sum_i \sum_k w_{ik} x_{ik}$, donde «*UV represents comprehensive urban vitality*», con pesos
  determinados por el coeficiente del modelo de regresión. Confirma «maximizan un índice de vitalidad».
- *«no contrasta su resultado contra un mapa observado posterior»* — sostenido. El horizonte de optimización es
  2025, posterior a la publicación (2021), de modo que no existe mapa observado con el cual contrastar; la
  comparación que el artículo reporta es entre dos simulaciones («*the urban vitality of the optimized urban form
  scheme was 4.8% higher than the simulated natural expansion scheme*»). En el texto completo no aparece ninguna
  métrica de concordancia con un mapa observado (búsqueda de «Kappa», «FoM» y «observed»: cero coincidencias
  sustantivas; las dos apariciones de «accuracy» se refieren a la exactitud de clasificación de la base NLUD-C y a
  una frase de trabajo futuro).

**Verificación del punto de riesgo señalado:** la tesis **no** atribuye a `Wang2021` ninguna afirmación sobre
costo o complejidad computacional. Revisadas las dos apariciones de la clave en el documento, la crítica que se le
dirige es de *propósito* (normativo frente a predictivo) y de *evidencia* (ausencia de contraste con observación),
ambas correctas. Nada que corregir.

---

## `White1993`

**Metadatos: CORRECTOS.** Crossref (`10.1068/a251175`): White, R.; Engelen, G.; *Cellular Automata and Fractal
Urban Form: A Cellular Modelling Approach to the Evolution of Urban Land-Use Patterns*; Environment and Planning A:
Economy and Space; 1993; vol. 25; núm. 8; pp. 1175–1199. Coincidencia exacta.

| # | Archivo:línea | Etiqueta |
|---|---|---|
| 1 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:18` (líneas 21–24) | CORRECTA |

**Afirmación.** «Su aplicación al crecimiento urbano se consolidó con \textcite{White1993}, cuyo modelo mostró que
reglas no globales, es decir locales, basadas en distancia y vecindad reproducen estructuras de uso de suelo con
propiedades fractales comparables a las medidas en ciudades reales.»

Verificada punto por punto contra el resumen del editor:

> *«In this paper, a cellular automaton is developed to model the spatial structure of urban land use over time.
> For realistic parameter values, the model produces fractal or bifractal land-use structures for the urbanized
> area and for each individual land-use type. **Data for a set of US cities show that they have very similar
> fractal dimensions.** The cellular approach makes it possible to achieve a high level of spatial detail and
> realism…»*

La cláusula más expuesta —«comparables a las medidas en ciudades reales»— es la que la fuente respalda de forma
más explícita. El verbo «se consolidó» también es el adecuado: no reclama prioridad para White y Engelen (que la
tesis asigna correctamente a Tobler y Couclelis en la oración anterior), sino consolidación de la aplicación
urbana.

**Punto de prioridad histórica verificado expresamente.** Se me pidió comprobar si la perturbación estocástica del
potencial de transición corresponde a White y Engelen **1993** y no a 1997. **Corresponde a 1993.** La tesis lo
asigna bien. En `cap02_marco_teorico/cap02_marco_teorico.tex:91` se lee: «*Esta extensión [los AC probabilísticos],
base de los modelos urbanos basados en AC desde \textcite{White1993} y de marcos calibrados como SLEUTH
\parencite{Clarke1998}…*», y en `cap03:23` se atribuye a `White1997` únicamente el acoplamiento posterior a un SIG
y a modelos regionales, que es lo correcto.

Evidencia: literatura revisada por pares que reproduce la formulación atribuye el término explícitamente a la
publicación de 1993. El potencial de transición se compone de idoneidad, accesibilidad y efecto de vecindad
multiplicados por un término estocástico $\xi = 1 + (-\log \rho)^{\sigma}$, donde $\rho$ es una variable aleatoria
uniforme en $(0,1)$ y $\sigma$ controla el tamaño de la perturbación, «*(White and Engelen, 1993)*». Declaración de
alcance: esto lo verifiqué en un artículo posterior que reproduce la ecuación y cita la fuente, no en el PDF de
1993 (SAGE cerrado, sin copia abierta); la atribución de la tesis coincide con la de esa literatura.

---

# Resumen contable

| Etiqueta | Conteo |
|---|---|
| CORRECTA | **16** |
| SOBREEXTENDIDA | **1** |
| INCORRECTA | **1** |
| NO VERIFICABLE | **0** |
| **Total de afirmaciones del lote** | **18** |

| Metadatos | Conteo |
|---|---|
| Referencias con metadatos correctos | **11 / 11** |
| Referencias con errores de metadatos | 0 |
| **REFERENCIAS INEXISTENTES** | **0** |

## Hallazgos que requieren edición

1. **INCORRECTA** — `cap03_estado_del_arte/cap03_estado_del_arte.tex:477`, fila de `JimenezLopez2021` en la Tabla
   `tab:cuadrante-mexico`: la columna R3 marca `\checkmark`. La fuente declara distribución «bajo demanda», y la
   nota al pie de la propia tabla justifica una valoración «parcial» para ese mismo trabajo. Cambiar `\checkmark`
   por `parcial` en esa celda.

2. **SOBREEXTENDIDA** — `cap03_estado_del_arte/cap03_estado_del_arte.tex:349` (líneas 355–358): `Legendre1993` se
   cita como fuente de la «sobreestimación del desempeño» en «validación de modelos espaciales». El artículo trata
   la violación del supuesto de independencia en las **pruebas estadísticas**, no la validación de modelos
   predictivos. Repartir las dos citas según la redacción propuesta en la sección correspondiente de este informe.

## Verificaciones de prioridad histórica solicitadas (todas favorables al texto actual)

- `Couclelis1985` «formalizó en términos teóricos»: correcto, el artículo es una generalización del principio de
  espacio celular basada en teoría de modelos discretos para uso geográfico.
- `White1993` como consolidación de la aplicación urbana y como origen de la perturbación estocástica del
  potencial de transición: correcto; 1997 aparece solo para el acoplamiento con SIG y modelos regionales.
- `Clarke2008` «más de cien aplicaciones en la década transcurrida»: correcto y literal («*over one hundred*»,
  «*over 100 different cities and regions*»).
- `Clarke2008` y Gigalopolis: correcto; el capítulo dice «*open source since its outset*» y que el código fuente
  completo y los datos de prueba se descargan del sitio del proyecto. No se le atribuye la publicación del código.
- `Wang2021`: no se le atribuye ninguna afirmación de costo computacional.
- `Ma2019`: la cita cubre solo la exigencia de conjuntos anotados grandes; el hardware gráfico queda fuera del
  alcance de la cita y va acompañado de un descargo explícito.
- `Beven2006`: la equifinalidad se enuncia como el artículo la enuncia, sin atribuirle el caso SLEUTH.
