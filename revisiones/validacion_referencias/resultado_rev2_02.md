# Resultado rev2 — lote 02 de 6 (10 referencias)

Revalidación literal del archivo `rev2_02.md` contra el texto que los `.tex` contienen hoy.
Fuentes consultadas por referencia: se declaran en cada apartado.

---

## 1. Aburas2016

**Metadatos: CORRECTO.** Crossref (`10.1016/j.jag.2016.07.007`) devuelve exactamente:
Aburas, Maher Milad; Ho, Yuek Ming; Ramli, Mohammad Firuz; Ash'aari, Zulfa Hanan.
*The simulation and prediction of spatio-temporal urban growth trends using cellular automata models: A review*, **International Journal of Applied Earth Observation and Geoinformation 52: 380–389 (2016)**. Coinciden autores, título, revista, año, volumen y páginas.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03_estado_del_arte.tex:526` (texto en 529–531) | «Las revisiones sistemáticas de Sante2010 y **Aburas2016** documentan la heterogeneidad de los métodos de **calibración y validación** entre los modelos que examinan.» | **SOBREEXTENDIDA** |

**Evidencia.** Crossref y OpenAlex no exponen resumen para este DOI; ScienceDirect devuelve 403. El resumen del editor sí está disponible íntegro en el registro de NASA/ADS (`2016IJAEO..52..380A`). Dice literalmente: *«The aim of this paper is to provide a basis for a literature review of urban Cellular Automata (CA) models... The general characteristics of simulation models of urban growth and urban CA models are described, and **the different techniques used in the design of these models are classified**. **The strengths and weaknesses of the various models are identified** based on the analysis and discussion of the characteristics of these models.»*

Dos problemas:

1. El artículo se presenta a sí mismo como *«a basis for a literature review»* (revisión narrativa). Llamarlo **«revisión sistemática»** le atribuye un protocolo (criterios de inclusión, corpus delimitado, procedimiento replicable) que el resumen no declara. Compárese con `Wahyudi2016`, que sí da un corpus contado (88 aplicaciones, 1993–2012), o con `Sante2010`, que sí revisa explícitamente calibración y validación.
2. Lo que el resumen sostiene es la **clasificación de técnicas de diseño** y el contraste de **fortalezas y debilidades**, no la heterogeneidad de los métodos de *calibración y validación*. Es plausible que el cuerpo del artículo trate calibración/validación, pero no pude acceder al texto completo, de modo que esa parte concreta queda sin verificar.

**Redacción corregida propuesta** (línea 529–531):

```latex
Las revisiones de \textcite{Sante2010} y \textcite{Aburas2016} clasifican las técnicas
empleadas en el diseño de los modelos y contrastan sus fortalezas y debilidades, sin que
emerja un procedimiento común de calibración y validación entre los trabajos que examinan.
```

Si se quiere conservar la afirmación fuerte sobre calibración y validación, basta con dejarla apoyada solo en `Sante2010`, que sí la sostiene explícitamente, y citar `Aburas2016` para la clasificación de técnicas.

---

## 2. Arfiansyah2024

**Metadatos: CORRECTO.** Crossref (`10.1007/s41324-024-00599-5`): Arfiansyah, Dody; Hawken, Scott; Zlatanova, Sisi; Han, Hoon. *Cellular automata modelling to simulate patterns of urban growth for Nusantara: Indonesia's new capital*, **Spatial Information Research 32(6): 829–849 (2024)**. Coinciden los ocho campos.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03_estado_del_arte.tex:31` | Aplicó el enfoque de AC a **Nusantara, la capital planificada de Indonesia**, un caso de ciudad de nueva creación. | **CORRECTA** |

**Evidencia.** Resumen del editor vía Crossref (JATS): *«This paper uses cellular automata (CA) modelling to simulate possible patterns of urban growth for Nusantara–Indonesia's new capital. The modelling uses criteria such as projected population growth and planned development stages... The study's cellular automata approach and methodology can be employed for urban planning and biodiversity impact assessment in similar contexts of **new city development**.»* La caracterización de «ciudad de nueva creación» es literal del resumen.

**Observación de redacción, no de fuente.** La conjunción de la frase está rota: «El enfoque conserva vigencia: **puesto que** \textcite{Arfiansyah2024} lo aplicó a Nusantara...». `puesto que` introduce una causa, pero aquí se quiere dar un ejemplo. Sugerencia: «El enfoque conserva vigencia: \textcite{Arfiansyah2024} lo aplicó a Nusantara, la capital planificada de Indonesia, un caso de ciudad de nueva creación.»

---

## 3. Chen2022

**Metadatos: CORRECTO.** Crossref (`10.1016/j.envsoft.2022.105362`): Chen, Changjie; Judge, Jasmeet; Hulse, David. *PyLUSAT: An open-source Python toolkit for GIS-based land use suitability analysis*, **Environmental Modelling & Software 151: 105362 (2022)**. Coinciden los siete campos.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03_estado_del_arte.tex:238` | «el conjunto de herramientas en Python para análisis de aptitud del suelo de \textcite{Chen2022}», presentado como componente que apunta hacia implementaciones de código abierto. | **CORRECTA** |

**Evidencia.** El propio título registrado en Crossref contiene los tres elementos atribuidos: *open-source*, *Python toolkit* y *land use suitability analysis*. No hizo falta ir más allá del registro bibliográfico, porque la afirmación de la tesis no excede el título.

---

## 4. Chihuahua2023

**Metadatos: CORRECTO con una reserva menor.** Crossref (`10.33064/iycuaa2023904103`) confirma autores (García-Ramírez, Pedro; Alatorre-Cejudo, Luis Carlos; Bravo-Peña, Luis Carlos), título completo, revista (*Investigación y Ciencia de la Universidad Autónoma de Aguascalientes*), año 2023 y número 90.
**Reserva:** Crossref **no registra volumen ni páginas** para este artículo; la entrada `.bib` declara `volume = {31}` y `pages = {e4103}`. El portal de la revista lo indexa como «Núm. 90 (2023)» sin volumen visible, y `e4103` corresponde al identificador de artículo (coincide con el sufijo del DOI). No es un error demostrado, pero conviene verificar el «Año 31» en la carátula del PDF antes de la entrega; si no aparece, lo prudente es borrar el campo `volume`.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03_estado_del_arte.tex:426` | Cadena **SVM + regresión logística + CA-Markov**; **diez** núcleos urbanos; **cinco** cuencas nombradas una a una; proyecciones a **2030, 2040 y 2050**. | **CORRECTA** |
| 2 | `cap03_estado_del_arte.tex:477` (tabla `tab:cuadrante-mexico`) | Fila: R1 parcial, R2 ✓, R3 no, R4 parcial. | **CORRECTA** (juicio propio del autor, consistente con la fuente) |

También verifiqué el resto del párrafo citante (líneas 430–441), que forma parte de la misma atribución:

| Afirmación en 430–441 | Etiqueta | Evidencia textual |
|---|---|---|
| Clasificación SVM sobre imágenes de **2010, 2015 y 2020** | CORRECTA | Resumen: «Se produjeron mapas de uso de suelo años 2010, 2015 y 2020 utilizando Máquina de Soporte Vectorial (SVM)» |
| Matriz de confusión sobre **385 puntos** de verificación, precisión global y Kappa | CORRECTA | «El número de puntos para verificar la precisión fue de 385 (77 puntos por clases de uso de suelo)... La precisión del productor, la precisión del usuario, la precisión global y el coeficiente Kappa se utilizaron para medir la precisión de la clasificación» |
| Se publican las ecuaciones de los diez núcleos con sus coeficientes | CORRECTA | «El modelo de RL y el coeficiente de cada una de las variables independientes se muestran en la tabla 4»; «para las **diez ciudades** los coeficientes Kappa mostraron valores altos» |
| **Siete** núcleos por encima del umbral de 0,2 y **tres** por debajo | CORRECTA | Un núcleo con R²MF = 0,52 (Cd. Jiménez) + seis entre 0,26 y 0,33 = siete; «por el contrario, en Chihuahua, Santa Eulalia y Juan Aldama los modelos mostraron valores inferiores a 0.2». Y en discusión: «en 7 de 10 zonas urbanas estudiadas y regular en las tres zonas urbanas» |
| CA-Markov dentro del **Land Change Modeler de IDRISI Selva**, software de código cerrado | CORRECTA | «Se utilizó la herramienta Modelador de Cambios en el Terreno (LCM) en software **IDRISI Selva 17.0**»; «la CA-Markov... localizada en el software **IDRISI Selva 17**». **No** dice TerrSet en ningún lugar del texto |
| No se reporta métrica alguna de concordancia entre simulación y mapa observado | CORRECTA | Búsqueda exhaustiva sobre el texto completo: la única «validación» declarada es la del modelo de RL mediante pseudo R² de McFadden y ROC (§«La validación del modelo fue óptima...»), y la de la clasificación mediante Kappa y precisión global. No aparece Kappa, FoM ni matriz de confusión aplicados a la salida del CA-Markov |

**Fuente consultada:** texto completo del artículo en el *galley* HTML de la revista (`revistas.uaa.mx/investycien/article/view/4103/5550`), acceso abierto.

**Nota para el autor (no es un error, es un riesgo de lectura).** El resumen del artículo afirma en bloque que «El modelo de RL resultó ser consistente mostrando Pseudo R² de McFadden **superior a 0.2**», lo que contradice el cuerpo, donde tres núcleos quedan por debajo. La tesis sigue el cuerpo, que es lo correcto. Si alguien contrasta la tesis solo contra el resumen podría creer que hay discrepancia; conviene tenerlo presente si un sinodal lo pregunta.

---

## 5. Gorelick2017

**Metadatos: CORRECTO.** Crossref (`10.1016/j.rse.2017.06.031`): Gorelick, Noel; Hancher, Matt; Dixon, Mike; Ilyushchenko, Simon; Thau, David; Moore, Rebecca. *Google Earth Engine: Planetary-scale geospatial analysis for everyone*, **Remote Sensing of Environment 202: 18–27 (2017)**. Coinciden los siete campos.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap05_modelo_crecimiento_urbano.tex:44` | «en este trabajo **no** se empleó Google Earth Engine~\parencite{Gorelick2017} ni descarga masiva por API». | **CORRECTA** |

**Evidencia.** La cita no atribuye ninguna afirmación sustantiva a la fuente: la usa solo para identificar la plataforma que el trabajo declara **no** haber utilizado. El título registrado en Crossref basta para respaldar esa identificación. No hay riesgo de sobreextensión.

---

## 6. Huacuz2018Metropolizacion

**Metadatos: CORRECTO.** Crossref (`10.29105/contexto12.16-6`): Huacuz Elías, Rafael de Jesús; Vázquez Cruz, Rubí del Rocío. *El proceso de metropolización en Querétaro 1990-2010*, **CONTEXTO. Revista de la Facultad de Arquitectura de la Universidad Autónoma de Nuevo León, 12(16): 79–91 (2018)**. Coinciden los ocho campos, incluido el `url` de Redalyc.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap05_modelo_crecimiento_urbano.tex:44` | «su dinámica de crecimiento **reciente**, bien documentada para la zona metropolitana~\parencite{Huacuz2018Metropolizacion}» | **CORRECTA** (con matiz) |

**Evidencia.** Resumen completo obtenido de OpenAlex: *«Aplicando el enfoque del análisis territorial el estudio se enfoca al proceso de metropolización en la **Zona Metropolitana de Querétaro**, la cual ha destacado por **lo acelerado que se ha expandido en su superficie de urbanización a través del tiempo**... muestra la manera en que un territorio de escala metropolitana se va cubriendo de urbanizaciones y cómo puede explorarse comparativamente entre las décadas **1990-2010**.»*

El respaldo es directo: la fuente documenta la dinámica de expansión de la ZMQ. El único matiz es el adjetivo **«reciente»**: el estudio cubre 1990–2010 y se publicó en 2018, de modo que no habla de la década de 2010 ni de la de 2020. No es un error, pero si se quiere precisión se puede escribir «su dinámica de expansión metropolitana, documentada para el período 1990–2010~\parencite{Huacuz2018Metropolizacion}».

---

## 7. JimenezLopez2018

**Metadatos: CORRECTO.** Sin DOI (la revista no asigna). Verificado contra dos fuentes independientes:
- El **PDF completo** depositado en el repositorio institucional de la UAEMex (`ri.uaemex.mx/bitstream/handle/20.500.11799/112650/...pdf`), cuyo encabezado da los tres autores en el orden del `.bib` («Eduardo Jiménez López - Tania Chávez Soto - Carlos Garrocho», El Colegio Mexiquense A.C.) y el **ISSN 1852-8031** declarado en el pie de página.
- El **índice del número** en el sitio de la revista (`revistageosig.wixsite.com/geosig/geosig-12-2018`), que lista «Modelando la expansión urbana con autómatas celulares: aplicación de la estación de inteligencia territorial (CHRISTALLER) — Eduardo Jiménez López, Tania Chávez Soto, Carlos Garrocho, **II:1-26**», lo que confirma tanto el rango de páginas como el campo `note = {Sección II: Metodología}`.

La referencia **existe** y es de acceso abierto. Toda la validación siguiente se hace contra el **PDF completo**, no contra el resumen.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03_estado_del_arte.tex:443` | AC deterministas sobre rejilla raster; barrido exhaustivo de las **256** reglas; ciudades **Querétaro, San Luis Potosí y Toluca**, **2003–2017**; para Querétaro la **regla 192** con **Kappa 0,53** y **Jaccard 0,76**. | **CORRECTA** |
| 2 | `cap03_estado_del_arte.tex:477` (tabla) | Fila: R1 parcial, R2 ✓, R3 parcial, R4 parcial. | **CORRECTA** (juicio propio, consistente con la fuente) |
| 3 | `cap03_estado_del_arte.tex:493` | «describen CHRISTALLER como software de **código abierto**». | **CORRECTA** |
| 4 | `cap03_estado_del_arte.tex:504` | «Querétaro ya ha sido modelada con autómatas celulares». | **CORRECTA** |
| 5 | `cap05_modelo_crecimiento_urbano.tex:358` | «existe un antecedente de simulación con AC para Querétaro entre **2003 y 2017**». | **CORRECTA** |
| 6 | `cap06_resultados_analisis.tex:244` | No reporta FoM sino Kappa, Jaccard y dimensión fractal; Kappa 0,53 y Jaccard 0,76 sobre **catorce años** y sobre la **clase urbana completa**; el mapa de 2017 sirve a la vez para seleccionar la regla entre 256 y para medir su ajuste. | **CORRECTA** |
| 7 | `cap07_conclusiones.tex:84` | Mismo argumento, con «las cifras publicadas corresponden al máximo de una búsqueda y no a una prueba independiente». | **CORRECTA** |

**Evidencia textual, punto por punto.** Este es el caso delicado del lote, así que lo desgloso con cita literal del PDF.

- **Período y ciudades.** «En este trabajo usamos software y herramientas de simulación para instrumentar modelos de AC que repliquen la expansión de la mancha urbana **entre 2003 y 2017**, de tres grandes ciudades mexicanas: **Querétaro, San Luis Potosí y Toluca**.»
- **Determinismo y raster.** «La lógica del modelo de AC que se utiliza en la parte experimental de este trabajo se basa en **Reglas de Transición deterministas**»; «las condiciones iniciales provienen de **mapas raster** a los que se les realizó un proceso de filtrado para binarizarlas»; y en conclusiones, «con **modelos de AC deterministas** para el periodo 2003-2017».
- **Barrido de 256 reglas.** «tenemos **256 Reglas de Transición** posibles»; «se ejecutó... la expansión de las manchas urbanas para 2017 con **cada una de las 256 Reglas** de Transición»; «CHRISTALLER hace un **barrido completo** de las 256 Reglas de Transición, estima los indicadores de bondad de ajuste e **identifica la Regla de Transición que mejor replica** la expansión de la mancha urbana de cada ciudad».
- **Regla 192 y cifras de Querétaro.** «Para Querétaro la Regla de Transición que mostró el mejor ajuste entre la imagen satelital de 2017 y la proyección del modelo de AC fue la **192**; para San Luis Potosí fue la **218** y para Toluca la Regla **222**.» El Cuadro 3 («Bondad de ajuste entre la imagen satelital y los resultados del modelo de AC») da: Querétaro **0.53 / 0.76**, San Luis Potosí 0.64 / 0.91, Toluca 0.56 / 0.68. Las dos cifras de la tesis son exactas y la asignación regla→ciudad también.
- **Métricas reportadas.** El artículo reporta exactamente tres indicadores: «la **Dimensión Fractal**, el **Índice de Kappa de Cohen** y el **Índice de Jaccard**». No aparece FoM en ninguna parte. La tesis acierta al decir que no hay magnitud común.
- **Horizonte de catorce años.** «Los tres valores indican una alta capacidad del modelo de AC para replicar el fenómeno de expansión urbana en las tres ciudades, en los **catorce años** de análisis.»
- **Proyección a 2031 sin intervalo observado posterior.** «permite proyectar... para un periodo similar al de las simulaciones: un horizonte de catorce años, esto es, **al año 2031**», y las Figuras 21–23 se rotulan «Mapa del Modelo de Autómata Celular, **2031**». (Aviso menor: las conclusiones del propio artículo dicen «proyectar la expansión urbana de las tres ciudades a **2030**». Es una inconsistencia **interna de la fuente**. La tesis usa 2031, que es lo que sostienen los resultados y las figuras; es la lectura correcta, pero conviene saberlo por si un sinodal cita las conclusiones.)
- **Código abierto.** Resumen del artículo: «Este trabajo vincula Sistemas de Información Geográfica (SIG) con **software especializado de código abierto** desarrollado en El Colegio Mexiquense (la Estación de Inteligencia Territorial: CHRISTALLER)»; en inglés, «**open source software**». Y en el cuerpo: «su diseño prevé su operación como un **sistema abierto de uso gratuito**». La afirmación de la tesis es literal.

**Sobre el argumento central de no comparabilidad: SE SOSTIENE, y no está exagerado.**

La tesis afirma que el mapa observado de 2017 cumple dos funciones a la vez — criterio de selección de la regla ganadora entre 256 y mapa de referencia contra el que se reporta el ajuste de esa regla. El artículo dice exactamente eso, sin ambigüedad: el barrido «estima los indicadores de bondad de ajuste e identifica la Regla de Transición que mejor replica la expansión», la regla ganadora es «la que mostró el mejor ajuste **entre la imagen satelital de 2017** y la proyección del modelo», y el Cuadro 3 publica como resultado la bondad de ajuste **de esas mismas reglas ganadoras contra el mismo mapa de 2017**. No existe en el artículo ningún conjunto de evaluación independiente ni ninguna partición temporal adicional. La formalización $\delta^{*} = \arg\max_{\delta} M(\widehat{2017}_{\delta}, 2017_{\text{obs}})$ seguida de la publicación de $M(\widehat{2017}_{\delta^{*}}, 2017_{\text{obs}})$ describe fielmente el procedimiento.

También es exacto que «cada ciudad recibe su propia regla óptima (192, 218 y 222)», con el orden Querétaro–San Luis Potosí–Toluca correctamente pareado.

**Y tampoco está suavizado en exceso.** La tesis dice «El diagnóstico no invalida el trabajo, que es metodológicamente explícito en cuanto a su procedimiento». Es una caracterización justa: los autores describen el barrido con total transparencia y no presentan las cifras como predicción fuera de muestra; el propio artículo advierte que las proyecciones «de ninguna manera son una predicción del futuro». La crítica de la tesis se dirige a la interpretación de las métricas, no a una ocultación, y así está formulada. No veo margen de mejora en el equilibrio de este pasaje.

**Una salvedad técnica que conviene conocer (no exige corregir nada).** Las 256 reglas del artículo son las reglas elementales de Wolfram para un AC **unidimensional de tres bits** («tenemos 256 Reglas de Transición posibles para simular la evolución de un modelo de AC de tres bits en el **espacio lineal**»), aplicadas después sobre mapas raster bidimensionales. La frase de la tesis, «instrumentan autómatas celulares deterministas sobre rejilla raster y, mediante un barrido exhaustivo de las 256 reglas de transición posibles», reproduce con fidelidad lo que la fuente dice de sí misma, así que no hay nada que corregir. Solo tenlo presente: la tensión entre «256 reglas» (1-D) y «rejilla raster» (2-D) es de la fuente, no de la tesis.

### Afirmación adyacente que sí conviene revisar

| Archivo:línea | Afirmación | Etiqueta |
|---|---|---|
| `cap03_estado_del_arte.tex:497–500` (nota al pie de la tabla) | «registra que **la distribución se hace bajo solicitud al grupo desarrollador** y no mediante un repositorio público con historial de versiones» | **NO VERIFICABLE** |

Ninguna de las dos fuentes citadas dice que CHRISTALLER se distribuya «bajo solicitud». Al contrario, `JimenezLopez2018` anuncia que el sistema «esté disponible en **acceso abierto** (en el primer trimestre de 2018)» y el trabajo de 2021 del mismo grupo remite a un sitio web propio (`christaller.org.mx`). El enunciado de la tesis es una comprobación empírica del canal de distribución, no una atribución a los autores, y como tal no pude confirmarlo con las fuentes disponibles.

**Redacción corregida propuesta** — sustituir el mecanismo no verificado por el hecho comprobable:

```latex
Nota sobre R3. Los trabajos de \textcite{JimenezLopez2018,JimenezLopez2021} describen
CHRISTALLER como software de código abierto. La valoración «parcial» no cuestiona esa
descripción: registra que los artículos no señalan un repositorio público con historial de
versiones desde el cual un tercero pueda obtener el código y reejecutar el procedimiento
sin intermediarios, que es lo que R3 exige.
```

Si el autor sí comprobó por su cuenta que el acceso es bajo solicitud, la alternativa es dejar la frase actual y añadir la fecha y la vía de consulta en la propia nota. Tal como está, es la única frase del lote que un revisor podría pedir que se documente.

---

## 8. Montejano2023

**Metadatos: CORRECTO.** Crossref (`10.22198/rys2023/35/1734`): Montejano-Escamilla, Jorge Alberto; Caudillo-Cos, Camilo Alberto; Ávila-Jiménez, Felipe Gerardo; Tapia-McClung, Rodrigo; Barrera-Alarcón, Itzia Gabriela. *Expansión y crecimiento urbanos en México, 1975-2020*, **Región y Sociedad 35: e1734 (2023)**. Coinciden los cinco autores en orden, título, revista, año, volumen y número de artículo.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap05_modelo_crecimiento_urbano.tex:44` | «dentro de la expansión urbana que el país arrastra **desde los años setenta**~\parencite{Montejano2023}» | **CORRECTA** |

**Evidencia.** Resumen en SciELO México: *«se presenta una estimación del aumento de la población y de la expansión urbana en México... **entre 1975 y 2020**, utilizando sobre todo fuentes de percepción remota... se concluye que mientras que el **suelo edificado total creció tres veces** en cuarenta años, la población tan solo se duplicó... **se confirma que hay una tendencia generalizada a la expansión urbana dispersa**.»*

La serie del artículo arranca en 1975 y documenta expansión sostenida a lo largo de todo el período, de modo que «desde los años setenta» está respaldado. La tesis usa la fuente para un encuadre general, sin atribuirle cifras, así que no hay riesgo numérico.

---

## 9. RamirezHernandez2021

**Metadatos: CORRECTO.** Crossref (`10.22201/iiec.9786073044349e.2021`) confirma título y autor. El libro **existe** y está en acceso abierto en el repositorio del IIEc (`libros.iiec.unam.mx/sites/libros.iiec.unam.mx/files/2021-04/ZMCM_RRH.pdf`), con portada: «Autor: **Roberto Ramírez Hernández**; Primera edición digital pdf, **abril 2021**; ISBN 978-607-30-4434-9; UNAM, Coordinación de Humanidades, **Instituto de Investigaciones Económicas**, Programa Universitario de Estudios sobre la Ciudad».
**Reserva menor, opcional:** Crossref lo tipifica como `edited-book` y lista como co-editores a CoHu y PUEC; la entrada `.bib` acredita solo al IIEc. La portada del PDF confirma que Ramírez Hernández es **autor** (no coordinador), así que `@book` con `author` es la elección correcta. Si se quiere exhaustividad, se puede ampliar `publisher` a «UNAM, Instituto de Investigaciones Económicas y Programa Universitario de Estudios sobre la Ciudad».

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03_estado_del_arte.tex:413` | Libro del IIEc-UNAM; **él mismo clasifica** su modelo como «econométrico con simulación espacial»; AC sobre **celdas territoriales** con **vecindad de Moore**; probabilidades de transición estimadas con **regresiones logísticas binomial y multinomial**; resueltas mediante **rutina de Monte Carlo**. | **CORRECTA** |
| 2 | `cap03_estado_del_arte.tex:477` (tabla) | Fila: R1 no, R2 ✓, R3 no, R4 parcial. | **CORRECTA** (juicio propio, consistente con la fuente) |

**Evidencia textual, desde el PDF completo del libro.**

- **Autoclasificación.** «El modelo propuesto **es catalogado como un modelo econométrico con simulación espacial**» (cap. IV). Y de nuevo: «es econométrico con simulación espacial y está basado en el cambio [de uso de suelo]». La atribución de la tesis («lo que **él mismo clasifica** como...») es literal, no una interpretación. Esto es importante: el verbo de atribución está bien elegido y resiste escrutinio.
- **Vecindad de Moore sobre celdas territoriales.** «Asumamos que se define por la **vecindad de Moore**»; «El tipo de vecindad es conocida como **vecindad de Moore**, misma que involucra **las ocho celdas contiguas** a $ct_{ijt}$ y sus respectivos usos de suelo»; y en el resumen del modelo: «La influencia espacial asume una vecindad como la definida por Moore... se consideran vecinas las ocho **celdas territoriales** que rodean cada celda territorial». El término «celdas territoriales» es el del propio libro.
- **Binomial y multinomial.** «...mediante ejercicios de **regresión logística binomial y multinomial**»; «regresiones logísticas **binomial y multinomial**, y así determinar las **probabilidades de transición**».
- **Monte Carlo.** «...se implementara una **rutina de simulación de tipo Montecarlo**, de acuerdo con las especificaciones planteadas ampliamente en este capítulo.» **Confirmado explícitamente: el modelo es econométrico con Monte Carlo y NO está inspirado en SLEUTH.** Busqué SLEUTH en el texto completo y no aparece; el único antecedente mexicano que el libro invoca es Delgado y Suárez-Lastra (2006), con regresión logística binomial.
- **1990–2010 → 2040.** El capítulo III se titula «Patrones de crecimiento y expansión de la zmcm **entre 1990 y 2010**» y el IV, «Modelo de crecimiento y expansión para la Ciudad de México y su zm **al 2040**».
- **Coeficientes publicados íntegros.** El libro reproduce las tablas completas de coeficientes de ambos mecanismos (intersección y coeficientes de `pt`, `vh`, `poi`, `poc`, `posg`, `ie`, `dcbd`, `dsub`, `MESES`, y las categorías `pdt`, `sr`, `rt`, para los seis estados de la multinomial), con la nota «Fuente: elaboración propia a partir de base de datos para modelo de simulación urbana y rutina de regresión logística multinomial del programa spss v. 18».
- **Dependencia de SPSS y SAS, con un solo fragmento del programa.** «rutina de regresión logística binomial del programa **spss v. 18**»; «fue necesario... diseñar un programa en un lenguaje de programación... **Se anexa un fragmento del programa** diseñado en lenguaje **sas v. 9.0** para ilustrar lo anterior» (Figura 4.2, rotulada «Fragmento del programa en lenguaje sas v. 9.0 con implementación del modelo de simulación»). La tesis dice exactamente esto.
- **Ausencia de métricas de concordancia.** Búsqueda sistemática sobre el texto completo de «kappa», «figure of merit», «matriz de confusión» y «bondad de ajuste»: **cero coincidencias**. La afirmación negativa de la tesis («no reporta ninguna métrica de concordancia entre el mapa simulado y un mapa observado, ni Kappa ni FoM ni matriz de confusión») está verificada por ausencia sobre el documento íntegro, que es el nivel de evidencia más alto que admite una negación de este tipo.

---

## 10. Torrens2000

**Metadatos: CORRECTO.** Sin DOI (documento de trabajo). Verificado en dos fuentes institucionales:
- **UCL Discovery** (`discovery.ucl.ac.uk/id/eprint/1371`): «Torrens, P.M.; (2000) *How cellular models of urban systems work (1. theory)*. (**CASA Working Papers 28**). **Centre for Advanced Spatial Analysis (UCL): London, UK**», con PDF en acceso abierto.
- **Sitio del Bartlett, UCL** (`ucl.ac.uk/bartlett/publications/2000/nov/casa-working-paper-28`): «CASA Working Paper 28 — How Cellular Models of Urban Systems Work (1. Theory). Authors: **Paul Torrens**. Publication Date: **1/11/2000**».

Coinciden autor, título, tipo, institución, número, ciudad, año y mes (noviembre). La entrada `.bib` es exacta.

| # | Archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03_estado_del_arte.tex:31` | «La mecánica interna de esos modelos, es decir de qué piezas se compone un modelo celular urbano y **cómo debe modificarse el formalismo del autómata para representar una ciudad**, está descrita **en el plano teórico**» por Torrens. | **CORRECTA** |

**Evidencia.** Resumen oficial en el sitio del Bartlett: *«This paper (part 1 of a two-part series) is intended to serve as an introduction to **how cellular models of urban system actually work, on a theoretical level**. Part 2 focuses on how to build urban CA models in a practical context... Section 4.1 outlines the advantages of using CA in urban studies. **Section 4.2 describes how CA must be modified for urban applications.**»*

La correspondencia es casi palabra por palabra, incluido el matiz «en el plano teórico», que el propio resumen marca al distinguir la parte 1 (teórica) de la parte 2 (práctica). Es la atribución mejor calibrada del lote.

---

# Resumen contable

**Afirmaciones listadas en el lote: 18.**

| Etiqueta | Conteo |
|---|---|
| CORRECTA | **17** |
| SOBREEXTENDIDA | **1** |
| INCORRECTA | **0** |
| NO VERIFICABLE | **0** |

**Referencias inexistentes: 0.** Las diez referencias del lote existen y fueron confirmadas en fuente primaria o institucional: ocho por DOI en Crossref, `JimenezLopez2018` por PDF completo en repositorio institucional más índice del número de la revista, y `Torrens2000` por UCL Discovery y el sitio del Bartlett.

**Metadatos:** 10 de 10 correctos. Dos reservas menores, ninguna bloqueante: el `volume = {31}` y `pages = {e4103}` de `Chihuahua2023` no están en Crossref (conviene cotejarlos con la carátula del PDF), y `RamirezHernandez2021` omite a PUEC y CoHu como coeditores.

**Único hallazgo SOBREEXTENDIDA**
- `cap03_estado_del_arte.tex:526` — `Aburas2016` se califica de «revisión sistemática» (el artículo se presenta como revisión narrativa) y se le atribuye documentar la heterogeneidad de los métodos de calibración y validación, extremo que el resumen del editor no sostiene: lo que declara es clasificar técnicas de diseño e identificar fortalezas y debilidades.

**Afirmación adicional señalada, fuera de las 18 del lote**
- `cap03_estado_del_arte.tex:497–500` — **NO VERIFICABLE**: «la distribución se hace bajo solicitud al grupo desarrollador». No lo sostiene ninguna de las dos fuentes citadas; `JimenezLopez2018` anuncia acceso abierto. Propuesta de reescritura en el apartado 7.

**Nota sobre los tres puntos de atención señalados en el encargo**
- `JimenezLopez2018`: el argumento de no comparabilidad **se sostiene íntegro** contra el PDF completo. Las cifras (0,53 y 0,76), la regla (192), el emparejamiento regla–ciudad (192/218/222 para Querétaro/SLP/Toluca), el horizonte (catorce años) y el doble uso del mapa de 2017 están verificados con cita literal. Ni exagerado ni suavizado.
- `Chihuahua2023`: el software es **IDRISI Selva 17.0**, tal como dice la tesis. «TerrSet» no aparece en el artículo.
- `RamirezHernandez2021`: **modelo econométrico con simulación espacial resuelto por Monte Carlo**, así catalogado por el propio autor. SLEUTH no se menciona en el libro.
