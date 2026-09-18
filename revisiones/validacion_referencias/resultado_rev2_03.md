# Resultado rev2 lote 03 — Validación de fuentes

11 referencias, 18 afirmaciones. Metadatos: 11/11 correctos, 0 inexistentes.
Etiquetas: 12 CORRECTA, 5 SOBREEXTENDIDA, 0 INCORRECTA, 1 NO VERIFICABLE.

## AgterbergCheng2002
Metadatos: CORRECTOS (Crossref 10.1023/A:1021193827501 — Natural Resources Research 11(4):249-255, 2002).

| Afirmación | Etiqueta |
|---|---|
| cap03:315 «la prueba exacta de AgterbergCheng2002» | CORRECTA |

Verificado con cita textual del artículo (copia abierta, ige.unicamp.br): "This new test is
exact and simpler to use than other tests including the Kolmogorov-Smirnov test and various
chi-squared tests." El adjetivo "exacta" es el de los autores.

## ArribasBel2014
Metadatos: CORRECTOS (Crossref 10.1016/j.apgeog.2013.09.012 — Applied Geography 49:45-53, 2014).

| Afirmación | Etiqueta |
|---|---|
| cap03:246 (tres fuentes: datos gubernamentales abiertos, sensores móviles, actividad empresarial en línea) | CORRECTA |

Resumen textual (Semantic Scholar): "three groups of data sources ... data collected from
mobile sensors carried by individuals, data derived from businesses moving their activity
online and government data released in an open format." Coincidencia exacta de las tres categorías.

## BonhamCarter1994
Metadatos: CORRECTOS. Pergamon, Oxford, 1994; Computer Methods in the Geosciences vol. 13;
398 pp. Confirmado en ScienceDirect/Elsevier (pp. 1-398). Libro sin DOI, verificado por catálogo.
Capítulo 9 = "Tools for Map Analysis: Multiple Maps", pp. 267-337.

| Afirmación | Etiqueta |
|---|---|
| cap03:258 WoE se consolida como procedimiento estándar en el cap. 9 | CORRECTA |
| cap03:315 pruebas por pares y ómnibus del cap. 9; exceso <= 15 % | CORRECTA |
| cap03:332 propone elegir cortes maximizando el contraste estudentizado | SOBREEXTENDIDA |
| cap03:349 varianza de pesos por conteos, sin estructura espacial | CORRECTA |

- cap03:258 — Documentación Arc-WofE (Geological Survey of Canada / Unicamp): "For the
  derivation of weights, see Bonham-Carter (1994, ch. 9)". Verificación indirecta pero explícita.
- cap03:315 — Agterberg y Cheng (2002), cita textual: "in Bonham-Carter (1994, p. 316), it is
  argued that T should not exceed n by more than 15%", y más adelante "the informal 15% rule of
  the earlier version of the omnibus test". La p. 316 está dentro del cap. 9 (267-337).
  La misma documentación Arc-WofE confirma ambas pruebas: "WofE allows the user to carry out a
  pair-wise conditional independence test, and an 'omnibus' test as described in
  Bonham-Carter (1994, ch. 9)".
- cap03:332 — En el cap. 9 el corte se elige en el máximo del contraste C (ejemplo: clase 5,
  1.25 km) y el contraste estudentizado C/s(C) es una columna auxiliar "helpful for choosing the
  cutoff distance, because it shows the contrast relative to the uncertainty due to the weights".
  El texto confunde maximizar C con maximizar C/s(C).
  CORRECCIÓN: «propone elegir los cortes en el máximo del contraste $C$, con el contraste
  estudentizado $C/s(C)$ como criterio auxiliar de significancia, criterio que optimiza la
  asociación medida sobre el propio conjunto de entrenamiento.»
- cap03:349 — Las expresiones de s(W+) y s(W-) del cap. 9 se construyen con conteos de celdas
  unitarias y de puntos de entrenamiento; no hay término de autocorrelación espacial. La
  sobreestimación se atribuye en la tesis a Legendre1993/Brenning2012, no a Bonham-Carter.

## Clarke1998
Metadatos: CORRECTOS (Crossref 10.1080/136588198241617 — IJGIS 12(7):699-714, 1998,
Clarke y Gaydos).

| Afirmación | Etiqueta |
|---|---|
| cap03:26 SLEUTH «aplicado primero» a San Francisco y Washington-Baltimore | SOBREEXTENDIDA |

El artículo cubre ambas áreas (está en el título), pero la aplicación a la bahía de San Francisco
se publicó antes: Clarke, Hoppen y Gaydos (1997), Environment and Planning B 24(2):247-261,
DOI 10.1068/b240247 (verificado en Crossref). El cap. 2 de la tesis ya lo formula bien.
CORRECCIÓN: «aplicado a la bahía de San Francisco y al corredor Washington-Baltimore
\parencite{ClarkeHoppen1997,Clarke1998}» (suprimir "primero").

## Gomez2020
Metadatos: CORRECTOS (Crossref 10.3390/rs12010109 — Remote Sensing 12(1):109; en línea
2019-12-28, número de 2020).

| Afirmación | Etiqueta |
|---|---|
| cap03:59 llevó el aprendizaje automático al modelado espaciotemporal del crecimiento urbano | CORRECTA |
| cap03:159 concordancias espaciales altas, sin contraste contra modelo de referencia | NO VERIFICABLE |

- cap03:59 — Resumen (Crossref): "models the population distribution as a spatiotemporal
  regression problem using machine learning ... combining this framework with free data from the
  Landsat archive and the Global Human Settlement Layer".
- cap03:159 — NO VERIFICABLE. MDPI bloqueó el texto completo en dos intentos (timeout y
  respuesta vacía). Solo se accedió al resumen, que no reporta ninguna métrica de concordancia y
  menciona los AC como contraste retórico, no cuantitativo. No se infiere el contenido de la
  sección de resultados. Si no se consigue el PDF, conviene suavizar a una formulación que no
  atribuya cifras ni ausencias de comparación a esta fuente.

## Hagenauer2022
Metadatos: CORRECTOS (Crossref 10.1080/13658816.2021.1871618 — IJGIS 36(2):215-235; en línea
2021, número de 2022).

| Afirmación | Etiqueta |
|---|---|
| cap03:63 introdujo una GWANN para relaciones no lineales en vivienda y calidad del aire, no como modelo de crecimiento urbano | CORRECTA |

Resumen (repositorio institucional Utrecht): "we propose a geographically weighted artificial
neural network (GWANN)" — el verbo "introdujo" corresponde a la afirmación de los propios
autores. Casos de aplicación confirmados: precios de vivienda en Austria (datos UniCredit Bank
Austria) y un modelo de regresión de usos de suelo para dióxido de nitrógeno en Austria
("Land use regression model for nitrogen dioxide in Austria", 142 estaciones, 2018). No es un
modelo de crecimiento urbano: correcto.

## INEGI2020Censo
Metadatos: CORRECTOS. Programa Censo de Población y Vivienda 2020, INEGI; la ruta
inegi.org.mx/.../programas/ccpv/2020/ es válida (documentos oficiales del programa accesibles).

| Afirmación | Etiqueta |
|---|---|
| cap05:40 la ZM de Querétaro rebasó el millón y medio de habitantes | CORRECTA |

Censo 2020: Querétaro 1 049 777 + Corregidora 212 567 + El Marqués 231 668 + Huimilpan 36 808
= 1 530 820. Verificación indirecta por dos fuentes que compilan el censo: breviario del COESPO
del Gobierno de Querétaro ("llegando a 1'530,820 habitantes en la última medición censal") y
Bustamante/Ramírez (IIEC-UNAM, tabla con base en censos INEGI 2000-2020). No se consultó el
tabulador del INEGI directamente.

## Ojala2002
Metadatos: CORRECTOS (Crossref 10.1109/TPAMI.2002.1017623 — IEEE TPAMI 24(7):971-987, 2002).

| Afirmación | Etiqueta |
|---|---|
| cap05:242 «los patrones binarios locales [LBP] de Ojala2002, robustos a cambios de iluminación» | SOBREEXTENDIDA |

La robustez es textual: "very robust in terms of gray-scale variations since the operator is, by
definition, invariant against any monotonic transformation of the gray scale" (resumen IEEE y
copia abierta del PDF). El problema es la atribución del operador: el LBP original es de Ojala,
Pietikäinen y Harwood (1996); el artículo de 2002 deriva "a generalized gray-scale and rotation
invariant operator presentation" (confirmado en Scholarpedia, artículo firmado por Pietikäinen).
La corrección ya está aplicada en cap02:40 («en la formulación circular generalizada de») pero
no se propagó al capítulo 5.
CORRECCIÓN: «en particular los patrones binarios locales [LBP] en la formulación circular
generalizada de \textcite{Ojala2002}, invariantes a cambios monótonos de iluminación y útiles
para distinguir la rugosidad del tejido construido».

## PontiusMillones2011
Metadatos: CORRECTOS (Crossref 10.1080/01431161.2011.552923 — IJRS 32(15):4407-4429, 2011).

| Afirmación | Etiqueta |
|---|---|
| cap03:372 Kappa confunde desacuerdo de cantidad y de asignación; recomiendan abandonarlo | CORRECTA |
| cap06:417 la descomposición separa ambas fuentes | CORRECTA |
| cap06:421 pie de tabla que atribuye la descomposición | CORRECTA |

Resumen textual: "baseline maps that can have two types of randomness: (1) random distribution
of the quantity of each category and (2) random spatial allocation of the categories ... we
recommend that the profession abandon the use of Kappa indices for purposes of accuracy
assessment and map comparison, and instead summarize the cross-tabulation matrix with two much
simpler summary parameters: quantity disagreement and allocation disagreement." Es el artículo
correcto y no hay cifras atribuidas a la fuente.

## SoaresFilho2002
Metadatos: CORRECTOS (Crossref 10.1016/S0304-3800(02)00059-5 — Ecological Modelling
154(3):217-235, 2002).

| Afirmación | Etiqueta |
|---|---|
| cap03:43 «Dinamica EGO» entre arquitecturas alternativas al autómata celular puro | SOBREEXTENDIDA |
| cap03:275 la formulación original calcula probabilidades por regresión logística | CORRECTA |

- cap03:43 — El artículo se autodescribe como "a stochastic cellular automata model", de modo
  que no es una alternativa al AC sino un AC con función de transición estimada. Además, el
  nombre "Dinamica EGO" es de la plataforma posterior; el artículo de 2002 presenta DINAMICA.
  CORRECCIÓN: «sistemas como CLUE-S \parencite{Verburg2002}, Dinamica, base del posterior
  Dinamica EGO \parencite{SoaresFilho2002}, y CA-Markov, en los que la función de transición no
  se fija por reglas sino que se estima estadísticamente a partir de datos históricos
  (regresión logística o pesos bayesianos).»
- cap03:275 — Resumen textual (ADS/Ecological Modelling): "the application of logistic regression
  to calculate the spatial dynamic transition probabilities". La genealogía de la tesis es
  correcta. Observación: que la arquitectura admita indistintamente regresión logística o WoE
  está documentado en Soares-Filho et al. (2001), DINAMICA software (SIBGRAPI,
  DOI 10.1109/sibgrapi.2001.963114: "the application of logistic regression or weights of
  evidence") y en la documentación de Dinamica EGO, no en el artículo de 2002. Conviene añadir
  esa cita a la cláusula correspondiente.

## vanVliet2016
Metadatos: CORRECTOS (Crossref 10.1016/j.envsoft.2016.04.017 — Environmental Modelling &
Software 82:174-182, 2016; seis autores en el orden del .bib).

| Afirmación | Etiqueta |
|---|---|
| cap03:388 31 % sin validación; cuando la reporta, una sola comparación de ajuste sobre los mismos datos de calibración | SOBREEXTENDIDA |

El 31 % es textual: "Of the reviewed model applications, thirty-one percent did not report any
validation" (resumen, repositorio VU Amsterdam y ScisPace). La segunda mitad no está sostenida
por el resumen, que afirma algo distinto: "Validation of model results is predominantly based on
locational accuracy assessment, while a small fraction of the applications assessed the accuracy
of the generated land-use or land-cover patterns". El solapamiento calibración/validación es lo
que la tesis atribuye acto seguido a pontius2008comparing, donde sí está documentado. Texto
completo de pago, sin copia abierta en el primer intento.
CORRECCIÓN: «el $31$\% de las aplicaciones examinadas no reporta validación alguna y, cuando la
reporta, se limita predominantemente a la exactitud locacional: solo una fracción pequeña evalúa
la exactitud de los patrones de uso y cobertura generados.»

## Nota sobre numeración de líneas
El lote tiene líneas desfasadas para las primeras citas del capítulo 3. Valores actuales:
Clarke1998 -> 26-27 (no 18); Gomez2020 -> 59 y 159 (no 56 y 156); Hagenauer2022 -> 63 (no 56);
ArribasBel2014 -> 246 (no 238). El resto coincide.

## Resumen contable
- Referencias revisadas: 11. Metadatos correctos: 11. Inexistentes: 0.
- Afirmaciones: 18 -> CORRECTA 12, SOBREEXTENDIDA 5, INCORRECTA 0, NO VERIFICABLE 1.
- Correcciones concretas propuestas: 5 (cap03:26, cap03:332, cap03:43, cap03:388, cap05:242).
- Fuentes consultadas: API de Crossref (9 DOIs + 1 búsqueda bibliográfica), API de Semantic
  Scholar (resúmenes), copia abierta del PDF de Agterberg y Cheng (2002) en ige.unicamp.br,
  documentación Arc-WofE del Geological Survey of Canada, catálogo ScienceDirect/Elsevier del
  libro de 1994, repositorios institucionales de VU Amsterdam y Utrecht, ADS para
  Soares-Filho et al. (2002), Scholarpedia y copia abierta del PDF de Ojala et al. (2002),
  breviario del COESPO de Querétaro y Bustamante (IIEC-UNAM) para el censo.
- Acceso fallido: texto completo de Gomez2020 (MDPI, bloqueo tras dos intentos) y de
  vanVliet2016 (de pago).
