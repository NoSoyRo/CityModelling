# VALIDACIÓN DE FUENTES — LOTE 8 (12 referencias)

## AgterbergCheng2002
**Metadatos:** CORRECTO
- Evidencia: `https://api.crossref.org/works/10.1023/A:1021193827501` → *Conditional Independence Test for Weights-of-Evidence Modeling*, Agterberg, Frederik P.; Cheng, Qiuming; *Natural Resources Research* 11(4):249–255, dic. 2002. Confirmado en ADS: `https://ui.adsabs.harvard.edu/abs/2002NRR....11..249A/abstract`
- Correcciones: ninguna.

**Afirmaciones:**
1. *(cap03:310 — «la prueba exacta de Agterberg y Cheng»)* — **CORRECTA.** El artículo dice literalmente: «This new test is **exact** and simpler to use than other tests including the Kolmogorov-Smirnov test and various chi-squared tests». URL: `https://www.ige.unicamp.br/sdm/ArcSDM2/documentation/CI_Agterberg.pdf`
   - Beneficio adicional: el mismo PDF **corrobora el 15 % que la tesis atribuye a Bonham-Carter (1994)**: «For example, in Bonham-Carter (1994, **p. 316**), it is argued that T should not exceed n by more than **15 %**», y lo llama «the informal 15 % rule of the earlier version of the omnibus test». La p. 316 cae dentro del capítulo 9, así que la frase de la tesis es defendible tal como está.

**Riesgo en examen:** Nulo. Es la cita mejor sustentada del lote.

---

## Almeida2003
**Metadatos:** CORRECTO
- Evidencia: Crossref confirma los siete autores en ese orden, *Computers, Environment and Urban Systems* 27(5):481–509, 2003. El PDF del artículo lleva el encabezado impreso «Computers, Environment and Urban Systems 27 (2003) 481–509»: `http://www.complexcity.info/files/2011/06/batty-ceus-2003.pdf`
- Correcciones: ninguna.

**Afirmaciones:**
1. *(cap03:270 — la ruta WoE «se desarrolló y estimó empíricamente» en las aplicaciones urbanas de Almeida 2003)* — **CORRECTA.** Resumen literal: «we propose a structure for simulating urban change based on **estimating land use transitions using elementary probabilistic methods which draw their inspiration from Bayes' theory and the related 'weights of evidence' approach**… applied to a medium-sized town, **Bauru**… **1979–1988**… statistical validation… multiple resolution fitting procedure». El cuerpo del artículo desarrolla $W^+$/$W^-$ citando a Bonham-Carter (1994). La versión de INPE explicita el vínculo con la arquitectura: «These land use change probabilities drive **a CA model DINAMICA**». URLs: `https://zenodo.org/records/3830970`, `http://urlib.net/sid.inpe.br/mtc-m12@80/2006/04.27.19.49`, PDF citado arriba.

**Riesgo en examen:** Nulo. La genealogía que narra la tesis es exactamente la que documentan los propios autores.

---

## Chihuahua2023
**Metadatos:** INCORRECTO (un campo)
- Evidencia: la cita oficial impresa en el propio PDF de la revista es «García-Ramírez, P., Alatorre-Cejudo, L. C., & Bravo-Peña, L. C. (2023). … *Investigación y Ciencia de la Universidad Autónoma de Aguascalientes*, **31(90), e4103**». URL: `https://revistas.uaa.mx/investycien/article/download/4103/3780`. Crossref confirma autores, año (2023-09-29), número 90 y que **no hay paginación tradicional**.
- Correcciones: `pages = {1--19}` → `pages = {e4103}` (artículo con número de artículo, no rango de páginas). Volumen 31 y número 90: correctos.

**Afirmaciones:**
1. *(cap01:119 — satisface parcialmente R1/R2, no R3/R4)* — **NO APLICA A VERIFICACIÓN EXTERNA.** Es un juicio de la tesis contra su propia rúbrica R1–R4, no un dato atribuido al artículo. Consistente con lo que el artículo hace y no hace.
2. *(cap03:421 — SVM + regresión logística + CA-Markov, diez núcleos, cinco cuencas nombradas, proyecciones 2030/2040/2050)* — **CORRECTA en todos sus términos.** Resumen: «utilizando **regresión logística (RL) y CA-Markov** incorporado en Modelador de Cambio en el Terreno (LCM) integrado en IDRISI. Se produjeron mapas de uso de suelo … utilizando **Máquina de Soporte Vectorial (SVM)**» y escenarios «**2030, 2040 y 2050**». Conclusiones: «particularmente en la cuenca **Laguna Bustillos y de los Mexicanos, río Conchos-Ojinaga, río Conchos-Presa de la Colina, río Conchos-Presa el Granero y río San Pedro**» — las cinco cuencas, con los mismos nombres. Y la tabla 5 enumera exactamente **diez** núcleos: Pedro Meoqui, Santa Eulalia, Manuel de Ojinaga, Delicias, Chihuahua, Cuauhtémoc, Juan Aldama, José Mariano Jiménez, Hidalgo del Parral y Santa Rosalía de Camargo. URL: `https://revistas.uaa.mx/investycien/article/download/4103/3780`
3. *(cap03:472 — celda «parcial / ✓ / no / parcial» en tab:cuadrante-mexico)* — **CORRECTA** en lo verificable: el artículo **no valida** el modelo contra un mapa observado independiente (reporta Pseudo R² de McFadden > 0.2 y ROC de la regresión logística, es decir bondad de ajuste de la etapa de idoneidad, más Kappa de la *clasificación*), lo cual respalda el «no» en R3.

**Riesgo en examen:** Bajo. Solo hay que corregir `pages`.

---

## Clarke2008
**Metadatos:** CORRECTO
- Evidencia: el índice del propio libro lista «3 A Decade of Cellular Urban Modeling with SLEUTH: **47** — Unresolved Issues and Problems — Keith C. Clarke» (`https://www.lincolninst.edu/app/uploads/legacy-files/pubfiles/planning-support-systems-for-cities-and-regions-chp.pdf`); la página del autor y el preprint confirman «Ch. 3 … Lincoln Institute of Land Policy, Cambridge, MA, **pp 47-60**» (`https://people.geog.ucsb.edu/~kclarke/books.htm`, `http://www.ncgia.ucsb.edu/projects/gig/Pub/SLEUTHPapers_Nov24/Clarke_Lincoln2008.pdf`).
- Correcciones: ninguna.

**Afirmaciones:**
1. *(cap03:18 — «más de cien aplicaciones documentadas … lo que lo mantuvo durante dos décadas como la referencia del campo»)* — **PARCIAL.** La primera mitad es literal: «SLEUTH is a mature cellular automaton urban model, now applied to **over 100 different cities and regions**» y «At last count, the number of applications of the model to cities and regions was **over one hundred**». La segunda mitad **no** es de Clarke: el capítulo habla de **una década** («With close to 100 applications over a decade»; «SLEUTH has enjoyed a longer than typical lifetime as an urban model»), nunca de dos décadas ni de ser «la referencia del campo». URL: `http://www.ncgia.ucsb.edu/projects/gig/Pub/SLEUTHPapers_Nov24/Clarke_Lincoln2008.pdf`
   - Matiz recomendado: atribuir a Clarke solo el conteo y dejar el «dos décadas» como afirmación propia de la tesis, o suprimirlo.
2. *(cap03:86 — fuerza bruta sobre cinco coeficientes, «del orden de $a^{b}$ combinaciones posibles», «cada una evaluada con simulaciones Monte Carlo»)* — **INCORRECTA, y por dos motivos distintos.**
   - **(a) Defecto grave de redacción, no de fuente:** el `.tex` contiene literalmente un **marcador sin rellenar**, `$a^{b}$`. Verificado en `cap03_estado_del_arte/cap03_estado_del_arte.tex:87`. Con el dato del propio Clarke —«Five parameters control SLEUTH's behavior entirely, each with a possible **integer value between 0 and 100**»— el valor correcto es $101^{5} \approx 1{,}05\times10^{10}$ (o $\sim\!10^{10}$).
   - **(b) Atribución inexacta:** «cada una evaluada con simulaciones Monte Carlo» contradice a Clarke, que describe una calibración **por fases con acotamiento progresivo**, no exhaustiva: «Three phases are used in calibration… At first, **large increments** of the parameters are used… the highest scoring parameter sets are used to 'bracket' the next round… and a last round with unit increments». Clarke sí respalda «fuerza bruta» como método («Successive search using **brute force methods** (Silva and Clarke, 2002) or genetic algorithms») y sí respalda el Monte Carlo por conjunto («averaged over several Monte Carlo iterations»), pero no que se evalúen las $\sim\!10^{10}$ combinaciones.
   - Corrección propuesta: «…lo que da del orden de $10^{10}$ combinaciones posibles \parencite{Silva2002,Clarke2008}; la calibración no recorre ese espacio de forma exhaustiva, sino en tres fases de incrementos decrecientes, y cada conjunto evaluado se promedia sobre varias iteraciones Monte Carlo».

**Riesgo en examen:** **Alto por el `$a^{b}$`.** Un sinodal lo leerá como placeholder olvidado y pondrá en duda la revisión del capítulo completo. Es lo primero que hay que arreglar del lote.

---

## Cohen1960
**Metadatos:** CORRECTO
- Evidencia: `https://api.crossref.org/works/10.1177/001316446002000104` → Cohen, Jacob; *A Coefficient of Agreement for Nominal Scales*; *Educational and Psychological Measurement* 20(1):37–46, abril 1960.
- Correcciones: ninguna.

**Afirmaciones:**
1. *(cap02:301 — κ corrige la concordancia observada por la esperada por azar, con $\kappa=(p_o-p_e)/(1-p_e)$)* — **CORRECTA.** Es la definición canónica del artículo y el título mismo («nominal scales») coincide con el uso que la tesis le da (dos clases nominales urbano/no-urbano). La fórmula de $p_e$ que la tesis desarrolla en `eq:pe` es la especialización estándar a una matriz 2×2 y no se atribuye a Cohen como algo distinto.

**Riesgo en examen:** Nulo.

---

## Gomez2020
**Metadatos:** CORRECTO
- Evidencia: Crossref y el PDF del artículo: Gómez, Jairo A.; Patiño, Jorge E.; Duque, Juan C.; Passos, Santiago; *Remote Sensing* 12(1):109. Encabezado impreso «Remote Sens. **2020, 12, 109**», «Published: 28 December 2019». Citar el año 2020 (el del volumen) es correcto. URL: `https://res.mdpi.com/d_attachment/remotesensing/remotesensing-12-00109/article_deploy/remotesensing-12-00109.pdf`
- Correcciones: ninguna.

**Afirmaciones:**
1. *(cap01:117 — modelos de caja negra que «alcanzan concordancia espacial alta en los casos publicados, pero sus pesos internos no son legibles como evidencia espacial»)* — **PARCIAL, y el matiz importa.**
   - Lo que **sí** sostiene el artículo: la caracterización de opacidad es de los propios autores, «data-driven models are often **not as interpretable**, but they capture complex dynamics» (p. 3 del PDF).
   - Lo que **no** sostiene: «concordancia espacial alta» no se cumple en los dos casos. En Valledupar la huella urbana binaria da IoU = 7.96e-01 a 8.07e-01, accuracy = 0.975, F1 = 0.887; pero en **Rionegro** da **IoU = 3.74e-01 a 4.54e-01**, F1 = 0.624, ZNCC = 0.61. Los autores lo dicen sin rodeos: «The performance of the binary urban footprint **is not as good** as the population distribution… the estimation performance in Valledupar is **much better** than in Rionegro… In both cities, **the worst metric was the IoU**».
   - Además el artículo se presenta como habilitador de análisis de sensibilidad, no como caja negra cerrada: «it is very easy to train new machine learning models using different explanatory input variables **to assess their impact**».
2. *(cap03:55 — «llevó el aprendizaje automático al modelado espaciotemporal del crecimiento urbano a partir de percepción remota»)* — **CORRECTA.** El resumen describe exactamente eso: regresión espaciotemporal con aprendizaje automático sobre el archivo Landsat y GHSL. Casos: Valledupar y Rionegro, Colombia. Única cautela: el verbo «llevó» puede leerse como reclamo de primicia; el artículo no reclama ser el primero.
3. *(cap03:155 — «reportan **mejoras** de concordancia espacial en sus propios casos de estudio»)* — **INCORRECTA para Gomez2020.** El artículo **no compara su desempeño contra ningún modelo de AC ni contra baseline alguno**; reporta métricas absolutas de acuerdo. Sus ventajas declaradas frente al AC son cualitativas, no métricas: «Unlike widely used growth models based on cellular automata… **it does not require to define rules a priori**… Secondly, it is very easy to train new machine learning models». No hay «mejora» reportada.
   - Corrección propuesta: «…reportan concordancia espacial aceptable en sus propios casos de estudio, sin comparación contra modelos de autómata celular».

**Riesgo en examen:** **Medio-alto, y de un tipo incómodo.** El IoU de 0.374–0.454 de Gomez2020 en Rionegro es **inferior al IoU de 0.633 de esta tesis**. Si un sinodal abre el artículo, el ejemplo elegido como «caja negra de alta concordancia» resulta peor que el modelo propio y el contraste retórico se invierte. Conviene reformular el párrafo para que no dependa de que el aprendizaje automático sea el que gana en concordancia.

---

## HernandezGuerrero2015
**Metadatos:** CORRECTO
- Evidencia: la firma impresa en el artículo es «**Juan Hernández-Guerrero**» (Facultad de Ciencias Naturales, UAQ), con «Revista de Geografía Norte Grande, **61: 45-64 (2015)**» en la cabecera. URL: `https://pdfs.semanticscholar.org/674e/8e0c62ca6e2cbec0e0213e18b54e51b35444.pdf`. Crossref y el DOI SciELO confirman número 61, pp. 45–64.
- Correcciones: ninguna. (Dialnet lo indiza como «Juan Alfredo Hernández Guerrero»; el artículo publicado firma «Juan Hernández-Guerrero», así que el `.bib` sigue la fuente autorizada.)

**Afirmaciones:**
1. *(cap05:40 — «estudiado desde la calidad ambiental que **perciben sus habitantes**»)* — **INCORRECTA.** El estudio excluyó deliberadamente a los residentes. Texto literal: «Cabe señalar que los recorridos se efectuaron por **personas no residentes a las AGEB para evitar sesgos en la valoración**. Los participantes, mayores de 18 años, **recibieron capacitación previa** y fueron seleccionados… debido a sus conocimientos básicos sobre análisis visual del paisaje». Se levantaron 2 400 cuestionarios de **valoración visual por observadores capacitados** (6 por AGEB en 304 AGEB), con termómetro y anemómetro incluidos. No hay encuesta de percepción a habitantes. URL: `https://pdfs.semanticscholar.org/674e/8e0c62ca6e2cbec0e0213e18b54e51b35444.pdf`
   - Segundo matiz: el ámbito es el **área urbana del municipio de Querétaro (AUMQ)**, no la Zona Metropolitana.
   - Corrección propuesta: «…estudiado mediante valoración visual sistemática de la calidad ambiental del área urbana municipal por observadores capacitados~\parencite{HernandezGuerrero2015}».

**Riesgo en examen:** **Medio.** Es una descripción equivocada de lo que hizo el trabajo citado, del tipo que la Dra. Lárraga detecta de inmediato por ser literatura local. La corrección es de una línea.

---

## LandisKoch1977
**Metadatos:** CORRECTO
- Evidencia: `https://api.crossref.org/works/10.2307/2529310` y `https://pubmed.ncbi.nlm.nih.gov/843571/` → Landis, J. Richard; Koch, Gary G.; *The Measurement of Observer Agreement for Categorical Data*; *Biometrics* 33(1):159–174, marzo 1977.
- Correcciones: ninguna.

**Afirmaciones:**
1. *(cap02:315 — la escala completa)* — **PARCIAL. Cuatro de cinco bandas son correctas; la última está mal etiquetada.** La escala de Landis y Koch es: **< 0.00 → poor**; **0.00–0.20 → slight**; 0.21–0.40 → fair; 0.41–0.60 → moderate; 0.61–0.80 → substantial; 0.81–1.00 → almost perfect. URLs: `https://www.ncbi.nlm.nih.gov/sites/books/NBK52665/table/ch3.t5/?report=objectonly` y `https://ehealthinformation.ca/web/default/files/wp-files/isern-98-02.pdf` (reproduce la tabla completa).
   - La tesis dice «$\kappa < 0{,}20$, acuerdo **pobre**», lo que fusiona dos bandas distintas y traslada la etiqueta «poor» (reservada a κ < 0) al tramo 0.00–0.20, que es «slight» (leve/escaso).
   - Corrección propuesta: «…$0{,}00$–$0{,}20$, acuerdo leve; y $\kappa < 0$, acuerdo pobre».
   - Cautela adicional recomendable: los propios autores califican las divisiones de «clearly arbitrary». La tesis ya amortigua esto con «escala de interpretación **de uso común**», lo cual es correcto y prudente.
2. *(cap06:200 — Kappa 0,445 = acuerdo moderado, parte baja del rango 0,41–0,60)* — **CORRECTA.** 0,445 cae en la banda «moderate» (0.41–0.60) y está efectivamente en su tercio inferior. La banda usada aquí sí es la de Landis y Koch.

**Riesgo en examen:** Bajo, pero es exactamente el tipo de detalle que un sinodal estadístico marca. Arreglo de media línea en `cap02`.

---

## Legendre1993
**Metadatos:** CORRECTO
- Evidencia: `https://api.crossref.org/works/10.2307/1939924` → Legendre, Pierre; *Spatial Autocorrelation: Trouble or New Paradigm?*; *Ecology* 74(6):1659–1673, sept. 1993.
- Correcciones: ninguna.

**Afirmaciones:**
1. *(cap03:344 — «la **sobreestimación del desempeño** que resulta de tratar observaciones espacialmente dependientes como independientes está documentada en… \parencite{Legendre1993,Brenning2012}»)* — **PARCIAL.** Legendre respalda la premisa, no la conclusión. Resumen literal (depositado en Crossref por el editor): «autocorrelated data **violate the assumption of independence of most standard statistical procedures**… Then, **proper statistical testing** in the presence of autocorrelation is briefly discussed». Es decir: Legendre documenta la invalidación de las **pruebas de significancia** (error de tipo I inflado), no la sobreestimación de métricas de desempeño en validación de modelos espaciales, que es un tema posterior y propio de la validación cruzada espacial. URL: `https://api.crossref.org/works/10.2307/1939924` (campo `abstract`).
   - En la pareja de citas, **Brenning2012** es el que sostiene la parte de «sobreestimación del desempeño». La solución más limpia es separar las dos atribuciones: Legendre para la violación del supuesto de independencia, Brenning para el optimismo del desempeño estimado.
   - No se pudo leer el texto completo del artículo (Wiley/ESA de pago); la evidencia es el resumen oficial depositado por el editor, que es suficiente para el matiz señalado.

**Riesgo en examen:** Bajo. Es un afinamiento de atribución, no un error factual.

---

## Seto2012
**Metadatos:** CORRECTO
- Evidencia: `https://api.crossref.org/works/10.1073/pnas.1211658109` → Seto, Karen C.; Güneralp, Burak; Hutyra, Lucy R.; *PNAS* 109(40):16083–16088. Confirmado en PMC: `https://pmc.ncbi.nlm.nih.gov/articles/PMC3479537/`
- Correcciones: ninguna.

**Afirmaciones:**
1. *(cap01:16 — la superficie urbana se expande más rápido que la población «en particular esto se observa en México»)* — **PARCIAL, con un problema concreto en la cláusula sobre México.**
   - La primera parte **sí** aparece en Seto 2012: «Today, urban areas around the world are expanding on average **twice as fast than their populations** (2, 3)». Matiz: los propios autores lo presentan como hallazgo de terceros con referencias 2 y 3, no como resultado propio; es una cita de segunda mano, aceptable pero no ideal.
   - La cláusula sobre México **no** está respaldada por Seto 2012 y de hecho apunta en dirección contraria. Lo único que el artículo dice de México es: «total forecasted area of urban expansion in **Mexico is small**, but the probability that specific locations in Mexico will undergo urban expansion is high (Fig. 1C)». Eso es una relación inversa entre probabilidad y magnitud, no una tasa de expansión superficial superior a la demográfica. URL: `https://pmc.ncbi.nlm.nih.gov/articles/PMC3479537/`
   - Si el dato mexicano proviene de `UNHabitat2020` o `Angel2012`, conviene citarlos por separado para esa cláusula y dejar a Seto solo en la afirmación global.
2. Observación editorial: la oración de `cap01:16` contiene errores tipográficos y sintácticos graves («La urbanizacipon **en** un proceso social», «de **el** total de población», ausencia de puntuación entre cláusulas). Es la tercera oración del capítulo 1.

**Riesgo en examen:** **Medio.** El dato global es sólido; la cláusula sobre México es indefendible con esa fuente, y está en la apertura de la tesis, que es donde más se lee.

---

## SoaresFilho2002
**Metadatos:** CORRECTO
- Evidencia: `https://api.crossref.org/works/10.1016/S0304-3800(02)00059-5` → Soares-Filho, Britaldo Silveira; Cerqueira, Gustavo Coutinho; Pennachin, Cássio Lopes; *Ecological Modelling* 154(3):217–235, 2002. Confirmado en ADS: `https://ui.adsabs.harvard.edu/abs/2002EcMod.154..217S/abstract`
- Correcciones: ninguna. (El título en minúsculas «dinamica» es el estilo del editor; la mayúscula del `.bib` es aceptable.)

**Afirmaciones:**
1. *(cap03:41 — Dinamica EGO entre los sistemas LUCC cuya función de transición «emerge de regresiones logísticas o pesos bayesianos sobre datos históricos»)* — **CORRECTA para la parte de regresión logística**, que es la del artículo de 2002: «the application of **logistic regression** to calculate the spatial dynamic transition probabilities» (`https://ui.adsabs.harvard.edu/abs/2002EcMod.154..217S/abstract`). Los «pesos bayesianos» corresponden a la implementación posterior de Dinamica EGO, documentada en `https://dinamicaego.com/dokuwiki/doku.php?id=calc_w._of_e._probability_map`. Como enunciado de conjunto sobre una familia de sistemas, es defendible.
2. *(cap03:217 — «se distribuye de forma gratuita para uso académico y comercial»)* — **CORRECTA en el hecho, INCORRECTA en la atribución.** El artículo de 2002 no dice nada sobre licenciamiento ni distribución; además describe «DINAMICA», una generación anterior al producto «Dinamica EGO». El hecho sí se verifica en la licencia oficial: «The Product is licensed to you on an As-Is basis… for your personal, **academic/research and commercial** purpose, since not in conflict with the interests of CSR/UFMG» (`https://dinamicaego.com/license/`). Matiz que un sinodal puede usar: la página de inicio del mismo sitio se describe como plataforma «free, and **non-commercial**» (`https://dinamicaego.com/`), la licencia es revocable y prohíbe redistribución y obras derivadas, de modo que «gratuito» sí, pero «software libre» no.
   - Corrección propuesta: citar `https://dinamicaego.com/license/` (con fecha de consulta) en lugar de `\parencite{SoaresFilho2002}` para el enunciado de licenciamiento.
   - **Fuera de este lote pero en la misma oración:** «TerrSet liberaGIS… liberados sin costo en diciembre de 2024» no tiene ninguna cita. Debe llevar una.
3. *(cap03:270 — «la formulación original de SoaresFilho2002 calcula las probabilidades de transición mediante regresión logística, y la arquitectura admite indistintamente ese método o el de Pesos de Evidencia»)* — **PARCIAL.** La primera mitad es literal y correcta (ver cita del resumen arriba). La segunda mitad —que la arquitectura admite ambos métodos indistintamente— **no está en el artículo de 2002**, que solo implementa regresión logística; se documenta en el manual de Dinamica EGO, que por cierto retro-cita a «Soares-Filho et al. 2002, 2004» al describir el método WoE (`https://dinamicaego.com/dokuwiki/doku.php?id=lesson_18`). El enunciado es cierto respecto del software actual, pero no verificable contra la publicación de 2002.
   - Corrección propuesta: «…y la arquitectura, en sus versiones posteriores, admite indistintamente ese método o el de Pesos de Evidencia \parencite{SoaresFilho2004}», o añadir la cita al manual.

**Riesgo en examen:** Bajo-medio. Nada es falso; hay dos enunciados colgados de una fuente que no los contiene.

---

## vanVliet2016
**Metadatos:** CORRECTO
- Evidencia: `https://api.crossref.org/works/10.1016/j.envsoft.2016.04.017` y las fichas institucionales de VU y WUR → seis autores en ese orden, *Environmental Modelling & Software* **82:174–182**, 2016, sin número de fascículo (la ausencia de `number` en el `.bib` es correcta). URL: `https://research.vu.nl/en/publications/a-review-of-current-calibration-and-validation-practices-in-land-/`
- Correcciones: ninguna.

**Afirmaciones:**
1. *(cap03:383)* — se desglosa en dos mitades con veredictos distintos.
   - **Primera mitad — «el 31 % de las aplicaciones examinadas no reporta validación alguna»: CORRECTA, literal.** Resumen oficial: «Of the reviewed model applications, **thirty-one percent did not report any validation**». URL: `https://research.vu.nl/en/publications/a-review-of-current-calibration-and-validation-practices-in-land-/`
   - **Segunda mitad — «cuando la reporta, se limita en general a una sola comparación de ajuste, con frecuencia sobre los mismos datos usados para calibrar»: NO VERIFICABLE.** El resumen dice algo **distinto**: «Validation of model results is predominantly based on **locational accuracy assessment**, while a small fraction of the applications assessed the accuracy of the generated land-use or land-cover **patterns**». Eso contrasta exactitud locacional frente a exactitud de patrón; no afirma «una sola comparación» ni «sobre los mismos datos usados para calibrar».
   - Vías intentadas para leer el texto completo, sin éxito: resolver el DOI a ScienceDirect (HTTP 403); `https://research.wur.nl/files/8926094/A_review_of_current_calibration.pdf` (403); página de aterrizaje de WUR marcada GREEN en OpenAlex pero sin enlace a PDF en el HTML; `hdl.handle.net/2440/106689` de Adelaide (404); API de CORE (redirección sin resultados); `scholar.archive.org` (sin copia); Semantic Scholar devuelve `abstract: null` por supresión del editor.
   - Recomendación: o se reescribe la segunda mitad con lo que el resumen sí dice («la validación se apoya predominantemente en exactitud locacional y solo una fracción pequeña evalúa la exactitud del patrón generado»), o se consigue el PDF por BIDI-UNAM y se localiza la frase exacta antes de defenderla. Tal como está, es la afirmación más expuesta del lote porque el número del 31 % invita al sinodal a abrir el artículo.

**Riesgo en examen:** **Medio-alto en la segunda mitad.** La cifra estrella es correcta, lo que hace más probable que alguien verifique el resto de la oración.

---

# Resumen del lote

Revisadas las **12 referencias** del lote 8, con acceso al texto completo o al resumen oficial del editor en todas.

**Metadatos:** 1 de 12 con error → **Chihuahua2023**, `pages = {1--19}` → **`e4103`** (es artículo numerado, sin rango de páginas; cita oficial 31(90), e4103).

**Afirmaciones:** 22 revisadas → 12 correctas, 6 parciales, 3 incorrectas, 1 no verificable.

**Problemas graves, en orden de urgencia:**

1. **Clarke2008** — `cap03_estado_del_arte.tex:87`: el texto contiene el **marcador sin rellenar `$a^{b}$`**. Valor correcto: $\sim 10^{10}$ combinaciones ($101^5$, pues Clarke dice «each with a possible integer value between 0 and 100»). En la misma oración, «cada una evaluada con simulaciones Monte Carlo» contradice a Clarke, que describe calibración **en tres fases con acotamiento progresivo**, no exhaustiva.
2. **HernandezGuerrero2015** — `cap05_modelo_crecimiento_urbano.tex:40`: dice «la calidad ambiental que **perciben sus habitantes**». El artículo hizo lo contrario: «los recorridos se efectuaron por **personas no residentes** a las AGEB para evitar sesgos», con observadores capacitados. Corregir a «valoración visual por observadores capacitados» y precisar que el ámbito es el área urbana **municipal**, no la zona metropolitana.
3. **Gomez2020** — `cap03:155`: «reportan **mejoras** de concordancia espacial» es falso; el artículo **no compara contra ningún baseline**. Y en `cap01:117`, «concordancia espacial alta» solo aplica a Valledupar (IoU ≈ 0,80); en Rionegro el IoU cae a **0,374–0,454**, por debajo del IoU 0,633 de esta tesis — el ejemplo elegido como contraste se vuelve en contra.
4. **Seto2012** — `cap01:16`: «en particular esto se observa en México» no está en el artículo; lo único que dice de México es que «total forecasted area of urban expansion in Mexico **is small**». Mover esa cláusula a `UNHabitat2020`/`Angel2012`. La misma oración tiene errores tipográficos graves («urbanizacipon», «en un proceso»).
5. **LandisKoch1977** — `cap02:315`: «$\kappa < 0{,}20$, acuerdo **pobre**» mezcla dos bandas. La escala real es **0,00–0,20 = leve (slight)** y **< 0 = pobre (poor)**. Las otras cuatro bandas están bien.
6. **vanVliet2016** — `cap03:383`: el **31 % es literal y correcto**; la segunda mitad («una sola comparación de ajuste, con frecuencia sobre los mismos datos usados para calibrar») **no se pudo verificar** tras seis vías de acceso fallidas, y el resumen oficial dice otra cosa (predominio de exactitud locacional frente a exactitud de patrón).
7. **SoaresFilho2002** — `cap03:217` y `cap03:270`: dos enunciados cuelgan de una fuente que no los contiene (licenciamiento de Dinamica EGO, y que «la arquitectura admite indistintamente» ambos métodos). Ambos son ciertos, pero hay que citar `dinamicaego.com/license` y `SoaresFilho2004`. Además, «TerrSet liberaGIS, diciembre de 2024» va sin ninguna cita.
8. **Legendre1993** — `cap03:344`: sostiene la violación del supuesto de independencia, no la «sobreestimación del desempeño»; esa parte es de **Brenning2012**. Separar las dos atribuciones.

**Sin observaciones:** AgterbergCheng2002 (que además **corrobora** el 15 % que la tesis atribuye a Bonham-Carter, p. 316), Almeida2003, Cohen1960 y las tres celdas de Chihuahua2023 en el cuerpo del texto (SVM + RL + CA-Markov, **diez** núcleos, **cinco** cuencas con nombres exactos, proyecciones 2030/2040/2050: todo verificado literalmente).

**VEREDICTO: REQUIERE CORRECCIONES.**
