# Resultado de validación — Lote 1 de 8

Fuentes consultadas y abiertas:
- Crossref API: https://api.crossref.org/works/10.1007/s00168-007-0138-2
- Texto completo renderizado del artículo: https://doi.org/10.1007/s00168-007-0138-2
- HAL API (registro hal-03061493): https://api.archives-ouvertes.fr/search/?q=halId_s:hal-03061493
- OpenAlex: https://api.openalex.org/works/doi:10.1007/s00168-007-0138-2
- Semantic Scholar: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1007/s00168-007-0138-2
- Corroboración secundaria del rango de FoM: https://www.tandfonline.com/doi/full/10.1080/15481603.2020.1829376
- Definición de componentes A/B/C/D: https://doi.org/10.1007/978-3-030-90998-7_9

No se logró abrir un PDF de acceso abierto del artículo (Clark University commons, WUR research portal,
CiteSeerX, ResearchGate y HAL no exponen archivo; HAL declara `openAccess_bool: false`). El texto
completo verificado proviene de la versión renderizada de la página del editor vía DOI, que incluye
resumen, introducción, métodos, resultados, discusión, conclusiones, apéndice y referencias. No fue
posible leer los valores numéricos de la Figura 4 ni de las Tablas 1 y 2, lo cual se indica donde importa.

## pontius2008comparing

**Metadatos:** CORRECTO

- evidencia: Crossref (https://api.crossref.org/works/10.1007/s00168-007-0138-2) devuelve
  `"type": "journal-article"`, `"container-title": ["The Annals of Regional Science"]`,
  `"volume": "42"`, `"issue": "1"`, `"page": "11-37"`, `"published-print": 2008-03`,
  `"publisher": "Springer Science and Business Media LLC"`, título exacto
  «Comparing the input, output, and validation maps for several models of land change»,
  y los 20 autores en el mismo orden que la entrada `.bib`: Pontius Jr (suffix "Jr", given
  "Robert Gilmore"), Boersma, Castella, Clarke, de Nijs, Dietzel, Duan, Fotsing, Goldstein, Kok,
  Koomen, Lippitt, McConnell, Mohd Sood, Pijanowski, Pithadia, Sweeney, Trung, Veldkamp, Verburg.
  HAL confirma de forma independiente volumen 42, número 1, páginas 11-37 y revista.
- correcciones: ninguna obligatoria. Refinamientos opcionales, cosméticos, no exigibles:
  `Clarke, Keith` -> `Clarke, Keith C.`; `McConnell, William` -> `McConnell, William J.`;
  `Pijanowski, Bryan` -> `Pijanowski, Bryan C.`. Son las iniciales con que el artículo publicado
  lista a esos autores; Crossref registra las formas breves, de modo que la entrada actual es
  defendible tal como está.
- advertencia útil: existe literatura que cita mal la paginación de este artículo como 11-27
  (por ejemplo el benchmarking de Cybergeo, https://univ-tlse2.hal.science/hal-01447925v1/file/cybergeo-26610-benchmarking-of-lucc-modelling-tools-by-various-validation-techniques-and-error-analysis_cor.pdf).
  La tesis tiene la paginación correcta: 11-37.

**Afirmaciones:**

1. cap01_introduccion:154 (el archivo de entrada indica 143; la frase está en 154-156) —
   «La validación en ventanas desplazadas, en lugar de un único período final, hace posible
   detectar el sobreajuste a una fase específica del crecimiento urbano.» — **PARCIAL**.
   El artículo no propone, no ensaya y no evalúa la validación en ventanas temporales
   desplazadas: cada una de las 13 aplicaciones cubre un solo intervalo. Sí sostiene las dos
   premisas de las que la tesis deriva su conclusión: «One of the most important general lessons
   is that the selection of the place, time, and format of the data must be taken into
   consideration when interpreting the model's performance, because these characteristics can
   have profound influence on the modeling results» y «It encourages scientific rigor because it
   asks the investigators to expose the degree to which calibration information is separated from
   validation information». El término sobreajuste aparece una sola vez y acotado a SLEUTH en
   Santa Barbara: «Ongoing research demonstrates that the model can over-fit the data, leading to
   a prediction of less change than observed». URL: https://doi.org/10.1007/s00168-007-0138-2
   Recomendación: citar a Pontius para la premisa, no para la propuesta. Por ejemplo, «el período
   evaluado influye de forma determinante en el desempeño medido \parencite{pontius2008comparing},
   de modo que la validación en ventanas desplazadas permite detectar el sobreajuste a una fase
   específica del crecimiento urbano», dejando la segunda oración sin cita, como aportación propia.

2. cap01_introduccion:191 — «métricas (FoM, Kappa e IoU) dentro del rango de la literatura LUCC,
   por encima del subconjunto de modelos con FoM inferior a 0,15» — **PARCIAL**.
   La parte de FoM es exacta: «All six of the applications that have a figure of merit less than
   15% have an observed net change of less than 10%». Pero el artículo no reporta Kappa en ningún
   punto (búsqueda de la cadena «kappa» en el texto completo: cero ocurrencias) ni reporta IoU
   como métrica separada, de modo que no aporta rango de referencia para esas dos métricas.
   Además, el propio FoM de Pontius es un índice de intersección sobre unión restringido a las
   celdas de cambio, mientras que la IoU de la tesis es sobre la clase urbana completa: son
   cantidades distintas. URL: https://doi.org/10.1007/s00168-007-0138-2
   Recomendación: «reproduce el patrón de crecimiento con un FoM por encima del subconjunto de
   aplicaciones con FoM inferior a 0,15 \parencite{pontius2008comparing}, y con Kappa e IoU
   consistentes con lo reportado en la literatura LUCC».

3. cap02_marco_teorico:321 — «El Figure of Merit (FoM), que Pontius et al. consolidaron como medida
   de referencia para comparar resultados entre estudios, evalúa exclusivamente el subconjunto de
   píxeles donde ocurre o se predice un cambio.» — **CORRECTA**.
   Sobre el verbo «consolidaron», que es el correcto y no «propusieron»: el artículo atribuye
   explícitamente la métrica a terceros, «The figure of merit is the ratio of the intersection of
   the observed change and predicted change to the union of the observed change and predicted
   change (Klug et al. 1992; Perica and Foufoula-Georgiou 1996)», y la promueve como lenguaje
   común, «we encourage scientists to use the concepts and techniques of this paper in order to
   communicate with a common language that is scientifically rigorous, generally applicable, and
   intellectually accessible». La tesis además lo explicita en cap03:124, «que el propio artículo
   toma de la literatura previa sobre comparación de campos espaciales», lo cual blinda la
   atribución. Sobre el subconjunto de píxeles: la Ecuación 1 es FoM = B/(A+B+C+D) donde los
   cuatro términos son cambio observado o cambio predicho, por lo que las celdas de permanencia
   correctamente predicha quedan fuera. URL: https://doi.org/10.1007/s00168-007-0138-2

4. cap03_estado_del_arte:123 — «reunieron trece aplicaciones de nueve modelos de cambio de uso de
   suelo sobre doce sitios y las compararon con un mismo procedimiento estadístico» — **CORRECTA**.
   Cita textual: «This paper applies methods of multiple resolution map comparison to quantify
   characteristics for 13 applications of 9 different popular peer-reviewed land change models» y
   «The 13 contributions include applications to 12 different locations, since two models apply to
   The Netherlands, albeit with the data formatted differently». Matiz de interés, no un error:
   el envío original fue mayor, «Seven different laboratories contributed maps from 18 different
   applications of 9 different land-change models to 12 different sites from around the world», y
   el artículo presenta «the most representative application from each model». La palabra
   «reunieron» de la tesis es compatible con ambos números.
   URL: https://doi.org/10.1007/s00168-007-0138-2

5. cap03_estado_del_arte:123 — «en esa muestra el FoM crece con el porcentaje de celdas que
   efectivamente cambian de estado, con un R^2 de 0,40 que asciende a 0,88 al excluir dos
   aplicaciones de formato heterogéneo» — **CORRECTA**, y numéricamente exacta.
   Cita textual: «Applications that have larger amounts of observed net change in the reference
   maps tend to have larger predictive accuracies as measured by the figure of merit; R-squared is
   40% for the increasing linear relationship (Fig. 6). R-squared is 88%, if we ignore the two CLUE
   applications to Honduras and Costa Rica, which are fundamentally different than the other
   applications, because the two CLUE applications have heterogeneous pixels that are very few and
   very coarse compared to the other applications». Las dos excluidas son Honduras y Costa Rica con
   CLUE, y el motivo es exactamente el formato de píxel heterogéneo.
   URL: https://doi.org/10.1007/s00168-007-0138-2
   Sobre la primera mitad de la frase, «La dispersión no se explica por las características del
   modelo»: el artículo la respalda de dos maneras, «We could not find other strong relationships
   with prediction accuracy when we considered many possible explanatory factors including those
   in Tables 1 and 2», siendo la Tabla 1 precisamente la de características de los modelos, y
   «Even if the goal of this exercise was to rank the models according to predictive power, it
   would be impossible given the information in this article». Conviene tener presente en el examen
   que un R^2 de 0,40 sobre las 13 aplicaciones deja el 60% de la varianza sin explicar, así que
   la formulación fuerte «no se explica por X sino por Y» es una lectura del hallazgo, no una
   cita. Si un sinodal aprieta, la defensa es la frase de los autores sobre no haber encontrado
   otras relaciones fuertes.

   Hallazgo adicional no numerado, misma oración de cap03:123 — «los valores se reparten entre
   0,01 y 0,59» — **PARCIAL / verificada solo de forma indirecta**. Ninguna oración del artículo
   enuncia ese rango; los valores individuales solo se leen en la Figura 4, que no pude abrir. El
   rango está corroborado por literatura secundaria que cita este artículo: «An experiment of 13
   modeling applications showed that FOM ranges from 1% to 59% (Pontius et al. 2008a), where only
   half of the applications have FOMs greater than 20% and 4 of these exceed 30%», en
   https://www.tandfonline.com/doi/full/10.1080/15481603.2020.1829376, lectura repetida en
   https://doi.org/10.3390/rs12101675 y https://doi.org/10.3390/land11122333. Del artículo mismo
   solo consta el límite superior, «Perinet is the only application where the amount of correctly
   predicted change is larger than the sum of the various types of error, i.e., figure of merit is
   greater than 50%». Recomendación: conservar el rango, que es correcto y ampliamente citado, pero
   redactarlo como lectura de la Figura 4 para no atribuirlo a una frase del artículo.

6. cap03_estado_del_arte:155 — «la comparación multi-sitio de Pontius muestra que la dispersión del
   FoM entre trece aplicaciones no se explica por la sofisticación del modelo, sino por la magnitud
   del cambio neto del período evaluado» — **CORRECTA**, con el mismo matiz de R^2 = 0,40 anotado
   en el punto 5. Cita de respaldo en conclusiones: «The most synthetic result is that LUCC models
   that are applied to landscapes that have larger amounts of observed net change tend to have
   higher rates of predictive accuracy as indicated by the figure of merit, for the voluntary
   sample of applications that we analyzed». Nótese «voluntary sample»: los autores acotan la
   generalidad, «This is a voluntary sample, so it is not assured to be representative of all
   land-change modeling». La tesis ya dice «en esa muestra» y «multi-sitio», lo cual es adecuado.
   URL: https://doi.org/10.1007/s00168-007-0138-2

7. cap03_estado_del_arte:383 — «la mayoría de las aplicaciones empleó información posterior al
   tiempo inicial, de modo que sus resultados mezclan bondad de ajuste y validación» —
   **PARCIAL**, por deriva en el emparejamiento de los términos.
   Lo que dice el artículo, en la discusión: «Most applications used some information subsequent to
   time 2 to simulate the change between time 1 and time 2. Therefore many of the results reflect
   the goodnessof-fit of a mix of both calibration and validation». Dos precisiones.
   Primera: lo que se mezcla, según los autores, es calibración y validación, y el resultado es una
   bondad de ajuste de esa mezcla. La tesis dice que se mezclan «bondad de ajuste y validación»,
   que no es el par correcto. Segunda: la oración de la discusión, tal como aparece en el texto
   renderizado, dice «subsequent to time 2», no «time 1»; el criterio del ejercicio sí está
   formulado respecto al tiempo inicial en la introducción, «The invitation requested that each
   LUCC model generates its prediction map based on information at or before time 1, meaning that
   the LUCC model should not use information subsequent to time 1», y varias aplicaciones usan la
   cantidad correcta del mapa de referencia de tiempo 2 para calibrar. La afirmación de la tesis
   es sustantivamente correcta en cualquiera de las dos lecturas, porque «posterior al tiempo
   inicial» cubre ambas. No pude dirimir si «time 2» es errata del artículo publicado o del
   renderizado, al no acceder al PDF. URL: https://doi.org/10.1007/s00168-007-0138-2
   Recomendación: «la mayoría de las aplicaciones empleó información posterior al tiempo inicial,
   de modo que sus resultados reflejan la bondad de ajuste de una mezcla de calibración y
   validación». Es la formulación literal de los autores y elimina la objeción.

8. cap03_estado_del_arte:408 — «no admite comparación cuantitativa con el rango de desempeño
   documentado en Pontius et al.» — **CORRECTA** en lo que concierne a esta referencia. La cita se
   usa solo como fuente del rango empírico de FoM, que quedó verificado en los puntos 4, 5, 10 y 11.
   La afirmación sobre la ausencia de métricas en RamirezHernandez2021 es ajena a esta referencia y
   corresponde validarla contra esa otra obra.

9. cap05_modelo_crecimiento_urbano:376, pie de la figura F8 — «Componentes del Figure of Merit:
   acierto (B), falsa alarma (A) y omisión (C). Solo las celdas en transición entran al cálculo.
   Imagen elaborada con IA generativa, con base en la definición de Pontius et al.» — **PARCIAL**.
   El contenido conceptual es fiel: solo las celdas en transición entran al cálculo, conforme a la
   Ecuación 1. El problema es la asignación de letras, que contradice a la fuente citada. Pontius
   et al. definen: «where A is the area of error due to observed change predicted as persistence,
   B area of correct due to observed change predicted as change, C area of error due to observed
   change predicted as wrong gaining category, and D area of error due to observed persistence
   predicted as change», con FoM = B/(A+B+C+D). Es decir, en Pontius A es la omisión y D la falsa
   alarma. La tesis, en cap02:332-336, define B como acierto, C como omisión y A como comisión, es
   decir invierte A y C respecto de la fuente y reasigna la falsa alarma de D a A.
   URLs: https://doi.org/10.1007/s00168-007-0138-2 y, para la nomenclatura estándar de los cuatro
   componentes, https://doi.org/10.1007/978-3-030-90998-7_9 («MISSES (A)», «HITS (B)»,
   «WRONG HITS (C)», «FALSE ALARMS (D)»).
   La notación de la tesis es internamente consistente y matemáticamente equivalente en el caso
   binario, donde no existe categoría de destino equivocada y por tanto el término C de Pontius es
   cero, de modo que B/(A+B+C) de la tesis iguala a B/(A+B+C+D) de Pontius. Pero un sinodal que
   abra el artículo encontrará que A significa lo contrario. Recomendación: adoptar la notación de
   Pontius (A omisión, B acierto, D falsa alarma) o añadir una nota al pie en cap02 que diga que
   la notación se adapta al caso binario, en el que el término de categoría de destino equivocada
   es nulo, y que las letras A y C no coinciden con las del artículo original. Es una corrección de
   bajo costo que cierra una vía de ataque innecesaria.

10. cap06_resultados_analisis:200 — «por encima del subconjunto de aplicaciones con FoM inferior a
    0,15 y muy por debajo del único caso que supera 0,50 en esa muestra multi-sitio» —
    **CORRECTA**. Ambos extremos verificados textualmente: «All six of the applications that have a
    figure of merit less than 15%...» y «Perinet is the only application where ... figure of merit
    is greater than 50%». El valor propio de 0,317 está efectivamente entre 0,15 y 0,50.
    Matiz de forma: la expresión «banda intermedia» es una construcción de la tesis, no una
    categoría de Pontius, que no define bandas. Aquí la tesis la define en el mismo enunciado con
    los dos hechos verificados, así que es transparente y defendible.
    URL: https://doi.org/10.1007/s00168-007-0138-2

11. cap06_resultados_analisis:227, Tabla tab:literature_comparison — «Trece aplicaciones de nueve
    modelos. Espectro empírico amplio: seis aplicaciones con FoM inferior a 0,15 y una sola por
    encima de 0,50; el FoM puede variar entre 0 y 1.» — **CORRECTA**.
    Las tres cifras quedan verificadas. Sobre la escala: el artículo enuncia el recorrido en
    porcentaje, «The figure of merit can range from 0%, meaning no overlap between observed and
    predicted change, to 100%, meaning perfect overlap», que expresado como razón es 0 a 1. La
    tesis usa la razón de forma consistente en toda la tabla y declara la equivalencia en cap06:200
    y en cap07:20. URL: https://doi.org/10.1007/s00168-007-0138-2

12. cap06_resultados_analisis:502 — «Su FoM promedio de 0,317 se interpreta frente al espectro
    multi-sitio documentado por Pontius et al.» — **CORRECTA**. Uso legítimo y acotado a FoM.

13. cap06_resultados_analisis:520, H4 — «El FoM promedio de 0,317, el Kappa de 0,445 y el IoU de
    0,633 se ubican en la banda intermedia del espectro multi-sitio de Pontius et al.» —
    **PARCIAL**, por la misma razón del punto 2: el artículo no reporta Kappa ni IoU, de modo que
    no puede proveer banda para esas dos métricas. El propio capítulo ya interpreta bien el Kappa
    por separado en cap06:200, «El Kappa promedio de 0,445 corresponde a un acuerdo moderado según
    la escala de LandisKoch1977», que es la atribución correcta.
    Recomendación: «El FoM promedio de 0,317 se ubica en la banda intermedia del espectro
    multi-sitio de \textcite{pontius2008comparing}; el Kappa de 0,445 corresponde a un acuerdo
    moderado en la escala de \textcite{LandisKoch1977} y el IoU de 0,633 se reporta como
    complemento». URL: https://doi.org/10.1007/s00168-007-0138-2

14. cap07_conclusiones:20 — «equivalente a aproximadamente 31,7% en la escala porcentual usada por
    Pontius et al.; ... (seis aplicaciones con FoM inferior a 0,15 y una sola por encima de 0,50),
    el resultado se ubica en una banda intermedia de dicho marco» — **CORRECTA**. La escala
    porcentual es en efecto la del artículo, «Figure 4 expresses these statistics as percents», y
    las dos cifras del paréntesis están verificadas. La conversión 0,317 a 31,7% es correcta.
    URL: https://doi.org/10.1007/s00168-007-0138-2

15. cap07_conclusiones:26 — «El FoM promedio de 0,317, el Kappa de 0,445 y el IoU de 0,633 ... se
    sitúan en la banda intermedia del espectro multi-sitio de Pontius et al.» — **PARCIAL**,
    idéntico al punto 13, con la misma corrección sugerida.

16. cap07_conclusiones:62 — «un FoM promedio de 0,317 y un Kappa promedio de 0,445 ..., valores en
    la banda intermedia documentada por Pontius et al.» — **PARCIAL**. Dos matices: el Kappa no
    proviene de este artículo, y la expresión «banda documentada por» atribuye a los autores una
    categorización de bandas que no formulan. Recomendación: «un FoM promedio de 0,317, en la banda
    intermedia del espectro documentado por \textcite{pontius2008comparing}, y un Kappa promedio de
    0,445, de acuerdo moderado en la escala de \textcite{LandisKoch1977}».

17. cap07_conclusiones:64, tabla de hipótesis — «con FoM, Kappa e IoU en la banda intermedia del
    espectro de Pontius et al.» — **PARCIAL**, idéntico al punto 13. En una tabla de hipótesis, que
    es lo primero que un sinodal lee, conviene dejar solo FoM asociado a Pontius.

18. cap07_conclusiones:84 — «el contraste del modelo WoE-AC con estudios internacionales mediante
    métricas estandarizadas (FoM, IoU), que contextualiza los resultados en el espectro empírico
    multi-sitio de Pontius et al.» — **CORRECTA**. La oración no afirma que la IoU pertenezca al
    espectro de Pontius, solo que el contraste sitúa los resultados en él. Nota: el espectro que
    aporta el artículo es exclusivamente de FoM.

19. cap07_conclusiones:126 — «ubican al modelo WoE-AC en la banda intermedia del espectro
    multi-sitio de Pontius et al.; el contraste con modelos de caja negra es cualitativo, porque se
    refiere a sitios, clases y horizontes distintos, de modo que no permite cuantificar el costo de
    la interpretabilidad» — **CORRECTA**, y es la formulación más sólida de toda la tesis en esta
    materia, porque coincide con la advertencia central de los autores: «model assessment must
    focus primarily on the performance of each model relative to its own data and its own null
    model, and then secondarily in relation to other data and other models. Even if the goal of
    this exercise was to rank the models according to predictive power, it would be impossible
    given the information in this article, because each model is applied to different data, and the
    data have a large influence on the results». URL: https://doi.org/10.1007/s00168-007-0138-2

20. cap07_conclusiones:173 — «el FoM promedio de 0,317 sobre cinco ventanas quinquenales desplazadas
    se ubica en una banda intermedia del espectro multi-sitio documentado en la literatura» —
    **CORRECTA**. Solo FoM, y «documentado en la literatura» es una atribución prudente.

21. front/abstract.tex:24 — «values within the intermediate band of the multi-site spectrum
    documented by Pontius et al. (six of thirteen applications with FoM below 0.15 and only one
    above 0.50)» — **PARCIAL**. El paréntesis es exacto, «six of thirteen» coincide con «All six of
    the applications that have a figure of merit less than 15%» sobre 13 aplicaciones, y «only one
    above 0.50» con la frase de Perinet. El problema es el sujeto: «values» abarca FoM, Kappa,
    accuracy e IoU, y el artículo solo documenta FoM. Recomendación: «reaches a Figure of Merit of
    0.317, within the intermediate band of the multi-site FoM spectrum documented by
    \textcite{pontius2008comparing} (six of thirteen applications below 0.15 and only one above
    0.50), together with a Kappa coefficient of 0.445, an accuracy of 0.722 and an IoU of 0.633,
    sustained across all five windows». URL: https://doi.org/10.1007/s00168-007-0138-2

22. front/resumen.tex:24 — versión en español de la anterior — **PARCIAL**, por la misma razón y
    con la corrección paralela. Las cifras del paréntesis, «seis de trece» y «una sola por encima de
    0,50», son exactas.

**Riesgo en examen:** bajo en metadatos y en cifras, que resisten verificación línea por línea,
incluidos los dos valores de R^2 y el conteo de seis aplicaciones bajo 0,15, que están textualmente
en el artículo; el riesgo real, moderado y fácil de cerrar, está en dos puntos de atribución que un
sinodal con el artículo en la mano puede señalar: presentar a Pontius et al. como fuente de banda de
referencia también para Kappa e IoU, métricas que ese artículo no reporta, y usar letras A y C en los
componentes del FoM con significado invertido respecto de la Ecuación 1 del artículo citado.
