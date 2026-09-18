# Resultado rev2 lote 01 — validación de fuentes

Fecha: 2026-09-18. Revalidación literal del texto vigente en los `.tex`.
Fuentes consultadas: API de Crossref (metadatos), texto completo del artículo de
Pontius et al. (2008) servido por Springer Nature Link, PDF abierto del capítulo de
Tobler (1979) alojado en people.geog.ucsb.edu, y resumen JATS de White y Engelen (1997)
registrado en Crossref. Las citas textuales en inglés provienen de esas copias.

---

## 1. `pontius2008comparing`

### Veredicto de metadatos: CORRECTO (verificado en Crossref, DOI 10.1007/s00168-007-0138-2)

| Campo | `referencias.bib` | Crossref | Coincide |
|---|---|---|---|
| Título | Comparing the input, output, and validation maps for several models of land change | idéntico | Sí |
| Revista | The Annals of Regional Science | The Annals of Regional Science | Sí |
| Volumen / número / páginas | 42 / 1 / 11--37 | 42 / 1 / 11-37 | Sí |
| Año | 2008 | número de marzo de 2008 (publicación en línea 16-08-2007) | Sí |
| Editorial | Springer | Springer Science and Business Media LLC | Sí |
| Autores | 20 nombres | los mismos 20, en el mismo orden | Sí |
| DOI | 10.1007/s00168-007-0138-2 | resuelve al artículo correcto | Sí |

No hay nada que corregir en la entrada.

### Base factual verificada en el artículo

Citas textuales del texto completo, para que sirvan de respaldo al redactar:

- «This paper applies methods of multiple resolution map comparison to quantify
  characteristics for **13 applications of 9 different** popular peer-reviewed land change
  models.»
- «The 13 contributions include applications to **12 different locations**, since two
  models apply to The Netherlands, albeit with the data formatted differently.»
- «The figure of merit can range from **0%** [...] **to 100%**.»
- «Figure 4 shows that **Perinet is the only application** where the amount of correctly
  predicted change is larger than the sum of the various types of error, i.e., figure of
  merit is **greater than 50%**.»
- «R-squared is **40%** for the increasing linear relationship (Fig. 6). R-squared is
  **88%**, if we ignore the **two CLUE applications** to Honduras and Costa Rica, which
  are fundamentally different [...] because the two CLUE applications have **heterogeneous
  pixels**.»
- «**All six** of the applications that have a figure of merit **less than 15%** have an
  observed net change of less than 10%.»
- «Figure of merit = B / (A + B + C + D), where **A** is the area of error due to observed
  change predicted as persistence, **B** area of correct due to observed change predicted
  as change, **C** area of error due to observed change predicted as wrong gaining
  category, and **D** area of error due to **observed persistence predicted as change**.»
- «The invitation requested that each LUCC model generates its prediction map based on
  information at or before time 1 [...] For applications where the criterion was not
  satisfied, we asked the participant to describe how the model uses **information
  subsequent to time 1** for calibration.» Y en la discusión: «**Most applications** used
  some information subsequent to time [1] to simulate the change between time 1 and time
  2. Therefore **many of the results** reflect the goodness-of-fit of a **mix of both
  calibration and validation**.»
- No aparece la palabra *kappa* en ninguna parte del artículo.
- Cautela propia de los autores: «model assessment must focus primarily on the performance
  of each model relative to its own data and its own null model, and then secondarily in
  relation to other data and other models. **Even if the goal of this exercise was to rank
  the models according to predictive power, it would be impossible** given the information
  in this article.»

### Tabla de afirmaciones

| # | archivo:línea | Afirmación (resumen) | Etiqueta |
|---|---|---|---|
| 1 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:124` | Trece aplicaciones de nueve modelos sobre doce sitios, comparadas con el mismo procedimiento | CORRECTA |
| 2 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:131`–`135` | La dispersión responde al cambio neto observado; $R^2=0{,}40$, sube a $0{,}88$ al excluir dos aplicaciones de formato heterogéneo | CORRECTA |
| 3 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:161` | La dispersión del FoM «no se explica por la **sofisticación** del modelo» | SOBREEXTENDIDA |
| 4 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:388` | La mayoría de las aplicaciones usó información posterior al tiempo inicial; sus resultados mezclan ajuste y validación | CORRECTA |
| 5 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:424` | El libro de Ramírez Hernández no admite comparación con el rango de desempeño documentado por Pontius | CORRECTA |
| 6 | `cap05_modelo_crecimiento_urbano/cap05_modelo_crecimiento_urbano.tex:380` | Las letras del pie de figura siguen la ecuación del cap. 2, «no la notación original» de Pontius | CORRECTA |
| 7 | `cap06_resultados_analisis/cap06_resultados_analisis.tex:200` | $0{,}317$ en banda intermedia: por encima del subconjunto bajo $0{,}15$ y por debajo del único caso sobre $0{,}50$ | CORRECTA |
| 8 | `cap06_resultados_analisis/cap06_resultados_analisis.tex:235` | Tabla comparativa: trece aplicaciones de nueve modelos; seis bajo $0{,}15$, una sobre $0{,}50$; FoM entre $0$ y $1$ | CORRECTA |
| 9 | `cap06_resultados_analisis/cap06_resultados_analisis.tex:502` | $0{,}317$ se interpreta frente al espectro multi-sitio | CORRECTA |
| 10 | `cap06_resultados_analisis/cap06_resultados_analisis.tex:520` | «ese estudio **no reporta Kappa ni IoU**» | SOBREEXTENDIDA |
| 11 | `cap07_conclusiones/cap07_conclusiones.tex:20` | $31{,}7$\% en la escala porcentual de Pontius; seis bajo $0{,}15$, una sobre $0{,}50$; banda intermedia | CORRECTA |
| 12 | `cap07_conclusiones/cap07_conclusiones.tex:26` | $0{,}317$ en banda intermedia, por encima de $0{,}15$ y por debajo de $0{,}50$ | CORRECTA |
| 13 | `cap07_conclusiones/cap07_conclusiones.tex:62` | FoM $0{,}317$ dentro de la banda intermedia documentada | CORRECTA |
| 14 | `cap07_conclusiones/cap07_conclusiones.tex:79` (el lote indicaba `:64`) | H4: FoM en la banda intermedia del espectro | CORRECTA |
| 15 | `cap07_conclusiones/cap07_conclusiones.tex:84` | Sexta aportación: contextualizar los resultados en el espectro empírico multi-sitio | CORRECTA |
| 16 | `cap07_conclusiones/cap07_conclusiones.tex:126` | Banda intermedia; el contraste con caja negra es cualitativo por sitios, clases y horizontes distintos | CORRECTA |
| 17 | `cap07_conclusiones/cap07_conclusiones.tex:173` | FoM $0{,}317$ en banda intermedia del espectro multi-sitio | CORRECTA |

### Problemas y redacción propuesta

**Afirmación 3 — `cap03:161` — SOBREEXTENDIDA.**
El artículo no habla de «sofisticación». Lo que dice es que no halló otras relaciones
fuertes con la exactitud al considerar los factores de las Tablas 1 y 2, y además declara
explícitamente que ordenar los modelos por poder predictivo «would be impossible» con esa
información. Atribuirle una conclusión sobre la sofisticación le pone en la boca una
comparación que el propio artículo se niega a hacer.

> Redacción propuesta: «...por otro, la comparación multi-sitio de
> \textcite{pontius2008comparing} no encontró relaciones fuertes entre la exactitud
> predictiva y las características de los modelos que documenta, mientras que sí la
> encontró con la magnitud del cambio neto del período evaluado.»

**Afirmación 10 — `cap06:520` — SOBREEXTENDIDA.**
«No reporta Kappa» es exacto: la palabra no aparece en el artículo. «Ni IoU» es
resbaladizo, porque el FoM de Pontius *es* una intersección sobre unión, definida sobre las
celdas de cambio: «the ratio of the intersection of the observed change and predicted
change to the union of the observed change and predicted change». Un sinodal puede señalar
la contradicción aparente. Lo defendible es precisar sobre qué clase se calcula cada índice.

> Redacción propuesta: «...ese estudio no reporta Kappa, y su FoM es ya una intersección
> sobre unión restringida a las celdas de cambio, distinta del IoU sobre el estado urbano
> final que aquí se reporta; de modo que el Kappa de $0{,}445$ se interpreta frente a la
> escala de \textcite{LandisKoch1977} y el IoU de $0{,}633$ se ofrece como complemento
> sobre el estado urbano final.»

La misma corrección aplica a `cap07_conclusiones/cap07_conclusiones.tex:26`, donde se lee
«ninguna de las dos aparece en aquel estudio».

**Afirmación 4 — `cap03:388` — CORRECTA, con una nota de precaución.**
La versión impresa de la discusión dice «information subsequent to time **2**», lo que
choca con el criterio que el propio artículo fija en la metodología («information
subsequent to time 1») y con la evidencia que documenta (LTM, CLUE-S y CLUE usan el cambio
neto correcto tomado del mapa de referencia del tiempo 2). La lectura de la tesis es la
sustantivamente correcta, pero conviene anclarla en el hecho documentado y no en la frase
con errata, y conservar el cuantificador del original («many of the results»).

> Redacción sugerida (opcional, refuerza la cita): «...la mayoría de las aplicaciones
> empleó información posterior al tiempo inicial, en varios casos el cambio neto observado
> en el mapa de referencia del tiempo final, de modo que muchos de sus resultados mezclan
> bondad de ajuste y validación.»

### Advertencia global sobre el uso de esta referencia (no es un error, es un riesgo)

Los autores piden explícitamente que la evaluación de un modelo se haga «primarily [...]
relative to its own data and its own null model», y sólo en segundo término contra otros
estudios. La tesis ya declara que el contraste es cualitativo, lo cual está bien, pero en
ningún punto reporta si el WoE-AC supera a su propio modelo nulo de persistencia, que es el
contraste que ese artículo señala como primario: allí sólo 7 de 13 aplicaciones lo lograron
a resolución fina, y en 12 de 13 el error superó al cambio correctamente predicho. Añadir
esa comparación, barata de calcular con los mapas que ya existen, fortalecería H4 mucho más
que la ubicación en la «banda intermedia».

### Afirmaciones adicionales detectadas en las mismas líneas (fuera de las 17 del lote)

**A. `cap03:128`–`131`: «los valores se reparten entre $0{,}01$ y $0{,}59$» — NO VERIFICABLE.**
Ese par de cifras vive en la Figura 4 del artículo, que no es texto y no pude leer. Sí
encontré corroboración indirecta convergente: dos artículos revisados por pares citan
literalmente ese rango atribuyéndolo a Pontius et al. 2008 («An experiment of 13 modeling
applications showed that FOM ranges from 1% to 59%», Tandfonline 2020; «the figure of merit
observed in other land change models ranged from 1% to 59%», MDPI IJGI 2015). El riesgo de
que sea falso es bajo, pero no lo verifiqué contra el original. Si se conserva, lo prudente
es escribir «entre el $1$\% y el $59$\% según la Figura~4 de ese estudio».

**B. `cap05:380`: «Solo las celdas en transición entran al cálculo» — INCORRECTA.**
Contradice la definición de Pontius y al propio Capítulo 3 de la tesis. En
$\text{FoM}=B/(A+B+C+D)$, el término $D$ es «observed persistence predicted as change»:
celdas donde *no* hubo transición observada, y que sin embargo entran en el denominador.
El Capítulo 3, en `cap03:135`–`138`, lo dice bien: «sí penaliza las celdas estables que el
modelo predice como cambio, porque entran en el denominador». El pie de figura dice lo
contrario.

> Redacción propuesta: «Las celdas donde se observó y se predijo permanencia quedan fuera
> del cálculo; las celdas estables que el modelo predice como cambio sí entran, en el
> denominador.»

---

## 2. `Tobler1979`

### Veredicto de metadatos: CORRECTO (Crossref, DOI 10.1007/978-94-009-9394-5_18, y el propio PDF)

Crossref devuelve «Cellular Geography», *Philosophy in Geography*, pp. 379-386, 1979,
Springer Netherlands (sello moderno de D. Reidel), autor W. R. Tobler. La nota al pie del
PDF original confirma la entrada palabra por palabra: «From: S. Gale and G. Olsson (eds.),
Philosophy in Geography, 379-386. [...] Copyright ©1979 by D. Reidel Publishing Company,
Dordrecht, Holland». Editorial, ciudad, editores, páginas y año coinciden.

### Tabla de afirmaciones

| # | archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:18`–`20` | El uso de autómatas celulares para representar fenómenos geográficos fue propuesto por Tobler | CORRECTA |

Verificado leyendo el PDF completo (seis páginas). El capítulo clasifica los modelos de
cambio de uso de suelo en cinco tipos y destaca el quinto, «the geographical model», en el
que el uso de suelo de una celda depende del de sus vecinas:
$g^{t+\Delta t}_{ij}=F(g^{t}_{ij},n_{ij})$. Dedica secciones enteras a la definición de
vecindad (incluida la de cinco celdas y la de $(2p+1)(2q+1)$) y a la regla de transición,
usa explícitamente el «Life» de Conway como ilustración y cita en la bibliografía a Codd
(1968), *Cellular Automata*. Es, sin ambigüedad, la propuesta de un formalismo de autómata
celular para fenómenos geográficos.

Dos matices que no invalidan la frase pero conviene conocer por si un sinodal los levanta:
Tobler nunca escribe la expresión «cellular automata» en el cuerpo del capítulo, y el texto
es «a condensed translation of a lecture [...] presented [...] Wien, 17 April 1975», de modo
que la formulación es de 1975 aunque la publicación sea de 1979. Además, su propio trabajo
de 1970 sobre Detroit ya simulaba crecimiento urbano sobre una malla de celdas. Si se quiere
blindar el verbo de prioridad, basta con «fue formulado en términos celulares por
\textcite{Tobler1979}».

---

## 3. `White1997`

### Veredicto de metadatos: CORRECTO (Crossref, DOI 10.1068/b240235)

Roger White y Guy Engelen, «Cellular Automata as the Basis of Integrated Dynamic Regional
Modelling», *Environment and Planning B: Planning and Design*, 24(2):235-246, 1997, SAGE.
Todos los campos de la entrada coinciden.

### Tabla de afirmaciones

| # | archivo:línea | Afirmación | Etiqueta |
|---|---|---|---|
| 1 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:23`–`25` | Acopló el autómata a un sistema de información geográfica y a modelos regionales | CORRECTA |

Respaldo textual, del resumen depositado por el editor en Crossref: «We present an
integrated model of regional spatial dynamics consisting of a **cellular automaton-based
model of land use linked both to a geographic information system (GIS) and to standard
nonspatial models of regional economics and demographics**, as well as to a simple model of
environmental change.» La afirmación de la tesis es una paráfrasis fiel. El orden temporal
también es correcto: extiende el esquema de \textcite{White1993}, de los mismos autores.

Sólo se verificó el resumen, no el texto completo; basta para esta afirmación, que es
puramente descriptiva del diseño del modelo. Para el dato de que la aplicación es la isla de
Santa Lucía, que la tesis no menciona, el resumen también lo confirma, por si se quisiera
añadir.

---

## Resumen contable

Sobre las 19 afirmaciones del lote:

| Etiqueta | Conteo |
|---|---|
| CORRECTA | 17 |
| SOBREEXTENDIDA | 2 |
| INCORRECTA | 0 |
| NO VERIFICABLE | 0 |
| **Total** | **19** |

Hallazgos adicionales, detectados en las mismas líneas pero fuera de las 19 afirmaciones
enumeradas en el lote:

| Etiqueta | Conteo | Ubicación |
|---|---|---|
| INCORRECTA | 1 | `cap05_modelo_crecimiento_urbano/cap05_modelo_crecimiento_urbano.tex:380` |
| SOBREEXTENDIDA | 1 | `cap07_conclusiones/cap07_conclusiones.tex:26` |
| NO VERIFICABLE | 1 | `cap03_estado_del_arte/cap03_estado_del_arte.tex:128` |

Ninguna de las tres referencias del lote es inexistente ni tiene errores de metadatos.
