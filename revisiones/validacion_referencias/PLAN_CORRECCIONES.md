# Plan de correcciones derivado de la validación de referencias

Estado: COMPLETO Y APLICADO. Cubre los ocho lotes, es decir las 79 referencias que tenía el
`.bib` al iniciar la validación. Cinco entradas nuevas (`Lloyd1982`, `Smith1978`,
`PackardWolfram1985`, `Woebbecke1995`, `Meyer1998`) se añadieron como consecuencia de la
propia validación; con las dos inexistentes fuera, el total queda en 82.

Cada entrada indica el archivo y la línea, qué está mal, qué dice realmente la fuente y
cuál es la corrección. Las marcadas como VERIFICADO POR EL ORQUESTADOR se comprobaron una
segunda vez de forma independiente, sin confiar en el reporte del validador.

---

## Estado de avance

Todo aplicado, y todo verificado de forma independiente antes de tocar el `.tex`.

En la primera tanda: las dos referencias inexistentes (A1 y A2), la regresión del commit
posterior a la entrega (B1 y B2), el río Mixcoac (C1), las tres fases de SLEUTH (D2), el
algoritmo de K-Means (D2bis), el acrónimo y los coeficientes de SLEUTH (D2ter), el código
de Gigalopolis (D2quater), HSV (D2quinquies), Gong y Weng (F3), el cuantificador de van
Vliet (F2 y F4bis), «decenas de modelos» (F4ter), la fila de R1 (G1bis), la nota de R3 en
la tabla (I2) y cuatro fichas del `.bib`.

En la segunda tanda: las siete atribuciones falsas restantes (D1 y D3 a D8), las tres
escalas mal transcritas (E1 a E3), las ocho sobreextensiones (F1 y F4 a F9 más G2), las
trece fichas pendientes del `.bib` (sección H) y las cifras oficiales de la NOM (I1). La
única entrada que se cierra sin cambio es F3bis: la descripción de CLUE-S como idoneidad
por regresión logística más asignación por demanda es la canónica del sistema y está
documentada en su manual de acceso libre; lo que no la sostiene es el resumen del artículo,
que no es lo mismo que estar mal.

Estado final verificado: la tesis compila limpia en 132 páginas, sin citas indefinidas ni
referencias cruzadas rotas. La auditoría bibliográfica sale en 82 entradas, 82 citadas, 82
impresas, cero huérfanas, cero fantasmas y cero duplicados. El corrector ortográfico no
reporta dedazos, tildes faltantes ni errores gramaticales.

---

## A. Referencias que no existen

Son el hallazgo más serio. Una entrada bibliográfica que no corresponde a ninguna
publicación real no se puede defender de ninguna manera: si un sinodal la busca, no la
encuentra.

### A1. `aguilar2003urbanization` — `cap01_introduccion.tex:22` APLICADO

VERIFICADO POR EL ORQUESTADOR contra el índice completo de *Cities* 20(1), 2003, vía Crossref.

El número real contiene siete artículos, en las páginas 1-2, 3-21, 23-29, 31-39, 41-49,
51-64 y 65-74. No existe ninguno titulado «Urbanization, population growth, and employment
in México», ni ninguno firmado por Aguilar en solitario, ni ninguno en las páginas 3-19 que
declara la entrada. Lo que sí existe en ese hueco es Aguilar, Ward y Smith Sr,
«Globalization, regional development, and mega-city expansion in Latin America: Analyzing
Mexico City's peri-urban hinterland», pp. 3-21, DOI 10.1016/S0264-2751(02)00092-6.

Ese artículo real tampoco sirve como sustituto: trata de la **megaciudad** de México y su
periferia periurbana, es decir lo contrario de una ciudad intermedia.

Se revisó también un candidato que existe y sí es de Aguilar, «Crecimiento urbano y
especialización económica en México», con María Isabel Vázquez, *Investigaciones
Geográficas* núm. 42, 2000, DOI 10.14350/rig.59116. No sirve: analiza componentes
principales sobre población ocupada por sector en 1990, en 101 ciudades, para identificar
funciones económicas dominantes. Es especialización funcional, no concentración del
crecimiento, y no dice nada sobre la literatura de modelado.

APLICADO. Entrada borrada del `.bib`. La oración sostenía dos cosas y se resolvieron por
separado:

1. que las ciudades intermedias mexicanas concentran buena parte del crecimiento reciente
   del país. **Suprimida**, por no tener fuente real que la respalde;
2. que están poco representadas en la literatura de modelado urbano. **Conservada**, ahora
   con la cifra verificada de `Wahyudi2016` (ochenta y ocho aplicaciones clasificadas por
   región, al menos la mitad en Estados Unidos y China, escasas en América Latina) y una
   remisión al capítulo 3, que es donde la tesis documenta el vacío para el caso mexicano.

### A2. `DelgadoLopez2018` — `cap05_modelo_crecimiento_urbano.tex:44` APLICADO

VERIFICADO POR EL ORQUESTADOR contra el índice completo de *Estudios Demográficos y Urbanos*
33(1), 2018, vía Crossref.

El número tiene veinte entradas y ninguna es de Delgado y López. Las páginas 173-205 que
declara la entrada ni siquiera están libres: caen dentro del artículo de Ojeda y González
Ramírez, que ocupa 169-211.

La propia entrada ya traía una señal de alarma: el campo `note` dice «DOI no localizado en el
repositorio de Estudios Demográficos y Urbanos».

APLICADO. Entrada borrada del `.bib` y cita retirada de `cap05:44`. No hizo falta reescribir
nada: esa afirmación ya iba acompañada de `Huacuz2018Metropolizacion`, verificada, que
documenta la expansión de la zona metropolitana de Querétaro entre 1990 y 2010. La frase se
sostiene igual con una sola cita.

---

## B. Regresión introducida después de la entrega

VERIFICADO POR EL ORQUESTADOR con `git show 7cd82e5` y comparación contra el PDF entregado.

El PDF que se envió a la Dra. Lárraga está limpio: dice «del orden de 10^10 combinaciones
posibles» con el intervalo entero [0, 100]. El commit `7cd82e5` («wip», 5 de septiembre),
posterior a la entrega, reescribió el capítulo 3 y en esa reescritura se perdieron datos y
entraron erratas.

### B1. Marcador sin rellenar — `cap03_estado_del_arte.tex:87` APLICADO

El texto dice literalmente `$a^{b}$` donde debería ir el tamaño del espacio de búsqueda.
Clarke (2008) dice «Five parameters control SLEUTH's behavior entirely, each with a possible
integer value between 0 and 100», de donde salen 101^5 ≈ 1,05 × 10^10.

Acción: restituir `$10^{10}$` y el intervalo entero [0, 100], que es el dato que justifica la
cifra y que la reescritura eliminó.

### B2. Erratas de la misma reescritura APLICADO

- `cap03:86` «Alo largo» → «A lo largo»; «razgos» → «rasgos».
- `cap03` «crecimeinto» (dos veces), «temrinos», «metrica», «despues», «conclusion», «tambien».
- `cap03:16` «modelado urbano computaciones» → «computacional».
- `cap01:16` «La urbanizacipon en un proceso social» → «La urbanización es un proceso social»;
  «de el total de población» → «del total de la población».
- `cap01:38` «rios» → «ríos»; «ejemlplo» → «ejemplo»; «por ejemplo … por ejemplo» repetido.

Nota: el corrector `tools/revisar_ortografia.py` no cazó estas porque busca transposiciones
de letras contiguas, y «razgos» o «urbanizacipon» son sustituciones, no transposiciones.
Conviene ampliarlo.

---

## C. Afirmaciones falsas sobre hechos

### C1. El río Mixcoac — `cap01_introduccion.tex:38` APLICADO

VERIFICADO POR EL ORQUESTADOR por búsqueda independiente.

La tesis dice «el rio Mixcoac que desemboca en lagos cercanos a la ciudad de México». Es
falso en presente: el Mixcoac fue entubado, kilómetro y medio de cauce, y hoy es la avenida
que lleva su nombre. El Churubusco, al que alimentaba, fue desviado en 1952 precisamente
para que dejara de abastecer los lagos de Xochimilco, Mixquic y Tláhuac.

Además el ejemplo no está en Seto et al. (2012), que es un artículo conceptual sobre China,
India y retos globales, y no menciona México.

Acción: eliminar el ejemplo completo. La cláusula que sí cita a Seto («los efectos alcanzan
a los territorios con los que la ciudad se conecta mediante flujos») se sostiene sola, aunque
conviene ajustar el léxico: el artículo habla de flujos de personas, bienes y servicios
económicos, y de capital, no de «recursos».

---

## D. Atribuciones a fuentes que no dicen lo que se les atribuye

Son el tipo de error más peligroso en un examen, porque se comprueban abriendo el artículo.

### D1. Moore y Von Neumann atribuidos a Wolfram — `cap02_marco_teorico.tex:99` APLICADO

El pie de la figura de vecindades dice «con base en la definición de Wolfram (1984)». Ese
artículo de *Nature* no contiene las palabras «Moore» ni «Neumann» ni una sola vez; su única
noción de vecindad es el radio r sobre una línea de sitios.

Acción: atribuirlo a Packard y Wolfram (1985), *Two-dimensional cellular automata*, J. Stat.
Phys. 38:901-946, DOI 10.1007/BF01010423, que sí trata reglas bidimensionales de 5 y 9
vecinos. Requiere entrada nueva en el `.bib`. Alternativa mínima: suprimir la atribución.

### D2. Las tres fases de SLEUTH atribuidas a Clarke y Hoppen — `cap03:90` APLICADO

Confirmado por dos lotes independientes. El artículo de 1997 no contiene «coarse», «fine» ni
«brute force». El esquema gruesa/fina/final aparece con esos nombres en Silva y Clarke
(2002) —«Results from the three phases of the calibration mode (Coarse, Fine, and Final
calibrations)»— y es a ese artículo al que Dietzel y Clarke (2007) se lo acreditan.

Aplicado: la atribución pasa a `Silva2002`, y `ClarkeHoppen1997` queda para la
automodificación, que es lo que ese artículo sí formula y que la tesis ya usa bien seis
líneas más abajo.

Queda una observación menor sin aplicar: el intervalo entero [0, 100] y las 10^10
combinaciones son correctos (101^5 ≈ 1,05 × 10^10), pero proceden de la documentación de
SLEUTH (Jantz et al. 2003; pSLEUTH), no del artículo de Lisboa y Oporto. Si se quiere
blindar del todo, añadir una de esas fuentes al `\parencite`.

### D2bis. El algoritmo de K-Means atribuido a MacQueen — `cap02:61` APLICADO

Es el hallazgo más expuesto de todo el ejercicio ante un sinodal de computación, porque la
confusión MacQueen/Lloyd es de manual.

La tesis decía que el K-Means «propuesto por MacQueen» procede asignando cada observación
al centroide más próximo, recalculando los centroides y repitiendo. Ese es el algoritmo por
lotes, y MacQueen no solo no lo propone: se lo acredita a otras personas.

VERIFICADO POR EL ORQUESTADOR sobre el PDF original del Quinto Simposio de Berkeley. En la
sección 3.6, titulada «A two-step improvement procedure», p. 294: «The method of obtaining
partitions with low within-class variance which was suggested by Forgy and Jennrich (see
section 1.1) works as follows. Starting with an arbitrary partition into k sets, the means
of the points in each set are first computed. Then a new partition of the points is formed
by the rule of putting the points into groups on the basis of nearness to the first set of
means.» Y en la p. 282: «a simple and elegant method […] was noticed by Edward Forgy and
Robert Jennrich, independently of one another, and communicated to the writer sometime in
1963».

El k-means propio de MacQueen es secuencial, de una pasada y en línea, p. 283: «the k-means
procedure consists of simply starting with k groups each of which consists of a single
random point, and thereafter adding each new point to the group whose mean the new point is
nearest».

Aplicado: la frase distingue ahora las dos cosas. A MacQueen se le acredita el nombre y el
criterio de varianza intraclase, que sí son suyos, y la variante por lotes que el código
realmente usa se atribuye a Forgy y Jennrich vía el propio MacQueen, con la formalización
de Lloyd (1982), entrada nueva en el `.bib`. El pie del algoritmo del apéndice pasó a
llamarse «K-Means por lotes» para que no contradiga al texto.

### D2ter. El acrónimo SLEUTH y los cinco coeficientes — `cap02:93` APLICADO

Dos anacronismos en una sola oración, y el segundo contradecía a la propia tesis. El
acrónimo SLEUTH no aparece en Clarke y Gaydos (1998); se fija en Silva y Clarke (2002), que
es exactamente lo que el capítulo 3 dice dos capítulos más adelante. Y los cinco
coeficientes de crecimiento se formulan en Clarke, Hoppen y Gaydos (1997); a Clarke y Gaydos
(1998) la literatura le atribuye los cuatro tipos de crecimiento, no los coeficientes.

Aplicado: la oración reparte ahora cada cosa a su fuente, y de paso el capítulo 2 deja de
contradecir al 3.

### D2quater. El código de SLEUTH atribuido al artículo de 1998 — `cap03:227` APLICADO

Un artículo de 1998 en *IJGIS* no publica código fuente. El código en C y los datos de
prueba se distribuyen por el sitio del proyecto Gigalopolis (NCGIA/UCSB), y quien lo declara
de forma citable es Clarke (2008): «the model has been open source since its outset, and a
complete set of source code and test data can be downloaded from the website».

Aplicado: el hecho se mantiene, que es lo que sostiene el argumento de apertura del código,
pero la cita pasa a `Clarke2008` y el texto nombra el proyecto en vez del artículo.

### D2quinquies. HSV atribuido a Gonzalez y Woods — `cap02:34` APLICADO

VERIFICADO POR EL ORQUESTADOR sobre el índice detallado que publican los propios autores:
«HSV» aparece cero veces y «HSI» cinco. El capítulo de color del libro cubre RGB, CMY/CMYK,
HSI y CIELAB. HSV no está.

Importante: no se corrigió cambiando HSV por HSI. El extractor de características de la
tesis usa HSV de verdad, así que sustituirlo habría hecho que el marco teórico dejara de
describir el código. Lo que se hizo fue darle a HSV su fuente propia, Smith (1978), que es
su origen, y dejar a Gonzalez y Woods a cargo de RGB y CIELAB, que sí están en el libro.

De paso, «luminancia $L^*$» pasó a «luminosidad $L^*$»: el libro dice *lightness*, y $L^*$
es una función no lineal de la luminancia, no la luminancia misma.

### D3. Exceso de verde y exceso de rojo atribuidos a Tucker — `cap02_marco_teorico.tex:36` APLICADO

NGRDI sí es de Tucker (1979). ExG es de Woebbecke, Meyer, Von Bargen y Mortensen (1995),
DOI 10.13031/2013.27838, y ExR de Meyer et al. (1998).

Acción: separar las tres atribuciones. Requiere dos entradas nuevas en el `.bib`.

### D4. Percepción de los habitantes atribuida a Hernández Guerrero — `cap05:40` APLICADO

La tesis dice «la calidad ambiental que perciben sus habitantes». El artículo dice lo
contrario: «los recorridos se efectuaron por personas no residentes a las AGEB para evitar
sesgos en la valoración», con observadores capacitados, 2 400 cuestionarios de valoración
visual. Además el ámbito es el área urbana del municipio de Querétaro, no la zona
metropolitana.

Acción: «estudiado mediante valoración visual sistemática de la calidad ambiental del área
urbana municipal por observadores capacitados».

### D5. FLUS descrito como regresión logística o pesos bayesianos — `cap03:41` APLICADO

FLUS usa una red neuronal artificial, y el artículo contrapone explícitamente ambas cosas:
«it is stronger than other simple methods such as the logistic regression». La propia tesis
lo dice bien 250 líneas después, en `cap03:288`, así que además es una inconsistencia interna.

Acción: sacar FLUS de esa cláusula.

### D6. Prueba por pares atribuida a Bonham-Carter 1989 — `cap03:310` APLICADO

La documentación oficial del método atribuye tanto la prueba χ² por pares como la ómnibus al
capítulo 9 de Bonham-Carter (1994). El capítulo de 1989 no fue accesible y no hay forma de
dar la página.

Acción: mover la atribución a `BonhamCarter1994`, que además ya está citada ahí por la regla
del 15 %.

### D7. Autocorrelación espacial atribuida a Bonham-Carter 1994 — `cap02:201` APLICADO

El capítulo 9 documenta independencia condicional entre capas, no autocorrelación espacial
entre celdas. La tesis ya cita correctamente `Legendre1993` y `Brenning2012` para ese punto
en `cap03:344`.

Acción: usar ahí las mismas dos referencias.

### D8. Licenciamiento de Dinamica EGO atribuido a Soares-Filho 2002 — `cap03:217` APLICADO

El artículo de 2002 no dice nada sobre licencias. El hecho es cierto y está en
`https://dinamicaego.com/license/`. Además, en la misma oración, «TerrSet liberaGIS, liberado
sin costo en diciembre de 2024» va sin ninguna cita.

Acción: citar la licencia oficial con fecha de consulta, y añadir fuente para TerrSet.

---

## E. Escalas y definiciones mal transcritas

### E1. Quinto tramo del Information Value — `cap02_marco_teorico.tex:186` APLICADO

La tesis dice que por encima de 0,50 el IV «se considera muy fuerte». Siddiqi lo califica de
**sospechoso**, y su recomendación práctica es investigar la variable y posiblemente
excluirla.

VERIFICADO POR EL ORQUESTADOR sobre los datos del modelo: seis de las siete variables caen en
ese tramo.

| variable | IV |
|---|---|
| `distance_urban` | 3,27 |
| `neighbor_density_3x3` | 2,90 |
| `neighbor_density_5x5` | 1,68 |
| `neighbor_density_7x7` | 1,24 |
| `local_fragmentation` | 1,23 |
| `urban_gradient` | 1,16 |
| `nearest_cluster_size` | 0,14 |

Atenuante importante: la tesis **no publica** estos valores en ninguna tabla. El IV solo
aparece dentro de la ecuación de transición, como peso normalizado. De modo que basta con
corregir la etiqueta del tramo.

Dos argumentos de defensa, por si preguntan en el examen:

1. El umbral de Siddiqi viene del scoring crediticio, donde un IV alto delata fuga de
   información porque la variable ya contiene la respuesta. Aquí no hay fuga: los predictores
   se derivan del mismo mapa binario que el objetivo, y que la urbanización ocurra junto a lo
   ya urbano es el mecanismo que se quiere capturar, no un artefacto.
2. El IV entra normalizado, ω_k = IV_k / Σ IV_l, así que solo importan las proporciones entre
   variables. La magnitud absoluta no altera el modelo.

### E2. Banda inferior de la escala de Landis y Koch — `cap02:315` APLICADO

La tesis dice «κ < 0,20, acuerdo pobre». La escala real separa dos bandas: 0,00-0,20 es
«leve» (slight) y solo κ < 0 es «pobre» (poor). Las otras cuatro bandas están bien.

### E3. Letras de los componentes del FoM — `cap02:332-336`, `cap05:380`, `cap06:449` APLICADO

VERIFICADO POR EL ORQUESTADOR en cuatro artículos independientes que reproducen la ecuación
original.

Pontius define A como la omisión (cambio observado predicho como permanencia) y D como la
falsa alarma (permanencia observada predicha como cambio). La tesis usa A como falsa alarma y
C como omisión, o sea invertidas respecto de la fuente que cita.

Es matemáticamente equivalente en el caso binario, porque el término C de Pontius (categoría
de destino equivocada) es cero y B/(A+B+C) iguala a B/(A+B+C+D). Pero el pie de figura dice
«con base en la definición de Pontius» y las letras dicen lo contrario que él.

Las letras están dibujadas dentro de tres imágenes; solo una se regenera con script
(`generate_cap02_figures.py`), las otras dos se hicieron con IA generativa.

Acción recomendada, de bajo costo: nota al pie en `cap02` declarando que la notación se
adapta al caso binario, en el que el término de categoría de destino equivocada es nulo, y
que las letras A y C no coinciden con las del artículo original.

---

## F. Fuentes sobreextendidas

No son falsedades, son citas que abarcan más de lo que la fuente sostiene.

### F1. Kappa e IoU atribuidos al espectro de Pontius APLICADO

Afecta a `cap01:191`, `cap06:520`, `cap07:26`, `cap07:62`, `cap07:64`, `front/abstract.tex:24`
y `front/resumen.tex:24`.

Pontius et al. (2008) no reporta Kappa (cero ocurrencias de la palabra en el texto) ni IoU
como métrica separada. Solo documenta FoM.

Acción: dejar a Pontius únicamente el FoM de 0,317, el Kappa de 0,445 en la escala de Landis
y Koch (como ya se hace bien en `cap06:200`) y el IoU de 0,633 como complemento.

### F2. «En particular esto se observa en México» — `cap01:16` APLICADO

Ninguna de las tres referencias de esa cita lo sostiene. UN-Habitat 2020 sí documenta que la
huella urbana crece más rápido que la población (1,8 veces frente a 1,2 entre 1990 y 2015 en
una muestra de 200 ciudades), pero no formula nada específico de México. Seto 2012 dice de
México justo lo contrario en magnitud: «total forecasted area of urban expansion in Mexico is
small».

Acción: suprimir el remate o respaldarlo con fuente mexicana.

### F3. Series históricas multidecadales atribuidas a Gong 2013 — `cap02:24` APLICADO

FROM-GLC es un mapa global de época única: tres cuartas partes de las escenas son de
alrededor de 2010, una cuarta de alrededor de 2000, y solo 18 escenas son anteriores a 1998.
No respalda «series históricas con décadas de profundidad».

En la misma oración había un segundo problema: `Weng2002` quedaba pegado a la mención de
«la familia Landsat», y su resumen no nombra ningún sensor, de modo que un lector podía
entender que el artículo declara haber usado Landsat. Lo verificable de Weng es que es un
estudio de cambio de uso de suelo entre 1989 y 1997 en el delta del Zhujiang.

Aplicado: Gong queda a cargo de la cartografía global a decenas de metros, que es lo que su
título declara, y Weng a cargo de los estudios de cambio de uso de suelo que comparan un
mismo sitio en fechas separadas por años. Ninguna de las dos citas afirma ya nada que su
fuente no sostenga.

### F3bis. CLUE-S descrito con detalle no verificable — `cap03:288` CERRADO SIN CAMBIO

La tesis dice que CLUE-S «acopla una idoneidad derivada de regresión logística con una
asignación iterativa gobernada por la demanda sectorial de suelo». La descripción es la
canónica del sistema, pero el resumen de Verburg et al. (2002), que es lo único accesible
sin suscripción, no nombra ninguna técnica estadística. Riesgo bajo: no es una cifra ni un
hallazgo ajeno, y la afirmación se reparte entre cuatro sistemas. Si se quiere blindar,
el manual de CLUE-S es de acceso libre y sí documenta ambos componentes.

### F4. Gomez 2020 como ejemplo de caja negra con alta concordancia — `cap01:117`, `cap03:155` APLICADO

En Rionegro, uno de sus dos casos, el IoU de Gomez et al. es de 0,374 a 0,454, **por debajo
del 0,633 de esta tesis**. Y el artículo no compara contra ningún baseline, así que no
«reporta mejoras».

Acción: reformular para que el contraste no dependa de que el aprendizaje automático gane en
concordancia. El argumento fuerte de la tesis es la interpretabilidad, no la superioridad
métrica del rival.

### F4bis. El cuantificador de la validación de un solo período — `cap01:117` APLICADO

Nota posterior: el lote 3a elevó este punto de «no verificable» a **incorrecto**, y con
razón. No es solo que las fuentes no sostuvieran el cuantificador: es que Clarke y Gaydos
(1998) hacen lo contrario de lo que se les atribuía. Calibran con mapas históricos sobre
fechas de control repartidas en décadas; Clarke (2008) las enumera: 1965, 1980, 1990 y 2006.
Atribuirles validación en un solo quinquenio era falso, no impreciso. La reescritura ya
hecha lo resuelve, porque el cuantificador pasó a van Vliet y Clarke quedó solo como ejemplo
de modelo de caja blanca, que es lo que es.

La frase decía que los modelos de caja blanca «en la mayoría de los casos se validan sobre
un único período quinquenal», atribuido a Clarke, Santé y Aburas. Es el cuantificador que
sostiene la justificación central de la tesis, las cinco ventanas, y ninguna de las tres
fuentes lo respalda. Santé y Aburas además no son modelos, son revisiones.

VERIFICADO POR EL ORQUESTADOR: el dato sí existe y es exhibible, pero en van Vliet et al.
(2016), ya presente en el `.bib`. Resumen literal, leído en el repositorio de Wageningen:
«Validation of model results is predominantly based on locational accuracy assessment […]
Of the reviewed model applications, thirty-one percent did not report any validation.»

Aplicado: el peso del dato pasa a `vanVliet2016` con la cifra literal del 31 %, y Santé y
Aburas quedan nombradas como lo que son, las revisiones que sistematizan a los modelos.

### F4ter. «Las decenas de modelos que examinan» — `cap03:511` APLICADO

Cuantificador sin verificar sobre un artículo de pago sin copia abierta. Suprimido, sin
pérdida de contenido.

### F5. Coste computacional atribuido a Wang 2021 — `cap01:117` APLICADO

El artículo no reporta tiempos, coste ni hardware. Solo que el ajuste se estabilizó tras unas
650 iteraciones. Para el coste computacional existe `Jantz2003`, que sí lo documenta con cita
textual («Over a week of processing time was required to complete the calibration», sobre un
clúster Beowulf de 16 nodos).

### F6. Hardware gráfico atribuido a Ma 2019 — `cap07:95` APLICADO

El artículo menciona GPU una sola vez, al narrar el triunfo de AlexNet en ImageNet 2012. La
parte de los conjuntos anotados sí está bien sostenida.

### F7. «Dos décadas» atribuido a Clarke 2008 — `cap03:18` APLICADO

El conteo de más de cien aplicaciones es literal. «Dos décadas» no: el capítulo habla de una
década.

### F8. LBP «propuestos por» Ojala 2002 — `cap02:40` APLICADO

El propio artículo remite el operador original a Ojala, Pietikäinen y Harwood (1996); de 2002
es la generalización circular (P, R).

### F9. Prioridad del AC probabilístico — `cap02:91` APLICADO

La perturbación estocástica del potencial de transición es de White y Engelen (1993), no de
1997. `White1993` ya está en el `.bib`.

---

## G. Contradicciones internas

### G1. Suárez y Delgado cumplen R4 — `cap01:119` frente a `cap03:472` APLICADO

`cap01:119` dice que no cumplen R3 ni R4. La tabla de `cap03:472` dice R1 = no y R4 = sí. La
tabla tiene razón: usan regresión logística binomial y reportan e interpretan los
coeficientes uno por uno, lo cual es exactamente lo que pide R4.

Acción: armonizar `cap01:119` con la tabla.

### G1bis. Los tres trabajos mexicanos frente a R1–R4 — `cap01:119` APLICADO

Ver G1. La frase decía «satisfacen parcialmente R1 y R2, pero no R3 ni R4», y la tabla de
`cap03:480-482` dice otra cosa en tres de las cuatro columnas: R2 lo cumplen los tres por
completo, R4 lo cumple Suárez y Delgado y lo cumplen parcialmente los otros dos, y solo R3
es «no» para los tres.

El lote 3b afinó además la columna R1, que en una primera pasada quedó mal redactada. Se
leyó el PDF completo del libro de Ramírez Hernández, 310 páginas, y no aparece «kappa», ni
figura de mérito, ni IoU, ni comparación simulado contra observado; la única mención de
validación es el epígrafe metodológico «h) Validación de resultados y calibración de
parámetros». O sea que R1 = no para ese trabajo, como dice la tabla, y el único de los tres
con R1 parcial es Chihuahua. La frase quedó ajustada a eso.

De paso, el mismo lote confirmó por texto literal la caracterización que la tesis hace de
ese libro en `cap03:408`, incluida la afirmación de que no es un modelo inspirado en SLEUTH:
el libro se autodescribe como «un modelo econométrico con simulación espacial basado en una
simulación de Montecarlo, aplicación de autómatas celulares y mecanismos de transición
probabilística mediante modelos Logit», y SLEUTH solo aparece en su revisión de literatura,
siempre como trabajo ajeno.

### G2. Población de la zona metropolitana — `cap01:69` APLICADO

El millón y medio de habitantes se atribuye a CONAPO 2018, pero ese documento reporta
1 323 640 para 2015 y habla de rebasar el millón. La cifra de 1,5 millones es del Censo 2020.

Acción: dejar la población a cargo de `INEGI2020Censo` y que CONAPO respalde la delimitación
y la tasa de crecimiento del 2,8 % anual, que sí es suya.

---

## H. Correcciones al `.bib` APLICADAS

Todas las filas de esta tabla están aplicadas y verificadas contra Crossref o contra la
fuente institucional que publica el documento.

| clave | campo | corrección |
|---|---|---|
| `aguilar2003urbanization` | toda la entrada | borrar, no existe |
| `DelgadoLopez2018` | toda la entrada | borrar, no existe |
| `CONABIO2024Porque` | `year` | 2024 → 2022 |
| `CONAPO2018Delimitacion` | `author` | un solo grupo de llaves con «and» dentro; biblatex imprime «and SEDATU and INEGI» como nombre literal. Separar en tres grupos y poner SEDATU primero |
| `CONAPO2018Delimitacion` | tipo | `@online` → `@report`, con `edition` y `address` |
| `Seto2012teleconnections` | `author`, `doi` | listar los diez autores; añadir 10.1073/pnas.1117622109 |
| `Waddell2002` | `title` | falta la coma serial: «Transportation, and Environmental» |
| `HernandezGuerreroOsorno2018` | `author` | sin guion en los apellidos, protegidos con llaves |
| `Herold2003` | `author` | `Nicholas C. Goldstein` → `Noah C. Goldstein` |
| `Gong2013` | `author` | `Huang, Xiaomei` → `Huang, Xiaomeng` APLICADO |
| `JimenezLopez2018` | `volume`, `number` | `volume = {12}` → `volume = {10}` más `number = {12}`; añadidos `issn` y `note`. APLICADO |
| `Batty2005` | `address`, `isbn` | añadir `Cambridge, MA` e `isbn = {9780262025836}` |
| `Montejano2023` | `number` | `number = {e1734}` → `pages = {e1734}`, que es como lo registra Crossref |
| `Chihuahua2023` | `pages` | `1--19` → `e4103`, es artículo numerado |
| `ONUHabitat2016TendenciasMexico` | `year` | 2016 → 2017, la página está fechada así |
| `White1997` | `number` | falta `2` |
| `UNHabitat2020` | varios | añadir `publisher`, `address` e `isbn = {978-92-1-132872-1}` |
| `YinYan1988` | varios | opcional: `editor = {Bonnard, C.}`, `publisher = {Balkema}`, `address = {Rotterdam}` |
| nuevas | — | `PackardWolfram1985`, `Woebbecke1995`, `Meyer1998` |

---

## I. Mejoras al alza, opcionales

### I1. Cifras oficiales de la NOM — `cap01:24` APLICADO

La norma dice literalmente que «solo 567 municipios (22.94%) cuentan con un instrumento en
materia de ordenamiento territorial y/o desarrollo urbano, de los cuales casi 45%, fueron
publicados hace más de 15 años». Sustituir «cobertura incompleta» y «una parte» por esas
cifras convierte una frase impresionista en un dato de fuente primaria.

### I2. Nota sobre el «bajo demanda» de Jiménez López — `cap03:472` APLICADO

La tabla marca R3 como «parcial» para Jiménez López et al. (2018), pero el artículo se
autodescribe en su propio resumen como «software especializado de código abierto». Dicho
así, la celda parece contradecir a la fuente. VERIFICADO POR EL ORQUESTADOR: esa frase está
literalmente en el resumen del PDF original.

Aplicado: se añadió una nota al pie de la tabla que deja explícito que «parcial» no
cuestiona la descripción de los autores, sino que registra que la distribución es bajo
solicitud y no por repositorio público versionado, que es lo que R3 exige para que un
tercero pueda reejecutar sin intermediarios.
