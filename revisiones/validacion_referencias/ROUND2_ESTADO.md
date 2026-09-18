# Segunda ronda de validación de referencias

Objetivo: dejar el aparato de citas cerrado antes de que Rodrigo reescriba el estilo del
documento. La primera ronda se hizo sobre el texto anterior; esta se hace sobre el texto ya
corregido, de modo que no repite, comprueba.

Alcance: el documento completo. Nueve lotes.

| lote | tramo | refs | afirmaciones | estado |
|---|---|---|---|---|
| `rev2_01` | cap03 al apéndice | 3 | 19 | CERRADO |
| `rev2_02` | cap03 al apéndice | 10 | 18 | CERRADO |
| `rev2_03` | cap03 al apéndice | 11 | 18 | CERRADO |
| `rev2_04` | cap03 al apéndice | 11 | 18 | CERRADO |
| `rev2_05` | cap03 al apéndice | 11 | 18 | CERRADO |
| `rev2_06` | cap03 al apéndice | 11 | 18 | CERRADO |
| `rev2pre_01` | cap01, cap02, resumen y abstract | 18 | 24 | CERRADO |
| `rev2pre_02` | cap01, cap02, resumen y abstract | 19 | 24 | CERRADO |
| `rev2pre_03` | cap01, cap02, resumen y abstract | 18 | 23 | CERRADO |

Total: 180 afirmaciones atribuidas a terceros.

Reglas impuestas a cada validador: si no se pudo abrir la fuente, la etiqueta es NO
VERIFICABLE y no se infiere nada; hay que declarar qué se consultó en cada caso (Crossref,
resumen del editor, PDF abierto); presupuesto de tres o cuatro consultas de red por
referencia, porque en la primera ronda dos agentes se quedaron sin recursos a media tarea.

---

## Lote 1: Pontius, Tobler, White 1997 — CERRADO

Reporte completo en `resultado_rev2_01.md`.

Conteo: 17 CORRECTAS, 2 SOBREEXTENDIDAS, 0 INCORRECTAS, 0 NO VERIFICABLES sobre las 19
afirmaciones enumeradas. Ninguna referencia inexistente. Metadatos exactos en las tres,
incluidos los veinte autores de Pontius en el orden correcto.

Se verificaron contra el texto original, palabra por palabra, las cifras de las que depende
la lectura de todos los resultados: trece aplicaciones de nueve modelos sobre doce sitios,
$R^2$ de 0,40 que sube a 0,88 al excluir las dos aplicaciones de CLUE con píxeles
heterogéneos, seis aplicaciones por debajo de 0,15 y una sola por encima de 0,50.

### Correcciones aplicadas

**1. `cap05:380` — contradicción interna, corregida.** El pie de la figura del FoM decía
«Solo las celdas en transición entran al cálculo». Contradecía la Ecuación del capítulo 2 y
el párrafo de `cap03:137`, que dicen bien que las celdas estables predichas como cambio sí
entran en el denominador. El pie quedó alineado con los otros dos capítulos.

**2. `cap03:161` — sobreextensión, corregida.** Se atribuía a Pontius que la dispersión del
FoM «no se explica por la sofisticación del modelo». El artículo no dice eso y declara
explícitamente que ordenar los modelos por poder predictivo «would be impossible» con su
información. Ahora dice que no halló relaciones fuertes entre la exactitud y los factores de
los modelos que documenta, y que sí la halló con la magnitud del cambio neto.

**3. `cap06:520` y `cap07:26` — sobreextensión, corregida.** Se afirmaba que el estudio «no
reporta Kappa ni IoU». Lo de Kappa es exacto: la palabra no aparece en el artículo. Lo del
IoU no, porque el FoM de Pontius es en sí una intersección sobre unión, restringida a las
celdas de cambio. El texto ahora distingue las dos métricas en lugar de negarlas: el FoM es
una intersección sobre unión sobre las celdas de cambio, el IoU de la tesis se mide sobre el
estado urbano final.

**4. `cap03:129` — cifra trazable.** El rango de 0,01 a 0,59 vive en la Figura 4 del
artículo, que es una imagen. No se pudo abrir el PDF (Springer tras el muro; los
repositorios de la VU, WUR y HAL tras Cloudflare). La corroboración indirecta sí existe y se
comprobó: varios artículos revisados por pares citan «1% to 59%» atribuyéndolo a Pontius
2008, aunque lo describen erróneamente como rango «teórico». La tesis lo usaba bien, como
rango empírico de las trece aplicaciones. Se añadió «según su Figura~4» para que el lector
sepa dónde verificarlo.

---

## Lotes 2, 3 y 4 — CERRADOS

Reportes en `resultado_rev2_02.md`, `resultado_rev2_03.md` y `resultado_rev2_04.md`.

Conteo agregado sobre 54 afirmaciones: 45 CORRECTAS, 7 SOBREEXTENDIDAS, **0 INCORRECTAS**,
2 NO VERIFICABLES. Ninguna referencia inexistente; los 32 metadatos coinciden con Crossref o
con el catálogo del editor.

Las cuatro piezas de las que depende el argumento de la tesis salieron limpias, todas con
cita literal contra la fuente primaria:

- **El 31 % de van Vliet es textual**: «Of the reviewed model applications, thirty-one
  percent did not report any validation». Es el cuantificador que justifica el protocolo de
  cinco ventanas.
- **La crítica a Jiménez López et al. (2018) se sostiene íntegra** y se verificó contra el
  PDF completo: la regla 192 para Querétaro, el Kappa de 0,53, el Jaccard de 0,76, el
  emparejamiento 192/218/222 con Querétaro, San Luis Potosí y Toluca, y sobre todo el doble
  uso del mapa de 2017, que sirve a la vez para elegir la regla entre 256 y para medir su
  ajuste. Ni exagerada ni suavizada.
- **El criterio del 15 % de Bonham-Carter existe y está en el capítulo 9**, p. 316, citado
  literalmente por Agterberg y Cheng (2002).
- **Los seis tramos de la escala de Landis y Koch** coinciden uno por uno con la tabla de la
  p. 165 del facsímil.

### Correcciones aplicadas

| dónde | qué estaba mal | qué se hizo |
|---|---|---|
| `cap03:26` | SLEUTH «aplicado primero» a San Francisco por Clarke y Gaydos (1998); esa aplicación se publicó un año antes | se quitó «primero» y se co-cita `ClarkeHoppen1997` |
| `cap03:37` | «El enfoque conserva vigencia: puesto que…» dejaba la oración sin cláusula principal, y «lo aplicó» en singular con cuatro autores | reescrita sin la conjunción causal y en plural |
| `cap03:43` | Dinamica se presentaba como «alternativa al autómata celular puro» cuando el artículo se autodescribe como *a stochastic cellular automata model*; además «Dinamica EGO» es el nombre de la plataforma posterior, no del sistema de 2002 | ahora dice que conservan rejilla y vecindad pero estiman la función de transición, y el nombre queda como «Dinamica, base de la posterior plataforma Dinamica EGO» |
| `cap03:159` | se atribuía a Gómez et al. «concordancias espaciales altas» y la ausencia de contraste contra un modelo de referencia; no fue posible abrir el texto completo para verificarlo | se sustituyó por lo que sí es verificable: que cada trabajo reporta sobre su propio caso, con su sitio, sus clases y su horizonte |
| `cap03:332` | Bonham-Carter «propone elegir los cortes maximizando el contraste estudentizado»; en realidad el corte se elige en el máximo del contraste $C$ y $C/s(C)$ es una columna auxiliar de significancia | corregido a la formulación del libro |
| `cap03:388` | la segunda mitad de la frase de van Vliet («una sola comparación de ajuste, con frecuencia sobre los mismos datos usados para calibrar») no está en la fuente; el solapamiento entre calibración y validación es de Pontius, que se cita acto seguido | ahora dice exactitud locacional predominante y fracción pequeña que evalúa patrones, que es lo que la fuente afirma |
| `cap03:399` | «uno de los primeros ejercicios de pronóstico espacialmente explícito para la ZMCM» es un reclamo de prioridad que Suárez y Delgado no formulan | pasó a «plantean un ejercicio de pronóstico espacialmente explícito» |
| `cap03:490` | la celda R3 de Jiménez López et al. (2021) decía `\checkmark`, pero la nota al pie justifica «parcial» para los dos trabajos del mismo grupo | la celda pasó a `parcial` |
| `cap03:498` | la nota al pie afirmaba que CHRISTALLER «se distribuye bajo solicitud al grupo desarrollador»; ninguna de las dos fuentes lo dice y el artículo de 2018 anuncia acceso abierto | la nota ahora se apoya en el hecho comprobable: los artículos no señalan un repositorio público con historial de versiones |
| `cap03:529` | Santé y Aburas llamadas «revisiones sistemáticas» y acreditadas con documentar la heterogeneidad de calibración y validación; Aburas se presenta como revisión narrativa y su resumen declara clasificar técnicas de diseño | reescrito a lo que ambas sostienen, con la conclusión sobre calibración y validación formulada como ausencia de procedimiento común |
| `cap05:242` | la corrección de los LBP aplicada en `cap02:40` no se había propagado al capítulo 4 | misma formulación: «en la formulación circular generalizada de» |
| `cap02:315` | el tramo superior de Landis y Koch estaba escrito como $\kappa > 0{,}81$, con lo que 0,81 exacto quedaba fuera de toda categoría | intervalo cerrado $0{,}81$--$1{,}00$, más la advertencia de los autores de que las divisiones son convencionales |
| `referencias.bib` | `HernandezGuerrero2015` con apellido compuesto mal segmentado e inicial del segundo nombre ausente | «{Hernández Guerrero}, Juan Alfredo», coherente con `HernandezGuerreroOsorno2018` |

### Lo único que queda sin verificar de estos tres lotes

Dos afirmaciones, ambas por muro de pago y ninguna con indicio de ser falsa:

1. `cap03:529` — `Sante2010`: el resumen confirma la revisión de 33 modelos de AC urbanos y la
   clasificación de técnicas, pero el pasaje concreto sobre calibración y validación está en el
   texto completo, tras Elsevier. La reescritura de esa frase ya no le atribuye más de lo
   comprobable.
2. `cap03:159` — `Gomez2020`: MDPI devuelve 403 al PDF y el depósito de la Universidad EAFIT
   solo contiene una hoja de resumen en español, no el artículo. La frase se reformuló para no
   depender de la sección de resultados.

### Avisos sin acción

- Las conclusiones de Jiménez López et al. (2018) dicen «2030» donde sus propios resultados y
  figuras dicen «2031». Es una inconsistencia interna de la fuente. La tesis usa 2031, que es
  la lectura correcta, pero conviene saberlo si un sinodal cita las conclusiones.
- Las 256 reglas de ese trabajo son las elementales de Wolfram para un autómata unidimensional
  de tres bits, aplicadas después sobre raster bidimensional. La tensión viene de la fuente y la
  tesis la reproduce con fidelidad.

---

### Punto descartado por decisión del autor

El artículo pide evaluar un modelo primero contra su propio modelo nulo de persistencia y
solo en segundo término contra otros estudios, y documenta que en doce de sus trece
aplicaciones el error superó al cambio correctamente predicho. La tesis no reporta ese
contraste. Se planteó incorporarlo y se decidió dejarlo fuera.

---

## Lotes 5, 6 y los tres de cap01-cap02 — CERRADOS

Los cuatro agentes de `rev2_05`, `rev2pre_01`, `rev2pre_02` y `rev2pre_03` corrieron en
modo de solo lectura y entregaron su reporte en el mensaje. El acta consolidada, con el
hallazgo y su disposición uno por uno, está en `resultado_rev2_05_y_pre.md`. El de
`rev2_06` está en `resultado_rev2_06.md`.

Saldo de esos cuatro lotes: 104 afirmaciones, 90 correctas, 18 con observación, ninguna
referencia inexistente. `rev2pre_03` salió limpio, con las 23 afirmaciones correctas.

Se corrigieron dos errores que yo mismo había introducido esa misma mañana al aplicar la
primera ronda: la prioridad histórica de Lloyd frente a MacQueen en el K-Means, y la
atribución del índice de exceso de rojo a un artículo de Meyer que trata de textura y no
del índice. El que corresponde, Meyer, Hindman y Laksmi (1999), es el que contiene la
formulación 1,4R−G que efectivamente calcula el código.

---

# Cierre de la segunda ronda

Los nueve lotes están cerrados. El documento completo quedó revisado: cada referencia
contrastada contra su fuente y cada afirmación que la tesis le atribuye verificada contra
lo que la fuente dice.

Saldo total sobre las 180 afirmaciones:

| | |
|---|---|
| Correctas | 150 |
| Ajustadas por sobreextensión | 22 |
| Corregidas por incorrectas | 6 |
| Sin fuente accesible, reformuladas para no depender de lo no verificado | 2 |
| Referencias inexistentes | 0 |

Estado del documento tras aplicar todo: compila sin errores, 135 páginas, 82 entradas en
el `.bib`, 82 impresas y 82 citadas. Cero huérfanas, cero fantasma, cero duplicadas, cero
dedazos y cero errores gramaticales de los que detectan las herramientas.

Con esto, el aparato de citas queda cerrado y el texto puede reescribirse en estilo sin
riesgo de arrastrar una atribución falsa. La única precaución al reescribir: no convertir
en verbo de prioridad («X propuso», «X demostró») lo que hoy está redactado como cita
neutra, porque varias de esas citas neutras lo son justamente para no reclamar una
prioridad que la fuente no sostiene. Los casos concretos están anotados en los reportes
de cada lote.
