# Auditoría triple de la tesis — 20 de agosto de 2026

**Autor del documento:** José Rodrigo Moreno López
**Directora:** Dra. María Elena Lárraga Ramírez
**Alcance:** revisión exhaustiva de los seis capítulos vigentes, front matter y apéndice contra `revisiones/SESION DRA/` (NOTAS y TRANSCRIPTION.TXT) y contra los artefactos de datos del repositorio.

Se corrieron tres auditorías independientes y se verificó cada hallazgo a mano antes de corregir:

1. **Concordancia con la sesión del 4-ago-2026** — 46 observaciones registrables.
2. **Coherencia del arco narrativo** — hipótesis, objetivos, doctrinas del modelo.
3. **Cifras, labels y fuentes** — contra los cinco `validation_results.json`, `quinquenal_best_config.json` y `woe_pooled_1984_2010.pkl`.

## Estado del documento

| Ítem | Valor |
|---|---|
| Páginas | 111 |
| Errores de LaTeX | 0 |
| Referencias / citas indefinidas | 0 |
| Compilación | `pdflatex → biber → pdflatex ×2`, exit 0 |
| Concordancia con la sesión | 31 resueltas, 12 parciales, 1 no resuelta, 1 pendiente de la carta del jurado |

## Correcciones aplicadas en esta ronda

### Coherencia del arco narrativo

| Problema | Corrección |
|---|---|
| La hipótesis general del Cap. 1 decía «protocolo multitemporal **independiente**», lo que contradecía el solapamiento de las ventanas y no coincidía con la del Cap. 6 | Enunciado idéntico en ambos capítulos, sin «independiente» |
| El Cap. 1 afirmaba que la calibración se hizo sobre las cinco ventanas de validación | Se separan los dos actos: calibración en 1984–2010, evaluación de estabilidad en las cinco ventanas |
| El Cap. 5 justificaba el umbral con el Recall de las ventanas de validación | El Recall > 0,91 se presenta como resultado de validación, no como criterio de ajuste |
| R2 prometía «evaluar alternativas de ocupación del suelo», que el Cap. 6 niega | R2 se limita a comparar patrones de expansión bajo supuestos declarados |
| La transferibilidad se declaraba como hecho | «Potencialmente transferible, previa recalibración; no verificado empíricamente» |
| El cierre del Cap. 6 ubicaba el aporte en el modelo y la ciudad | El aporte es el protocolo; el modelo WoE-AC de la ZMQ es el caso de uso |
| El Cap. 6 concluía que el costo de la interpretabilidad «no es prohibitivo» | El contraste se declara cualitativo y no permite cuantificar ese costo |
| El Cap. 3 describía el modelo propio y adelantaba un resultado del Cap. 5 | Cuatro pasajes reescritos como criterios de diseño que la literatura deja abiertos |

### Cifras contrastadas contra los artefactos

| Cifra | Antes | Ahora | Fuente de verdad |
|---|---|---|---|
| Punto de equilibrio WoE | ρ ≈ 0,35–0,40 | ρ ≈ 0,10–0,15 | `woe_pooled_1984_2010.pkl`: cambio de signo entre los bins 3 y 4 de las tres densidades |
| Forma de la relación WoE–densidad | «monotónica creciente» | unimodal, con máximo +0,93 en ρ ≈ 0,37–0,55 y caída a −13,78 en vecindad saturada | mismo pkl |
| Bins con evidencia negativa | 1–2 | 1–3, más los bins 9–10 por saturación | mismo pkl |
| Columna «IV» de `tab:iv_weights` | pesos normalizados duplicados, total 1,0003 | IV crudos (3,2697 … 0,1388, total 11,6135) y pesos normalizados (total 1,000) | mismo pkl |
| Concentración de las cuatro variables dominantes | 0,783 (78,3 %) | 0,782 (78,2 %) | normalización exacta desde el pkl |
| Error del clasificador | 5–10 % / 270 000–540 000 px (Cap. 5) contra 6–12 % (Cap. 6) | 6–12 % / 325 000–650 000 px en ambos | complemento de la coherencia interna 88–94 % |
| Kappa 0,445 en la escala de Landis y Koch | «cercano al límite superior» | «en la parte baja de ese rango (0,41–0,60)» | escala definida en el Cap. 2 |
| Fracción media 2016–2020 | ~51 % | ~50 % | media de los `.npy`: 50,48 % |
| Factor de sobre-predicción | «en todos los períodos, entre 2 y 8 veces» | «en las cuatro ventanas con crecimiento positivo, entre 2,3 y 7,2; en 2011–2016 el observado es negativo y el cociente no está definido» | `growth_statistics` de los cinco JSON |
| Ventana de FoM mínimo | «la de menor crecimiento neto» | «la única con crecimiento neto observado negativo» | −120 074 px en 2011→2016 |
| Resolución efectiva | 10 m/píxel y 11 m/píxel en el mismo capítulo | **26,7 m/píxel** (ver nota) | registro contra Sentinel-2: `data/processed/georreferencia_capturas.json` |
| Varianza retenida por PCA | «cerca del 96 %», sin artefacto | 96,1 % (rango 95,3–96,8 %) | **nuevo artefacto** `data/processed/pca_variance_explained.json` |

**Nota sobre la resolución efectiva.** La primera pasada de esta auditoría unificó las dos cifras del capítulo en 11 m/píxel, calculando 625 km² entre 5 419 008 píxeles. La ventana de 625 km² resultó ser el dato equivocado, no la resolución: es incompatible con la proporción de la rejilla de 1792 × 3024. En una segunda pasada se midió la geometría real registrando las capturas contra el mosaico Sentinel-2 *cloudless*: la ventana mide 80,8 × 47,9 km, unos 3 870 km², y el píxel es cuadrado de 26,7 m de lado. El detalle está en `revisiones/CUMPLIMIENTO_SESION_DRA_LARRAGA.md`, sección 6.6.

### Figura corregida

El pie de `fig:woe_weights` describía «pesos WoE por bin de densidad de vecindad» con una «relación monotónica creciente». La imagen en `figures/woe_weights.png` muestra otra cosa: Information Value por variable (izquierda) y rango de pesos WoE por variable (derecha). Se reescribió el pie para describir la figura real, y la interpretación por bins se movió al cuerpo del texto, donde ahora corresponde al pkl.

### Otras correcciones

- Cita añadida a la afirmación comparativa sobre Querétaro del Cap. 1, que era el dato cuantitativo sin fuente que la Dra. señaló en 00:16:10. Se sustituyó por un enunciado verificable con `CONAPO2018Delimitacion` e `INEGI2020Censo`.
- Citas añadidas a CLUE-S, Dinamica EGO y FLUS en su primera mención (Cap. 3), y a la cifra de 10⁹ combinaciones de SLEUTH.
- Cinco tablas que nadie referenciaba ahora se invocan desde el párrafo que las introduce: `tab:cuadrante-mexico`, `tab:simulation_config`, `tab:literature_comparison`, `tab:error_contribution`, `tab:hypothesis_validation`.
- Etiquetas OE1–OE5 explícitas en el Cap. 1, para que la trazabilidad con el Cap. 6 sea verificable.
- Seis `\cite` sueltos convertidos a `\parencite` / `\textcite`, para uniformar el estilo con el resto del cuerpo.
- Nueve siglas añadidas a la lista: ABM, BAU, CLUE-S, CONAPO, FLUS, NDBI, PMOTDU, ROC, TOD.
- Tiempo verbal de la síntesis de conclusiones pasado a pretérito, como pidió la Dra. en 00:21:49.
- Corrección de «expanderse» → «expandirse» y de «trabajaba» → «trabaja».
- «De forma independiente al entrenamiento» → «en régimen de *hold-out* respecto del período de entrenamiento», para no invitar a la lectura de que las ventanas son independientes entre sí.

## Falsas alarmas verificadas

Dos hallazgos de las auditorías no procedían, y se dejaron como estaban:

- **FoM de 0,17 en ensayos con umbrales menores** (Cap. 6). Sí tiene respaldo: `analysis/ga_calibration/best_ac_config.json` registra umbral 0,3996 con FoM 0,1725 y 2 203 593 transiciones simuladas contra 391 095 observadas, que es exactamente la sobre-predicción masiva que describe el texto.
- **Varianza del 96 % por PCA.** La cifra era correcta; lo que faltaba era el artefacto. Se calculó sobre ocho años de la serie con el propio `FeaturePreprocessor` y se guardó el resultado.

Tampoco se atenuó la frase sobre atractores espaciales del Cap. 5: ese lenguaje proviene de la propuesta de la propia Dra. Lárraga para esa sección.

## Pendiente

### Bloquea la entrega

1. **Título contra la carta del jurado.** La Dra. advirtió que incluso una preposición distinta causa problemas administrativos. Requiere la carta para verificar carácter por carácter. Es el único punto que sigue abierto y no se puede cerrar sin ese documento.

### Requiere decisión

- Los cuatro bloques de pseudocódigo de StandardScaler/PCA, K-Means y SVM siguen en el Cap. 2, que es el capítulo del que la Dra. pidió sacar el detalle procedimental. El Cap. 4 no tiene ninguno.
- «Validación de hipótesis» aparece en el Cap. 5 y en el Cap. 6.
- El Cap. 3 tiene tres diagramas de pipeline que compiten entre sí.
- 16 entradas de `referencias.bib` no se citan. No afectan al PDF, porque biblatex no imprime lo no citado, pero varias de ellas (Batty2005, Torrens2000, Angel2012) servirían para elevar la densidad de citas del Cap. 2, que es otra observación parcial de la Dra.
- Falta el párrafo que justifique por qué WoE y no regresión logística o redes neuronales para poblar la regla de transición del AC. El argumento existe, pero está en el Cap. 1, no en el Cap. 2, donde la Dra. lo pidió.
- Las afirmaciones sobre licenciamiento de Dinamica EGO, IDRISI y TerrSet en el Cap. 3 no llevan fuente.

## Cierre de la segunda ronda

Se resolvieron los dos puntos que bloqueaban la entrega y que sí dependían del documento.

### Imágenes del marco teórico (NOTAS:33)

El Cap. 2 pasó de una figura a cinco. Tres son nuevas y se dibujan por código, con `generate_cap02_figures.py`, de modo que son reproducibles y no requieren la leyenda de imagen generativa:

| Figura | Contenido | Ubicación |
|---|---|---|
| 2.1 | Dimensiones y teselaciones del espacio celular: 1D, cuadrada, hexagonal y triangular | donde se define el espacio celular |
| 2.2 | Vecindad de Moore frente a Von Neumann | donde se define la vecindad |
| 2.3 | Condiciones de frontera: periódica, fija y reflejante | donde se definen las fronteras |
| 2.5 | Matriz de confusión y componentes del FoM | donde se define la matriz |

La 2.2 es la antigua `F3_moore_ai`, que estaba en el Cap. 4. Se movió al lugar donde se define el concepto; el Cap. 4 ahora la referencia en lugar de repetirla, con lo que desaparece la figura duplicada.

La notación de la figura 2.5 se alineó con las ecuaciones del capítulo, TP/TN/FP/FN, para que el lector no tenga que traducir entre dos convenciones.

### Trazabilidad de las imágenes con IA (NOTAS:38)

Los once pies declaran ahora de dónde sale el contenido, no solo que se generaron con IA. Dos remiten a fuente externa: vecindades a Wolfram y componentes del FoM a Pontius. Los demás remiten a la construcción propia que ilustran, con la ecuación o la sección concreta. Las dos figuras del Cap. 6 declaran que se construyeron a partir de los resultados de validación propios, lo que responde a la pregunta de fondo de la Dra.: de dónde salieron esos números.

### Recorte de las conclusiones

«Implicaciones para la planificación urbana» ocupaba unas cinco páginas con tres tablas. Se redujo a dos párrafos dentro de trabajo futuro, con la salvedad explícita de que no se calibraron corridas contrafactuales por política. «Accesibilidad computacional» no se eliminó: es la evidencia de H3 y se movió a Alcances, conservando su etiqueta para que la tabla de validación de hipótesis siga apuntando a ella.

### Limpieza de rastros de IA

Se eliminaron los siete guiones largos decorativos de la prosa, sustituidos por dos puntos, coma o punto según el caso. Los `---` que quedan son de dos tipos legítimos: celdas de tabla que marcan un criterio no cumplido y comentarios de LaTeX. Se verificó también que no aparezcan las muletillas del perfil de voz: cero «es importante destacar», cero «cabe señalar», ningún párrafo que abra con «en este trabajo», cero superlativos. Los tres «robusto» que sobreviven son técnicos: la invariancia del LBP a cambios de iluminación.

### Estado del PDF

112 páginas, cero errores, cero referencias o citas indefinidas, cero \hbox desbordadas por encima de 30 pt.

### Pseudocódigo fuera del marco teórico

Los cuatro bloques de pseudocódigo de StandardScaler, PCA, K-Means y SVM salieron del Cap. 2 y pasaron al Apéndice A, sección A.2, en el orden en que se ejecutan. El Cap. 2 conserva el fundamento conceptual de cada método y remite al apéndice con una frase; el Cap. 4 ya no dice que las definiciones están en el marco teórico, sino que el fundamento está ahí y el pseudocódigo en el apéndice. Las cinco etiquetas `alg:` resuelven correctamente y el apéndice queda con los cinco algoritmos numerados de forma corrida.

De paso se corrigió un detalle que la Dra. habría marcado: los encabezados salían en inglés. Ahora dicen «Algoritmo», «Entrada», «Salida», «para … hasta … hacer», «fin para» y «devolver».

El PDF queda en 113 páginas, cero errores y cero referencias indefinidas.

## Tercera pasada: lo que las auditorías dejaron abierto

Las dos auditorías se ejecutaron sobre el estado anterior del documento, de modo que casi todo lo que reportan ya estaba corregido. Quedaban tres cosas reales.

**El diagrama del pipeline empezaba en el paso 2.** La figura de preprocesamiento del Cap. 4 numeraba sus cajas del 2 al 6, sin paso 1: el nodo del insumo se había eliminado en algún momento y la numeración quedó huérfana. Ahora corre del 1 al 5, de extracción de características a matriz binaria final.

**La escala del FoM estaba mezclada.** Nueve pasajes reportaban el umbral de referencia de Pontius, unos como fracción y otros como porcentaje, y en dos casos ambas escalas convivían en la misma oración. Todo quedó en fracción, que es la escala en que la tesis reporta sus propias métricas: FoM inferior a 0,15 y por encima de 0,50.

**Los escudos de la portada no existen.** `figures/escudo_unam.png` y `figures/logo_pcic.png` están protegidos con `\IfFileExists`, así que no rompen la compilación, pero la portada sale en modo solo texto. Requiere los archivos.

El PDF queda en 113 páginas, cero errores y cero referencias indefinidas.

---

## Cuarta pasada: auditoría del arco narrativo

Esta ronda revisó la coherencia entre capítulos, no la ortografía ni las cifras. Encontró dos huecos estructurales reales y varios desajustes de nomenclatura. Todo lo que sigue quedó corregido.

### El hueco grave: la descomposición de Pontius se prometía y no se entregaba

La descomposición del error entre desacuerdo de cantidad y de asignación figuraba como componente de la aportación central en cuatro lugares del documento: las aportaciones del Cap. 1, el protocolo del Cap. 3, una afirmación del Cap. 4 que decía explícitamente que «se reportan» esas métricas, y el balance del Cap. 6. En el capítulo de resultados las palabras «cantidad» y «asignación» no aparecían una sola vez.

Se calculó. Los datos ya estaban en los cinco `validation_results.json`: con TP, FP y FN de cada ventana la descomposición binaria de Pontius y Millones es aritmética directa. El resultado no solo llena el hueco, refuerza el diagnóstico del modelo:

| Ventana | FoM | Cantidad | Asignación | Cantidad / total |
|---|---|---|---|---|
| 2011–2016 | 0,222 | 31,1 % | 2,4 % | 92,9 % |
| 2012–2017 | 0,345 | 17,4 % | 7,6 % | 69,7 % |
| 2013–2018 | 0,355 | 17,1 % | 8,5 % | 66,7 % |
| 2014–2019 | 0,287 | 26,8 % | 3,5 % | 88,6 % |
| 2015–2020 | 0,378 | 17,8 % | 6,7 % | 72,6 % |
| **Promedio** | **0,317** | **22,0 %** | **5,7 %** | **79,4 %** |

El 79,4 % del desacuerdo es de cantidad. Es decir: el modelo urbaniza demasiadas celdas pero las coloca mayormente donde corresponde. Eso explica la sobre-predicción sistemática, es coherente con el Recall alto frente a la Precision moderada, y da una implicación metodológica clara: el margen de mejora está en estimar la magnitud del crecimiento, no en la regla que decide la ubicación.

La verificación que da confianza en el cálculo es que el desacuerdo total coincide exactamente con el complemento de la exactitud global ya publicada en la tabla de métricas, en las cinco ventanas. Se añadió la Tabla 5.12 al Cap. 5, las dos fórmulas al Cap. 2 con sus etiquetas, la referencia Pontius y Millones (2011) al `.bib` (DOI verificado: 10.1080/01431161.2011.552923) y el artefacto reproducible en `data/processed/pontius_decomposition.json`.

### El segundo hueco: los requerimientos R1–R4 nunca se aplicaban al propio modelo

El Cap. 1 los declaraba «la forma operativa de comprobar si un modelo cierra esa brecha» y el Cap. 3 los usaba para calificar toda la literatura mexicana. Después no volvían a aparecer: un grep de `R1|R2|R3|R4` solo daba resultados en esos dos capítulos. El modelo propio se evaluaba con otro criterio.

Se añadió la sección 6.1.4, «Cumplimiento de los requerimientos del modelo», que aplica los cuatro al trabajo con el mismo rigor: R1 parcial (con la reserva medida de la sobre-predicción de magnitud), R2 cumplido en su formulación mínima, R3 y R4 cumplidos. El veredicto de R1 no se maquilla: se apoya en la descomposición recién calculada.

### Contradicciones y nomenclatura

**Transferibilidad.** El Cap. 4 afirmaba que las siete variables derivadas del mapa binario hacen al modelo «transferible a cualquier ciudad». Cuatro pasajes del Cap. 1 y del Cap. 6 dicen lo contrario: que la transferencia no se verificó y que los pesos WoE exigirían recalibración. Reformulado: reduce los insumos necesarios, y la transferencia no se verificó aquí.

**Las preguntas de investigación no eran rastreables.** El Cap. 1 las enunciaba en prosa como «la primera, la segunda...» y el Cap. 6 las respondía como P1–P4, rótulos que no existían en ningún otro lugar. Peor: la sigla vecina R1–R4 ya estaba tomada por los requerimientos. Ahora están etiquetadas P1 a P4 en el Cap. 1.

**El resumen del Cap. 3 renombraba los requerimientos.** Hablaba de «reproducibilidad, robustez temporal, transferibilidad e interpretabilidad»; de los cuatro reales solo coincidía uno. Ahora usa los nombres literales de la Tabla 1.1 con su identificador. El título de la subsección 3.2.4 pasó de «Transferibilidad» a «Reproducibilidad: software propietario y barreras de acceso», que es lo que esa subsección argumenta.

**Tres limitaciones anunciadas, cuatro desarrolladas.** El Cap. 3 decía «tres limitaciones estadísticas» y desarrollaba cuatro párrafos. Y el Cap. 2, que remite ahí, enumeraba tres distintas: le faltaba la multicolinealidad. Ambas listas quedaron alineadas en cuatro.

**El SVM es supervisado.** Cuatro pasajes llamaban «clasificación no supervisada» a la cadena completa K-Means + SVM. Se sustituyó por «sin verdad terreno etiquetada» y «SVM sobre pseudoetiquetas», que es lo que ocurre. El Cap. 2 ya lo decía con precisión.

### Afirmaciones sin respaldo

**Atractores espaciales.** El párrafo de cierre del capítulo de resultados afirmaba que el sistema «no converge a un estado estacionario único» y que sus trayectorias «se organizan en torno a atractores espaciales locales». No hay análisis de convergencia ni de sensibilidad al estado inicial en ninguna parte del documento: era vocabulario de sistemas dinámicos sin medición, en el lugar donde más pesa. Reescrito en términos de lo observado, con la salvedad explícita de que es una regularidad de las cinco corridas y no un resultado de convergencia.

**Una métrica adversa relativizada.** El apartado de fortalezas sostenía que la reproducción cualitativa de los patrones de difusión «puede ser tan relevante como la concordancia píxel a píxel», sin atribuirlo a nadie: justo aquello en lo que el modelo rinde peor quedaba minimizado por una afirmación conveniente. Ahora se declara como límite del alcance de la evaluación: la similitud morfológica se aprecia de forma visual, sin métrica que la cuantifique, y medirla queda como extensión.

**El FoM de 0,17 solo aparecía en conclusiones.** Se reportó en el Cap. 5, donde se describe la exploración de umbrales, con las cifras del artefacto: umbral 0,400, 2.203.593 transiciones simuladas frente a 391.095 observadas, FoM 0,172.

**Una propiedad atribuida a terceros sin verificar.** El Cap. 4 afirmaba que «en buena parte de la literatura el WoE entra al autómata como un mapa de idoneidad estático». El Cap. 3 no establece eso de Dinamica EGO, CLUE-S ni FLUS. Se reformuló sin la comparación: la formulación propia se describe por lo que es, sin atribuir a otros una propiedad no documentada.

### Duplicaciones y jerarquía

**La validación de hipótesis estaba enterrada.** En el índice, el contraste de H1–H4 aparecía como subsección 5.6.4, dentro de «Análisis espacial del error del modelo». Se promovieron a sección: 5.7 fortalezas y debilidades, 5.8 validación de hipótesis, 5.9 síntesis.

**Dos diagramas del mismo pipeline.** El Cap. 4 tenía una figura PNG y un TikZ del mismo flujo de preprocesamiento, separados por menos de dos páginas. El nombre del archivo PNG, `pipeline_rgb_features_binary_cap02.png`, delataba que era resto de cuando el contenido vivía en el Cap. 2. Se eliminó el PNG y quedó el TikZ, que es vectorial y más informativo.

**Las 23 características se describían tres veces.** Con fórmula por componente, otra vez en prosa dos páginas después, y una tercera en la tabla del apéndice. Igual el StandardScaler + PCA y el 96,1 % de varianza. Se eliminaron las dos subsubsecciones redundantes y su contenido único, la varianza retenida y su rango, se integró donde corresponde. Se conservó la aclaración sobre NGRDI y NDVI, que sí aporta.

**SLEUTH.** La auditoría pedía recortar su descripción del Cap. 2 por ser contenido de estado del arte. Se revisó y no procede: el Cap. 3 remite explícitamente al Cap. 2 para la definición de sus componentes, de modo que moverla habría creado una referencia circular. El reparto actual es correcto y es el que pidió la Dra.: el Cap. 2 define, el Cap. 3 critica la calibración. Solo se explicitó ese reparto en la frase de cierre.

### Otros ajustes

El Cap. 2 valoraba una decisión propia en su resumen («la razón por la que resulta apropiado acoplarlo»); ahora solo define. El único plural de primera persona del documento, «en nuestro conocimiento», pasó a «hasta donde alcanza esta revisión». El recorrido del Cap. 1 omitía la sección de preprocesamiento del Cap. 2 y el Apéndice A completo: ambos añadidos. El resumen del Cap. 6 anunciaba una sección de implicaciones para la planificación que ya no existe como tal. Los umbrales de Pontius que sobrevivían en porcentaje quedaron en fracción, incluidos los valores de Tang. El archivo muerto `cap04_area_estudio.tex`, que contenía texto idéntico al del Cap. 4 y no compilaba, se movió a `_deprecated/`.

### Estado del PDF

115 páginas, cero errores, cero referencias indefinidas, cero citas indefinidas, cero cajas desbordadas por encima de 20 pt. La bibliografía impresa tiene 44 entradas, todas citadas: las 21 entradas no citadas del `.bib` no llegan al PDF, de modo que no hay referencias zombie visibles para el lector.

### Lo único que sigue abierto

Los escudos de la portada. `figures/escudo_unam.png` y `figures/logo_pcic.png` siguen sin existir; están protegidos con `\IfFileExists`, así que la portada compila en modo solo texto. Y el título contra la carta del jurado, que requiere el documento.

### Hallazgo propio de esta ronda: «Cuadro» contra «Tabla»

Al revisar el render de la tabla nueva apareció algo que ninguna auditoría había marcado y que la Dra. habría visto de inmediato: babel-spanish rotula los flotantes de tabla como «Cuadro 5.12», pero los 34 reenvíos del cuerpo del texto dicen «Tabla 5.12». Un lector que siguiera cualquier referencia a una tabla buscaba un rótulo que no existía, en las 21 tablas del documento. Se unificó en el preámbulo con `\addto\captionsspanish`, junto con el título del índice de tablas, y se renombró la sección A.3 del apéndice, que también decía «Cuadros». El PDF ya no contiene la palabra «Cuadro» en ninguna de sus 115 páginas.

---

## Quinta pasada: verificación contra los datos y el código

Esta ronda no revisó redacción. Contrastó cada cifra del documento contra los cinco `validation_results.json`, el pickle `woe_pooled_1984_2010.pkl`, los `.npy` y el código de `src/tesis_ac/`. El resultado fue tranquilizador en lo numérico y grave en lo metodológico: las cifras publicadas reproducen los datos al último decimal, pero aparecieron cuatro problemas que no son de estilo.

### Una figura decía lo contrario que el capítulo que la contenía

El panel derecho de `woe_weights.png` estaba mal graficado. En `generate_cap05_figures.py` los anchos de las barras se calculaban en el orden del IV descendente, pero el borde izquierdo se pasaba invertido con `left=woe_min[::-1]`. Cada barra recibía el ancho de una variable y el origen de otra. Seis de las siete quedaban desplazadas.

La consecuencia visible: la figura mostraba que la distancia al área urbana alcanzaba evidencia positiva de $+14{,}28$ y que el gradiente urbano era negativo en todo su rango. El rango real de la distancia es $[-14{,}28; +0{,}85]$, y la página 78 del propio documento afirma —correctamente— que los máximos son $+0{,}93$, $+0{,}87$ y $+0{,}80$. Es decir, el texto y su figura se contradecían, y la contradicción era de un orden de magnitud. Se corrigió el generador y se regeneró la figura: ahora las siete barras terminan justo a la derecha del cero, como corresponde.

### El Information Value se usaba en todo el documento y no se definía en ninguna parte

El IV es el mecanismo de ponderación de la ecuación central del modelo, el contenido de la tabla de pesos y el sustento de H1. No aparecía una sola vez en el marco teórico. Se añadió como subsección 2.3.2, con su fórmula y la escala convencional de interpretación en cinco tramos, que además ya estaba documentada en el código del propio repositorio.

Esto arregló de paso una contradicción: el Cap. 4 citaba la ecuación de la suma simple para decir que el modelo *no* usa una suma simple. Ahora existe la ecuación de la suma ponderada por IV y la referencia apunta a ella.

### El 85 % del IV de la variable más importante venía de un bin sin datos

Al añadir la escala de interpretación se hizo visible un problema de fondo: los IV reportados están un orden de magnitud por encima del umbral de «muy fuerte». La causa es que el muestreo incluye como negativos las celdas que ya eran urbanas, y esas celdas forman un bin sin ninguna transición observada cuyo peso lo fija el suavizado `max(n_positive, 1)`, no los datos.

Descompuesto por bin: en `distance_urban`, $2{,}786$ de sus $3{,}270$ de IV provienen de ese bin, un $85{,}2$ %. En el radio $3\times3$, $1{,}617$ de $2{,}897$, un $55{,}8$ %. Las otras cinco variables no tienen bins degenerados. Si se descuenta la contribución, la jerarquía se reordena y `distance_urban` cae del primer al **sexto** lugar.

Eso toca a H1, que afirma que la distancia y la densidad concentran el mayor IV. Se optó por declarar la salvedad en lugar de reentrenar, porque los resultados publicados son los que produjo la simulación y cambiarlos exigiría rehacer las cinco ventanas. Se añadió la Sección 5.2.x con la descomposición completa, se matizó el veredicto de H1 en el Cap. 5 y en el Cap. 6 —lo que se sostiene en cualquier lectura es que la difusión por contigüidad gobierna la simulación; lo que no se sostiene sin la salvedad es la primacía de la distancia sobre la densidad— y se apoyó la interpretación de saturación en el bin 9, que tiene $186\,635$ transiciones que lo sostienen, en lugar del bin 10, que no tiene ninguna.

### Las variables del WoE están promediadas en el tiempo, y el texto decía otra cosa

El Cap. 4 y el Cap. 5 describían un emparejamiento período a período. El código hace algo distinto: promedia las siete variables espaciales sobre los 26 períodos y repite ese único mapa 26 veces para contrastarlo con 26 mapas de transición diferentes. El *pooling* es de transiciones, no de pares (variable, transición).

Se reformularon los dos pasajes para decir lo que el código hace y se retiró la afirmación de que acumular 26 períodos reduce el error de estimación de cada bin, porque el número de valores distintos de la covariable es el de una sola rejilla, $5\,419\,008$. Emparejar cada configuración anual con su transición y restringir el muestreo a celdas elegibles quedaron como dos refinamientos explícitos en el trabajo futuro.

### La observación de la Dra. que seguía sin atender

NOTAS:73 pedía una subsección dedicada al WoE con sus ventajas y limitaciones, y la justificación de por qué se acopla a un autómata celular en lugar de usar regresión logística o una red neuronal. La sección de Pesos de Evidencia era un bloque plano de siete párrafos. Ahora tiene cuatro subsecciones: formulación, Information Value, ventajas y limitaciones, y aplicación al acoplamiento. La justificación frente a las alternativas está donde la Dra. la pidió, con el argumento correcto: la regresión logística entrega un coeficiente por variable y no un peso por intervalo, y los pesos internos de una red no admiten lectura geográfica.

Se aplicó el mismo criterio a las dos secciones que seguían planas: Autómatas Celulares se dividió en cuatro subsecciones y las métricas de clasificación global quedaron separadas por métrica —exactitud, precisión, recall, F1 y Kappa—, que es lo que pedía NOTAS:74.

### Convenciones numéricas

La coma hacía de separador decimal y de separador de miles a la vez, y en dos frases chocaba consigo misma. Se convirtieron 66 números a espacio fino, distinguiendo uno por uno los miles de los decimales: `$2\,699\,088$` frente a `$0{,}317$`.

Con el signo de porcentaje ocurrió lo contrario de lo esperado. Unificar en `\,\%` rompió la compilación con 59 errores de *incompatible glue units*, porque babel-spanish ya inserta el espacio fino por su cuenta mediante `\es@sppercent` y el `\,` adicional falla en modo matemático y en argumentos móviles. La convención correcta era la contraria: se normalizaron las 77 ocurrencias a `\%` a secas.

### Errores de lengua

Cinco en el Cap. 3: una frase con la sintaxis quebrada y desacuerdo de género («dos directivas que parecen contrapuestos»), «un tanto desmotivante» como juicio coloquial sobre el trabajo de Pontius, un *hedge* invertido («claramente rara vez sucede»), un pronombre acentuado que la ortografía vigente no admite, y «pionería», que no existe en español. Dos en el Cap. 2: preguntas retóricas con el verbo en indicativo donde pedía subjuntivo, reformuladas en modo asertivo, que además es el registro del resto del documento.

### Otros ajustes

El mismo argumento sobre el desbalance de clases usaba el 90 % en un capítulo y el 95 % en otro: unificado. Las afirmaciones sobre licenciamiento de Dinamica EGO, IDRISI, TerrSet y SLEUTH ahora se atribuyen a los términos de distribución de cada uno y llevan cita. `Tang2024` era la única comparación cuantitativa con otro modelo y aparecía por primera vez en el capítulo de resultados: se presentó antes, en el estado del arte, con su contexto árido. Se redujo de cinco a tres la repetición literal del posicionamiento frente a Pontius. Se retiró el *hedge* «numéricamente» de tres pies de figura. Se corrigió un redondeo ($71{,}9$ % por $72{,}0$ %), una etiqueta heredada que nombraba una sección inexistente, dos `\path` con el guion bajo escapado que imprimía la barra invertida, un `\figref` único frente a 40 `Figura~\ref`, y la descripción de `red_std`, `green_std`, `blue_std` y `contrast`, que el documento llamaba desviaciones estándar y el código calcula como medias locales. Se añadieron siete acrónimos usados y no listados: EPSG, ExG, ExR, IIEc, NIR, WGS84 y ZMCM.

### Estado del PDF

119 páginas, cero errores, cero referencias indefinidas, cero citas indefinidas, cero cajas desbordadas por encima de 20 pt. Ninguna coma como separador de miles en el texto renderizado. Los dos únicos guiones largos que quedan están en títulos de artículos de la bibliografía, tal como se publicaron.

### Lo que sigue abierto

Los escudos de la portada y el título contra la carta del jurado, igual que en la ronda anterior. Se añade una decisión de fondo: si se quiere que los IV sean comparables con la escala convencional y que la jerarquía de H1 no dependa de una salvedad, hay que reentrenar el WoE restringiendo el muestreo a celdas elegibles y rehacer las cinco ventanas de validación. El documento actual es defendible con la salvedad declarada; sin ella, no lo era.

---

## Sexta pasada: la salvedad del IV, resuelta sin reentrenar

Decisión tomada: no se reentrena el WoE. Al buscar el argumento que sostuviera esa decisión apareció uno mucho más fuerte que la simple declaración de la salvedad.

Los dos bins degenerados **no pueden intervenir en la simulación**, y no por una precaución del código sino por la construcción de las variables. El autómata evalúa como candidatas solo las celdas no urbanas (`src/tesis_ac/ca/rules.py`, `step()`). Una celda no urbana tiene distancia al frente urbano $\geq 1$, mientras que el bin degenerado de `distance_urban` cubre $[0;\ 0{,}0385]$. Y la densidad de vecindad usa una ventana $3\times3$ que incluye el centro, de modo que si el centro es no urbano la densidad no supera $8/9 \approx 0{,}889$, mientras que el bin degenerado exige densidad exactamente unitaria. Ambos son inalcanzables.

Comprobado sobre las diez rejillas del período de validación: de $28\,917\,357$ celdas candidatas, **cero** caen en cualquiera de los dos bins. El artefacto `data/processed/woe_bin_reachability.json` guarda la verificación año por año, las razones estructurales y los bins con sus pesos.

Con eso el problema queda acotado a lo que realmente es. Los pesos de $-14{,}28$ y $-13{,}78$ no tocan ninguna transición simulada. Lo único que los bins degenerados afectan es el IV y, por tanto, la ponderación relativa $\omega_k$ entre variables. Esa ponderación es la que produjo todos los resultados del Cap. 5 y se validó fuera de muestra en cinco ventanas que no intervinieron en su estimación. No es un error a corregir: es una configuración con desempeño documentado.

Se reescribió la Sección 5.2.3 con este argumento, se firmó el veredicto de H1 —la salvedad afecta al orden entre las dos primeras variables, no al enunciado— y se reformuló el trabajo futuro para decir que los dos refinamientos depuran la interpretación de los IV sin alterar la mecánica de la simulación.

De paso se corrigió algo que la auditoría no había visto. El texto sostenía la lectura de saturación en un bin ($-1{,}42$ con $186\,635$ transiciones) que también es inalcanzable en simulación, porque cubre $[0{,}9231;\ 1{,}0)$ y el tope de una candidata es $0{,}889$. La unimodalidad se enuncia ahora dentro del rango que la simulación efectivamente visita: el peso sube a $+0{,}93$ en $\rho \approx 0{,}37$--$0{,}55$ y baja a $+0{,}21$ en $\rho \approx 0{,}74$--$0{,}92$, que es el intervalo más alto que una celda candidata puede alcanzar. El mecanismo de difusión queda sostenido por bins con millones de observaciones, sin depender de ninguno degenerado.

### Estado del PDF

120 páginas, cero errores, cero referencias indefinidas, cero citas indefinidas.
