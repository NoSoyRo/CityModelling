# Cumplimiento de la sesión con la Dra. Lárraga

**Sesión:** 4 de agosto de 2026, 18:04 CST (1 h 43 min)  
**Fuentes:** `revisiones/SESION DRA/NOTAS`, `revisiones/SESION DRA/TRANSCRIPTION.TXT`  
**Estado del documento:** 129 páginas, cero errores de compilación, cero referencias y citas indefinidas  
**Fecha de este reporte:** 2 de septiembre de 2026 (última verificación independiente)

Este reporte recorre los **25 acuerdos** de la lista de próximos pasos y los **33 detalles** de la transcripción. Cada punto lleva la evidencia en el archivo, con línea, para que pueda verificarse sin abrir el PDF. Las rutas son relativas a `report/tesis/tesis_indice_nuevo/caps_larraga/`.

---

## Resumen


|                                 | Acuerdos (NOTAS) | Detalles (transcripción) |
| ------------------------------- | ---------------- | ------------------------ |
| Resueltos                       | 24 de 25         | 29 de 29 aplicables      |
| Pendientes de un insumo externo | 1                | 0                        |
| Fuera del alcance de la tesis   | 0                | 4                        |


El único punto abierto es la verificación del título contra la carta del jurado, que depende de un documento que no está en el repositorio.

Dos puntos se resolvieron corrigiendo cifras, no solo añadiendo texto, y merecen lectura aparte. La geometría de las capturas estaba mal declarada y se midió (§6.6): la ventana no son 625 km² sino 3 870, el píxel no mide de 8 a 14 m sino 26,7, y el marco no está orientado al norte. Y la única cifra de terceros que sostenía una comparación de desempeño se cotejó contra el artículo original (§7.7).

Después de esos dos, el estado del arte se auditó fuente por fuente (§7.10) y apareció un antecedente que obligó a precisar el alcance de la aportación (§7.11). Son las dos secciones que conviene leer si solo hay tiempo para una parte de este reporte.

### Verificación del 22 de agosto

Se revisó el documento otra vez sin dar por buenas las conclusiones de esta tesis, comprobando cada punto contra los archivos. 

Los demás puntos resistieron la revisión: cero primera persona, estilo IEEE, doce figuras con IA atribuidas, ligas entre capítulos, hipótesis general con cuatro particulares, orden de alcances, limitaciones y trabajo futuro, cero citas rotas, y correspondencia párrafo por párrafo entre el resumen y el abstract. Un recuento automático confirma ahora cero figuras y cero tablas sin referencia.

### Revisión final del 23 de agosto

Una última pasada sobre el PDF, buscando restos de nombres de archivo del repositorio en la prosa, encontró uno en la discusión de fortalezas: las comparaciones visuales se remitían a `summary_XXXX_to_XXXX.png` con comodín en vez de a las figuras del documento. Ahora se citan como Figuras 5.4 a 5.8 (`cap06:500`). Era el último lugar donde el texto mandaba al lector a un archivo interno en lugar de a una referencia cruzada.

### Bibliografía: dieciséis fuentes que estaban sin citar

Este punto no salió de la sesión, pero conviene dejarlo cerrado. La bibliografía tenía 60 entradas activas y solo 44 estaban citadas en el texto. Las 16 restantes no eran material ajeno al tema: eran fuentes pertinentes que se habían quedado sin anclar a ninguna afirmación. Ya están las 60 citadas, sin entradas huérfanas ni citas rotas.

En dos casos la integración añadió argumento y no solo la cita. UrbanSim (`cap03:41`) ahora explica por qué no se siguió la ruta de la microsimulación de mercados de suelo: exige microdatos socioeconómicos de hogares, empleo y transporte que rara vez existen con esa resolución para una ciudad intermedia. Y Alonso (`cap05:17`) queda anclado a una omisión declarada del modelo, la ausencia de un término de decaimiento por distancia a centros de empleo, de modo que una simplificación pasa a ser una decisión razonada frente a la teoría clásica de la renta del suelo.

El resto se reparte así: la expansión urbana global y sus efectos en la introducción (Angel, Seto sobre teleconexiones, CONABIO sobre pérdida de hábitat, ITDP sobre externalidades del transporte y el Observatorio de Ciudades sobre vivienda en Querétaro); el marco conceptual y la mecánica de los modelos celulares en el estado del arte (Batty, Torrens), junto con una aplicación reciente (Arfiansyah y colegas, sobre la capital planificada de Indonesia), el aprendizaje automático aplicado al crecimiento urbano (Gómez y colegas) y las herramientas abiertas disponibles (Chen y colegas con PyLUSAT, Arribas-Bel sobre fuentes de datos urbanos); la exigencia de declarar el comportamiento de cada agente en los modelos basados en agentes (Li y Liu); y la heterogeneidad del paisaje urbano de Querétaro en el capítulo del modelo (Hernández-Guerrero 2015 y 2018).

Antes de citar la página de CONABIO se verificó contra la fuente, porque era la única de las nuevas que afirmaba una causa principal.

---

## 1. Reestructuración del documento

### 1.1 Fusión de capítulos: de siete a seis

La Dra. pidió fusionar los capítulos cortos para evitar la fragmentación (NOTAS:39, NOTAS:41, transcripción 00:79) y renombrar el capítulo del modelo (NOTAS:42).

*Área de estudio*, el *preprocesamiento* que vivía en el marco teórico y el capítulo del *modelo* son ahora un solo capítulo. `main.tex:25-31` declara seis capítulos y ya no incluye el archivo de área de estudio. Los títulos actuales:


| Cap. | Título                                   | Archivo                                                                 |
| ---- | ---------------------------------------- | ----------------------------------------------------------------------- |
| 1    | Introducción                             | `cap01_introduccion/cap01_introduccion.tex:5`                           |
| 2    | Marco teórico                            | `cap02_marco_teorico/cap02_marco_teorico.tex:6`                         |
| 3    | Estado del arte                          | `cap03_estado_del_arte/cap03_estado_del_arte.tex:6`                     |
| 4    | **Modelo de crecimiento urbano híbrido** | `cap05_modelo_crecimiento_urbano/cap05_modelo_crecimiento_urbano.tex:7` |
| 5    | Resultados y análisis                    | `cap06_resultados_analisis/cap06_resultados_analisis.tex:6`             |
| 6    | Conclusiones                             | `cap07_conclusiones/cap07_conclusiones.tex:5`                           |


Los nombres de carpeta conservan la numeración anterior por historial de versiones; el número de capítulo que ve el lector es el de la tabla.

### 1.2 Orden de secciones del capítulo del modelo

La Dra. pidió abrir con las consideraciones del modelo (NOTAS:43, transcripción 00:79) y seguir con la arquitectura, el área de estudio y el preprocesamiento. El orden actual coincide exactamente:

```
14:  \section{Consideraciones del modelo}
21:  \section{Arquitectura del modelo}
33:  \section{Área de estudio}
79:  \section{Preprocesamiento de datos}
289: \section{Definición del modelo}
347: \section{Validación}
```

### 1.3 Separar conceptos de metodología propia

Este fue el punto que la Dra. repitió más veces (transcripción 00:65, 00:66, 00:70, 00:71). El marco teórico no debía mezclar conceptos con decisiones del trabajo, y toda mención a «el presente modelo» debía salir de ahí.

El flujo de trabajo, el arreglo de datos y el pipeline aplicado se trasladaron al Cap. 4. Una búsqueda de «presente modelo», «este modelo» y «nuestro modelo» en `cap02_marco_teorico.tex` devuelve **cero ocurrencias**. El resumen del capítulo lo declara: «La aplicación de estos conceptos al caso estudiado corresponde al Capítulo 4» (`cap02:10`).

---

## 2. Front matter

### 2.1 Título (NOTAS:20, transcripción 00:50): pendiente de la carta del jurado

El título se revirtió al acordado y está en `front/portada.tex:27-29`:

> Modelado del crecimiento urbano en la Zona Metropolitana de Querétaro mediante Weight of Evidence y Autómatas Celulares

La Dra. advirtió que una preposición distinta causa problemas administrativos en la aceptación del documento. **Falta cotejarlo carácter por carácter contra la carta del jurado**, que no está en el repositorio. Es el único punto de la sesión que sigue abierto.

### 2.2 Agradecimientos con la beca (NOTAS:21, transcripción 00:51): resuelto

Requisito obligatorio para la liberación de la beca. `front/agradecimientos.tex:6`:

> Se agradece a la Secretaría de Ciencia, Humanidades, Tecnología e Innovación (SECIHTI) por el apoyo otorgado a través de la beca para los estudios de maestría que hicieron posible este trabajo.

Se usa el nombre vigente de la dependencia. Las notas de la reunión la registran como «CECI» y «CCTI», ambas transcripciones fonéticas del mismo organismo. El archivo se incluye en `main.tex:14`.

### 2.3 Resumen realzando la contribución (NOTAS:22, NOTAS:40, transcripción 00:52, 00:54): resuelto

La Dra. pidió que el resumen enfatizara la contribución científica por encima de la aplicación local, y que presentara el modelo híbrido. El resumen abre por la estrategia de integración, calibración y validación, y sitúa Querétaro como caso de uso, no como objeto. Las palabras clave incluyen «validación multitemporal» (`front/resumen.tex:36`).

### 2.4 Reescribir el resumen en inglés al final del proceso (NOTAS:26): resuelto

El acuerdo es sobre el momento de redactarlo, no sobre su lugar en el documento: rehacerlo una vez cerrada la versión en español, para que ambos digan lo mismo. Se reescribió después. `front/abstract.tex` abre por la estrategia y el protocolo de cinco ventanas desplazadas, no por el caso de estudio, igual que el resumen. Cotejados párrafo por párrafo, coinciden en estructura y en las cuatro cifras que reportan: FoM 0,317, Kappa 0,445, exactitud 0,722 e IoU 0,633.

---

## 3. Capítulo 1: Introducción

### 3.1 Estructura de secciones (NOTAS:23, NOTAS:27, transcripción 00:55): resuelto

La Dra. dictó las secciones: planteamiento del problema, justificación, antecedentes, objetivos, contribución y organización. También aclaró que la introducción no lleva resumen, porque es el prólogo. Estructura actual:

```
8:   \section{Planteamiento del problema}
97:  \section{Justificación}
101: \section{Antecedentes}
109: \section{Objetivos}
147: \section{Preguntas de investigación e hipótesis}
180: \section{Aportaciones del trabajo}
212: \section{Organización del documento}
```

No hay entorno `resumen` en este capítulo. Es el único que no lo tiene.

### 3.2 Antecedentes y justificación separados (NOTAS:28, transcripción 00:58): resuelto

La sección que se llamaba «Literatura existente y brecha metodológica» se partió en dos: la literatura quedó en *Antecedentes* (`cap01:101`) y la brecha pasó a ser la *Justificación* (`cap01:97`). Es el reparto que pidió la Dra.

### 3.3 Categorizar en lugar de listar (transcripción 00:62): resuelto

La instrucción fue no decir «se revisaron trece trabajos» sino categorizar los modelos. Una búsqueda del patrón «se revisaron/analizaron N trabajos/estudios» en los seis capítulos devuelve **cero ocurrencias**. Los antecedentes agrupan por familia: redes neuronales, autómatas celulares, híbridos y Pesos de Evidencia.

### 3.4 Referencias en todo dato cuantitativo (NOTAS:24, transcripción 00:57): resuelto

Barrido de cifras (millones, habitantes, porcentajes, kilómetros) en el Cap. 1: **ninguna sin cita**. El capítulo tiene nueve citas. La afirmación sobre el dinamismo de Querétaro, que antes iba sin fuente, se reformuló con apoyo en `CONAPO2018Delimitacion` e `INEGI2020Censo`.

### 3.5 Alcances y limitaciones fuera de la introducción (transcripción 00:60, 00:61): resuelto

La Dra. pidió que fueran al cierre, no al inicio. `cap01:177` conserva el rastro del traslado como comentario, y las secciones viven ahora en el Cap. 6 (`cap07:87` Alcances, `cap07:115` Limitaciones, `cap07:141` Trabajo futuro), en el orden exigido.

### 3.6 Hipótesis: una general con particulares (NOTAS:31, transcripción 00:64): resuelto

La Dra. sugirió reducir las seis hipótesis a una general desglosada en particulares, planteada como afirmación validable. `cap01:165` enuncia la general y `cap01:167` abre las cuatro particulares, cada una declarada falsable con los resultados del Cap. 5. H1 a H4 corren en paralelo a los objetivos específicos.

### 3.7 Enfoque de embudo y contribución realzada (transcripción 00:53, 00:54): resuelto

El planteamiento abre por el problema general del análisis de imágenes de archivo y desciende hasta la propuesta. La sección de aportaciones (`cap01:180`) se abre reconociendo que el acoplamiento WoE-AC no es una novedad metodológica y sitúa la contribución donde la Dra. la puso: en la metodología de evaluación y en las condiciones de reproducibilidad.

### 3.8 Objetivos coherentes con los resultados (NOTAS:30, transcripción 00:63): resuelto

El objetivo general (`cap01:113`) y los cinco específicos se verificaron contra el Cap. 5. El Cap. 6 dedica una subsección al cumplimiento de cada objetivo específico (`cap07:40`). La correspondencia queda explícita, no al criterio del lector.

---

## 4. Capítulo 2: Marco teórico

### 4.1 Definición formal del autómata celular (transcripción 00:69): resuelto

Se pidió una definición formal con dimensiones, vecindades y tipos de frontera, en lugar de centrarse en la aplicación del proyecto. La sección quedó subdividida:

```
76:  \subsection{Espacio celular y teselaciones}
89:  \subsection{Autómatas celulares probabilísticos}
95:  \subsection{Vecindades}
```

El autómata se define por la cuádrupla $(Q, \mathcal{N}, \delta, q_0)$. Se cubren las dimensiones 1D, 2D y superiores, y las teselaciones cuadrada, hexagonal y triangular. Las vecindades tratadas son las de Moore y Von Neumann. Las condiciones de frontera son tres: periódica, fija y reflejante, con la justificación de por qué en aplicaciones urbanas se prefiere la fija (`cap02:106`).

### 4.2 Imágenes ilustrativas en el marco teórico (NOTAS:33, transcripción 00:69): resuelto

El capítulo tiene cinco figuras. Tres se generaron por programa con `generate_cap02_figures.py` y son reproducibles: el espacio celular con las cuatro teselaciones, las tres condiciones de frontera y la matriz de confusión. Las dos restantes son la comparación de vecindades y el concepto de WoE.

### 4.3 Sección de Pesos de Evidencia con ventajas y limitaciones (transcripción 00:73): resuelto

Este era el último punto de la sesión sin atender. Se resolvió en esta ronda. La Dra. pidió una subsección que explicara el método con sus ventajas y limitaciones, y que **justificara la combinación de WoE con autómatas celulares** para calibrar las reglas de transición. La sección era un bloque plano de siete párrafos. Ahora:

```
134: \section{Pesos de Evidencia}
140:   \subsection{Formulación del método}
176:   \subsection{Information Value y ponderación de variables}
197:   \subsection{Ventajas y limitaciones}
205:   \subsection{Aplicación: del WoE a la regla de transición del autómata}
```

Las ventajas son tres y las limitaciones cuatro, declaradas una por una. La justificación del acoplamiento está donde la Dra. la pidió, y con el argumento frente a las alternativas. Una regresión logística entrega un coeficiente por variable, no un peso por intervalo: impone la forma de la relación en vez de estimarla. Los pesos internos de una red neuronal no admiten lectura geográfica. El cierre enlaza con el estado del arte: «diversos estudios han integrado ya el WoE con modelos de autómatas celulares para calibrar sus reglas de transición».

La subsección de *Information Value* se añadió en esta misma ronda por una razón adicional: el IV es el mecanismo de ponderación de la ecuación central del modelo y no estaba definido en ninguna parte del documento.

### 4.4 Métricas con introducción y subsecciones (transcripción 00:70, 00:74): resuelto

La Dra. pidió un párrafo que introdujera la necesidad de evaluación cuantitativa antes de presentar las métricas, y subsecciones individuales «para conceptos como la matriz de confusión, exactitud y precisión».

El párrafo introductorio está en `cap02:217`. La estructura:

```
221: \subsection{Matriz de confusión}
251: \subsection{Métricas de clasificación global}
       \subsubsection{Exactitud global}
       \subsubsection{Precisión}
       \subsubsection{Recall}
       \subsubsection{F1-Score}
       \subsubsection{Coeficiente Kappa de Cohen}
317: \subsection{Métricas de cambio}
347: \subsection{Descomposición del error}
```

La exactitud y la precisión tienen subsección propia, como se pidió literalmente. FoM, Kappa, IoU y la descomposición de Pontius se definen aquí, por ser conceptos estándar y no antecedentes.

### 4.5 SLEUTH definido en el marco teórico (transcripción 00:78): resuelto

La Dra. pidió definir el concepto aquí y dejar el análisis crítico para los antecedentes. `cap02:93` desglosa el acrónimo capa por capa y describe los cinco coeficientes de crecimiento. El problema de calibración, equifinalidad y sobreajuste se discute como antecedente en el Cap. 3.

### 4.6 Densidad de citas y narrativa, no glosario (NOTAS:33, transcripción 00:67, 00:68): resuelto

Cada concepto es una subsección con su cita de origen, encadenada con la siguiente. El capítulo abre declarando qué va a definir y por qué (`cap02:10`), y cierra remitiendo al estado del arte.

### 4.7 Balance de longitud: resuelto

La regla era que el capítulo de conceptos no fuera más largo que el del modelo. El pseudocódigo de StandardScaler, PCA, K-Means y SVM se movió al Apéndice A, dejando en el cuerpo el fundamento conceptual con punteros a los algoritmos.

---

## 5. Capítulo 3: Estado del arte

### 5.1 Análisis crítico por clasificaciones (transcripción 00:62, 00:76): resuelto

La instrucción fue ir de lo general a lo particular con clasificaciones, ofreciendo un análisis crítico que identificara las brechas, en lugar de listar autores. Estructura:

```
14:  \section{Evolución del modelado urbano computacional}
49:  \section{Problemas metodológicos persistentes}
205: \section{Pesos de Evidencia y modelos probabilísticos espaciales}
294: \section{Validación espacial en modelos LUCC}
317: \section{Vacíos específicos en el contexto mexicano}
379: \section{Síntesis: posicionamiento del presente trabajo}
```

Cada bloque cierra con lo que la familia de modelos resuelve y lo que deja abierto. La síntesis final deriva cuatro requerimientos, R1 a R4, que el Cap. 6 retoma para evaluar el propio modelo (`cap07:44`).

### 5.2 Resumen del capítulo sin tablas ni figuras (transcripción 00:75): resuelto

Verificado: el entorno `resumen` de este capítulo contiene **cero** referencias a tablas o figuras.

### 5.3 Cierre con la brecha y por qué importa (transcripción 00:80): resuelto

El capítulo no se limita a decir que algo no se ha hecho: justifica por qué importa y traduce cada brecha en un requerimiento. El giro humilde que la Dra. autorizó para esta sección se redactó como «hasta donde alcanza esta revisión», en lugar del plural de primera persona.

### 5.4 Liga con el capítulo siguiente (transcripción 00:75): resuelto

El cierre anuncia el contenido del Cap. 4 y retoma las tres decisiones de diseño que el capítulo justificó. Los cinco capítulos que tienen sucesor cierran con una liga de este tipo.

---

## 6. Capítulos 4, 5 y 6

### 6.1 Diagrama de arquitectura como abstracción (NOTAS:44, transcripción 00:80): resuelto

La Dra. pidió un esquema gráfico del flujo entre componentes, entendido como abstracción y no como implementación técnica, tomando la tesis de Fernando como ejemplo. La sección `cap05:21` presenta la figura `fig:pipeline-woe-ac`, que recorre las cinco etapas de los datos a la predicción. El UML se quedó en el apéndice, que es donde corresponde una vista de implementación.

### 6.2 Consideraciones del modelo al inicio (NOTAS:43): resuelto

`cap05:14` abre el capítulo declarando qué considera el modelo y qué no: el espacio celular, la celda como píxel de imagen de archivo, el tratamiento de la frontera y la ausencia de decaimiento por distancia.

### 6.3 Conclusiones en pretérito (NOTAS:25, transcripción 00:60): resuelto

La Dra. distinguió entre la introducción, en presente, y las conclusiones, en pretérito. El Cap. 6 usa pretérito en todo el capítulo: «Esta investigación abordó el problema de modelar…» (`cap07:20`). Los verbos que describían acciones de investigación en presente se corrigieron.

### 6.4 Orden alcances → limitaciones → trabajo futuro (transcripción 00:61): resuelto

El orden exigido, con el trabajo futuro derivado de las limitaciones:

```
87:  \section{Alcances}
115: \section{Limitaciones del modelo}
141: \section{Trabajo futuro}
```

Las limitaciones se desglosan en alcance del estudio, conceptuales, empíricas y de transferibilidad. Cada línea de trabajo futuro responde a una de ellas.

### 6.5 Validación de hipótesis alineada: resuelto

`cap07:58` consolida el veredicto de la hipótesis general y de las cuatro particulares en una tabla única, con la evidencia de cada una. El Cap. 5 desarrolla el razonamiento y el Cap. 6 recoge el veredicto, sin repetir el argumento.

### 6.6 Representación en píxeles y resolución de la fuente: resuelto, con corrección de cifras

Tres pasajes de la sesión piden lo mismo desde ángulos distintos. Se citan por su línea en `TRANSCRIPTION.TXT`, no por número de tema. La Dra. pidió «una introducción de cómo representar los píxeles y por qué se representan así» y precisó que eso es una decisión metodológica propia y va en el capítulo del trabajo (línea 1012). Pidió también hablar «de qué son las imágenes satelitales y imagen de alta resolución, un poco de Google Earth y sus imágenes históricas como fuente de información geoespacial» (línea 1041). Y volvió sobre ello al describir cómo enunciar las consideraciones del modelo: «el espacio se divide en subáreas que representan tal, las cuales se toman en píxeles con la finalidad de tal» (línea 1960).

El apartado existe y está en el capítulo correcto (`cap05:17` abre con la rejilla, la celda y la frontera; `cap05:54-60` desarrolla la fuente, el encuadre y la resolución). Lo que faltaba era que las cifras fueran ciertas.

**El problema detectado.** El capítulo declaraba una ventana de 25 × 25 km, unos 625 km², sobre una rejilla de 1792 × 3024 píxeles. Esa combinación es aritméticamente imposible: la rejilla tiene proporción 1,6875, así que una ventana cuadrada exigiría píxeles de 8,3 m en horizontal y 14 m en vertical. De ahí venía el rango declarado de «8 a 14 m/píxel» y los «115 m² por píxel». Nada de eso estaba medido.

**Cómo se resolvió.** Las capturas son imágenes de pantalla del visor: no llevan georreferencia ni barra de escala, así que la extensión no se puede leer del archivo y hay que reconstruirla. Se registró cada una de las 37 capturas contra el mosaico Sentinel-2 *cloudless*, que es la misma familia de imagen que el visor despliega a ese nivel de zoom, según la atribución «Image Landsat / Copernicus» impresa en las propias capturas. El registro dio solución sólida en 17 imágenes; las 20 que quedaron fuera son todas anteriores a 2005, donde el mosaico histórico no aporta rasgos suficientes.


| Parámetro            | Valor declarado antes       | Valor medido                                                       |
| -------------------- | --------------------------- | ------------------------------------------------------------------ |
| Ventana              | 25 × 25 km, 625 km²         | **80,8 × 47,9 km, 3 870 km²**                                      |
| Tamaño de píxel      | 8 a 14 m (anisotrópico)     | **26,7 m, cuadrado**                                               |
| Superficie por píxel | 115 m²                      | **713 m²**                                                         |
| Centro               | 20,59° N, 100,39° W         | **20,6144° N, 100,3843° W**                                        |
| Orientación          | implícitamente norte arriba | **girada: el norte queda a 22° en sentido horario de la vertical** |


El giro se confirmó por tres vías independientes que concuerdan: dos registros contra fuentes distintas, que dan −21,61° y −21,69°, y la medición directa de la pista del aeropuerto de Querétaro, cuyo rumbo verdadero es 101° y que en la captura aparece a 33,9°, lo que implica 22,9°.

**Por qué la corrección fortalece el capítulo.** Los 26,7 m/píxel coinciden con la resolución nativa de Landsat, 30 m, que es justo lo que declara la atribución del visor. La versión anterior afirmaba entre 8 y 14 m/píxel sobre un mosaico Landsat, es decir, reclamaba más detalle del que la fuente contiene. `cap05:60` ahora explica esa coherencia y añade la escala física de la regla de transición: las vecindades de 3×3, 5×5 y 7×7 píxeles miden 80, 134 y 187 m de lado.

**Lo que además quedó demostrado.** El criterio de consistencia geométrica entre años que el capítulo afirmaba pasó de ser una intención a ser una medición. Entre capturas el tamaño de píxel varía 0,4 %, el giro del marco tiene una desviación de 0,37° y el centro se mantiene dentro de 230 m sobre una ventana de 80 km (`cap05:58`). En consecuencia, la fila de georreferenciación de la tabla de incertidumbre del apéndice bajó de riesgo «bajo-medio» a «bajo», con la medición como respaldo (`apendice:164`).

El procedimiento es reproducible: `tools/registrar_s2.py` hace el registro, `tools/verificar_encuadre.py` rehace la escena con los parámetros estimados para comprobarlos, y los resultados por año quedan en `data/processed/georreferencia_capturas.json`. Todo ello citado en `apendice:21`.

---

## 7. Cambios globales

### 7.1 Estilo de citación numérico (NOTAS:36, transcripción 00:77): resuelto

La Dra. señaló que el estilo numérico IEEE se prefiere en ingeniería por ahorrar espacio y facilitar la identificación por orden de aparición. `standalone_preamble.tex:165` declara `style=ieee`. La bibliografía impresa tiene 79 entradas, todas citadas, numeradas por aparición.

### 7.2 Modo impersonal en todo el documento (NOTAS:35, transcripción 00:74): resuelto

Búsqueda de primera persona (consideramos, proponemos, nuestro, nuestra, realizamos, obtuvimos, hicimos, creemos, analizamos, presentamos) en los seis capítulos y el front matter: **cero ocurrencias**.

### 7.3 Etiquetar las imágenes generadas con IA (NOTAS:37, transcripción 00:72): resuelto

Doce figuras se generaron con IA y las doce llevan la leyenda en el pie: dos en el Cap. 2, ocho en el Cap. 4 y dos en el Cap. 5.

### 7.4 Fuentes de las imágenes con IA (NOTAS:38, transcripción 00:72): resuelto

La Dra. pidió identificar la fuente original cuando la imagen contuviera datos específicos o se basara en literatura existente. Las doce leyendas declaran su origen conceptual. Unas remiten a la formulación propia de la ecuación o del algoritmo del capítulo; otras, a la fuente publicada. La figura de la matriz de confusión del FoM cita a Pontius. Las dos del Cap. 5 declaran que se construyeron con los resultados de validación propios. Una búsqueda del pie sin atribución devuelve **cero ocurrencias**.

### 7.5 Toda figura referenciada e interpretada: resuelto

No basta con colocar la figura: cada una se referencia en el texto y se acompaña de su lectura. Se detectaron y corrigieron cinco tablas que no se referenciaban en el cuerpo.

Una verificación posterior encontró un caso más que se había escapado: la Figura `fig:metricas-quinquenales`, con el desempeño comparado de las cinco ventanas, estaba colocada en el Cap. 5 sin que ningún párrafo la invocara. Se añadió su lectura en `cap06:244`, que es además la que faltaba en el capítulo: las tres métricas coinciden en señalar las dos ventanas más débiles, que son las de menor crecimiento neto observado, mientras que entre las tres restantes el orden cambia según la métrica, de modo que la jerarquía fina no es robusta. Se explica también por qué el FoM se dispersa en un factor de 1,70 entre ventanas y el IoU apenas en 1,15: el IoU incorpora las celdas que ya eran urbanas al inicio y el FoM excluye por construcción el acierto sobre la permanencia.

Estado verificado con un recuento automático: cero figuras y cero tablas de nivel superior sin referencia, y en las figuras compuestas por subfiguras la figura madre siempre se referencia.

### 7.6 Ligas entre capítulos (transcripción 00:75): resuelto

Verificado uno por uno: los cinco capítulos con sucesor cierran anunciando el siguiente.

### 7.7 Cifras de terceros cotejadas contra la fuente (transcripción 00:57): resuelto

La Dra. exigió referencia en todo dato cuantitativo. El requisito fuerte no es que la cifra lleve cita, sino que la cita diga lo que se le atribuye. La única cifra de terceros que sostenía una comparación de desempeño y no se había cotejado era la de Tang, que aparece en `cap03:302` y en la tabla comparativa de `cap06:236`.

Cotejada contra el artículo (*Scientific Reports* 14:21106, doi 10.1038/s41598-024-71709-4): reporta FoM de 0,4303 y 0,3764 con un autómata acoplado a búsqueda gravitacional en Urumqi. Las dos cifras, el método, la ciudad y la entrada bibliográfica son correctos.

Se añadió un matiz que faltaba y que conviene al argumento: 0,430 corresponde a la fase de calibración, de 2000 a 2010, y 0,376 a la de validación, de 2010 a 2020. La tesis las daba como «2010» y «2020» sin distinguirlas. La cifra comparable con un esquema de prueba independiente es la segunda, y el promedio de 0,317 de esta tesis sale precisamente de un *hold-out* temporal.

### 7.8 Figuras de arquitectura rehechas y verificadas contra los datos

El diagrama de arquitectura del Cap. 4 se sustituyó por uno operativo que declara, para cada etapa, lo que recibe, lo que hace y lo que entrega, con la ruta del artefacto y un ejemplo real (`cap05:30`). Se usa la versión en blanco y negro, `framework_woe_ac_bn.png`, por ser la adecuada para impresión; la variante en color queda en el repositorio sin uso. Se añadió además un sub-pipeline de clasificación que muestra la cadena aplicada a un recorte de 8×8 celdas (`cap05:248`). Las dos llevan la leyenda de origen con IA.

Como las dos figuras traen cifras y rutas dentro de la imagen, se cotejó cada dato contra el repositorio antes de integrarlas. Salieron exactos los 1792×3024 píxeles y los 5,4 millones de celdas, los 23 descriptores, el PCA a 8 dimensiones, los 26 pares consecutivos, el umbral 0,75 con vecindad 0,50, los cinco pasos anuales, el arranque de 2015 en 2 145 356 celdas urbanas que predice 3 827 553 para 2020, las métricas 0,378, 0,498 y 0,669 de la ventana 2015-2020, el promedio de 0,317 y la afirmación de que `distance_urban` concentra cerca del 28 % del peso, que medida sobre el modelo entrenado da 28,2 %.

Se detectaron y corrigieron tres discrepancias en las imágenes. La primera era un error de dato: el ejemplo de la etapa E2 atribuía a `2011.npy` un conteo de 2 145 356 celdas urbanas, que en realidad es el de 2015; el valor correcto de 2011 es 2 699 088, el 49,8 % del mapa. Las otras dos contradecían al texto: la varianza retenida por el PCA figuraba como 95 % cuando el promedio de la serie es 96,1 %, y el tamaño de celda como 30 m cuando la geometría medida da 26,7 m. Las tres quedaron corregidas en el archivo de la figura.

### 7.9 Convención de etiquetas, documentada

La figura operativa dejó a la vista un paso del pipeline que el texto no explicaba. El agrupamiento K-Means no nombra sus clases, de modo que la etiqueta de urbano puede salir invertida en un año respecto de otro. El código lo resuelve tomando el 30 % central de la rejilla y verificando qué etiqueta domina ahí: si domina el cero, invierte el mapa. Estaba implementado en `src/tesis_ac/historical/standardize_labels.py` y no aparecía en la tesis. Se añadió su descripción en `cap05:253`, con la ruta del módulo.

---

### 7.10 Auditoría de fuente por fuente del estado del arte

La Dra. pidió que toda cifra llevara referencia. Llevado al extremo, el requisito es más exigente: que la referencia diga lo que se le atribuye. Se auditó por eso el Capítulo 3 completo, afirmación por afirmación, yendo al texto de cada fuente y no a lo que la literatura derivada dice de ella. La bibliografía pasó de 71 a 79 entradas, todas citadas, y el capítulo cambió en veinte lugares.

Lo que sobrevivió intacto es la cifra que sostiene la comparación de desempeño. Pontius y colegas reportan en efecto seis de trece aplicaciones con FoM por debajo de 0,15 y una sola por encima de 0,50, con un rango de 0,01 a 0,59. Y Tang reporta 0,4303 y 0,3764, como ya se había cotejado.

El hallazgo de mayor consecuencia toca al aparato conceptual del propio método. El *Information Value* no pertenece a la formulación de Pesos de Evidencia de Bonham-Carter: el capítulo 9 de su libro trabaja con los pesos, el contraste y su varianza, y no menciona el estadístico ni su escala de interpretación. La fórmula y los cinco tramos que usa la tesis son la convención de las tarjetas de puntuación de riesgo de crédito atribuida a Siddiqi (2006). El riesgo era doble, porque en la literatura geoespacial *information value* ya nombra un método bivariado distinto, el de Yin y Yan (1988), que se compara con el WoE como rival. La matemática de la tesis es correcta, pues el estadístico equivale a la divergencia de Kullback-Leibler simetrizada entre las dos distribuciones condicionales; lo que estaba mal era la atribución. Corregida en `cap02:138`, `cap02:179`, `cap02:186` y `cap03:262`.

Se corrigieron además cinco atribuciones que las fuentes no sostenían. La equifinalidad se atribuía a Clarke; ahora va a Beven (2006), de donde viene el término, y la discusión sobre calibración óptima de SLEUTH a Dietzel y Clarke. La automodificación y el esquema de búsqueda en tres fases van a Clarke y Hoppen (1997), y la sensibilidad a la escala a Jantz y Goetz (2005). La crítica al coeficiente Kappa se atribuía a Pontius y colegas (2008), artículo donde la palabra «Kappa» no aparece; corresponde a Pontius y Millones (2011), que se titula precisamente *Death to Kappa*. La concentración geográfica de la literatura en China y Estados Unidos se atribuía a dos revisiones que no desglosan por región; ahora se respalda con Wahyudi y Liu (2016), que clasifica ochenta y ocho aplicaciones por región. Y la autocorrelación espacial se atribuía a Bonham-Carter, que en realidad trata la celda unitaria y llega a una conclusión casi opuesta; el respaldo correcto es la literatura de validación de modelos espaciales.

Dos afirmaciones estaban desactualizadas y una invertida. Sobre el licenciamiento, Dinamica EGO es gratuito incluso para uso comercial y TerrSet se liberó sin costo en diciembre de 2024, así que el argumento de barrera económica ya no existe; reformulado como código cerrado, que es lo que en realidad exige el requerimiento R3 y no depende del precio. Sobre Wang (2021), el capítulo reprochaba que sus cromosomas carecieran de lectura geográfica, cuando cada gen es una celda del territorio; la objeción defendible, que es la que ahora aparece, es que se trata de una optimización normativa sin validación contra un mapa observado. Y el espacio de búsqueda de SLEUTH es del orden de 10¹⁰ combinaciones, no 10⁹.

En la literatura mexicana el capítulo era injusto con dos trabajos. Suárez y Delgado sí calibran sobre cambio celular histórico, en 15 670 celdas de una hectárea con 82,9 % de aciertos, y publican todos sus coeficientes, de modo que su función de asignación es auditable; se les subió el requerimiento R4 de «no» a cumplido. Y el modelo de Ramírez Hernández no está inspirado en SLEUTH: es econométrico con simulación Monte Carlo, y el libro solo menciona SLEUTH al reseñar a terceros.

### 7.11 Un antecedente que obligó a precisar el alcance

La auditoría encontró un trabajo que el capítulo no citaba y que afecta al posicionamiento. Jiménez López, Chávez y Garrocho, de El Colegio Mexiquense, simularon con autómatas celulares la expansión de Querétaro entre 2003 y 2017, con un barrido de las 256 reglas de transición posibles, y reportan para la regla ganadora un Kappa de Cohen de 0,53 y un índice de Jaccard de 0,76. El mismo grupo generalizó el método en 2021 en *Estudios Demográficos y Urbanos* y declara código abierto. Ambos datos se verificaron contra el PDF original y contra Crossref.

La tesis afirmaba que ningún trabajo mexicano cumple los cuatro requerimientos y que Querétaro es una ciudad poco estudiada. Lo primero se sostiene, porque esos trabajos validan una sola ventana temporal por ciudad y su regla de transición es binaria sobre el estado de la vecindad, sin pesos por variable. Lo segundo no. Se optó por citarlos, incorporarlos a la tabla de posicionamiento y reformular la brecha: la aportación no está en el caso de estudio sino en la combinación de validación en cinco ventanas desplazadas, implementación abierta y una función de transición con pesos individualmente auditables. Ajustado en `cap03`, y en las aportaciones de `cap05` y `cap07`, donde se retiró el calificativo de ciudad poco estudiada.

Preferimos que el capítulo cite a ese grupo y acote su propia aportación, antes que sostener un «ningún trabajo» que cualquier sinodal refuta en dos minutos.

### 7.12 Por qué las cifras de ese antecedente no son un punto de comparación

Citar ese trabajo abrió un riesgo que conviene cerrar de forma explícita, porque sus números son, en apariencia, mejores que los de esta tesis: Kappa de 0,53 frente a un promedio de 0,445, y Jaccard de 0,76 frente a un IoU promedio de 0,633. Leídos sin más, invitarían a concluir que el antecedente supera al modelo propuesto.

La diferencia no está en la métrica sino en el diseño experimental. En ese trabajo el mapa observado de 2017 cumple dos funciones a la vez: es el criterio con el que se elige la regla de transición entre las 256 posibles, y es también el mapa contra el cual se mide el ajuste de la regla elegida. Se resuelve un máximo sobre el espacio de reglas y después se publica ese máximo como desempeño, de modo que la cifra reportada no es una estimación independiente. El sesgo optimista de un procedimiento así crece con el tamaño del espacio explorado, y aquí el espacio es la totalidad de las reglas. A eso se suma que la regla seleccionada se usa para proyectar a 2031 sin ningún intervalo posterior observado que permita comprobar si conserva su desempeño, y que cada ciudad recibe su propio óptimo (192 para Querétaro, 218 para San Luis Potosí, 222 para Toluca), por lo que las tres aplicaciones demuestran capacidad de ajuste individual y no generalización entre sitios.

El esquema de esta tesis es el contrario, y está documentado en `data/processed/quinquenal_best_config.json` y en `cap06:156`: los pesos WoE, su ponderación por *Information Value* y el umbral de transición se fijan con la serie de 1984 a 2010, y las cinco ventanas de 2011 a 2020 se evalúan sin reajustar nada. Los parámetros quedan congelados antes de observar el período de evaluación, y se reporta el rango completo de las cinco ventanas en lugar del valor más favorable. Esa es la razón por la que las dos familias de cifras no son comparables, y quedó declarada en `cap03`, en la tabla comparativa de `cap06:243` y en las aportaciones de `cap07`.

Para que el contraste sea verificable conviene dejar por escrito cuántas constantes tiene la regla de transición, porque son dos y sólo dos. La línea que combina evidencias es una sola en cada script de validación (`run_all_quinquenal_validations.py:197` y `validate_quinquenal_2011_2016.py:232`) y en ella entran el umbral $\theta = 0{,}75$ y el peso de vecindad $\alpha = 0{,}50$. Nada más. El umbral está justificado en `cap06:156` por la sobre-predicción que producían los valores del rango 0,40 a 0,60. El peso de vecindad es un valor redondo fijado a priori, que da a la vecindad la mitad del peso de la probabilidad WoE, y es el mismo para los cinco períodos: no se reeligió por ventana ni se ajustó contra los mapas de evaluación. Frente a una regla distinta por ciudad elegida contra el mapa que después se usa para medir, la diferencia de diseño se sostiene.

Un detalle del repositorio, por si alguien consulta los archivos de configuración. `data/processed/ga_calibrated_params.json` conserva tres campos del experimento exploratorio con algoritmos genéticos, y la validación publicada usa uno. El umbral de 0,596 queda sustituido por 0,75 en el propio script (`run_all_quinquenal_validations.py:58`). El campo `distance_weight` con valor 1,963 se lee y se escribe en la bitácora, pero no interviene en ningún cálculo: su única aparición es la copia al diccionario de parámetros (`run_all_quinquenal_validations.py:306`). Por eso no figura en el documento. El modelo de esta tesis no tiene peso de distancia, y el peso de vecindad de 0,50 es el único valor que hereda de ese experimento.

---

## 8. Puntos de la transcripción fuera del alcance de la tesis

Cuatro pasajes de la sesión no corresponden a cambios en el documento. Se registran para que quede constancia:

- **Anecdotario sobre la escritura de tesis** (00:56) y **consejos generales** (00:81). Recomendaciones de método de trabajo. La indicación operativa que sí se aplicó es leer el texto como evaluador externo, que gobernó las auditorías.
- **Verificación de versiones del documento** (00:59). Resuelto durante la propia sesión.
- **Consideraciones para la aplicación al doctorado** (00:82). Ajeno a la tesis.
- **Compartir la tesis de Fernando y enviar comentarios finales** (NOTAS:45-46). Tareas de la Dra., no del documento. La tesis de referencia se usó como modelo para el diagrama de arquitectura.

---

## 9. Lo que sigue abierto

De la sesión con la Dra. queda un solo punto, y depende de un documento que no está en el repositorio.

**Cotejar el título contra la carta del jurado.** El título está en su forma acordada en `front/portada.tex:27-29`. La Dra. advirtió que una diferencia de una sola preposición causa problemas administrativos en la aceptación del documento. La carta no está en el repositorio. La comparación carácter por carácter requiere ese insumo.

### Pendientes de formato institucional, ajenos a la sesión

Estos tres no salieron de la reunión con la Dra., pero conviene cerrarlos antes de entregar.

**Faltan los dos escudos**, `figures/escudo_unam.png` y `figures/logo_pcic.png`. Están protegidos con `\IfFileExists`, así que la portada compila en modo solo texto y el documento no falla, pero hay que colocarlos antes de imprimir.

**Hoja de restricciones de uso de la UNAM: resuelta.** Se añadió `front/restricciones.tex` y se incluye en `main.tex:13`, entre la portada y la dedicatoria, de modo que queda como página 2 igual que en la tesis de Israel Velázquez, del mismo posgrado y con la misma asesora. El texto se reproduce literalmente como lo publica la Dirección General de Bibliotecas, sin corregirlo. Falta únicamente el banner institucional que la DGB coloca en el encabezado: si se consigue esa imagen y se guarda como `figures/banner_dgb.png`, la página la incluye sola.

**Dos defectos de maquetación detectados y corregidos.** La portada se desbordaba por 17 pt y empujaba la línea del Instituto y la fecha a una segunda página numerada «ii»; se rehízo con espaciado elástico para que quepa siempre, incluso al agregar los escudos. Y la figura de las siete variables espaciales, montada como `wrapfigure`, arrancaba con cuatro renglones libres cuando necesitaba catorce, de modo que se salía por el borde inferior de la página y pisaba el folio; se movió dos párrafos y ahora envuelve dentro de la caja. Se verificó sobre las coordenadas del PDF que ningún texto ni imagen queda fuera de los márgenes en todo el documento.

---

## Anexo: correspondencia punto por punto

### Acuerdos de la lista de próximos pasos


| #   | Línea    | Acuerdo                                            | Estado                 | Evidencia                       |
| --- | ------------ | ------------------------------------------ | -------------------- | ------------------------------------- |
| 1   | NOTAS:20 | Cambiar el título al acordado                      | Aplicado, falta cotejo | `front/portada.tex:27`          |
| 2   | NOTAS:21 | Agradecimientos con la beca                        | Resuelto               | `front/agradecimientos.tex:6`   |
| 3   | NOTAS:22 | Reescribir el resumen                              | Resuelto               | `front/resumen.tex`             |
| 4   | NOTAS:23 | Reestructurar la introducción                      | Resuelto               | `cap01:8,97,101,109,180,212`    |
| 5   | NOTAS:24 | Referencias en datos cuantitativos                 | Resuelto               | Cap. 1 sin cifras sin cita      |
| 6   | NOTAS:25 | Conclusiones en pretérito                          | Resuelto               | `cap07:20,87,115,141`           |
| 7   | NOTAS:26 | Actualizar el abstract en inglés                   | Resuelto               | `front/abstract.tex`            |
| 8   | NOTAS:27 | Corregir la estructura de la introducción          | Resuelto               | Secciones y transiciones        |
| 9   | NOTAS:28 | Antecedentes y justificación separados             | Resuelto               | `cap01:97,101`                  |
| 10  | NOTAS:29 | Adaptar el protocolo a la introducción             | Resuelto               | Cap. 1 en presente              |
| 11  | NOTAS:30 | Objetivos coherentes con los resultados            | Resuelto               | `cap07:40`                      |
| 12  | NOTAS:31 | Hipótesis: una general con particulares            | Resuelto               | `cap01:165,167`                 |
| 13  | NOTAS:32 | Mover el preprocesamiento del marco teórico        | Resuelto               | `cap05:75`                      |
| 14  | NOTAS:33 | Enriquecer el marco teórico                        | Resuelto               | 5 figuras, subsecciones citadas |
| 15  | NOTAS:34 | Reorganizar la estructura de capítulos             | Resuelto               | `main.tex:25-31`                |
| 16  | NOTAS:35 | Modo impersonal                                    | Resuelto               | Cero primera persona            |
| 17  | NOTAS:36 | Citación numérica por aparición                    | Resuelto               | `standalone_preamble.tex:165`   |
| 18  | NOTAS:37 | Etiquetar las imágenes con IA                      | Resuelto               | 12 figuras                      |
| 19  | NOTAS:38 | Fuentes de las imágenes con IA                     | Resuelto               | 11 leyendas con origen          |
| 20  | NOTAS:39 | Unir los capítulos cortos                          | Resuelto               | Seis capítulos                  |
| 21  | NOTAS:40 | Resumen con el modelo híbrido                      | Resuelto               | `front/resumen.tex`             |
| 22  | NOTAS:41 | Fusionar área, preprocesamiento y modelo           | Resuelto               | `cap05:33,75,285`               |
| 23  | NOTAS:42 | Renombrar a «Modelo de crecimiento urbano híbrido» | Resuelto               | `cap05:7`                       |
| 24  | NOTAS:43 | Consideraciones del modelo al inicio               | Resuelto               | `cap05:14`                      |
| 25  | NOTAS:44 | Diagrama de arquitectura                           | Resuelto               | `cap05:21`                      |


### Detalles de la transcripción


| Línea | Detalle                                                       | Estado               | Evidencia                       |
| ------ | ------------------------------------------------------- | -------------------- | -------------------------------------- |
| 50    | Título y carta del jurado                                     | Pendiente de cotejo  | `front/portada.tex:27`          |
| 51    | Beca obligatoria en agradecimientos                           | Resuelto             | `front/agradecimientos.tex:6`   |
| 52    | Realzar la contribución en el resumen                         | Resuelto             | `front/resumen.tex`             |
| 53    | Enfoque de embudo                                             | Resuelto             | `cap01:8`                       |
| 54    | Contribución = protocolo multitemporal                        | Resuelto             | `cap01:180`                     |
| 55    | Secciones de la introducción                                  | Resuelto             | `cap01:8-212`                   |
| 56    | Anecdotario                                                   | Fuera de alcance     | No aplica                       |
| 57    | Referencias en datos y figuras                                | Resuelto             | Cap. 1                          |
| 58    | Antecedentes y justificación                                  | Resuelto             | `cap01:97,101`                  |
| 59    | Versiones del documento                                       | Resuelto en sesión   | No aplica                       |
| 60    | Presente en intro, pretérito en conclusiones                  | Resuelto             | `cap07:20`                      |
| 61    | Alcances → limitaciones → trabajo futuro                      | Resuelto             | `cap07:87,115,141`              |
| 62    | Categorizar, no listar estudios                               | Resuelto             | Cero «N trabajos»               |
| 63    | Objetivos coherentes con resultados                           | Resuelto             | `cap07:40`                      |
| 64    | Hipótesis como afirmación validable                           | Resuelto             | `cap01:165`                     |
| 65    | Separar marco teórico de metodología                          | Resuelto             | `cap05:75`                      |
| 66    | Propósito del marco teórico                                   | Resuelto             | `cap02:10`                      |
| 67    | No glosario, conceptos vinculados                             | Resuelto             | Subsecciones encadenadas        |
| 68    | Marco teórico muy referenciado                                | Resuelto             | Cita por concepto               |
| 69    | AC: dimensiones, vecindades, fronteras                        | Resuelto             | `cap02:76,95,106`               |
| 70    | Quitar «el presente modelo»                                   | Resuelto             | Cero ocurrencias                |
| 71    | Transiciones entre temas                                      | Resuelto             | Texto introductorio por sección |
| 72    | Referenciar contenido generado con IA                         | Resuelto             | 11 leyendas                     |
| 73    | **Subsección WoE con ventajas, limitaciones y justificación** | Resuelto             | `cap02:134-205`                 |
| 74    | Métricas: introducción y subsecciones                         | Resuelto             | `cap02:217,221,251`             |
| 75    | Ligas entre capítulos; resumen sin tablas                     | Resuelto             | Cinco cierres; cero refs        |
| 76    | Literatura de lo general a lo particular                      | Resuelto             | `cap03:14-377`                  |
| 77    | Estilo numérico IEEE                                          | Resuelto             | `standalone_preamble.tex:165`   |
| 78    | Definir en marco teórico, brechas en antecedentes             | Resuelto             | `cap02:93`                      |
| 79    | Fusionar capítulos; consideraciones al inicio                 | Resuelto             | `cap05:14`                      |
| 80    | Arquitectura como abstracción                                 | Resuelto             | `cap05:21`                      |
| 81    | Leer como evaluador externo                                   | Aplicado como método | Auditorías                      |
| 82    | Aplicación al doctorado                                       | Fuera de alcance     | No aplica                       |


