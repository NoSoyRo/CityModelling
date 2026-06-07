# Guion de presentación — Coloquio final 2026-II

**Tema:** Modelado del crecimiento urbano de la ZMQ con Weight of Evidence + Autómatas Celulares
**Duración objetivo:** ~20 minutos · 22 diapositivas
**Archivo:** `ppt_coloquio_final_2026.pdf`

> Sugerencia de ritmo: ~45–60 s por diapositiva de contenido. No leas las viñetas; úsalas como ancla. Lo que está entre comillas es lo que puedes decir casi literal; lo demás son notas.

---

## ¿Qué significa cada métrica? (léelo antes, no lo digas literal en la presentación)

> Esta sección es para que **tú** entiendas y puedas explicar con tus palabras si te preguntan. No son notas del guion.

### FoM — Figure of Merit

Imagínate que predices exactamente qué manzanas nuevas se van a construir en 5 años.

- **FoM = celdas que predijiste bien ÷ (todas las que predijiste + todas las que en realidad crecieron − las que acertaste)**
- Si aciertas 30 de 100 nuevas construcciones y predijiste 40 en total, FoM ≈ 0.30.
- **Lo clave**: no te da puntos por las zonas que ya eran urbanas y siguieron siendo urbanas (que es el 95 % del mapa). Por eso es la métrica más honesta para crecimiento urbano: solo mide lo que cambió.
- **Referencia**: Pontius (2008) revisó 13 modelos publicados; 6 de ellos tienen FoM menor a 0.15 y solo 1 supera 0.50. Tu promedio de **0.317** está en la banda intermedia.

### Kappa

Imagínate que un mono lanzara dardos al mapa al azar y también "predijera" cuánto creció la ciudad.

- **Kappa mide qué tan mejor que ese mono es tu modelo.** 0 = igual que el azar. 1 = perfecto.
- 0.40–0.60 = acuerdo "moderado a sustancial" (escala de Landis & Koch).
- Tu promedio de **0.445** está en ese rango: el modelo es claramente mejor que tirar monedas al aire.

### IoU — Intersection over Union

Dibuja en papel el área que predijiste que iba a crecer. Dibuja también el área que en realidad creció.

- **IoU = el área donde se traslapan ÷ el área total que cubrieron las dos zonas juntas.**
- Si los dos polígonos son idénticos, IoU = 1. Si no se tocan, IoU = 0.
- Tu promedio de **0.633** significa que predicho y observado se solapan el 63 % de su extensión combinada. Buena correspondencia geográfica.

### Accuracy

De todos los píxeles del mapa (urbano + campo), ¿qué porcentaje clasificaste bien?

- **Cuidado**: si el 95 % del mapa ya es campo, un modelo que siempre predice "campo" tendría 95 % de accuracy y sería inútil. Por eso Accuracy no es la métrica principal aquí.
- Tu **0.722** es correcto en el sentido de que incluye tanto la permanencia urbana como los cambios. Es un número honesto, no inflado.

### Recall

De todas las celdas que en realidad se urbanizaron, ¿cuántas capturó tu modelo?

- Si se urbanizaron 100 celdas y tu modelo encontró 91, Recall = 0.91.
- **Alto Recall = pocas omisiones.** Pero un modelo que predice "todo se urbaniza" tendría Recall = 1.0 y sería terrible. Por eso se analiza junto con Precision.
- Tus valores están por encima de **0.91 en todas las ventanas** — el modelo no se pierde casi ningún crecimiento real.

### F1

Es el promedio equilibrado entre Recall y Precision.

- Si eres muy generoso prediciendo crecimiento, Recall sube pero Precision baja (predices muchas celdas que no crecen).
- **F1 penaliza los extremos**: un buen F1 significa que aciertas mucho del crecimiento real Y que tus predicciones son confiables.

---

## SECCIÓN: Planteamiento del problema

## 1 · Portada  *(~30 s)*

"Buenas tardes. Voy a presentar el estado integral de mi tesis: un modelo para simular el crecimiento urbano de la Zona Metropolitana de Querétaro, de 1984 a 2020, usando un autómata celular cuya función de transición es el método Weight of Evidence."

- Preséntate y di que será un overview de todo el proyecto, del problema a las conclusiones.

---

## 2 · Agenda  *(~20 s)*

"El recorrido es: planteamiento del problema, objetivos e hipótesis, estado del arte, los datos y el modelo, los resultados de la validación, y las conclusiones."

- No te detengas; es solo un mapa de ruta.

---

## 3 · El problema  *(~75 s)*

"México sí tiene instrumentos de ordenamiento territorial, pero su cobertura real es limitada: solo el **20.5 %** de los municipios tiene un instrumento vigente y dos tercios de los planes están desactualizados. Al mismo tiempo, entre el 43 y el 80 % de la población vive en zonas urbanas."

"El resultado es lo que se ve en la imagen: un crecimiento **disperso** de la mancha urbana de Querétaro entre 1984 y 2020 que rebasa la capacidad de planeación."

- Cierra con: "Elegí Querétaro porque es una ciudad intermedia muy dinámica, con 37 años de serie histórica y datos de acceso libre."

---

## 4 · Requerimientos R1–R4  *(~60 s)*

"Para que un modelo sea útil para planear, definí cuatro requerimientos: **R1**, simular con precisión y medirlo con FoM, Kappa e IoU; **R2**, dar mapas útiles para planeación; **R3**, ser reproducible sin software propietario; y **R4**, que la función de transición sea **auditable**."

"El más restrictivo resultó ser R4: muchos modelos buenos en precisión son cajas negras."

- R4 es el hilo conductor de toda la tesis; recálcalo.

---

## SECCIÓN: Preguntas e hipótesis

## 5 · Preguntas e hipótesis  *(~70 s)*

"Las preguntas son: ¿puede reproducir el patrón con métricas comparables?, ¿qué variables discriminan mejor?, ¿es estable en las cinco ventanas? y ¿es replicable sin software propietario?"

"De ahí salen **seis hipótesis**: H1, que la evidencia espacial es predictiva; H2, que la calibración importa; H3, que proximidad y densidad dominan; H4, que el desempeño es estable; H5, que una **caja blanca iguala a las cajas negras sin perder interpretabilidad**; y H6, que todo corre con herramientas abiertas sin hardware especializado."

"La apuesta de fondo, abajo, es la clave: interpretabilidad y reproducibilidad **sin** sacrificar desempeño."

---

## SECCIÓN: Estado del arte

## 6 · Treinta años de modelado urbano  *(~70 s)*

"El campo arranca con reglas locales de vecindad —White y Engelen— y SLEUTH de Clarke como referencia por dos décadas. Luego llegan los modelos LUCC con mapas de idoneidad estadística, y después el aprendizaje automático y las metaheurísticas, que mejoran la precisión pero sustituyen parámetros legibles por activaciones de red o cromosomas no auditables."

"La tensión real del campo no es 'AC contra machine learning', sino **precisión** contra **trazabilidad** de cada decisión espacial."

---

## 7 · Problemas abiertos → decisiones de diseño  *(~80 s)*

"Detecté cinco problemas abiertos en la literatura y cada uno justifica una decisión de mi diseño:"

- "Calibración por fuerza bruta → equifinalidad; yo calibro solo **dos** parámetros con significado espacial, y el AC usa decaimiento y vecindario ponderado."
- "Validación en una sola ventana → no distingue robustez de sobreajuste; yo uso **cinco** ventanas quinquenales."
- "Opacidad de la función de transición → uso **WoE**, donde cada peso es interpretable."
- "Dependencia de software propietario → todo en **Python** abierto y modular."
- "Datos satelitales costosos o restringidos → uso **datos de acceso libre** (Google Earth histórico), sin imágenes comerciales, para que la réplica sea viable incluso con bajo presupuesto."

"Esta es la columna vertebral metodológica de la tesis: cada problema de la literatura se convierte en una decisión de diseño justificada."

---

## 8 · Vacío en el contexto mexicano  *(~60 s)*

"Revisé los tres trabajos representativos para ciudades mexicanas. Ninguno cumple los cuatro requerimientos a la vez: o validan sobre totales y no a nivel de celda, o no liberan código, o usan software propietario."

"Esa intersección vacía —validación multi-período, reproducibilidad abierta e interpretabilidad auditable— es exactamente la **brecha** que mi trabajo atiende."

---

## SECCIÓN: Datos y modelo WoE-AC

## 9 · Preprocesamiento  *(~75 s)*

"El reto: con solo RGB, el suelo urbano y el suelo desnudo se parecen mucho. Por eso extiendo cada píxel a **23 features** de textura y color —LBP, Sobel, entropía, LAB, HSV—, porque NDVI o NDBI no se pueden calcular sin infrarrojo."

"Luego estandarizo, reduzco con PCA a 8 componentes que retienen el 96 % de la varianza, agrupo con K-Means en dos clases, y refino la frontera con un SVM lineal que alcanza 88–94 % de exactitud. El resultado es el mapa binario urbano/no-urbano de cada año."

*"La precisión del 88–94 % mide la consistencia del SVM al replicar los clusters de K-Means en un 20 % hold-out. No es una validación contra etiquetas manuales. La validación de la clasificación en sentido absoluto es indirecta: si los mapas binarios fueran sistemáticamente erróneos, el WoE no aprendería pesos coherentes y el FoM del autómata no alcanzaría 0.317 en cinco ventanas independientes."*

- Clave: "K-Means da **consistencia**: el mismo criterio en los 37 años."

---

## 10 · WoE como función de transición  *(~75 s)*

"Una transición es una celda que pasa de no-urbana a urbana entre un año y el siguiente. Para cuantificar la evidencia uso WoE en vez de regresión logística, porque no impone una forma lineal: discretiza cada variable y estima un peso bayesiano por intervalo, capturando la no linealidad real."

"Uso **siete variables derivadas solo del mapa binario** —distancia al frente, densidades a tres escalas, fragmentación, tamaño de cluster y gradiente—, agrego 26 pares de años (más de 8 millones de transiciones) para que los pesos sean estables, y combino por Information Value. La gráfica muestra la relación monótona que valida la difusión."

---

## 11 · ¿Por qué un autómata celular?  *(~70 s)*

"Aquí está la decisión central. El mapa WoE es **estático**: dice qué tan apta es cada celda, pero no cómo se propaga el crecimiento. El autómata celular aplica la regla **iterativamente** y recalcula las variables en cada paso anual."

"Cuando una celda se urbaniza, sube la densidad de sus vecinas y aumenta su probabilidad en el siguiente paso: es **contagio borde a borde**. La regla combina WoE, la vecindad de Moore, un umbral y una componente estocástica."

- Remate: "Que las variables más predictivas sean de proximidad confirma que el crecimiento es por contagio, no aleatorio."

---

## 12 · Definición matemática de la transición  *(~70 s)*

"Formalizo la regla. La probabilidad de que la celda $(i,j)$ transite en el paso $t$ es una sigmoide aplicada a la suma de dos términos."

"El primer término es la **evidencia espacial**: sumo los pesos WoE de las siete variables, pero cada uno ponderado por su **Information Value** normalizado —ahí está IV en la función de transición—; así las variables más discriminativas, como la distancia al frente con cerca del 28 %, pesan más."

"El segundo término es la vecindad de Moore multiplicada por **alfa**, el peso de la vecindad, calibrado en 0.50. Y finalmente la celda transita a urbana si esa probabilidad supera el **umbral theta**, calibrado en 0.75. Bajar theta sobre-predice; ahí está el umbral que controla cuánto crece la mancha."

- Si preguntan por la sigmoide: "Acota la combinación a una probabilidad entre 0 y 1; la confirmación final es estocástica para no forzar determinismo."
- Remate: "Theta y alfa son los dos parámetros calibrados; IV no se calibra, se estima de los datos de sintonización."

---

## SECCIÓN: Resultados

## 13 · Information Value  *(~55 s)*

"Estos son los pesos. La distancia al área urbana y las densidades de vecindad concentran cerca del **75 %** del poder predictivo, coherente con un mecanismo de difusión espacial."

"Y lo importante: cada peso es **auditable** geográficamente. El modelo no es una caja negra; cumple R4."

---

## 14 · Resultados: cinco ventanas quinquenales  *(~90 s)*

"Esta diapositiva muestra las seis métricas de validación a lo largo de las cinco ventanas. Los parámetros calibrados son umbral **θ = 0.75** y peso de vecindad **α = 0.50**, con variables ponderadas por IV. Sintonización 1984–2010, validación en cinco ventanas independientes 2011–2020."

"Lo que más quiero que noten: **ninguna ventana se desploma ni se dispara**. Eso demuestra que el modelo captura una dinámica estructural, no un sobreajuste a un período particular."

- **FoM** (la más exigente): solo mide la zona de cambio, ignora la permanencia. Promedio **0.317**, rango 0.222–0.378. Triple del piso de Pontius.
- **Kappa**: acuerdo más allá del azar. Promedio **0.445**, rango moderado–sustancial.
- **IoU**: solapación espacial del cambio. Promedio **0.633** — los polígonos predicho y observado se solapan el 63 %.
- **Accuracy**: píxeles bien clasificados totales. Promedio **0.722** — no es la principal pero confirma coherencia global.
- **Recall**: fracción del crecimiento real capturada. Por encima de **0.91 en todas las ventanas** — pocas omisiones.
- **F1**: equilibrio entre precisión y recall. Consistente con el patrón de ligera sobre-predicción.
- Si preguntan por el 2011–2016 (FoM 0.222, el más bajo): "La clasificación RGB tuvo ligera variabilidad ese año; no es contracción urbana real. En las cuatro ventanas restantes el FoM sube y se estabiliza."

---

## 15 · Validación visual: cinco ventanas quinquenales  *(~60 s)*

"Aquí la validación se vuelve visual. Cada grupo de imágenes muestra cuatro mapas: **estado inicial** (de dónde parte), **observado** (lo que realmente pasó), **predicho** (lo que dijo el modelo) y la **diferencia** —verde para los aciertos, naranja para las falsas alarmas, azul oscuro para las omisiones."

"Pueden ver que el patrón morfológico se replica bien: el crecimiento en borde de mancha, la dirección de expansión. Lo que falla son los saltos aislados —leapfrog— que el modelo no anticipa porque no tiene variables de infraestructura vial."

"Abajo, las cuatro métricas resumen para que tengan el número junto a la imagen:"

- **FoM 0.317**: aciertas el 31.7 % de las celdas nuevas (triple del piso de Pontius).
- **Kappa 0.445**: mucho mejor que el azar — como superar consistentemente a un mono lanzando dardos al mapa.
- **IoU 0.633**: el área que predijiste y el área que creció de verdad se traslapan en el 63 %.
- **Accuracy 0.722**: 72 de cada 100 píxeles del mapa total están bien clasificados.

---

## 16 · Posicionamiento frente a la literatura  *(~55 s)*

"Para darle contexto: en el estudio multi-sitio de Pontius (2008), 6 de 13 modelos publicados tienen FoM **bajo 15 %** y solo uno supera 50 %. Mi promedio de 31.7 % queda en **banda intermedia** del espectro empírico global."

"El punto clave es el tercero: logro ese desempeño usando **solo siete variables derivadas del propio mapa binario**, sin drivers físicos externos (pendiente, vialidades, empleo). El modelo de Tang en Urumqi llega al 37–43 % pero requiere datos de pendiente, vialidades y empleo."

"Interpretabilidad y desempeño intermedio sin datos externos: eso es el posicionamiento."

---

## 17 · Análisis del error  *(~60 s)*

"Soy explícito con las dos caras del modelo."

**Fortalezas:**

- Recall alto: pocas omisiones del crecimiento real.
- Fidelidad morfológica al patrón de difusión borde a borde.
- Parsimonia: siete variables, dos parámetros calibrados.
- Corre en unas **3 horas en CPU**, sin GPU ni cluster.

**Debilidades:**

- **Sobre-predicción sistemática**: Precision baja a 0.59–0.70. El umbral y la vecindad tienden a expandir; el error de clasificación se amplifica por contagio.
- No modela el crecimiento tipo *leapfrog* (saltos aislados sin continuidad de borde).
- La clase "urbano" incluye algo de suelo árido por usar solo RGB.
- Se omiten variables externas (vialidades, pendiente, equipamiento).
- Si preguntan por qué no bajas el umbral para corregir la sobre-predicción: "Bajar theta de 0.75 a 0.60 eleva el FoM a 0.17 —lo empeora— porque se dispara a predecir todo como urbano. El umbral es el freno."

---

## SECCIÓN: Conclusiones

## 18 · Validación de hipótesis (H1–H6)  *(~90 s)*

> Esta es la diapositiva donde más te van a preguntar. Aterriza el "porqué" de cada una con su dato.

"Reviso las seis hipótesis contra la evidencia:"

- **H1 — Confirmada.** "FoM mayor a 0.10 en al menos 4 de 5 ventanas. Se cumplió en **las cinco**, promedio 0.317 — triple del piso. Eso prueba que la evidencia espacial histórica contiene información predictiva, no ruido."
- **H2 — Confirmada.** "Con θ = 0.75 el FoM promedia 0.317; con θ ≈ 0.60 caía a 0.17 por sobre-predicción masiva. La diferencia es **directamente** la calibración."
- **H3 — Confirmada.** "Distancia al frente (IV = 0.28) más densidades de vecindad concentran el **75 %** del poder predictivo. Relación monótona creciente confirma difusión por contagio."
- **H4 — Parcial.** "Morfológicamente estable en cinco ventanas, pero la métrica píxel-exacta varía con el crecimiento neto de cada período. El *leapfrog* no predicho degrada la estabilidad numérica."
- **H5 — Sustentada.** "FoM 31.7 % en banda intermedia de Pontius, con solo siete variables del mapa y modelo auditable. Comparable a modelos más complejos sin sacrificar interpretabilidad."
- **H6 — Sustentada.** "Pipeline completo en Python abierto, datos libres, 3 horas en CPU. Sin GPU ni software propietario."

> Si un sinodal nota que H5/H6 no están en la tabla de validación del Cap. 7: reconócelo y di que están discutidas en el cuerpo del capítulo y que integrarlas a la tabla es un ajuste editorial pendiente.

---

## 19 · Contribuciones originales  *(~55 s)*

"Las cinco contribuciones que identificamos:"

1. **Integración WoE-AC** para ciudad intermedia mexicana con datos libres — válida y potencialmente publicable.
2. **Validación quinquenal múltiple** (5 períodos independientes) — evalúa estabilidad temporal con más rigor que una sola ventana.
3. **Pipeline reproducible en Python** — clasificación → WoE → AC → evaluación, documentado y abierto.
4. **Uso explícito de evidencia histórica** (37 años, ZMQ) — alineado con la tradición LUCC.
5. **AC con decaimiento y vecindario ponderado** — regla de transición no trivial.

- Si tienes que elegir una para destacar: el **protocolo de validación quinquenal** — nadie más lo hace en el contexto mexicano.

---

## 20 · Implicaciones y trabajo futuro  *(~55 s)*

"En aplicación: provisión anticipada de infraestructura, reserva de suelo, zonificación proactiva, y es transferible a otras ciudades intermedias previa recalibración."

"En trabajo futuro: incorporar variables externas (pendiente, vialidades), usar imágenes con infrarrojo para NDVI real, calibración metaheurística o bayesiana, y WoE temporal."

"El mensaje central: modelos **parsimoniosos, interpretables y validados en múltiples períodos** sí pueden ser útiles para la planeación, sin hardware especializado ni cajas negras."

---

## 21 · Cronograma de actividades 2026-II  *(~45 s)*

"Para cerrar, este es el cronograma del semestre. La **investigación** —calibración, validación quinquenal y análisis de métricas— y la **escritura** de los siete capítulos quedaron concluidas entre enero y marzo."

"Ahora mismo estoy en el **Bloque 3: la iteración por capítulos con la Dra. Lárraga** —revisión, corrección, validación de referencias y compilación final—, que corre hasta finales de mayo. Todo esto en paralelo con los **dos cursos** que impartí en la Facultad de Ingeniería."

- Remate: "El objetivo es enviar la tesis a sinodales al cierre del semestre."

---

## 22 · Gracias  *(~10 s)*

"Muchas gracias. Quedo atento a sus preguntas y comentarios."

---

## Anexo: preguntas probables y respuestas cortas

- **¿Esto es realmente un autómata celular?** "Sí: estados discretos urbano/no-urbano, vecindad de Moore, una regla de transición local aplicada iterativamente y recálculo de variables paso a paso. El WoE es la función de transición, no un postprocesamiento."
- **¿Por qué no usar deep learning?** "Por R4: las redes mejoran precisión pero no son auditables. Un plan municipal necesita justificar por qué se predice una transición; el WoE lo da con el Information Value."
- **¿El FoM de 0.317 es bueno?** "Es banda intermedia en el espectro empírico de Pontius. De 13 modelos publicados, 6 tienen FoM menor a 0.15 y solo 1 supera 0.50. El nuestro está en la mitad superior usando solo variables del propio mapa."
- **¿Por qué la sobre-predicción?** "Umbral y vecindad tienden a expandir; además el error de clasificación se amplifica por contagio. Es la principal debilidad reconocida y guía el trabajo futuro."
- **¿Por qué solo RGB?** "Decisión deliberada de reproducibilidad (R3): Google Earth histórico no da infrarrojo. Acota la interpretación del área absoluta, pero no invalida la validación interna de la dinámica."
- **¿Por qué Querétaro?** "Ciudad intermedia sin estudios CA previos, serie de 37 años y datos libres. Aplicar el método sobre una ciudad ya estudiada por cajas negras no abriría espacio metodológico nuevo."
- **¿Qué es exactamente el FoM?** "En palabras simples: de todas las celdas que cambiaron (las que yo predije más las que realmente crecieron), ¿cuántas acerté? Es el único indicador que ignora las zonas que no cambiaron, que son el 95 % del mapa y que harían que cualquier modelo se vea bien."
- **¿Qué diferencia hay entre Kappa e IoU?** "Kappa mide acuerdo estadístico global descuentando el azar, en todo el mapa. IoU mide solo la solapación geométrica de las zonas de cambio — es más espacialmente específico. Los dos se complementan."

