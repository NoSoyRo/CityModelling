# Plan de reestructuración de la tesis — Sesión Dra. Lárraga (4-ago-2026)

> Fuente: `revisiones/SESION DRA/NOTAS` + `revisiones/SESION DRA/TRANSCRIPTION.TXT` (1h43m).
> Se ejecutará con Sonnet, fase por fase. Este documento es la versión maestra editable.
> **Modo: cumplimiento total de lo pedido por la Dra.**

## Decisiones ya tomadas
1. **Se fusionan capítulos como pidió la Dra: de 7 a 6.** *Área de estudio* + *preprocesamiento* (hoy en Marco teórico) + *Modelo* se unen en un solo capítulo **"Modelo de crecimiento urbano híbrido"**.
2. Las **hipótesis** pasan a **una general + particulares** (paralelo a objetivo general + específicos).
3. La **citación** cambia de APA a **IEEE numérico**, ordenado por aparición.

## Estructura final (6 capítulos)
1. Introducción
2. Marco teórico (solo conceptos)
3. Estado del arte (antecedentes)
4. **Modelo de crecimiento urbano híbrido** (Consideraciones → arquitectura → área de estudio → preprocesamiento → definición del modelo → validación)
5. Resultados y análisis
6. Conclusiones

## Regla de oro (aplica a todo el documento)
- Leer cada capítulo **como evaluador externo**, no como autor.
- Ir **de lo general a lo particular** (embudo).
- **Ligar** el cierre de cada capítulo con el inicio del siguiente.
- **No mezclar** conceptos (marco teórico) con metodología/decisiones propias (capítulos de trabajo).
- **Delimitar** con claridad dónde empieza la aportación propia.
- **Toda figura** se referencia **e interpreta** en el texto (no basta colocarla).
- **Todo dato cuantitativo** lleva referencia.
- **Modo impersonal** en todo el texto.
- Vigilar el **balance de longitud**: el capítulo de conceptos NO debe ser más largo que el capítulo del modelo.

---

## Fase 0 — Cambios globales (transversales)
- **Citación IEEE numérico.** En `standalone_preamble.tex` (~L147-157) cambiar `style=apa` → `style=ieee` (o `numeric-comp`) con `sorting=none`. Verificar que `\textcite`/`\parencite`/`\cite` compilen. Ejecutar con el agente `estilo-citacion` (atómico y reversible).
- **Modo impersonal.** Auditar y eliminar 1ª persona; permitido el giro humilde "en nuestro conocimiento" solo en el estado del arte. Agente `pulidor-estilo`.
- **Datos cuantitativos con referencia.** Barrido con `bibliografia` / `validador-fuentes`. **Usar siempre las referencias ya existentes y verificadas del `.bib` actual.**
- **Imágenes generadas con IA.** Pie "Imagen elaborada con IA"; revisar el chat de origen para detectar fuentes publicadas copiadas y citarlas. Agente `figures-tesis`.
- **Toda figura referenciada e interpretada** en el texto.
- **Ligas entre capítulos.** 1-2 frases de transición al cierre de cada capítulo.
- **Resúmenes de capítulo (chapter abstracts).** Patrón "En este capítulo se presentan/…", sin referenciar tablas ni figuras.
- **Coherencia objetivo–hipótesis–resultados.** Verificar al final con `verificador` y `auditor-coherencia`.

## Fase 1 — Front matter
- **Título** (`front/portada.tex`): ya revertido al acordado. Verificar contra la carta de jurado antes de imprimir (una preposición distinta la invalida).
- **Agradecimientos** (crear `front/agradecimientos.tex` + incluir en `main.tex`): sección **obligatoria** que agradezca a **SECIHTI** "por el apoyo otorgado a través de la beca para los estudios de maestría" (requisito de liberación de beca). Parte personal opcional.
- **Resumen** (`front/resumen.tex`): reescribir **realzando la contribución científica original** por encima de la aplicación local. Encuadre de la Dra.: *"estrategia original de integración, calibración y validación para construir un modelo reproducible de crecimiento urbano a partir de imágenes históricas de libre acceso, evaluado mediante un protocolo de validación multitemporal independiente."* Objetivo: que quien lea resumen+intro+conclusiones diga "quiero leerlo" (apto para concursos).
- **Abstract** (`front/abstract.tex`): reescribir en inglés **después** de cerrar el resumen en español.

## Fase 2 — Cap. 1 Introducción (`cap01_introduccion/cap01_introduccion.tex`)
- **Quitar `\begin{resumen}`**: la introducción es el único capítulo sin resumen (es el prólogo).
- **Estructura objetivo** (en este orden): Planteamiento del problema → Justificación → Antecedentes → Objetivos (general + particulares) → Contribución principal → Organización del documento. Adaptar el protocolo a **presente** (trabajo ya realizado).
- **Embudo**: abrir de lo general (las imágenes tipo Google Earth existen pero requieren análisis; reto mundial, cada país se comporta distinto, México incluido) → aterrizar en la propuesta. Querétaro como **caso de uso**, no decisión personal.
- Renombrar `\section{Contexto y problema}` → **Planteamiento del problema** (agregar el artículo inicial que falta).
- Separar `\section{Literatura existente y brecha metodológica}` en **Antecedentes** (literatura) + **Justificación** (la brecha). En Antecedentes **no** decir "se revisaron 13 trabajos"; del conjunto completo, **categorizar** (redes neuronales, autómatas celulares, híbridos, WoE) en prosa con referencias, sin listar autor por autor.
- **Hipótesis**: convertir H1–H6 en **una hipótesis general** (afirmación validable) que se desglosa en particulares. *(Se propondrán 2-3 redacciones de la general al llegar aquí.)*
- **Mover `\section{Alcances y limitaciones}` completo al Cap. 6 (Conclusiones).**
- **Contribución principal**: encuadre realzado (protocolo de validación multitemporal + estrategia original de integración/calibración/validación).
- **Figura de Querétaro**: introducirla con dato del INEGI referenciado; meter ahí el párrafo de crecimiento urbano (no entrar "en seco").
- **Objetivo general**: coherente con los resultados del Cap. 5.
- Actualizar la sección "Organización del documento" a **6 capítulos**.

## Fase 3 — Cap. 2 Marco teórico (`cap02_marco_teorico/cap02_marco_teorico.tex`)
- **Reescribir el resumen**: quitar "sigue el orden del capítulo uno"; usar "en este capítulo se presentan los conceptos necesarios para que el lector entienda la propuesta".
- **Sacar metodología/decisiones propias**: `\section{Preprocesamiento de datos}` (flujo del trabajo, arreglo $M_{t,i,j}$, pipeline aplicado) → **mover al nuevo Cap. 4 (Modelo híbrido)**. Aquí solo quedan **definiciones conceptuales** con cita: imágenes satelitales / Google Earth como fuente geoespacial, clasificación de cobertura (binaria urbano/no-urbano), espacio de color RGB, LBP, K-Means, SVM, StandardScaler, PCA. Cada una como subsección, con **ligas** (no glosario) e indicando supervisado/no supervisado.
- **Autómatas celulares** (~L190): añadir **definición formal completa** antes de la específica (dimensiones 1D/2D/nD, espacio celular y geometría, vecindades Moore/von Neumann, tipos de frontera) con **imágenes ilustrativas** referenciadas ("tomado de…"). Quitar toda mención a "el presente modelo" → Cap. 4.
- **Nueva sección WoE**: traer la **definición conceptual** desde el Cap. 3, con ventajas y limitaciones, y **justificar** por qué combinar WoE con AC para calibrar reglas de transición. La figura de WoE se referencia e interpreta.
- **Definir SLEUTH** aquí (concepto); el análisis de lo hecho con SLEUTH va al Cap. 3.
- **Métricas** (~L257): párrafo introductorio ("la evaluación cuantitativa requiere métricas…") antes de definir; **subseccionar** (matriz de confusión; métricas de clasificación: exactitud, precisión, sensibilidad, F1) con qué mide / para qué / limitaciones. Incluir **FoM, Kappa, IoU y descomposición de Pontius** aquí (conceptos estándar, no antecedentes).
- **Referencias**: capítulo densamente citado. **Usar siempre las referencias ya existentes y verificadas del `.bib` actual.**
- **Balance de longitud**: no dejar que este capítulo crezca más que el capítulo del modelo.
- **Liga final** hacia el Cap. 3.

## Fase 4 — Cap. 3 Estado del arte (`cap03_estado_del_arte/cap03_estado_del_arte.tex`)
- **Resumen**: sin referencias a tablas ni figuras.
- **Mover conceptos al Cap. 2**: `\section{Pesos de Evidencia y modelos probabilísticos espaciales}` (~L208) y definiciones conceptuales de validación/métricas → Cap. 2. Aquí solo **lo que se ha hecho** (obras).
- **Análisis crítico, no lista**: nada de "fulano hizo X, mengano hizo Y". De lo general a lo particular con **clasificaciones** (redes neuronales, autómatas celulares, híbridos → WoE para crecimiento urbano → combinaciones WoE+AC). Cada bloque: qué logran y qué **no** cubren, referencias agrupadas.
- **SLEUTH**: discutir lo hecho y el problema de calibración/equifinalidad/sobreajuste **como antecedente** (ya definido en Cap. 2), reordenado hacia la brecha.
- **Cierre con la brecha + por qué importa**: justificar relevancia (no basta "no se ha hecho"), giro humilde "en nuestro conocimiento no existe…". Liga: "en el siguiente capítulo se presenta un nuevo modelo que…".
- Citas en **IEEE numérico**, abundantes.
- **Liga final** hacia el Cap. 4 (Modelo híbrido).

## Fase 5 — Cap. 4 NUEVO: Modelo de crecimiento urbano híbrido (fusión)
> Fusiona: `cap04_area_estudio` + preprocesamiento (venido del Cap. 2) + `cap05_modelo_crecimiento_urbano`.
> Implementación sugerida: consolidar todo en el archivo del modelo (`cap05_modelo_crecimiento_urbano/…`), renombrar el `\chapter`, y **quitar `\input{cap04_area_estudio…}` de `main.tex`**. Actualizar todos los `\ref`/`\label` (`ch:area-estudio`, `ch:modelo-crecimiento-urbano`, etc.).
- **Título del capítulo**: **Modelo de crecimiento urbano híbrido**.
- **Resumen**: "en este capítulo se presenta un nuevo modelo híbrido que combina…", con el objetivo del modelo y qué toma en cuenta. Marcar **explícitamente que aquí empieza la aportación propia**.
- **Orden de secciones**:
  1. **Consideraciones del modelo** (qué considera y qué no: espacio, celdas, imágenes de Google Earth en píxeles, fronteras, ausencia de decaimiento por distancia, etc.). Abrir con esto.
  2. **Arquitectura del modelo (abstracción)**: "graphical abstract" de cajas con el flujo datos→predicción (datos Querétaro → preprocesamiento/binarización → WoE-AC → actualización de estados → predicción). NO el UML del apéndice. Cada caja → subsección. **Crear imágenes súper explicativas por paso** (ChatGPT/Gemini, el que mejor funcione), imágenes nuevas de la abstracción completa; **está bien crearlas de cero**. Agente `figures-tesis`.
  3. **Área de estudio** (localización, fuente de datos, imágenes).
  4. **Preprocesamiento de datos** (venido del Cap. 2 + pipeline que ya vivía en área de estudio; consolidar sin duplicados: LBP+K-Means+SVM, normalización, binarización).
  5. **Definición del modelo** (ecuación de transición ya validada contra el código; WoE ponderado por IV → sigmoide → vecindad de Moore → umbral estocástico; sin decaimiento por distancia).
  6. **Validación** (protocolo quinquenal múltiple).
- **Liga** hacia resultados.

## Fase 6 — Cap. 5 Resultados (`cap06_resultados_analisis/cap06_resultados_analisis.tex`)
- **Renumerar** a Capítulo 5 (actualizar `\ref`).
- **Modo impersonal** y **coherencia objetivo↔resultados** (lo prometido en Cap. 1 se muestra aquí).
- Mantener métricas verificables (FoM 0.317, Kappa 0.445, etc.) de los cinco períodos.
- **Liga** hacia conclusiones.

## Fase 7 — Cap. 6 Conclusiones (`cap07_conclusiones/cap07_conclusiones.tex`)
- **Renumerar** a Capítulo 6 (actualizar `\ref`).
- **Pretérito**: "en este trabajo se presentó un modelo para…", "el objetivo fue…", "los resultados obtenidos indicaron que…".
- **Recibir Alcances y Limitaciones** del Cap. 1. Orden exigido: **Alcances → Limitaciones → Trabajo futuro** (el futuro se deriva de las limitaciones).
- Alinear la **validación de hipótesis** con la nueva hipótesis general + particulares.
- Consolidar cierres redundantes ("Mensaje central" / "Palabras finales") en uno solo.

## Fase 8 — Cierre: compilación y verificación
- Ajustar `main.tex`: quitar el `\input` del capítulo fusionado y reordenar. Compilar (pdflatex → biber → pdflatex ×2) con `compilador`; 0 errores, 0 refs indefinidas.
- Correr `verificador` (labels/refs tras renumeración), `auditor-coherencia` (arco narrativo, hipótesis, objetivos), `auditor-larraga` (concordancia con `/revisiones` + esta sesión) y `acronimos`.
- Regenerar `\listoffigures`, `\listoftables` y bibliografía en IEEE.

---

## Revisión de cobertura — todos los puntos de la llamada
Marcado: [x] cubierto en el plan.

**Próximos pasos (NOTAS):**
- [x] Cambiar título al acordado (congruente con carta de jurado) → Fase 1 (ya hecho)
- [x] Incluir agradecimientos con beca (SECIHTI) → Fase 1
- [x] Reescribir resumen (contextualización + contribución original) → Fase 1
- [x] Reestructurar introducción (problema, justificación, antecedentes, organización) → Fase 2
- [x] Añadir referencias a todos los datos cuantitativos → Fase 0/2
- [x] Redactar conclusiones en pretérito (resumen, alcances, limitaciones, trabajo futuro) → Fase 7
- [x] Actualizar abstract en inglés al final → Fase 1
- [x] Corregir introducción (lectura coherente, transiciones) → Fase 2/0
- [x] Reorganizar literatura → antecedentes + justificación → Fase 2
- [x] Adaptar protocolo a la introducción (trabajo ya realizado, presente) → Fase 2
- [x] Validar objetivos coherentes con resultados → Fase 0/6
- [x] Revisar hipótesis (una general + particulares) → Fase 2
- [x] Mover metodología (preprocesamiento) del marco teórico a su capítulo → Fase 3/5
- [x] Enriquecer marco teórico (referencias, imágenes, conceptos, no glosario) → Fase 3
- [x] Reorganizar estructura (separar teoría de estado del arte) → Fase 3/4
- [x] Modo impersonal en toda la tesis → Fase 0
- [x] Estilo de citas APA → numérico IEEE por aparición → Fase 0
- [x] Etiquetar imágenes IA ("Imagen elaborada con IA") → Fase 0
- [x] Verificar fuentes de imágenes IA → Fase 0
- [x] **Unir capítulos cortos (área de estudio + preprocesamiento + modelo)** → Fase 5 (fusión adoptada)
- [x] Reescribir resumen del capítulo del modelo (modelo híbrido, objetivos) → Fase 5
- [x] Renombrar capítulo → "Modelo de crecimiento urbano híbrido" → Fase 5
- [x] Consideraciones del modelo al inicio → Fase 5
- [x] Crear diagrama de arquitectura del modelo → Fase 5

**Detalles adicionales de la transcripción:**
- [x] Contribución central = protocolo de validación multitemporal independiente → Fase 1/2
- [x] La introducción no lleva resumen → Fase 2
- [x] Figuras poco comunes en intro; si se usan, citarlas → Fase 2/0
- [x] Conexión (ligas) entre capítulos → Fase 0
- [x] Resumen del estado del arte sin tablas ni figuras → Fase 4
- [x] Ir de lo general a lo particular con clasificaciones → Fase 2/4
- [x] FoM es métrica estándar → va en Métricas (marco teórico) → Fase 3
- [x] Definir SLEUTH en marco teórico; brecha en antecedentes → Fase 3/4
- [x] Métricas con introducción + subsecciones → Fase 3
- [x] Autómatas: definición formal (dimensiones, vecindades, fronteras) + imágenes → Fase 3
- [x] WoE con ventajas/limitaciones + justificación WoE+AC → Fase 3
- [x] No mezclar "el presente modelo" en el marco teórico → Fase 3
- [x] Arquitectura = abstracción del modelo, no implementación (UML del apéndice) → Fase 5
- [x] Delimitar claramente trabajo propio vs ajeno → Fase 5 (+ regla de oro)
- [x] Alcances/limitaciones en conclusiones, no en intro → Fase 2/7
- [x] Intro en presente / conclusiones en pretérito → Fase 2/7
- [x] Hipótesis como afirmación validable → Fase 2
- [x] Toda figura referenciada e interpretada → Fase 0
- [x] Cuidar balance de longitud (conceptos ≤ trabajo) → regla de oro + Fase 3
- [x] Redacción humilde ("en nuestro conocimiento/opinión") → Fase 4
- [n/a] Tesis de Fernando como ejemplo de estructura/gráficos → material de referencia de la Dra.
- [n/a] Convocatoria de doctorado / becas octubre → fuera del alcance de la tesis

---

## Roadmap de ejecución (orden y dependencias)

Secuencia recomendada, con agente responsable y entregable/checkpoint por fase:

1. **Fase 1 (Front matter)** — `front-matter`. Entregable: agradecimientos (SECIHTI), resumen realzado. *Sin dependencias.* El abstract en inglés se hace al final (depende del resumen final).
2. **Fase 5 (Cap. 4 fusión)** — `analizador` → `implementador` + `figures-tesis`. Entregable: capítulo "Modelo de crecimiento urbano híbrido" armado (Consideraciones → Arquitectura → Área → Preprocesamiento → Definición → Validación); `main.tex` sin el `\input` de cap04; `\ref`/`\label` migrados. *Se hace temprano porque libera contenido conceptual del Cap. 2 y define la numeración.*
3. **Fase 3 (Cap. 2 Marco teórico)** — `redactor-academico` + `bibliografia`. Entregable: solo conceptos, AC formal, sección WoE, SLEUTH, métricas subseccionadas. *Depende de Fase 5 (a dónde se mueve el preprocesamiento).*
4. **Fase 4 (Cap. 3 Estado del arte)** — `redactor-academico` + `bibliografia`. Entregable: antecedentes por clasificaciones, brecha justificada, ligas. *Depende de Fase 3 (conceptos ya migrados).*
5. **Fase 2 (Cap. 1 Introducción)** — `redactor-academico`. Entregable: embudo, Planteamiento/Antecedentes/Justificación, hipótesis general+particulares, contribución realzada, Alcances/Limitaciones movidos fuera. *Depende de 3-4 para reflejar la nueva estructura.*
6. **Fase 6 (Cap. 5 Resultados)** — `implementador` + `verificador`. Entregable: impersonal, coherencia objetivo↔resultados, renumeración.
7. **Fase 7 (Cap. 6 Conclusiones)** — `redactor-academico`. Entregable: pretérito, Alcances→Limitaciones→Trabajo futuro, validación de hipótesis alineada.
8. **Fase 0 (Global)** — `estilo-citacion` (IEEE) → `pulidor-estilo` (impersonal) → `figures-tesis` (etiqueta IA). *Se corre casi al final para no re-tocar texto que aún cambiará; la conversión IEEE conviene tras estabilizar el `.bib`.*
9. **Fase 1b (Abstract EN)** — `front-matter`. Traducir el resumen ya cerrado.
10. **Fase 8 (Cierre)** — `compilador` → `verificador` → `auditor-coherencia` → `auditor-larraga` → `acronimos`. Entregable: PDF final 0 errores / 0 refs indefinidas, índices y bibliografía IEEE regenerados.

Checkpoint tras cada fase: compilar el capítulo standalone (o `main`) y confirmar 0 errores antes de avanzar.

## Propuestas de hipótesis general (elegir 1 en Fase 2)

**Opción A (general + particulares) — recomendada.**
> *General:* "Un modelo híbrido que acopla Pesos de Evidencia con un autómata celular, calibrado y validado mediante un protocolo multitemporal independiente sobre datos de libre acceso, permite simular de forma interpretable y reproducible el crecimiento urbano de una ciudad intermedia mexicana con un desempeño comparable al reportado en la literatura."
>
> *Particulares:*
> - H1: la regla de transición basada en WoE ponderada por Information Value produce un modelo interpretable, auditable variable por variable.
> - H2: el desempeño se sostiene en varias ventanas temporales independientes (estabilidad temporal), no solo en un período.
> - H3: el flujo completo es reproducible con herramientas de código abierto y datos de libre acceso.
> - H4: aplicado a la ZMQ, el modelo reproduce el patrón de crecimiento con métricas (FoM, Kappa, IoU) dentro del rango de la literatura.

**Opción B (centrada en el protocolo/contribución).**
> "La integración, calibración y validación de un acoplamiento WoE–AC bajo un protocolo de evaluación multitemporal independiente constituye una estrategia reproducible e interpretable para modelar el crecimiento urbano de ciudades intermedias, verificable con datos reales de la Zona Metropolitana de Querétaro."

**Opción C (una sola afirmación breve).**
> "El crecimiento urbano de una ciudad intermedia mexicana puede modelarse de forma interpretable y reproducible mediante un autómata celular cuya regla de transición se calibra con Pesos de Evidencia, alcanzando un desempeño estable a través de varias ventanas temporales independientes."

## Puntos a confirmar antes de ejecutar
- Elegir la redacción de la hipótesis general (Opción A / B / C de arriba).
- Que la carta de jurado (si ya se entregó) use el título actual antes de cerrar la portada.

*(R1 resuelto: se adopta la fusión de capítulos → cumplimiento total de lo pedido por la Dra.)*
