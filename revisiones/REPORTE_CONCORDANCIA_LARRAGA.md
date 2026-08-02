# Reporte de concordancia: tesis vs. requerimientos de la Dra. Lárraga

> Validación de que la **historia** que cuenta la tesis es consistente con lo que pide la Dra. Lárraga en sus documentos de revisión.
> Fuentes leídas: `Comentarios de la tesis.pages` (crítica maestra), los 7 `cap0X_puntos_revision.md`, y los `*_Sugerido.pages` por capítulo.
> Estado de compilación: **los 7 capítulos compilan sin errores** y el **documento completo (`main.tex`) compila a 94 páginas sin referencias ni citas indefinidas** tras las correcciones de esta sesión.

---

## 1. Resumen ejecutivo

La tesis **ya cuenta una historia coherente y defendible**, y esa historia **sí responde a la crítica central de la Dra. Lárraga**. Los cuatro reclamos más graves de su revisión están resueltos:

| Reclamo grave de la Dra. | Estado |
|---|---|
| Hipótesis citan un archivo interno del proyecto (`quinquenal_best_config.json`) | **RESUELTO** — 0 ocurrencias en las hipótesis del Cap. 1 |
| "NDVI aproximado" desde RGB (error de percepción remota) | **RESUELTO** — sustituido por NGRDI / proxy cromático; se niega explícitamente el NDVI |
| Pseudo-validación / circularidad (accuracy 88–94 %) | **RESUELTO** — reformulado como "coherencia interna del clasificador", no exactitud contra verdad terreno |
| Vender WoE+CA como novedad metodológica | **RESUELTO** — la novedad se reubicó en el **protocolo de validación reproducible multi-ventana** + ciudad intermedia mexicana, de forma consistente en Caps. 1, 3, 5 y 7 |

Lo que queda son **afinaciones de redacción y un puñado de decisiones de autor** (sobre todo en hipótesis y en compactar el Cap. 4), no fallas estructurales. El nivel actual corresponde al que la propia Dra. describió como *"se vuelve adecuado y hasta publicable con un poco más"*.

---

## 2. La historia que cuenta la tesis (arco narrativo)

La tesis se lee como un único argumento, capítulo a capítulo:

1. **Cap. 1 — Introducción.** Las ciudades intermedias mexicanas crecen de forma dispersa y poco planificada (caso ZMQ). Existe **una sola brecha científica concreta**: no hay un modelo de crecimiento urbano *interpretable celda a celda* cuya capacidad predictiva se haya evaluado en **varias ventanas temporales independientes** con código y datos abiertos para ese contexto. De ahí salen 4 requerimientos (R1–R4), 6 hipótesis (H1–H6) y la declaración explícita de que **WoE+CA no es, en sí mismo, la novedad**.

2. **Cap. 2 — Marco teórico.** Define formalmente AC, WoE y métricas (FoM, Kappa, IoU) con las citas fundacionales (Wolfram, White, Clarke, Bonham-Carter, Pontius, Cohen, Landis & Koch). Es el andamiaje conceptual, sin revisión de literatura.

3. **Cap. 3 — Estado del arte.** Ya **no es un catálogo** "autor→método→limitación": está reestructurado por **problemas metodológicos persistentes** (calibración/equifinalidad, validación/sobreajuste temporal, interpretabilidad/caja negra, transferibilidad/software propietario), con una sección propia de WoE bayesiano y Dinamica EGO, otra de validación espacial LUCC (Pontius, quantity/allocation disagreement) y cierra con el vacío mexicano. Construye la **necesidad** del trabajo.

4. **Cap. 4 — Área de estudio y datos.** Justifica la ZMQ (población INEGI/CONAPO, polo industrial) y la fuente Google Earth con **argumento positivo** (consistencia espacial/temporal del encuadre sobre calibración radiométrica), documenta resolución efectiva (8–14 m/píxel), una **tabla de incertidumbre por tipo de error** y **justifica el pipeline LBP+K-Means+SVM** (ausencia de verdad terreno → pseudoetiquetas). La ingeniería de software (UML/clases) está en el apéndice.

5. **Cap. 5 — Modelo.** Presenta el núcleo matemático: campo WoE ponderado por *Information Value* → sigmoide → suma de vecindad de Moore → umbral estocástico. **Coincide exactamente con el código validado** (sin decaimiento por distancia). Declara la novedad: protocolo reproducible, no el acoplamiento WoE-AC.

6. **Cap. 6 — Resultados.** Hilo modelo → calibración → validación → interpretación. Cinco ventanas quinquenales (FoM 0,317; κ 0,445; Acc 0,722; IoU 0,633), tasas de crecimiento **derivadas de los mapas binarios** (con fuente), FoM contextualizado en la banda intermedia de Pontius, y una **síntesis del desempeño** (régimen de alta cobertura/baja especificidad, atractores espaciales).

7. **Cap. 7 — Conclusiones.** Valida las 6 hipótesis con su enunciado literal, mapea los 5 objetivos específicos, reconoce limitaciones honestamente, encuadra las aplicaciones como **hipotéticas** y declara la **aportación central = protocolo de evaluación reproducible**, en simetría con el Cap. 5.

**El hilo es consistente:** la misma brecha del Cap. 1 → se documenta en la literatura (Cap. 3) → se aborda con el modelo (Cap. 5) → se prueba multi-período (Cap. 6) → se concluye sin sobrevender (Cap. 7).

---

## 3. Concordancia por capítulo (estado actual)

Leyenda: ✅ RESUELTO · 🟡 PARCIAL (afinación) · 🔵 decisión de autor.

### Cap. 1 — Introducción
| Crítica de la Dra. | Estado |
|---|---|
| Hipótesis sin archivos internos del proyecto | ✅ |
| Tono defensivo IA ("de manera prudente", etc.) | ✅ (0 ocurrencias) |
| Contribución diferenciada (no "WoE+CA es nuevo") | ✅ |
| Exceso de tablas administrativas | ✅ (1 sola tabla) |
| `\ref{ch:...}` y nombre del Cap. 4 alineado | ✅ |
| Brecha única concreta (no 4 genéricas) | ✅ |
| Narrativa continua (resumen/organización aún algo "índice") | 🟡 |
| Justificación como necesidad (falta 1 párrafo de aporte nacional) | 🟡 |
| Tiempo verbal (L128 pasado) | ✅ corregido esta sesión |
| Hipótesis fuertes/falsables: H5 y H6 siguen siendo débiles | 🔵 ver §5 |

### Cap. 2 — Marco teórico
| Crítica | Estado |
|---|---|
| Citas fundacionales (Wolfram, White, Clarke, Bonham-Carter, Pontius) | ✅ |
| Citas de Kappa (Cohen 1960, Landis & Koch 1977) | ✅ **añadidas esta sesión** |
| Notación decimal con coma | ✅ |
| Cierre con transición al Cap. 3 | ✅ |
| Muletilla "permite" | ✅ |
| Ecuaciones con `\label` no referenciadas | 🟡 (varias se referencian desde Caps. 5–6; las puramente definicionales pueden conservar label) |

### Cap. 3 — Estado del arte
| Crítica | Estado |
|---|---|
| Reestructurar por problemas metodológicos (3.1–3.5) | ✅ |
| Eliminar tabla IA "Enfoque/Fortalezas/Limitaciones" | ✅ |
| Sección de WoE bayesiano / Dinamica EGO | ✅ |
| Validación espacial LUCC (Pontius, disagreement) | ✅ |
| Reformular posicionamiento (no "rareza" WoE+CA) | ✅ |
| Afirmaciones generales ancladas a citas | ✅ |
| Exceso de "prudencia" | ✅ |
| Profundidad SLEUTH (sobreajuste explícito) | 🟡 |
| Sesgo/sensibilidad espacial de WoE como párrafo propio | 🟡 |
| GeoSOS/MOLUSCE: mención sin cita | 🟡 |

### Cap. 4 — Área de estudio y datos
| Crítica | Estado |
|---|---|
| Eliminar "NDVI aproximado" | ✅ |
| Circularidad accuracy 88–94 % → coherencia interna | ✅ |
| Justificar pipeline LBP+K-Means+SVM | ✅ |
| Google Earth con justificación positiva | ✅ |
| Resolución en m/píxel + variación temporal | ✅ (rango 8–14 m/píxel) |
| Tabla de incertidumbre (+ fila de compresión) | ✅ **fila de compresión añadida esta sesión** |
| Nombres de clases Python fuera del cuerpo | ✅ **movidos al apéndice esta sesión** |
| Citas (INEGI, CONAPO, Gorelick, Ojala, Herold) | ✅ (falta SEDATU, opcional) |
| Localización geográfica con texto propio | ✅ |
| Compactar el capítulo a la mitad | 🔵 ver §5 |

### Cap. 5 — Modelo
| Crítica | Estado |
|---|---|
| Sin "código mental" (rutas/scripts/JSON en el cuerpo) | ✅ |
| LBP+SVM remitido al Cap. 4 (no como parte del WoE-AC) | ✅ |
| Sin lenguaje de certeza ("garantiza") / sin "Otsu" sin tabla | ✅ |
| WoE no se redefine (se cita el Cap. 2) | ✅ |
| Ecuación = código validado (sin decaimiento) + aclaración explícita | ✅ **aclaración añadida esta sesión** |
| Novedad declarada (protocolo reproducible) | ✅ **frase añadida al resumen esta sesión** |
| Separar 3 niveles (conceptual/implementación/configuración) | 🔵 no se reestructura (decisión del autor) |

### Cap. 6 — Resultados
| Crítica | Estado |
|---|---|
| Tasas de crecimiento con fuente (mapas binarios) | ✅ |
| Caption "valida el mecanismo" → "coherente con" | ✅ |
| κ 0,445 = "moderado" (Landis & Koch) | ✅ |
| Hipótesis literales (6, no 4); H5 con citas | ✅ **cita de H5 alineada esta sesión** |
| Tablas/figuras de las 5 ventanas referenciadas | ✅ |
| §6.7 Síntesis del desempeño (régimen, atractores) | ✅ **ampliada y renombrada esta sesión** |
| Cierre con transición al Cap. 7 | ✅ |
| Grid search: remisión explícita al repositorio | ✅ **añadida esta sesión** |
| Espaciado fino `\,` antes de `%` en cifras matemáticas | ⛔ no aplicable (rompe la compilación con babel-spanish dentro de `$...$`; la convención del documento es `\%` sin `\,` en modo matemático) |
| §6.8 interpretación dinámica formal | 🔵 opcional (no requerido) |

### Cap. 7 — Conclusiones
| Crítica | Estado |
|---|---|
| Aportación central = protocolo (sección "Aportaciones del trabajo") | ✅ |
| §7.2 aplicaciones como hipotéticas (sujetas a recalibración) | ✅ |
| Subsección "Cumplimiento de objetivos específicos" (5 OE) | ✅ |
| Muletilla "permite/capta no-linealidades" | ✅ **reformulada esta sesión** |
| Mensaje central → "exploración morfológica, sin política calibrada" | ✅ **ajustado esta sesión (intro §7.2)** |
| FoM 0,222 (2011–2016) reconocido como límite inferior real | ✅ **matizado esta sesión** |
| Cierre genérico ("herramientas predictivas accesibles", ODS) | ✅ **reemplazado por cierre concreto (ZMQ, serie 37 años, protocolo)** |
| Hipótesis literales con párrafos secundarios completos | 🟡 (el enunciado núcleo es literal; los párrafos largos no caben en la tabla) |

---

## 4. Correcciones aplicadas en esta sesión

1. **Cap. 2:** añadidas las citas de Kappa — `Cohen1960` y `LandisKoch1977` (con sus entradas en `referencias.bib`).
2. **Cap. 4:** fila "Compresión / exportación" en la tabla de incertidumbre; se quitaron del cuerpo los nombres de clases Python (`FeatureExtractor`, `ImageClassifier`, módulo `tesis_ac...`), remitiendo al Apéndice de arquitectura.
3. **Cap. 5:** frase de novedad en el resumen (la aportación es el protocolo reproducible, no WoE-AC); aclaración explícita de que **no hay término de decaimiento por distancia** en la implementación validada.
4. **Cap. 6:** cita de H5 alineada literalmente con el Cap. 1; §6.7 renombrada a "Síntesis del desempeño del modelo" y ampliada con el lenguaje dinámico (régimen de alta cobertura/baja especificidad, atractores espaciales que no convergen a estado estacionario único); remisión explícita del grid search al repositorio.
5. **Cap. 7:** muletilla "capta no-linealidades" reformulada; mensaje central reencuadrado a "exploración morfológica de escenarios, sin sustituir verdad terreno ni política pública calibrada"; FoM 0,222 reconocido como límite inferior real (sin minimizarlo); cierre genérico/ODS reemplazado por un cierre concreto (ZMQ, serie 1984–2020, protocolo quinquenal reproducible).
6. **Cap. 1:** "desarrolla y valida" → "desarrolló y validó" (tiempo de tesis concluida).

Todos los capítulos se recompilaron sin errores tras estos cambios.

---

## 5. Decisiones de criterio — RESUELTAS en la segunda iteración

Los cinco puntos que se habían dejado como decisión del autor ya se implementaron:

- **Hipótesis H5 y H6 (Caso 1).** Reformuladas como afirmaciones falsables y sincronizadas en los tres capítulos (Cap. 1 enunciado, Cap. 6 validación, Cap. 7 tabla):
  - **H5** ahora afirma que restringir la transición a siete variables auditables (caja blanca) **mantiene el FoM medio dentro de la banda intermedia de Pontius** (por encima del subconjunto con FoM < 0,15), sin recurrir a representaciones no interpretables. Es falsable: si la interpretabilidad costara desempeño, el FoM caería por debajo de esa banda.
  - **H6** ahora es una afirmación medible: el pipeline **se ejecuta en CPU estándar en tiempo del orden de horas**, sin GPU ni software propietario.
- **Compactar el Cap. 4 (Caso 2).** Se fusionaron las subsecciones de adquisición (de 6 a 3), se convirtió la lista de criterios temporales en prosa, se eliminó la repetición de Google Earth y el **algoritmo de adquisición se movió al Apéndice de arquitectura** (`Algoritmo~\ref{alg:adquisicion}`). Tablas, m/píxel, incertidumbre y justificación del pipeline se conservaron.
- **Cap. 5: tres niveles (Caso 3).** Se insertó un párrafo-guía (sin reestructurar) que distingue modelo conceptual / implementación / configuración experimental.
- **Cap. 1: resumen y "Organización" (Caso 4).** Reescritos como prosa argumentativa continua: el resumen termina en la aportación; la sección "Organización" sustituye los verbos de índice ("Presenta/Expone/Describe") por el hilo del argumento capítulo a capítulo.
- **Cap. 3: añadidos finos (Caso 5).** Se añadió un párrafo de **sobreajuste de calibración SLEUTH** (`Jantz2003`, `Silva2002`), un `\paragraph{Sesgo y sensibilidad espacial}` sobre autocorrelación en WoE (`BonhamCarter1994`) y se quitaron las menciones sin cita de GeoSOS/MOLUSCE.

Todos los capítulos afectados (1, 3, 4, 5, 6, 7) recompilan sin errores tras esta segunda iteración.

### Nota técnica
El bullet de espaciado fino `\,` antes de `%` (Cap. 6) **no es aplicable** dentro de `$...$`: rompe la compilación con babel-spanish (`\es@sppercent`). La convención del documento es `\%` sin `\,` en modo matemático.

### Compilación del documento completo
La tesis estaba organizada solo como capítulos *standalone* (el `main.tex` previo se había eliminado al pasar a carpetas por capítulo). Se reconstruyó un **`main.tex` agregador** en `caps_larraga/` que une los 7 capítulos, el apéndice de arquitectura y una bibliografía global (APA, biber). Resultado de la compilación completa (`pdflatex` → `biber` → `pdflatex` ×2):

- **94 páginas**, PDF generado correctamente (`caps_larraga/main.pdf`).
- **0 referencias cruzadas indefinidas** (resuelven `\ref{ch:...}`, `\ref{eq:transicion}`, el nuevo `\ref{alg:adquisicion}` del apéndice y `\ref{ap:arquitectura}`).
- **0 citas indefinidas**, **0 etiquetas duplicadas**, **0 figuras faltantes**.

Comando para regenerarlo:
```bash
cd report/tesis/tesis_indice_nuevo/caps_larraga
pdflatex -interaction=nonstopmode main.tex && biber main && \
pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
```

---

## 6. Veredicto

La **historia de la tesis es consistente con la crítica de fondo de la Dra. Lárraga**: la novedad está bien ubicada (protocolo reproducible, no WoE-AC), los errores técnicos graves (NDVI, circularidad, hipótesis con archivos internos) están corregidos, el arco Cap. 1 → Cap. 7 sostiene un único argumento, y los cinco puntos de criterio que quedaban (H5/H6, poda del Cap. 4, tres niveles del Cap. 5, redacción del Cap. 1 y añadidos LUCC del Cap. 3) ya se implementaron y compilan. Quedan solo afinaciones menores opcionales (citar SEDATU, m/píxel por año específico, §6.8 dinámica formal), que no condicionan la defensa.
