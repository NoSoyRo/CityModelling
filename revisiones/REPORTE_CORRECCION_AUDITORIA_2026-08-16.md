# Reporte completo — tesis vs sesión Dra. Lárraga + auditoría interna

**Fecha:** 16 de agosto de 2026
**Autor del documento:** José Rodrigo Moreno López
**Directora:** Dra. María Elena Lárraga Ramírez
**Fuentes de la revisión:** `revisiones/SESION DRA/NOTAS`, `revisiones/SESION DRA/TRANSCRIPTION.TXT`, `.cursor/PLAN_REESTRUCTURACION_LARRAGA.md`, consistencia interna de los `.tex` y de `data/processed/`
**PDF:** `report/tesis/tesis_indice_nuevo/caps_larraga/main.pdf`

Leyenda: **HECHO** · **ACOTADO** (se bajó el claim; no se inventó el dato) · **PENDIENTE** (decisión tuya o trabajo opcional)

---

## 1. Estado del documento (hoy)

| Ítem | Valor |
|---|---|
| Páginas | **110** |
| Compilación | `pdflatex → biber → pdflatex ×2`, exit 0 |
| Refs / citas indefinidas | **0** |
| Capítulos en `main.tex` | **6** (cap01, cap02, cap03, cap05=Ch.4, cap06=Ch.5, cap07=Ch.6) + Apéndice A |
| Citación | IEEE numérico (`style=ieee`, `sorting=none`) |
| Portada | *Modelado del crecimiento urbano en la Zona Metropolitana de Querétaro mediante Weight of Evidence y Autómatas Celulares* |
| `pdftitle` | Alineado a esa portada |
| Directora en portada | Dra. María Elena Lárraga Ramírez (**no se tocó**) |

`cap04_area_estudio/` quedó huérfano a propósito: el área de estudio se fusionó en el capítulo del modelo. El archivo tiene aviso «NO COMPILAR» y **no** entra en `main.tex`.

---

## 2. Concordancia con la sesión del 4 de agosto de 2026

Lo que pidió la Dra. en esa reunión, y dónde quedó.

### Front matter

| Pedido | Estado | Dónde |
|---|---|---|
| Título = el acordado (una preposición distinta invalida la carta de jurado) | **HECHO** en portada. **PENDIENTE** que tú confirmes contra la carta | `front/portada.tex` |
| Agradecimientos con beca SECIHTI | **HECHO.** Secretaría de Ciencia, Humanidades, Tecnología e Innovación; beca de maestría | `front/agradecimientos.tex` |
| Nombre de la Dra. / PCIC correctos | **HECHO.** María Elena Lárraga Ramírez; Ciencia e Ingeniería | `front/agradecimientos.tex` |
| Resumen que realza la contribución (protocolo, no el acoplamiento) | **HECHO.** Sin «estrategia original» | `front/resumen.tex` |
| Abstract en inglés alineado al resumen | **HECHO** | `front/abstract.tex` |
| Lista de acrónimos | **HECHO**, orden alfabético | `front/acronimos.tex` |

### Estructura (7 → 6 capítulos)

| Pedido | Estado |
|---|---|
| Fusionar área de estudio + preprocesamiento + modelo | **HECHO.** Capítulo 4: «Modelo de crecimiento urbano híbrido» |
| Intro sin `\begin{resumen}` | **HECHO** |
| Orden intro: Planteamiento → Justificación → Antecedentes → Objetivos → Preguntas/hipótesis → Aportaciones → Organización | **HECHO** |
| Alcances y limitaciones fuera de la intro, en conclusiones | **HECHO.** Orden: Alcances → Limitaciones → Trabajo futuro |
| Intro en presente / conclusiones en pretérito | **HECHO** |
| Hipótesis: 1 general + H1–H4 (Opción A del plan) | **HECHO** |
| Marco teórico = solo conceptos (sin hiperparámetros de este trabajo) | **HECHO** |
| Estado del arte = obras, no autoevaluación del modelo | **HECHO** (iteración 2) |
| IEEE numérico por aparición | **HECHO** |
| Modo impersonal | **HECHO** |
| Figuras IA con pie «Imagen elaborada con IA» | **HECHO** en las que aplica |
| Toda figura referenciada en el texto | **HECHO** (incluida `fig:woe_weights` en la iteración 2) |
| Ligas entre capítulos | **HECHO** |
| Embudo de la intro (imagen histórica → mundo → México → ZMQ) | **HECHO** |

### Hipótesis y preguntas (arco único)

| | Cap. 1 | Cap. 5 (resultados) | Cap. 6 (conclusiones) |
|---|---|---|---|
| General | híbrido WoE-AC + protocolo multitemporal + datos libres | — | **Apoyada** |
| H1 interpretabilidad | IV en distancia + densidad | misma | **Apoyada** (IV 0,783) |
| H2 estabilidad | cinco ventanas desplazadas; el mínimo se asocia a menor crecimiento neto | misma | **Parcial** (FoM 0,222–0,378) |
| H3 reproducibilidad | abierto / CPU / sin GPU | misma | **Apoyada** |
| H4 desempeño comparable | banda intermedia de Pontius | misma | **Apoyada** (FoM 0,317) |
| Q4 | ¿el flujo corre en abierto/CPU? (lo que sí se midió) | — | P4 = esa pregunta. **No** hay réplica en otra ciudad |

Se borraron H5, H6 y cualquier «seis hipótesis».

---

## 3. Bloqueantes de la auditoría (cerrados)

Estos ocho rompían la defensa. Todos están cerrados en el `.tex`.

| # | Hallazgo | Qué se hizo | Archivo |
|---|---|---|---|
| 1 | Resultados validaban H1–H6 viejas | Reescrito contra H1–H4. H2 **parcial** | `cap06` |
| 2 | P1–P4 de conclusiones ≠ preguntas del Cap. 1 | Q4 bajó a «¿corre en abierto/CPU?». P1–P4 = esas cuatro. Se declara que no hay réplica en otra ciudad | `cap01`, `cap07` |
| 3 | Agradecimientos: Dra. / PCIC / SECIHTI mal | Nombres institucionales correctos. Portada no se tocó | `front/agradecimientos.tex` |
| 4 | Cap. 2 filtraba metodología (`k=8`, `V=7`, `θ=0,75`) | Algoritmos generalizados. Figura WoE sin umbral de este trabajo | `cap02` |
| 5 | Cap. 3 inventaba el modelo (GEE, PCA sobre 7 vars, fila «Presente trabajo») | GEE → escritorio. PCA es de imagen, no de WoE. Fila fuera de la tabla | `cap03` |
| 6 | El modelo se contradecía (recalcula / no actualiza / campo estático) | Doctrina única: **pesos** fijos (1984–2010); **variables** se recalculan cada paso | `cap05`, `cap07` |
| 7 | Cifras de terceros malas | 20,5 % / dos tercios / 43–80 % → texto cualitativo + citas SEDATU / ONU-Hábitat / INEGI (**ACOTADO**, no se inventó 22,9 %). Ma2019 y Hagenauer fuera de «mejores FoM». FoM «penaliza» → «excluye». Clarke2008 ya no «dice equifinalidad». 8–12 h GPU borradas | `cap01`, `cap03`, `cap07` |
| 8 | IV 0,775 / 75 % y caption de densidad | Distancia + 3 densidades = **0,783 (78,3 %)**. Caption: el IV más alto es **distancia** (0,282) | `cap06`, `cap07` |

---

## 4. Graves (cerrados)

| # | Hallazgo | Resolución |
|---|---|---|
| 9 | Ventanas «independientes» vs solapadas | Texto canónico: hold-out respecto de 1984–2010; **solapadas entre sí** (4 de 5 años). En el cuerpo: «desplazadas» |
| 10 | «índice de vegetación aproximado» / `ndvi_approx` | NGRDI. Feature `\texttt{ngrdi}` |
| 11 | Figuras con `\label` sin `\ref` | Añadidos refs (mapa binario, transición, variables, Moore, paso AC, ventanas, FoM, timeline, métricas, mapa de error, pesos WoE) |
| 12 | `pdftitle` = «Modelado Evolutivo…» | Alineado al título de portada |
| 13 | Ecuación PCA tautológica | \(\mathbf{z}=\mathbf{W}^{\top}\tilde{\mathbf{f}}\) |
| 14 | Ejemplo WoE: manchas → «vías» | «alejadas del frente urbano» + límites formales del WoE en Cap. 2 |
| 15 | «garantiza» / «estrategia original» / «mejor desempeño» / «Confirmada» | Sustituidos. Hipótesis: «apoyada», no «confirmada» |
| 16 | Rutas `.json` en hallazgos y preguntas | Sacadas del cuerpo. Quedan en el Apéndice A |
| 17 | «pesos de evidencia (IV)» | «pesos de evidencia y valores de información (IV)» |

«Independientes» se conservó solo donde sí aplica: las trece simulaciones de Pontius y la independencia estadística (condicional / observaciones).

---

## 5. Fuentes de terceros (detalle)

| Claim | Resolución |
|---|---|
| Pontius: 6 aplicaciones con FoM < 15 %, 1 > 50 % | Se mantiene (verificado) |
| Tang 43,03 / 37,64 Urumqi | Se mantiene |
| Ma2019 como modelo urbano con FoM | Quitado de ese claim. Queda como review de deep learning en teledetección |
| Hagenauer2022 como AC de crecimiento | Reencuadrado: GWANN de vivienda/NO₂, no crecimiento urbano |
| Chihuahua: 8 cuencas | Restauradas las 5, con guion. McFadden matizado (3 núcleos < 0,2) |
| Jantz «días en hardware contemporáneo» | «más de una semana en un clúster de la época». Tres fases → Silva2002 / Clarke2008 |
| Resumen «rango caja blanca / ciudades intermedias» | Acotado a la muestra Pontius, con cita |
| SEDATU 20,5 % / INEGI 43–80 % | **ACOTADO** a texto cualitativo + citas existentes. No se inventó un porcentaje sustituto |

---

## 6. Higiene (iteración 2)

| Ítem | Estado |
|---|---|
| Embudo intro (Google Earth → mundo → México → ZMQ) | **HECHO.** Citas: Seto2012, UNHabitat2020, Aguilar2003 |
| Cap. 3 ya no se autoevalúa («el presente trabajo responde / se ubica») | **HECHO.** Apunta al capítulo del modelo |
| Ventanas nuestras: todas «desplazadas» | **HECHO** (resumen, abstract, caps. 1, 3, 5, 6, 7) |
| Rutas `.json` fuera del cuerpo de resultados | **HECHO** → Apéndice A |
| Acrónimos en orden alfabético | **HECHO** |
| `Arfiansyah2024` metadatos | **HECHO** (Dody Arfiansyah, Hawken, Zlatanova, Han; SIR 32:829–849). Sigue sin citarse |
| `Li2007` autor | **HECHO:** Xia Li (no se cita en el cuerpo) |
| DOI Weng2002 / Gong2013 / Ma2019 | **HECHO** |
| Zombies AG (Goldberg, Holland, Fortin, North) | **HECHO:** comentados como reserva |
| `cap04_area_estudio` huérfano | **HECHO:** aviso en el `.tex` |
| `$95\,\%$` fuera de math | **HECHO** (`cap02`) |

---

## 7. Pasada de voz (solo lo reescrito)

No se pulió la tesis entera. Se humanizaron los párrafos que la auditoría había reescrito, para que no delataran plantilla.

**Qué se quitó**
- Guiones largos `---` de adorno
- Frases de plantilla: «El problema no es local», «México no es la excepción», «Queda abierta, por tanto, la vía», «se sitúa en la intersección», listas (i)(ii)(iii)
- Una cuarta «condición» que no estaba en el texto original. Volvieron las tres de siempre

**Qué se conservó**
- El embudo que pidió la Dra.
- Cifras, citas, labels, doctrina del modelo
- El resto de la prosa, que no se reescribió

---

## 8. Números que no se tocan (fuente verificable)

De `data/processed/validation_quinquenal_*_v3_weighted/validation_results.json` y `quinquenal_best_config.json`:

| Magnitud | Valor |
|---|---|
| FoM por ventana | 0,222 / 0,345 / 0,355 / 0,287 / 0,378 |
| FoM promedio | **0,317** |
| Kappa promedio | **0,445** |
| Accuracy promedio | **0,722** |
| IoU promedio | **0,633** |
| IV distancia + 3 densidades | **0,783 (78,3 %)** |
| IV distancia (el mayor) | 0,282 |
| Umbral θ | 0,75 |
| Peso de vecindad α | 0,50 |
| SVM vs K-Means | 88–94 % = coherencia interna, **no** verdad terreno |
| Variables WoE | 7 |
| Features de imagen | 23 (8+6+9); LBP R=3, P=24 |
| Ventanas | 2011–2016 … 2015–2020 (hold-out vs 1984–2010; solapadas entre sí) |

El 0,775 que aparece en la tabla de métricas del Cap. 5 es el **F1 promedio**, no el IV. No se confundió.

---

## 9. Doctrina que debe mantenerse en defensa

1. La novedad es el **protocolo de validación multi-temporal + reproducibilidad abierta**, no el acoplamiento WoE–AC.
2. **Pesos** WoE fijos (entrenados 1984–2010). Las **siete variables** se recalculan en cada paso del AC.
3. Querétaro es **caso de uso**, no la aportación.
4. Los mapas se leen como **escenarios**, no como planes calibrados.
5. NGRDI, no «NDVI aproximado».
6. No hay réplica en otra ciudad. Q4 mide si el flujo corre en abierto / CPU.

---

## 10. Lo que queda (tú o opcional)

| Ítem | Quién | Bloquea entrega |
|---|---|---|
| Confirmar título de `front/portada.tex` contra la **carta de jurado** | Tú | **Sí**, si la carta ya se entregó y el título no coincide letra por letra |
| Figuras de dimensiones / frontera del AC en Cap. 2 (Moore ya está en Cap. 4) | Opcional | No |
| Barrido de primera aparición de cada acrónimo | Opcional | No |
| `ACTUALIZACION_EJECUTIVA_LARRAGA.md` todavía dice «ventanas independientes» | Carta a la Dra., no el PDF | No (si se la reenvías, conviene alinearla) |

---

## 11. Veredicto

Los 8 bloqueantes de la auditoría están cerrados. La higiene de la iteración 2 está cerrada. La prosa que se había reescrito ya no lleva tipografía de plantilla. El PDF compila en **110 páginas**, 0 refs/citas indefinidas.

Para imprimir o enviar a la Dra.: el documento está listo, **salvo** que confirmes el título contra la carta de jurado.

No se inventó ningún número.
