# Revisión Dra. Lárraga — Capítulo 3: Estado del Arte

> **Diagnóstico general:** "Todavía tiene un tono muy 'ensamblado' y poco crítico. Sigue pareciendo texto muy asistido por IA, aunque probablemente editado por el alumno."
> El capítulo funciona como catálogo descriptivo (autor → método → limitación), no como estado del arte crítico.

---

## Objetivo que debe cumplir el capítulo

Construir la necesidad inevitable de la tesis respondiendo cuatro preguntas:
1. ¿Cómo evolucionó el campo?
2. ¿Qué sí resuelven los modelos actuales?
3. ¿Qué NO resuelven todavía?
4. ¿Por qué esta tesis es necesaria?

---

## Problemas a corregir

### 1. No hay estado del arte — hay resumen bibliográfico
- **Problema:** Formato dominante: "X hizo esto → limitación". Eso es review descriptiva, no estado del arte crítico.
- **Falta:**
  - Cómo evolucionó el campo
  - Qué problemas siguen abiertos
  - Qué contradicciones existen entre enfoques
  - Exactamente dónde se posiciona la tesis
- **Acción:** Reestructurar la narrativa alrededor de problemas metodológicos, no de tipos de modelos.
- [ ] PENDIENTE

---

### 2. Falta profundidad técnica en SLEUTH
- **Problema:** Solo dice "calibración laboriosa" sin explicar:
  - Brute force calibration / Monte Carlo
  - Self-modifying CA
  - Spread coefficients
  - Path dependence / equifinality
  - Sensibilidad a resolución espacial
  - Problemas de sobreajuste
- **Acción:** Profundizar o condensar SLEUTH dejando solo lo que construye tu argumento.
- [ ] PENDIENTE

---

### 3. 🚨 La discusión de WoE es débil — GRAVÍSIMO
- **Problema:** WoE es TU metodología. Pero el capítulo dedica mucho a SLEUTH, ANN, DL y poco a:
  - Weight of Evidence bayesiano
  - Dinamica EGO
  - Transición espacial probabilística
  - Independencia condicional
  - Problemas estadísticos de WoE
  - Sesgo espacial / multicolinealidad
  - Correlación entre drivers
- **Acción:** Añadir sección dedicada a WoE y modelos probabilísticos espaciales con literatura LUCC real.
- [ ] PENDIENTE

---

### 4. Afirmaciones peligrosamente generales
- "la literatura anglosajona dominante" → necesita respaldo
- "los modelos convencionales son reactivos" → ¿cuáles exactamente?
- "los modelos deep learning tienen interpretabilidad limitada" → generalización enorme y debatida
- **Acción:** Anclar cada afirmación a papers concretos o suavizar a "en muchos casos".
- [ ] PENDIENTE

---

### 5. La tabla comparativa parece hecha por IA
- **Problema:** `| Enfoque | Fortalezas | Limitaciones |` es extremadamente típica de texto generado. Simplifica demasiado: SLEUTH ≠ modelo homogéneo, DL ≠ categoría uniforme.
- **Acción:** Eliminar o reemplazar por discusión textual con matices. Si se mantiene, reducirla y añadir notas al pie.
- [ ] PENDIENTE

---

### 6. El posicionamiento metodológico es artificial
- **Problema:** "son relativamente escasos los estudios que integren WoE…" — WoE+CA tiene bastante literatura (Dinamica EGO, LUCC, deforestación, urbanización).
- **La Dra. es directa:** "el alumno está tratando de construir artificialmente una 'rareza' metodológica."
- **Acción:** Reformular el aporte hacia: contexto mexicano, validación multi-período, reproducibilidad, serie larga, benchmarking de estabilidad.
- [ ] PENDIENTE

---

### 7. Revisión desbalanceada
- **Demasiado:** generalidades urbanas, DL, aprendizaje automático
- **Muy poco:**
  - CA probabilísticos
  - LUCC / Dinamica EGO / CA-Markov / FLUS / CLUE-S
  - Validación espacial e incertidumbre
  - Métricas de cambio de uso de suelo
  - Error allocation vs quantity disagreement (Pontius)
  - Sensibilidad espacial
- **Acción:** Rebalancear hacia la literatura LUCC especializada.
- [ ] PENDIENTE

---

### 8. El capítulo evita conflicto científico — típico de IA
- **Problema:** Todo escrito como "una limitación es…" "otra aproximación es…" sin:
  - Crítica fuerte
  - Comparación de resultados contradictorios
  - Discusión de fallas reales
  - Problematización de supuestos
- **Acción:** Añadir postura técnica fundamentada. Una tesis SÍ puede tomar postura.
- [ ] PENDIENTE

---

### 9. Exceso de "prudencia" artificial
- Frases como "sin afirmar superioridad metodológica", "de manera descriptiva", "sin valorar mérito relativo" — demasiado frecuentes.
- **Acción:** Eliminar. La tesis puede y debe tomar postura técnica sustentada.
- [ ] PENDIENTE

---

### 10. No construye la necesidad del trabajo
- **Problema:** Al terminar el capítulo el lector piensa "ok… otro modelo WoE-CA más."
- **Acción:** Reestructurar para que el lector concluya: "este trabajo era necesario porque X, Y, Z no estaban resueltos."
- [ ] PENDIENTE

---

## Estructura recomendada por la Dra. (reescritura)

```
3.1 Evolución del modelado urbano computacional
    (condensado: White & Engelen, SLEUTH, CA clásicos)

3.2 Problemas metodológicos persistentes
    a) Calibración: heurística, equifinalidad, costo computacional
    b) Validación: sobreajuste temporal, poca evaluación multi-período
    c) Interpretabilidad: black-box, ANN, DL
    d) Transferibilidad: modelos entrenados en China/USA, contextos LATAM

3.3 WoE y modelos probabilísticos espaciales
    WoE bayesiano, suitability maps, Dinamica EGO,
    independencia condicional, limitaciones estadísticas,
    correlación entre variables, sensibilidad espacial

3.4 Validación espacial en modelos LUCC
    Kappa limitations, FoM, quantity/allocation disagreement,
    IoU, sensibilidad al cambio neto, Pontius como referente

3.5 Vacíos específicos en ciudades mexicanas/intermedias
    (al FINAL, no al principio — aquí aparece Querétaro)
```

---

## Literatura LUCC que la Dra. pide leer (de verdad, no solo citar)

- Pontius Jr. et al. (2008) — FoM, validation
- Soares-Filho et al. — Dinamica EGO
- Dinamica EGO documentation
- CA-Markov, CLUE-S, FLUS, GeoSOS, MOLUSCE
- SLEUTH calibration papers
- Uncertainty and validation in LUCC

---

## Qué eliminar

- [ ] Tabla `| enfoque | fortalezas | limitaciones |`
- [ ] Frases administrativas: "esta sección presenta", "de manera objetiva", "sin valorar", "con tono prudente", "se pretende"
- [ ] Explicaciones obvias: "el crecimiento urbano es importante"
- [ ] Exceso de historia urbana general (>8 páginas históricas no aportan)

---

## Checklist

- [ ] Reestructurar siguiendo el esquema 3.1-3.5 de la Dra.
- [ ] Añadir sección sustantiva sobre WoE bayesiano y Dinamica EGO
- [ ] Añadir sección de validación espacial (Pontius, FoM, disagreement)
- [ ] Eliminar tabla IA de fortalezas/limitaciones
- [ ] Reformular posicionamiento: no "WoE+CA es raro" sino "estos problemas no resueltos en México"
- [ ] Añadir postura crítica propia en al menos 2-3 puntos
- [ ] Cierre con transición explícita al Cap. 4
