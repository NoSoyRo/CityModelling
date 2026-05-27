# Revisión Dra. Lárraga — Capítulo 1: Introducción

> **Diagnóstico general:** El texto está construido como enumeración administrativa, no como introducción argumentativa de tesis concluida. Patrón típico de IA: frases genéricas, estructura excesiva, tono artificialmente prudente, ausencia de narrativa científica real.

---

## Objetivo que debe cumplir el capítulo

Al terminar de leer la introducción, el lector debe poder responder:

- ¿Existe un problema científico relevante y concreto?
- ¿Hay un vacío específico no resuelto en la literatura?
- ¿El enfoque propuesto es pertinente y justificado?
- ¿Se hizo una investigación sólida? 
- ¿Los resultados aportan algo verificable?

---

## Problemas a corregir

### 1. No hay narrativa científica

- **Problema:** Frases tipo "Esta sección presenta…", "Se exponen…", "Se justifica…", "Se enuncian…" — es índice comentado, no argumento.
- **Acción:** Reescribir como argumento científico continuo. El lector no debe sentir que lee un guión de secciones.
- PENDIENTE

---

### 2. El problema está mal planteado — brechas genéricas

- **Problema:** Las cuatro brechas (metodológica, temporal, contextual, operacional) son genéricas; parecen salidas de ChatGPT.
- **Lo que falta:** Demostrar qué falla específicamente, en qué papers, por qué importa, por qué la propuesta lo resuelve.
- **Ejemplo de brecha real (Dra.):**
  > "Los modelos CA aplicados a ciudades mexicanas suelen calibrarse heurísticamente y rara vez evalúan estabilidad temporal multi-período, lo que dificulta determinar si la capacidad predictiva observada corresponde a sobreajuste de corto plazo o a mecanismos espaciales robustos."
- **Acción:** Reemplazar las cuatro brechas genéricas por UNA brecha concreta y científicamente defendible.
- PENDIENTE

---

### 3. La justificación es débil

- **Problema:** "WoE permite…" "Los AC permiten…" es descripción técnica, no justificación científica.
- **Lo que falta responder:**
  - ¿Por qué importa modelar Querétaro específicamente?
  - ¿Qué aporta al urbanismo computacional nacional?
  - ¿Qué problema metodológico atiende que no resuelven los existentes?
  - ¿Por qué no bastan enfoques como ANN, DL, ABM?
- **Acción:** Reescribir la justificación como argumento de necesidad, no como descripción de técnicas.
- PENDIENTE

---

### 4. 🚨 Hipótesis muy mal formuladas — GRAVÍSIMO

#### H2 cita un archivo interno del proyecto

```
\path{quinquenal_best_config.json}
```

- **Problema:** Una hipótesis es una afirmación que surge ANTES de ver resultados. Citar un archivo interno delata que se redactó post-hoc.
- **Acción:** Eliminar toda referencia a archivos del proyecto. Reescribir como afirmación falsable.
- PENDIENTE — **URGENTE**

#### H1 no es hipótesis fuerte

- "produce simulaciones espacialmente coherentes con métricas en el rango reportado" = "esperamos que salga parecido a otros papers"
- **Problema:** No es falsable claramente, no establece relación causal.
- PENDIENTE

#### Problemas generales de H1-H4:

- No son claramente falsables
- No establecen relaciones causales
- Varias son descriptivas
- Algunas parecen redactadas después de ver resultados

---

### 5. Metodología como lista técnica, no diseño de investigación

- **Problema:** Menciona grid search, métricas, Google Earth, IoU, Kappa pero no explica:
  - ¿Por qué esas métricas y no otras?
  - ¿Por qué WoE y no regresión logística?
  - ¿Por qué AC y no ABM?
  - ¿Cuál es la lógica epistemológica del modelo?
- **Acción:** En un párrafo, justificar el diseño metodológico como decisión razonada.
- PENDIENTE

---

### 6. Suena a propuesta, no a tesis terminada

- **Problema:** Demasiado futuro: "se pretende", "se evaluará", "se plantea", "aportaciones previstas".
- **Acción:** Cambiar a pasado donde son resultados: "el modelo produjo", "se observó", "los resultados muestran", "se encontró que".
- PENDIENTE

---

### 7. Tono artificialmente defensivo — delata IA

- **Problema:** "tono prudente", "aportaciones previstas", "sujetas a verificación", "con el debido sustento empírico" — cada tres líneas.
- **Acción:** La ciencia real no se blindan lingüísticamente así. Eliminar el exceso de precautelas.
- PENDIENTE

---

### 8. Contribución metodológica parece pequeña

- **Problema:** WoE+CA ya existe ampliamente. Papers señalados por la Dra.:
  - "Spatially explicit simulation... WoE-CA model in India" (2024)
  - Clarke (1997) — fundacional CA urbano
  - ANN+CA (2005)
  - Grid search no es contribución. Validar con Kappa/FoM es estándar.
- **El verdadero valor debe venir de:**
  - Ciudades intermedias mexicanas poco estudiadas
  - Validación temporal robusta multi-período
  - Reproducibilidad abierta
  - Serie larga (37 años)
  - Análisis de estabilidad temporal
- **Acción:** Reformular la contribución alrededor de lo que SÍ es diferenciador. Responder explícitamente: **¿qué tiene diferente este trabajo?**
- PENDIENTE

---

### 9. Demasiadas tablas — organización excesiva

- **Problema:** Tabla de limitaciones + tabla de aportaciones + tabla de alcances = reporte técnico administrativo.
- **Acción:** Reducir tablas al mínimo necesario. Integrar la información al texto argumentativo.
- PENDIENTE

---

### 10. Organización del documento (sección final)

- **Problema:** Lista de capítulos con números hardcoded; no usa `\ref{ch:...}`.
- **Problema adicional:** Cap. 4 se llama "Área de estudio y datos" aquí pero el `\chapter{}` real dice "Área de estudio".
- **Acción:** Usar `Capítulo~\ref{ch:introduccion}`, etc. Alinear nombres.
- PENDIENTE

---

## Checklist de reescritura (orden recomendado)

- 1. Definir en una frase: ¿qué hace diferente este trabajo? (antes de escribir nada)
- 1. Reescribir párrafo 1 en pasado, con la contribución como punto de llegada narrativa
- 1. Reemplazar brechas genéricas por UNA brecha concreta con papers que la sostienen
- 1. Reescribir justificación como argumento de necesidad
- 1. Reescribir H1-H4: falsables, sin archivos internos, sin "rango reportado"
- 1. Cambiar "se pretende / se evaluará" → "el modelo produjo / se encontró"
- 1. Reducir tablas: fusionar o eliminar la de limitaciones y la de aportaciones
- 1. Pasada final: eliminar "se plantea", "se pretende", "de manera prudente", "con el debido sustento"

---

## Nota de la Dra. (textual)

> "Estás escribiendo 'como cree que escribe una tesis', no como investigador que domina el campo."

