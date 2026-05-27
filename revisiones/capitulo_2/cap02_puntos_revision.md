# Revisión Dra. Lárraga — Capítulo 2: Marco Teórico

> **Nota:** La Dra. no hizo comentarios extensos sobre el Cap. 2 de forma explícita. Los puntos aquí vienen de observaciones transversales de su revisión y de la revisión previa de la tesis. El capítulo cumple su función definicional, pero tiene áreas de mejora.

---

## Objetivo que debe cumplir el capítulo

Formalizar los conceptos que usa la tesis — AC, WoE, métricas — de forma precisa y autocontenida, sin revisión de literatura (eso va en Cap. 3).

---

## Problemas a corregir

### 1. Una sola cita en todo el capítulo
- **Problema:** Solo `BonhamCarter1994` está citado. Un marco teórico de AC y WoE sin citar a Wolfram, White & Engelen, Clarke, ni Pontius es indefendible.
- **Acción:** Añadir citas a las definiciones fundamentales:
  - `\textcite{Wolfram1984}` al definir clases de AC
  - `\textcite{White1997}` al hablar de AC probabilísticos urbanos
  - `\textcite{Clarke1998}` como referencia de AC calibrado
  - `\textcite{BonhamCarter1994}` ya está — mantener
  - `\parencite{pontius2008comparing}` al definir FoM
- [ ] PENDIENTE

---

### 2. Notación decimal inconsistente en el ejemplo de FoM
- **Problema:** El ejemplo numérico al final del capítulo usa punto decimal (`0.500`) en lugar de coma decimal española (`$0{,}500$`).
- **Acción:** Cambiar `0.500` → `$0{,}500$` en el ejemplo didáctico.
- [ ] PENDIENTE

---

### 3. ~14 ecuaciones con `\label` que nadie referencia
- **Problema:** `eq:woe`, `eq:woe_total`, `eq:kappa`, `eq:fom_jaccard`, etc. tienen etiqueta pero ningún párrafo del texto las invoca con `\ref{}`.
- **Acción:** Elegir una de dos opciones por ecuación:
  - Citarla desde Cap. 5 o Cap. 6 con `(Ec.~\ref{eq:woe})`
  - Eliminar el `\label{}` si nunca se referenciará
- [ ] PENDIENTE

---

### 4. Muletilla "permite" en definiciones
- **Problema:** "la técnica que **permite** derivar probabilidades…" — muletilla repetida.
- **Acción:** Sustituir por verbo preciso: "mediante la cual se derivan…"
- [ ] PENDIENTE

---

### 5. Sin párrafo de cierre — transición abrupta
- **Problema:** El capítulo termina en un ejemplo numérico sin síntesis ni puente al Cap. 3.
- **Acción:** Añadir 2-3 líneas: "Con estas definiciones formales, el capítulo siguiente revisa cómo se han aplicado enfoques similares y qué rangos de desempeño reporta la literatura."
- [ ] PENDIENTE

---

## Checklist

- [ ] Añadir citas a definiciones fundamentales (Wolfram, White, Clarke, Pontius)
- [ ] Corregir notación decimal del ejemplo FoM
- [ ] Resolver ecuaciones huérfanas (citar o quitar label)
- [ ] Eliminar muletilla "permite" en definiciones
- [ ] Añadir párrafo de cierre con transición al Cap. 3
