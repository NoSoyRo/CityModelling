# Revisión Dra. Lárraga — Capítulo 6: Resultados y Análisis

> **Diagnóstico general:** "Noto tres estilos mezclados: técnico, descriptivo urbano y 'explicativo tipo reporte'. Debe haber un hilo único: modelo → calibración → validación → interpretación."
> Nivel actual: 80-85% listo. Faltan síntesis global e interpretación dinámica formal.

---

## Objetivo que debe cumplir el capítulo

Presentar los resultados siguiendo el hilo: **modelo → calibración → validación → interpretación**, con una síntesis global integradora al final.

---

## Lo que SÍ está bien

- ✅ Tabla quinquenal con cifras correctas (FoM 0,317; κ 0,445; Acc 0,722; IoU 0,633)
- ✅ Nota metodológica sobre clase "urbano" ≠ mancha urbana real
- ✅ Reconocimiento honesto de la tabla de error como "cualitativa, no cuantificada"
- ✅ FoM contextualizado como "banda intermedia" respecto a Pontius

---

## Problemas a corregir

### 1. 🚨 Tabla de tasas de crecimiento sin fuente
- **Problema:** Tasas 1,8 %, 2,4 %, 3,0 %, 2,7 %/año sin referencia ni derivación explícita.
- **En defensa:** un sinodal preguntará de dónde vienen esos números.
- **Acción:** Calcularlas desde `standardized_maps` y documentar el método, o eliminar la tabla y describir cualitativamente.
- [ ] PENDIENTE — **URGENTE**

---

### 2. Caption "valida el mecanismo de difusión como driver principal" — sobreclaim
- **Problema:** Un caption de figura no puede afirmar causalidad exclusiva.
- **Acción:** "es coherente con un mecanismo de difusión descrito en la literatura, sin establecer causalidad exclusiva."
- [ ] PENDIENTE

---

### 3. κ = 0,445 calificado como "moderado a sustancial" — incorrecto
- **Problema:** Según la escala Landis & Koch citada en Cap. 2, 0,445 es **moderado** (0,41–0,60). No es "sustancial".
- **Acción:** Cambiar a "acuerdo moderado (próximo al umbral sustancial)".
- [ ] PENDIENTE

---

### 4. Enunciados de H1-H4 no son idénticos a Cap. 1
- **Problema:** La tabla de validación de hipótesis parafrasea, no transcribe. H4 pasa de "estable en métricas agregadas" a "robustez morfológica".
- **Acción:** Copiar textualmente los enunciados del Cap. 1 antes del veredicto. O usar `\newcommand{\hipUno}{...}` definido una sola vez.
- [ ] PENDIENTE

---

### 5. Grid search prometido — sin tabla de exploración
- **Problema:** Se afirma calibración por grid search pero no hay tabla del barrido (valores de θ probados, FoM por valor, criterio de selección).
- **Acción:** Añadir tabla mínima θ vs FoM en entrenamiento, o remisión explícita al script/log del repositorio.
- [ ] PENDIENTE

---

### 6. Tipografía inconsistente en captions y tablas
- Captions con punto decimal anglosajón: `FoM=0.222`, `+9.5%`, `0.965`
- `88--94\%` sin espacio fino
- `1984-2020` con guion simple en líneas 17, 23, 40 (debe ser `--`)
- **Acción:** Pasada completa: `$0{,}222$`, `$+9{,}5$\,\%`, `$0{,}965$`, `$88$--$94$\,\%`, `1984--2020`
- [ ] PENDIENTE

---

### 7. ~18 tablas y figuras con label sin referencia en el texto
- Todas las `tab:metrics_XXXX_XXXX` y `fig:summary_XXXX_XXXX` de las cinco ventanas quinquenales están huérfanas.
- **Acción:** Añadir `(véase Tabla~\ref{tab:metrics_2011_2016})`, `(Figura~\ref{fig:summary_2011_2016})`, etc.
- [ ] PENDIENTE

---

### 8. 🚨 Falta síntesis global integradora — CLAVE EN TESIS
- **Problema:** No hay sección tipo "Síntesis del desempeño del modelo".
- **La Dra. propone añadir §6.7:**
  > El modelo opera en un régimen de alta cobertura predictiva y baja especificidad relativa. El campo WoE favorece la expansión continua ante vecindades parcialmente urbanizadas (propagación conservadora en omisión, expansiva en FP). El sistema no converge a estado estacionario único — presenta atractores espaciales (núcleos y corredores urbanos) que persisten a través de las ventanas quinquenales.
- **Acción:** Añadir sección 6.7 de síntesis con interpretación dinámica.
- [ ] PENDIENTE

---

### 9. Falta interpretación matemática del resultado global (opcional pero potente)
- **La Dra. propone §6.8 (opcional):**
  > El modelo WoE-AC como sistema dinámico discreto no lineal: s_{t+1} = F(s_t, W), donde W induce campo de interacción espacial no homogéneo. Clase de: sistemas dinámicos en grafos ponderados, procesos de difusión sesgada, autómatas celulares no homogéneos.
- **Acción:** Si se quiere subir a nivel artículo, añadir §6.8.
- [ ] OPCIONAL

---

### 10. Capítulo sin cierre ni transición al Cap. 7
- **Acción:** Añadir párrafo final: "Estos hallazgos y limitaciones se sintetizan en las conclusiones, donde se contrastan las aportaciones previstas con la evidencia obtenida."
- [ ] PENDIENTE

---

## Checklist

- [ ] Calcular o eliminar tasas de crecimiento sin fuente
- [ ] Corregir caption "valida el mecanismo" → "es coherente con"
- [ ] Corregir κ 0,445: "moderado" no "moderado a sustancial"
- [ ] Igualar enunciados H1-H4 a los de Cap. 1 (textual)
- [ ] Añadir tabla de grid search (o remitir al repo)
- [ ] Pasada tipográfica: decimales, porcentajes, en-dashes en captions
- [ ] Añadir referencias cruzadas a las 18 tablas/figuras huérfanas
- [ ] Añadir §6.7 Síntesis del desempeño
- [ ] Añadir párrafo de cierre con transición al Cap. 7
- [ ] (Opcional) §6.8 Interpretación dinámica formal
