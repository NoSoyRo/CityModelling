# Revisión Dra. Lárraga — Capítulo 5: Modelo de Crecimiento Urbano

> **Diagnóstico general:** "El Capítulo 5 sí tiene aporte técnico real, pero está en riesgo porque mezcla implementación con modelo conceptual, tiene exceso de 'ingeniería de software' y no declara explícitamente la novedad científica del WoE-AC."
> Nivel actual: "tesis técnica fuerte, pero aún no tesis teóricamente blindada."

---

## Objetivo que debe cumplir el capítulo

Responder SOLO estas tres preguntas:
- **(A)** ¿Cuál es el modelo matemático?
- **(B)** ¿Cuál es la innovación exacta respecto a WoE-CA clásico?
- **(C)** ¿Qué se calibra y qué se evalúa?

---

## Lo que SÍ está bien (no tocar)

- ✅ Integración WoE → AC (válido y potencialmente publicable)
- ✅ Validación quinquenal múltiple (buena contribución metodológica)
- ✅ Pipeline reproducible
- ✅ Uso explícito de evidencia histórica (bien alineado con LUCC)
- ✅ AC con decaimiento + vecindario ponderado (no trivial)
- ✅ Honestidad sobre parámetros heredados (`neighbor_weight`, `distance_weight`)

---

## Problemas a corregir

### 1. 🚨 El capítulo está "sobrecargado de código mental"
- **Problema:** Rutas de archivos, scripts, nombres de funciones Python, JSON internos — todo eso pertenece a anexos o implementación, no a metodología científica.
- **La Dra.:** "No hay modelo formal. Todo eso es implementación para validar."
- **Acción:** Todo lo que sea ruta/script/JSON → apéndice técnico. En el capítulo solo: parámetros del modelo, ecuaciones, descripción metodológica.
- [ ] PENDIENTE

---

### 2. No se distinguen los tres niveles
- **Problema:** No queda claro qué es:
  - Modelo conceptual (lo más importante)
  - Implementación
  - Configuración experimental
- **Acción:** Separar explícitamente en secciones.
- [ ] PENDIENTE

---

### 3. Novedad científica no declarada explícitamente
- **Problema:** El sistema puede ser nuevo, pero no se formula: "¿qué parte EXACTA es nueva respecto a WoE-CA clásico?"
- **La contribución correcta (ejemplo de la Dra.):**
  > "acoplamiento reproducible WoE como campo probabilístico continuo dentro de un CA calibrado y validado en múltiples ventanas temporales en ciudad intermedia latinoamericana"
- **Acción:** Añadir una frase o párrafo explícito de novedad en la introducción del capítulo.
- [ ] PENDIENTE

---

### 4. AC muy "ingenierizado" — falta rigor formal
- **Problema:** Parece "un CA con muchas variables" en lugar de "un modelo probabilístico nuevo".
- **La Dra. propone el núcleo matemático compacto:**
  ```
  P_{i,j}(t) = σ( α·WoE_{i,j} + β·N_{i,j} + γ·D_{i,j} )
  ```
  Con notación limpia: S(x,t) = campo latente, φ_k(x) = evidencia WoE por variable, N(x,t) = influencia del vecindario con ponderación por distancia de Chebyshev.
- **Acción:** Presentar el modelo en 1 página estilo paper con notación homogénea y compacta.
- [ ] PENDIENTE

---

### 5. LBP+SVM mezclado en el capítulo equivocado
- **Problema:** LBP+KMeans+SVM es el pipeline de clasificación de imágenes, NO el modelo WoE-CA. Estar en el Cap. 5 hace que parezca "mezcla de dos tesis".
- **Acción:** Mover LBP+SVM a Cap. 4 (Área de estudio / datos) o al apéndice.
- [ ] PENDIENTE

---

### 6. "Este factor garantiza…" — lenguaje de certeza en modelo probabilístico
- **Problema:** "Este factor **garantiza** que zonas muy alejadas tengan probabilidad prácticamente nula."
- **Acción:** "Atenúa fuertemente la probabilidad en celdas alejadas (e.g., $d > 20$ celdas)."
- [ ] PENDIENTE

---

### 7. "Superior a la del Otsu simple" sin tabla de comparación
- **Problema:** Afirmación de comparación sin tabla ni referencia al repositorio.
- **Acción:** "en pruebas internas documentadas en el repositorio" o añadir la tabla.
- [ ] PENDIENTE

---

### 8. Desalineación de ecuaciones con Cap. 6
- **Problema:** Cap. 5 describe transición con decaimiento exponencial + umbral estocástico; Cap. 6 presenta ecuación compacta con sigmoide sin mencionar el decaimiento.
- **Acción:** Alinear ecuaciones entre ambos capítulos o explicitar qué términos quedan activos en la implementación validada.
- [ ] PENDIENTE

---

### 9. `eq:woe_definition` duplica `eq:woe` del Cap. 2
- **Problema:** Se define WoE dos veces con distintas etiquetas.
- **Acción:** Definir una vez en Cap. 2 y citar desde Cap. 5 con `\ref{eq:woe}`.
- [ ] PENDIENTE

---

## Estructura recomendada por la Dra.

```
5.1 Modelo conceptual (SOLO matemático)
    WoE, CA, función de transición
    Diagrama de arquitectura del modelo

5.2 Integración WoE–CA (LA CONTRIBUCIÓN)
    Ecuación global: P_{i,j}(t) = σ(α·WoE + β·N + γ·D)
    Interpretación de cada componente

5.3 Calibración
    Grid search: qué parámetros, qué rango, qué se optimiza

5.4 Validación
    Ventanas quinquenales, métricas, independencia temporal
```

---

## Núcleo matemático sugerido por la Dra. (verificar contra tu implementación)

```
Sea U_t(x) ∈ {0,1} el estado urbano de la celda x en el tiempo t.

Evidencia espacial:
  WoE_{k,j} = log[ P(X_k=j | U=1) / P(X_k=j | U=0) ]
  φ_k(x) = WoE_k(X_k(x))

Campo de propensión:
  S(x,t) = Σ_k w_k·φ_k(x) + w_n·N(x,t)

Vecindario (Moore ponderado por distancia Chebyshev):
  N(x,t) = Σ_{y∈N(x)} U_t(y)·d(x,y)^{-1} / Σ d(x,y)^{-1}

Probabilidad de transición:
  P(x,t) = 1 / (1 + exp(−S*(x,t)))

Actualización estocástica:
  U_{t+1}(x) = 1 si r < P(x,t), r ~ U(0,1)

Restricción global:
  Σ_x U_{t+1}(x) ≤ (1+ρ)·Σ_x U_t(x)
```

---

## Checklist

- [ ] Mover LBP+SVM al Cap. 4 o apéndice
- [ ] Mover rutas/scripts/JSON al apéndice
- [ ] Reestructurar en 5.1-5.4 según esquema de la Dra.
- [ ] Presentar núcleo matemático en ≤1 página con notación homogénea
- [ ] Declarar explícitamente la novedad en 1 frase
- [ ] Corregir "garantiza" → "atenúa fuertemente"
- [ ] Corregir comparación con Otsu (añadir tabla o suavizar)
- [ ] Alinear ecuaciones de transición con Cap. 6
- [ ] Unificar definición de WoE (una sola vez, en Cap. 2)
- [ ] Añadir párrafo de cierre con transición al Cap. 6
