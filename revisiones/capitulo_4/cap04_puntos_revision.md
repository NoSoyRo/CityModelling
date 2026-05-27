# Revisión Dra. Lárraga — Capítulo 4: Área de Estudio y Datos

> **Diagnóstico general:** "El capítulo está inflado artificialmente — mucha microestructura, poca densidad conceptual real. El mayor problema metodológico es el dataset de Google Earth manual." Parece tesis de ingeniería de software más que investigación espacial.

---

## Objetivo que debe cumplir el capítulo

Describir el área de estudio, justificar la fuente de datos y documentar el pipeline de preprocesamiento de manera que un lector externo pueda reproducirlo y evaluar su validez metodológica.

---

## Problemas a corregir

### 1. 🚨 "NDVI aproximado" desde RGB — ERROR TÉCNICO GRAVE
- **Problema:** NDVI requiere banda NIR + banda roja. Con RGB no hay NDVI real.
- **Lo que puede estar usando en realidad:** ExG, VARI, pseudo-NDVI, índices visibles, proxy cromático.
- **La Dra.:** "Un sinodal fuerte en percepción remota se lo destruye inmediatamente."
- **Acción:** Cambiar TODA mención de "NDVI aproximado" por:
  - "índice de vegetación aproximado basado en canales visibles" o
  - "proxy cromático de vegetación (ExG/VARI)"
- [ ] PENDIENTE — **URGENTE**

---

### 2. 🚨 Pseudo-validación — circularidad metodológica
- **Problema:** Accuracy 88-94% está calculado contra pseudoetiquetas de K-Means, no contra ground truth.
  - K-Means genera etiquetas → SVM aprende esas etiquetas → se mide qué tan bien reproduce esas mismas etiquetas.
  - Eso NO es accuracy de clasificación real. Es coherencia interna del clasificador.
- **Acción:** Cambiar "accuracy 88-94%" por "coherencia interna del clasificador" o "concordancia con las pseudoetiquetas de K-Means".
- [ ] PENDIENTE — **URGENTE**

---

### 3. Capítulo inflado artificialmente
- **Problema:** Subsecciones 4.3.1, 4.3.2, 4.3.3, 4.3.4… que podrían ser 3 páginas compactas. Típico de texto generado/expandido.
- **Acción:** Reducir el capítulo a la mitad. Compactar secciones de "criterios de selección", "metodología de adquisición", "selección de herramienta", "resultados de adquisición", "validación de representatividad".
- [ ] PENDIENTE

---

### 4. Dataset de Google Earth — lenguaje defensivo en lugar de justificación sólida
- **Problema:** El texto está lleno de: "no equivale a calibración radiométrica", "evaluación visual", "aproximado", "representativo". Eso transmite fragilidad, no solidez.
- **Cómo reencuadrarlo (ejemplo de la Dra.):**
  > "Dado que el objetivo del estudio es modelar expansión urbana macroscópica y no caracterización espectral fina, se privilegió la consistencia espacial y temporal del encuadre sobre la calibración radiométrica absoluta."
- **Acción:** Sustituir el lenguaje defensivo por justificación metodológica positiva.
- [ ] PENDIENTE

---

### 5. Resolución espacial ambigua
- **Problema:** "1792 × 3024 píxeles" NO es resolución espacial. Falta:
  - Metros/pixel
  - Escala efectiva
  - Variación temporal (probablemente cambia entre años en Google Earth)
  - Correspondencia geográfica real
- **La Dra.:** "Eso puede afectar muchísimo un CA espacial."
- **Acción:** Documentar resolución efectiva en m/pixel para al menos 3 años representativos, o reconocer explícitamente la variación como limitación cuantificada.
- [ ] PENDIENTE

---

### 6. Sin cuantificación de incertidumbre
- **Problema:** Nunca cuantifica:
  - Error de georreferenciación
  - Error temporal (¿las capturas son del mismo mes/estación?)
  - Error de mosaico
  - Error cromático
  - Error de compresión
  - Error de clasificación esperado
- Solo dice "se revisó visualmente" — débil para defensa.
- **Acción:** Añadir al menos una tabla o párrafo con estimación de incertidumbre por tipo de error, aunque sea cualitativa con nivel (bajo/medio/alto).
- [ ] PENDIENTE

---

### 7. Pipeline LBP+KMeans+SVM rarísimo — sin justificación
- **Problema:** La pregunta inevitable en defensa: ¿por qué no supervisado directo? ¿por qué no Random Forest/XGBoost/U-Net? ¿qué aporta exactamente esa combinación híbrida?
- **Acción:** Añadir un párrafo que justifique científicamente la elección:
  - Por qué LBP para textura urbana desde RGB
  - Por qué clustering previo para pseudoetiquetas (falta de ground truth)
  - Por qué SVM sobre las pseudoetiquetas
- [ ] PENDIENTE

---

### 8. Exceso de ingeniería de software
- **Problema:** Diagramas UML, rutas de archivos, clases, `architecture.puml`, JSON — en una tesis de modelación espacial sobra.
- **La Dra.:** "El alumno se siente más cómodo programando que haciendo modelación espacial."
- **Acción:** Mover UML y detalles de implementación al apéndice. Dejar solo la descripción metodológica en el cuerpo.
- [ ] PENDIENTE

---

### 9. Cero citas en todo el capítulo
- **Problema:** El cap. 4 no tiene una sola cita bibliográfica.
- **Qué citar:**
  - INEGI / CONAPO / SEDATU para datos de población y dinámica urbana de Querétaro
  - `\parencite{Gorelick2017}` (Google Earth Engine) para distinguir el flujo manual local
  - `\parencite{Herold2003}` o `\parencite{Ojala2002}` para justificar LBP
- [ ] PENDIENTE

---

### 10. Sección "Localización geográfica" es un stub
- **Problema:** Solo remite a una tabla posterior sin texto propio.
- **Acción:** Fusionar con §4.3 o redactar 1 párrafo con coordenadas y justificación espacial.
- [ ] PENDIENTE

---

## Lo que SÍ rescata la Dra.

- ✅ La honestidad metodológica (al menos reconoce limitaciones)
- ✅ La reproducibilidad (código, JSON, scripts, pipeline publicados)
- ✅ La serie larga de 37 años
- ✅ La validación temporal multi-período (lo más defendible del trabajo)

---

## Checklist

- [ ] Eliminar "NDVI aproximado" — reemplazar por proxy cromático / índice visible
- [ ] Cambiar "accuracy 88-94%" → "coherencia interna del clasificador"
- [ ] Reducir el capítulo a la mitad (compactar subsecciones)
- [ ] Reencuadrar Google Earth con justificación positiva, no defensiva
- [ ] Documentar resolución efectiva en m/pixel o reconocer variación
- [ ] Añadir estimación de incertidumbre por tipo de error
- [ ] Justificar científicamente el pipeline LBP+KMeans+SVM
- [ ] Mover UML/diagramas al apéndice
- [ ] Añadir citas (INEGI, Gorelick, Ojala)
- [ ] Completar sección de Localización geográfica
