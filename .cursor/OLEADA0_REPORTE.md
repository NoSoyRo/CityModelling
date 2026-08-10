# Oleada 0 — Cirugía estructural (completada)

Rama: `dev/reestructura-larraga`. Sin reescritura de prosa: solo movimiento de bloques, renumeración
y referencias cruzadas. Todo lo que requiere pluma quedó marcado con `% TODO-REDACCION:`.

## Resultado de compilación

| Métrica | Valor |
|---|---|
| Errores | 0 |
| Referencias indefinidas | 0 |
| Citas indefinidas | 0 |
| Warnings de biber | 0 |
| Overfull > 20 pt | 0 |
| Páginas | 105 |
| Capítulos | 6 + apéndice |

Índice resultante: 1 Introducción · 2 Marco teórico · 3 Estado del arte · **4 Modelo de crecimiento
urbano híbrido** · 5 Resultados y análisis · 6 Conclusiones · A Arquitectura e implementación.

## Movimientos de contenido

| Bloque | Origen | Destino |
|---|---|---|
| Capítulo completo "Área de estudio" | `cap04_area_estudio.tex` | Cap. 4, `\section{Área de estudio}` (secciones degradadas a subsecciones) |
| Pipeline de clasificación y análisis binario | `cap04_area_estudio.tex` | Cap. 4, `\section{Preprocesamiento de datos}` |
| Adquisición, arreglo $M_{t,i,j}$, las 23 características por píxel, figura del pipeline | `cap02_marco_teorico.tex` (§ Preprocesamiento) | Cap. 4, `\subsection{Representación de los datos y extracción de características}` |
| Párrafo de los "tres planos" (conceptual / implementación / configuración) | Cap. 5 antiguo | Cap. 4, `\section{Consideraciones del modelo}` |
| Figura del pipeline WoE-AC | Cap. 5 antiguo | Cap. 4, `\section{Arquitectura del modelo}` |
| Definición conceptual de SLEUTH (cinco coeficientes) | `cap03_estado_del_arte.tex` | Cap. 2, § Autómatas celulares |
| Definición de IoU y descomposición de Pontius | `cap03_estado_del_arte.tex` (§ Validación) | Cap. 2, § Métricas |
| `\section{Alcances y limitaciones}` | `cap01_introduccion.tex` | Cap. 6, como `\section{Alcances}` + `\subsection{Limitaciones de alcance del estudio}` |

Promociones de nivel: la definición de WoE, que vivía dentro de la sección de autómatas celulares del
Cap. 2, es ahora `\section{Pesos de Evidencia}`. "Trabajo futuro", que era subsección de Limitaciones
en Conclusiones, es ahora `\section`, con el orden exigido Alcances → Limitaciones → Trabajo futuro.

## Etiquetas

- Creadas: `sec:consideraciones-modelo`, `sec:arquitectura-modelo`, `sec:area-estudio`,
  `sec:preprocesamiento`, `ssec:calidad-clasificacion`, `sec:validacion-protocolo`, `sec:woe`.
- Eliminada del documento: `ch:area-estudio` (el archivo `cap04_area_estudio.tex` la conserva pero ya
  no se incluye desde `main.tex`).
- Repuntadas: seis referencias a `ch:area-estudio` en `cap01`, `cap03`, `cap07` y el apéndice, hacia
  `ch:modelo-crecimiento-urbano`, `sec:area-estudio` o `sec:preprocesamiento` según el contexto.
- Contadores de los `capXX_standalone.tex` renumerados. `cap05_standalone.tex` ahora incluye el
  apéndice, porque el capítulo fusionado referencia `ap:arquitectura`, `alg:adquisicion` y sus tablas.

## Marcadores `TODO-REDACCION` (insumo de la Oleada 1)

18 en total: 7 en el Cap. 4, 6 en el Cap. 2, 2 en el Cap. 3, 2 en el Cap. 6, 1 en el Cap. 1.
Los más importantes:

- **Cap. 2**: la sección de preprocesamiento quedó reducida a los algoritmos (StandardScaler, PCA,
  K-Means, SVM). Faltan por redactar las subsecciones conceptuales con cita: imágenes satelitales y
  Google Earth como fuente, clasificación de cobertura binaria, espacio de color RGB y
  transformaciones cromáticas, y LBP. También falta subseccionar las métricas y añadir ventajas,
  limitaciones y justificación en la nueva sección de WoE.
- **Cap. 4**: falta el resumen del capítulo con el patrón pedido y la marca de dónde empieza la
  aportación propia; falta desarrollar "Consideraciones del modelo"; la sección "Arquitectura del
  modelo" está vacía a la espera del diagrama de abstracción; hay duplicación por resolver entre la
  descripción formal de las 23 características (venida del Cap. 2) y la descripción en prosa que ya
  existía en el pipeline.
- **Cap. 6**: fusionar las limitaciones traídas de la introducción con las conceptuales, empíricas y
  de transferibilidad, que se solapan en tres puntos (RGB sin infrarrojo, representación binaria y
  ausencia de variables socioeconómicas).
- **Cap. 1**: reescribir "Organización del documento" para seis capítulos.

## Decisiones que conviene revisar

1. El archivo `cap04_area_estudio/cap04_area_estudio.tex` y su `cap04_standalone.tex` quedaron
   huérfanos, no borrados. Conviene eliminarlos cuando la fusión esté validada, para que nadie los
   edite por error.
2. Los algoritmos que quedaron en el Cap. 2 conservan parámetros específicos de este trabajo
   ($k=8$, $k=2$, muestra de 10 000 píxeles). Habría que generalizarlos allí y dejar los valores
   concretos en el Cap. 4.
3. La expansión del acrónimo SLEUTH que se añadió al Cap. 2 necesita cita explícita.
