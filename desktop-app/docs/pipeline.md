# Catálogo de etapas del pipeline

Este documento es la versión detallada de la tabla resumen en el `README`.
Cada etapa se describe con sus entradas, salidas, parámetros y el módulo
de `tesis_ac` que la implementa. Los IDs (`E0`–`E5`) son los mismos que
aparecen en la barra lateral de la aplicación.

---

## E0 — Imágenes crudas

| Campo | Valor |
|---|---|
| Entrada | (externo) Google Earth Engine / archivo histórico |
| Salida  | `data/raw/imagen_YYYY.png` (1984–2020) |
| Código  | — (importación manual) |

Las imágenes son capturas RGB de la Zona Metropolitana de Querétaro. Se
nombran por año y se asumen del mismo encuadre y resolución por
construcción.

---

## E1 — Clasificación binaria de una imagen

| Campo | Valor |
|---|---|
| Entrada | `data/raw/imagen_YYYY.png` |
| Salida  | `data/processed/batch_processing_*/year_YYYY/prediction_maps/svm_2_classes.npy` |
| Código  | `tesis_ac.pipeline.main_pipeline.SatelliteImagePipeline` |
| Parámetros | `n_clusters_list=[2]`, `n_pca_components=8`, `svm_kernel='linear'`, `sample_size=10000`, `random_state=42` |

La etapa se divide internamente en tres pasos visibles desde la UI:

1. **E1.a — Extracción de 23 features.** Por cada píxel se computan 23
   descriptores: bandas RGB e intensidad, brillo, medias locales 3×3 sobre
   cada canal (etiquetados como `*_std`), gradientes Sobel (x, y,
   magnitud), entropía local, contraste local (media en bloque 8×8),
   índices espectrales NDVI-aproximado, exceso de verde, exceso de rojo,
   componentes L*a*b* y componentes H, S, V.
2. **E1.b — Standardización + PCA(8).** La matriz `(n_pixels, 23)` pasa por
   un `StandardScaler` y un `PCA` que la proyecta a 8 dimensiones
   conservando una varianza típicamente del orden del 95 %.
3. **E1.c — K-Means(2) + SVM lineal.** Sobre el espacio PCA se ajusta un
   K-Means con $k=2$. Las pseudo-etiquetas se usan como objetivo para
   entrenar un SVM lineal con `sample_size` muestras balanceadas, que
   después predice la imagen entera.

---

## E2 — Estandarización de labels

| Campo | Valor |
|---|---|
| Entrada | `svm_2_classes.npy` (de E1) |
| Salida  | `data/processed/standardized_maps/YYYY.npy` |
| Código  | `tesis_ac.historical.standardize_labels` |
| Parámetros | `method='center'`, `center_ratio=0.3` |

K-Means no respeta convención alguna: el cluster 0 puede ser urbano o no
urbano según la imagen. Este paso aplica una heurística para garantizar
que `urbano=1` y `no_urbano=0`:

- **Método NDVI** (usado por el runner de la UI): el cluster con menor
  NDVI medio (menos vegetación) se marca como urbano.
- **Método centro 30 %** (usado por los scripts de batch): se examina el
  recuadro central; si el píxel mayoritario es 0 se invierte el mapa.

El mapa estandarizado es la entrada canónica para todas las etapas
posteriores.

---

## E3 — Entrenamiento WoE pooled (1984–2010)

| Campo | Valor |
|---|---|
| Entrada | `data/processed/standardized_maps/{1984..2010}.npy` |
| Salida  | `data/processed/woe_pooled_1984_2010.pkl` |
| Código  | `train_woe_pooled.py` + `tesis_ac.woe.woe.WoECalculator` |
| Parámetros | `binning_method='quantile'`, `max_bins=10`, `min_bin_size=50` |

Por cada par de años consecutivos `(t, t+1)`:

1. Se calculan **7 variables espaciales** sobre el estado urbano en `t`:
   distancia al frente urbano, densidades de vecinos (3×3, 5×5, 7×7),
   fragmentación local (varianza 5×5), tamaño del cluster urbano más
   cercano y gradiente Sobel.
2. Se identifica el conjunto de **transiciones**: píxeles que pasaron de 0
   a 1.

Las variables y transiciones de todos los pares se *acumulan* (pooling).
Sobre ese pool, `WoECalculator` discretiza cada variable en cuantiles y
calcula para cada bin:

$$
\mathrm{WoE}_i = \ln \frac{P(X \in \mathrm{bin}_i \mid \mathrm{transición})}{P(X \in \mathrm{bin}_i \mid \mathrm{no transición})}
$$

El `Information Value` total de la variable es

$$
\mathrm{IV} = \sum_i \left(P_+\big|_{\mathrm{bin}_i} - P_-\big|_{\mathrm{bin}_i}\right) \cdot \mathrm{WoE}_i.
$$

El pickle de salida contiene un `WoECalculator` con `woe_results: dict[str,
WoEResult]`. La aplicación lo reconoce automáticamente y lo abre con un
panel especializado que grafica los WoE por bin, los IV y las muestras
positivas/negativas por variable.

---

## E4 — Simulación AC quinquenal

| Campo | Valor |
|---|---|
| Entrada | `woe_pooled_1984_2010.pkl` + `standardized_maps/YYYY_inicial.npy` |
| Salida  | `data/processed/validation_quinquenal_*/yearly_predictions/*.npy` |
| Código  | `validate_quinquenal_*.py` + `tesis_ac.ca.rules` |
| Parámetros | `threshold=0.75`, `neighbor_weight` (calibrado), `distance_weight` (calibrado) |

Para cada ventana 5-anual (2011→2016, …, 2015→2020):

1. Se toma el mapa estandarizado del año inicial como estado.
2. En cada uno de los 5 pasos anuales:
   1. Se recalculan las 7 variables espaciales sobre el estado actual.
   2. Cada variable se transforma a su mapa WoE usando el `WoECalculator`
      entrenado.
   3. Los mapas WoE se suman ponderados por su IV normalizado.
   4. La suma se pasa por la sigmoide para obtener una probabilidad.
   5. Se añade la influencia de vecindad Moore ponderada por
      `neighbor_weight`.
   6. Una celda no urbana cambia a urbana si su probabilidad combinada
      supera `threshold` y un sorteo uniforme.

El resultado son 5 mapas predichos por ventana, guardados en
`yearly_predictions/`.

---

## E5 — Métricas y validación

| Campo | Valor |
|---|---|
| Entrada | predicción final + mapa observado del año meta |
| Salida  | `data/processed/validation_quinquenal_*/validation_results.json` |
| Código  | mismo script de E4, sección de métricas |

Métricas calculadas:

- **FoM (Figure of Merit)** sobre las celdas en cambio: $A / (A + B + C)$
  con $A$=aciertos, $B$=cambios perdidos, $C$=falsas alarmas.
- **Kappa de Cohen** sobre el conjunto completo.
- **IoU** (Jaccard) sobre la clase urbana.
- **Accuracy**, **Precision**, **Recall**, **F1**.
- **Descomposición Pontius**: `quantity_disagreement` y
  `allocation_disagreement`.
- **Conteos**: urbano observado, urbano predicho, $\Delta$ urbano observado
  y predicho.

El dashboard de métricas de la UI agrega las cinco ventanas y muestra la
media de FoM, Kappa, IoU y Accuracy, así como una tabla con todas las
columnas.
