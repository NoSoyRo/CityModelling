# Referencia de API

Todas las rutas viven bajo el mismo origen del backend (por defecto
`http://localhost:8765`). El frontend de desarrollo en `localhost:5173`
hace proxy automático a este origen.

## HTTP

### `GET /api/health`

```json
{
  "status": "ok",
  "project_root": "/Users/rod/Projects/MSC/Tesis/CityModelling",
  "time": "2026-05-20T16:42:33.000Z"
}
```

### `GET /api/stages`

Devuelve el catálogo estático de las seis etapas (E0–E5), tal y como se
muestra en la barra lateral. Cada elemento es un `StageDescriptor`:

```json
{
  "id": "E1",
  "title": "Clasificación binaria",
  "one_liner": "23 features por píxel → StandardScaler+PCA(8) → K-Means → SVM lineal.",
  "inputs": ["data/raw/imagen_YYYY.png"],
  "outputs": ["…/svm_2_classes.npy"],
  "code_module": "tesis_ac.pipeline.main_pipeline.SatelliteImagePipeline",
  "parameters": ["n_clusters_list=[2]", "n_pca_components=8", "..."]
}
```

### `GET /api/catalog`

Escanea el sistema de archivos en busca de artefactos del pipeline.
Devuelve una lista plana de `ArtifactRef`:

```json
{
  "artifacts": [
    {
      "id": "data/raw/imagen_1984.png",
      "path": "/.../data/raw/imagen_1984.png",
      "kind": "raw_image",
      "stage": "E0",
      "size_bytes": 1827483,
      "modified": "2025-03-14T13:59:00Z",
      "label": "Imagen RGB 1984",
      "tags": { "year": "1984" }
    }
  ],
  "total": 412,
  "scanned_paths": ["data/raw/", "..."]
}
```

### `POST /api/inspect`

Body: un `ArtifactRef`. Devuelve `InspectionResponse` con la representación
adecuada según el tipo de archivo:

- `.npy` → `numpy: NumpyStats` y opcionalmente `image_preview_b64`.
- `.json` → `json_content`.
- `.pkl` → `pickle_tree` y, si es el modelo WoE, `woe: WoeModelInspection`.
- `.png` → `image_preview_b64`.

### `GET /api/metrics`

Lee todos los `validation_results.json` de las carpetas
`validation_quinquenal_*` y devuelve un `MetricsDashboard`:

```json
{
  "runs": [
    {
      "window": "2011-2016",
      "fom": 0.2224,
      "kappa": 0.3491,
      "iou": 0.5810,
      "accuracy": 0.6814,
      "...": "…",
      "threshold": 0.75,
      "neighbor_weight": 0.5
    }
  ],
  "aggregate": { "fom": 0.317, "kappa": 0.445, "iou": 0.633, "accuracy": 0.722 }
}
```

### `POST /api/jobs`

Lanza un job. Body:

```json
{
  "kind": "classify_image",
  "parameters": {
    "image": "/.../data/raw/imagen_1990.png",
    "sample_size": 5000
  }
}
```

Respuesta:

```json
{ "job_id": "ab12cd34ef56", "started_at": "…", "kind": "classify_image" }
```

### `GET /api/jobs/{job_id}/history`

Repite la traza completa de eventos del job (útil si el cliente se reconectó
después de iniciar el job).

## WebSocket

### `GET ws://localhost:8765/ws`

Cada mensaje JSON es un `JobEvent`:

```json
{
  "job_id": "ab12cd34ef56",
  "ts": "2026-05-20T16:42:33.123Z",
  "level": "metric",
  "step": "pca",
  "message": "PCA(8) explicó 94.7% de varianza",
  "payload": {
    "X_pca_shape": [262144, 8],
    "explained_variance_ratio": 0.947,
    "elapsed_ms": 318
  }
}
```

Niveles posibles: `info`, `debug`, `warn`, `error`, `metric`, `artifact`,
`progress`, `done`.

El bus tiene historial bounded por job (5 000 últimos eventos). Los
suscriptores con cola llena pierden mensajes silenciosamente.
