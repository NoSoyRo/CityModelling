# Copilot Project Instructions (Tesis AC)

**Objetivo:** Generar código limpio y testeable para un pipeline de clasificación no supervisada (RGB + LBP + Sobel) → grid celular → Autómata Celular → Optimización con GA (DEAP) → métricas.

## Convenciones
- Lenguaje: Python 3.11+
- Estilo: PEP8, type hints, docstrings Google-style.
- Logging: usar `logging.getLogger(__name__)`.
- Config: TODOs y parámetros en `configs/*.yaml` (usar loader central `config.py`).
- IO raster: `rasterio`; vectorial: `geopandas`.
- ML: `scikit-learn`; GA: `deap`; imágenes: `scikit-image`.

## Módulos y responsabilidades
- `features/`: `lbp.py` (LBP), `edges.py` (Sobel), `stack.py` (concat features).
- `clustering/`: `cluster.py` (KMeans/GMM), `labeling.py` (mapa 3 clases).
- `grid/`: `raster_to_grid.py` (resample/quantize), `neighbors.py`.
- `woe/`: `woe.py` (calcular WoE inicial).
- `ca/`: `rules.py` (p_urb), `simulate.py` (evolución T pasos).
- `ga/`: `optimize.py` (DEAP, fitness = IoU vs. target).
- `eval/`: métricas espaciales y validación temporal.
- `viz/`: funciones de plotting (sin seaborn).

## Plantillas y tests
- Cada función pública debe tener pruebas en `tests/`.
- Evitar estados globales; preferir funciones puras y `@dataclass` para parámetros.
- Proveer ejemplos de uso en docstrings.

## Prompts listos para Copilot Chat

### Features/LBP
```
Genera src/tesis_ac/features/lbp.py con una función compute_lbp(image: np.ndarray, P: int=8, R: float=1.0, method='uniform') -> np.ndarray que retorne un mapa LBP por canal y apile canales; incluye docstring y test.
```

### Sobel
```
Crea src/tesis_ac/features/edges.py con compute_sobel_mag_and_dir(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray] usando scikit-image. Añade test con una imagen sintética.
```

### Stack
```
Implementa stack_features(rgb: np.ndarray, lbp: np.ndarray, sobel_mag: np.ndarray, sobel_dir: np.ndarray) -> np.ndarray normalizando features a [0,1]. Testea shapes.
```

### Clustering
```
En src/tesis_ac/clustering/cluster.py escribe cluster_kmeans(X: np.ndarray, n_clusters=3, random_state=42) que devuelva labels y modelo. Incluye fit_predict_image(rgb_path, config) que escribe labels.tif.
```

### Etiquetado heurístico
```
En labeling.py crea assign_semantic_labels(features, labels) que mapee clusters a {0:no-urbano,1:urbano,2:camino} con reglas simples (densidad LBP/gradiente). Escribe test con un caso toy.
```

### Raster→Grid
```
raster_to_grid.py: función to_grid(src_tif, dst_tif, cell_size) usando rasterio; conserva CRS; test con raster pequeño.
```

### Vecindarios
```
neighbors.py: get_neighbors_mask(shape, neighborhood='moore', radius=1) y count_urban_neighbors(grid); tests de borde.
```

### AC
```
rules.py: urbanization_probability(cell_state, n_urban, w_neighbors, w_lbp, w_sobel, threshold); simulate.py: simulate(grid0, features, params, T); tests con grid pequeño, T=3.
```

### WoE
```
woe.py: función para calcular WoE dado grid_t (estado) y bins de features → retorna pesos por bin. Test con dataset sintético.
```

### GA (DEAP)
```
optimize.py: configura toolbox, cromosoma [w_neighbors, w_lbp, w_sobel, threshold, radius, T], fitness=IoU contra grid_target. Incluye run_optimization(config) y guarda mejores parámetros y curva.
```

### Eval/Viz
```
eval/metrics.py: IoU, precisión, recall, kappa; viz/plots.py: función para curvas y mapas; tests básicos.
```

## Estilo de commits y ramas
- Ramas: main (estable), dev, feat/<modulo>, fix/<bug>
- Commits convencionales:
  - feat(features): add LBP uniform extractor
  - test(ca): add Moore neighborhood edge-case tests
  - fix(grid): handle CRS projection edge cases

## Backlog de prioridades
- **P0**: features/ + clustering/ + grid/ + ca/ básico + script run_classify_and_grid
- **P1**: woe/ + ga/ + eval/ + viz/ + script run_optimize  
- **P2**: docs/beamer + docs/article + CI/CD + data pipeline

## Definition of Done (MVP)
```bash
python -m tesis_ac.run_classify_and_grid configs/default.yaml
python -m tesis_ac.run_optimize configs/default.yaml
```
Obtener: processed/grid_init.tif, processed/grid_best.tif, reports/metrics.json, figs/curvas.png