# Tesis AC: Modelado de Crecimiento Urbano con Autómatas Celulares

Pipeline reproducible para modelar crecimiento urbano con Autómatas Celulares (AC) + Algoritmo Genético (AG) y conocimiento inicial por Weight-of-Evidence (WoE), alimentado por clasificación no supervisada de imágenes satelitales/históricas a un grid celular (urbano / no-urbano / vialidad).

## Objetivos

- **Clasificación no supervisada**: Extraer texturas (LBP), bordes (Sobel/gradientes) y bandas RGB para formar un vector por píxel y clusterizar (K-Means/GMM) → mapa discreto (urbano, no-urbano, camino).
- **Pipeline AC**: WoE inicial → AC (reglas con vecinos) → AG para optimizar pesos/umbrales/parámetros espaciales → Validación con históricos.
- **Entregables**: Presentación Beamer, artículo LaTeX, y código reproducible.

## Estructura del Proyecto

```
tesis-ac/
├── README.md
├── pyproject.toml
├── configs/
│   └── default.yaml           # Parámetros del pipeline
├── data/
│   ├── raw/                   # Imágenes originales
│   ├── interim/               # Recortes/normalizaciones
│   └── processed/             # Grids, etiquetas, máscaras
├── src/
│   └── tesis_ac/
│       ├── features/          # LBP, Sobel, stack features
│       ├── clustering/        # KMeans/GMM, etiquetado
│       ├── grid/              # Raster→grid, vecindarios
│       ├── woe/               # Cálculo WoE inicial
│       ├── ca/                # Autómata celular
│       ├── ga/                # Optimización (DEAP)
│       ├── eval/              # Métricas espaciales/temporales
│       └── viz/               # Visualización
├── tests/
├── docs/
│   ├── beamer/
│   └── article/
└── notebooks/                 # Exploración rápida
```

## Instalación

```bash
# Con poetry
poetry install

# O con pip
pip install -r requirements.txt
```

## Uso Rápido

### MVP Pipeline

```bash
# 1. Clasificación y grid inicial
python -m tesis_ac.run_classify_and_grid configs/default.yaml

# 2. Optimización con AG
python -m tesis_ac.run_optimize configs/default.yaml
```

### Outputs esperados
- `processed/grid_init.tif` - Grid inicial clasificado
- `processed/grid_best.tif` - Grid optimizado
- `reports/metrics.json` - Métricas de evaluación
- `figs/curvas.png` - Curvas de convergencia

## Metodología

1. **Preprocesar imágenes** → Normalización y recortes
2. **Features** → LBP + Sobel + estadísticos por píxel
3. **Clustering** → K-Means/GMM para 3 clases
4. **Postproceso** → Etiquetado heurístico simple
5. **Raster→Grid** → Discretización celular
6. **WoE inicial** → Pesos basados en evidencia
7. **AC** → Reglas con vecinos Moore/von Neumann
8. **AG** → Optimización de parámetros espaciales
9. **Validación** → Métricas vs. datos históricos

## Desarrollo

Ver `.copilot/INSTRUCTIONS.md` para convenciones de código y prompts para GitHub Copilot.

## Entregables

- [ ] Presentación Beamer (`docs/beamer/`)
- [ ] Artículo LaTeX (`docs/article/`)
- [ ] Pipeline reproducible (scripts principales)
- [ ] Tests unitarios (`tests/`)

## Estado Actual

- ✅ Estructura base del proyecto
- ⏳ P0: Módulos de features, clustering, grid, CA básico
- ⏳ P1: WoE, GA, evaluación
- ⏳ P2: Entregables LaTeX

## Licencia

Proyecto de tesis - Uso académico