# Tesis AC: Modelado de Crecimiento Urbano con Autómatas Celulares

Pipeline reproducible para modelar crecimiento urbano con Autómatas Celulares (AC) + Algoritmo Genético (AG) y conocimiento inicial por Weight-of-Evidence (WoE), alimentado por clasificación no supervisada de imágenes satelitales/históricas a un grid celular (urbano / no-urbano / vialidad).

## Objetivos

- **Clasificación no supervisada**: Extraer texturas (LBP), bordes (Sobel/gradientes) y bandas RGB para formar un vector por píxel y clusterizar (K-Means/GMM) → mapa discreto (urbano, no-urbano, camino).
- **Pipeline AC**: WoE inicial → AC (reglas con vecinos) → AG para optimizar pesos/umbrales/parámetros espaciales → Validación con históricos.
- **Entregables**: Presentación Beamer, artículo LaTeX, y código reproducible.

## Estructura del Proyecto

```
CityModelling/
├── README.md                  # Este archivo
├── USAGE.md                   # Guía de uso de scripts principales
├── pyproject.toml             # Configuración del paquete
│
├── configs/
│   └── default.yaml           # Parámetros del pipeline
│
├── data/
│   ├── raw/                   # Imágenes satelitales originales (1984-2020)
│   ├── interim/               # Procesamiento intermedio
│   └── processed/             # Mapas estandarizados, resultados AC
│
├── src/tesis_ac/              # 📦 Framework reutilizable
│   ├── woe/                   # Weight of Evidence
│   ├── ca/                    # Cellular Automata
│   ├── ga/                    # Genetic Algorithm
│   ├── features/              # Extracción de características
│   ├── eval/                  # Métricas de evaluación
│   ├── utils/                 # Utilidades
│   ├── historical/            # Procesamiento de transiciones
│   └── pipeline/              # Pipeline de clasificación de imágenes
│
├── analysis/                  # Resultados de análisis
│   ├── woe_weights.json       # Pesos WoE entrenados
│   ├── historical_transitions_corrected.pkl
│   └── ac_simulation/         # Resultados de simulaciones
│
├── report/                    # 📄 Documentación académica
│   └── tesis/                 # Tesis en LaTeX
│
├── notebooks/                 # Jupyter notebooks de exploración
├── tests/                     # Tests unitarios
│
└── 🎯 Scripts principales (raíz):
    ├── train_woe_model.py     # 1. Entrenamiento WoE
    ├── calibrate_ac_ga.py     # 2. Calibración con AG
    └── run_optimized_ac.py    # 3. Simulación final
```

## Instalación

```bash
# Con poetry
poetry install

# O con pip
pip install -r requirements.txt
```

## Uso Rápido

### Pipeline Completo (3 scripts principales)

```bash
# 1. Entrenar pesos Weight of Evidence
python train_woe_model.py

# 2. Calibrar AC con Algoritmo Genético
python calibrate_ac_ga.py

# 3. Ejecutar simulación optimizada
python run_optimized_ac.py
```

Ver **[USAGE.md](USAGE.md)** para guía detallada de uso.

### Outputs esperados
- `analysis/woe_weights.json` - Pesos WoE entrenados
- Parámetros optimizados del AC (threshold, iterations, etc.)
- Mapas simulados con métricas de validación (IoU, Kappa, FoM)
- Visualizaciones comparativas

## Metodología WoE-AC-AG

**Pipeline de 3 etapas:**

1. **Weight of Evidence (WoE)**
   - Cuantificación empírica de influencia de factores espaciales
   - Cálculo de Information Value por variable
   - Variables: vecindario urbano, edges, distancias, etc.

2. **Autómata Celular (AC)**
   - Simulación espacial con reglas de transición probabilistas
   - Basado en pesos WoE calculados
   - Vecindario Moore, estados binarios (urbano/no-urbano)

3. **Algoritmo Genético (AG)**
   - Optimización automática de parámetros del AC
   - Función objetivo: Figure of Merit (FoM)
   - Parámetros: threshold, iterations, max_growth_rate, etc.

**Validación:**
- Partición temporal: entrenamiento (1984-2014), validación (2015-2020)
- Métricas: IoU, Kappa, FoM, precisión/recall
- Validación espacial y temporal

Ver **tesis completa** en `report/tesis/book/main.pdf` para metodología detallada.

## Estado del Proyecto

- ✅ Framework WoE-AC-AG implementado y funcional
- ✅ Pipeline de clasificación de imágenes satelitales
- ✅ Calibración con Algoritmo Genético
- ✅ Validación temporal con datos históricos (1984-2020)
- ✅ Tesis completa (118 páginas) en LaTeX
- ✅ Tests unitarios básicos
- ✅ Documentación de uso

**Resultados principales:**
- IoU: 0.687 (mejora +14.9% vs AC tradicional)
- Kappa: 0.645 (mejora +21.6% vs baseline)
- Degradación temporal: -3.5% (validación 2015-2020)

## Documentación Adicional

- **[USAGE.md](USAGE.md)** - Guía de uso de scripts principales
- **[FRAMEWORK_EXPLICACION.md](FRAMEWORK_EXPLICACION.md)** - Explicación del framework
- **[MEJORAS_PRECISION.md](MEJORAS_PRECISION.md)** - Mejoras implementadas
- **`report/tesis/book/main.pdf`** - Tesis completa
- **`docs/VALIDACION_AVANZADA.md`** - Comparativa y protocolo de validación avanzada
- **`docs/PIPELINE_DE_IMAGEN_A_VALIDACION.md`** - Pipeline completo (de imágenes a validación, AC+WoE)

## Desarrollo

Para contribuir al proyecto:
1. Instalar dependencias: `pip install -e ".[dev]"`
2. Ejecutar tests: `pytest tests/`
3. Ver convenciones en `.copilot/INSTRUCTIONS.md`

## Licencia

Proyecto de tesis - Uso académico