"""Static description of the pipeline stages displayed in the sidebar.

The descriptors documented here mirror the catalogue in `docs/pipeline.md`
and are the single source of truth for the UI navigation tree.
"""

from __future__ import annotations

from .models import StageDescriptor, StageId


STAGES: list[StageDescriptor] = [
    StageDescriptor(
        id=StageId.E0,
        title="Imágenes crudas",
        one_liner="Capturas RGB anuales de Google Earth Engine, 1984–2020.",
        inputs=["(externo)"],
        outputs=["data/raw/imagen_YYYY.png"],
        code_module="—",
    ),
    StageDescriptor(
        id=StageId.E1,
        title="Clasificación binaria",
        one_liner=(
            "23 features por píxel → StandardScaler+PCA(8) → K-Means(k=2) → "
            "SVM lineal. Cada PNG se vuelve un mapa binario."
        ),
        inputs=["data/raw/imagen_YYYY.png"],
        outputs=[
            "data/processed/batch_processing_*/year_YYYY/prediction_maps/svm_2_classes.npy"
        ],
        code_module="tesis_ac.pipeline.main_pipeline.SatelliteImagePipeline",
        parameters=[
            "n_clusters_list=[2]",
            "n_pca_components=8",
            "svm_kernel='linear'",
            "sample_size=10000",
            "random_state=42",
        ],
    ),
    StageDescriptor(
        id=StageId.E2,
        title="Estandarización de labels",
        one_liner=(
            "Heurísticas NDVI y centro 30 % aseguran convención global "
            "urbano=1, no-urbano=0."
        ),
        inputs=["svm_2_classes.npy"],
        outputs=["data/processed/standardized_maps/YYYY.npy"],
        code_module="tesis_ac.historical.standardize_labels",
        parameters=["method='center'", "center_ratio=0.3"],
    ),
    StageDescriptor(
        id=StageId.E3,
        title="WoE pooled (1984–2010)",
        one_liner=(
            "7 variables espaciales por año, transiciones acumuladas, "
            "Information Value por bin. El modelo entrenado se serializa."
        ),
        inputs=[
            "data/processed/standardized_maps/{1984..2010}.npy",
        ],
        outputs=["data/processed/woe_pooled_1984_2010.pkl"],
        code_module="train_woe_pooled.py + tesis_ac.woe.woe.WoECalculator",
        parameters=[
            "binning_method='quantile'",
            "max_bins=10",
            "min_bin_size=50",
        ],
    ),
    StageDescriptor(
        id=StageId.E4,
        title="Simulación AC quinquenal",
        one_liner=(
            "5 pasos anuales: recalcular variables → WoE transform → "
            "ponderar por IV → sigmoide + vecindad Moore → umbral 0.75."
        ),
        inputs=[
            "data/processed/woe_pooled_1984_2010.pkl",
            "data/processed/standardized_maps/{2011..2020}.npy",
        ],
        outputs=[
            "data/processed/validation_quinquenal_*/yearly_predictions/*.npy"
        ],
        code_module="validate_quinquenal_*.py + tesis_ac.ca.rules",
        parameters=[
            "threshold=0.75",
            "neighbor_weight (calibrado)",
            "distance_weight (calibrado)",
        ],
    ),
    StageDescriptor(
        id=StageId.E5,
        title="Métricas y validación",
        one_liner=(
            "Comparación predicción vs observado: FoM, Kappa, IoU, "
            "descomposición cantidad/asignación."
        ),
        inputs=["predicted_YYYY.npy", "standardized_maps/YYYY.npy"],
        outputs=[
            "data/processed/validation_quinquenal_*/validation_results.json"
        ],
        code_module="validate_quinquenal_*.py (sección métricas)",
    ),
]


STAGES_BY_ID: dict[StageId, StageDescriptor] = {s.id: s for s in STAGES}
