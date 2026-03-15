"""Configuración central del proyecto (YAML -> dataclasses).

Este módulo define la estructura tipada de la configuración (paths, parámetros
de extracción de features, clustering, Autómata Celular, GA, evaluación y
visualización) y ofrece helpers para cargarla desde un archivo YAML.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuración para extracción de características."""
    lbp: Dict[str, Any]
    sobel: Dict[str, Any]


@dataclass
class ClusteringConfig:
    """Configuración para clustering."""
    clusters: int
    algorithm: str
    random_state: int


@dataclass
class CAConfig:
    """Configuración para el Autómata Celular (AC)."""
    neighborhood: str
    radius: int
    T: int
    weights: Dict[str, float]
    threshold: float


@dataclass
class GAConfig:
    """Configuración para el Algoritmo Genético (GA)."""
    pop_size: int
    n_generations: int
    crossover_prob: float
    mutation_prob: float
    bounds: Dict[str, tuple]


@dataclass
class EvaluationConfig:
    """Configuración para métricas/evaluación."""
    metrics: list
    validation_split: float


@dataclass
class VisualizationConfig:
    """Configuración para visualización."""
    dpi: int
    figsize: list
    colormap: str


@dataclass
class Config:
    """Configuración principal (paths + sub-configs)."""
    data_raw: Path
    data_processed: Path
    data_reports: Path
    data_figures: Path
    cell_size: float
    dataset: Dict[str, Any]
    features: FeatureConfig
    clustering: ClusteringConfig
    ca: CAConfig
    ga: GAConfig
    evaluation: EvaluationConfig
    visualization: VisualizationConfig


def load_config(path: str) -> Config:
    """Carga la configuración desde un archivo YAML.

    Argumentos:
        path: Ruta al YAML de configuración.

    Retorna:
        Objeto :class:`Config` con paths resueltos y sub-secciones tipadas.

    Lanza:
        FileNotFoundError: Si el archivo no existe.
        yaml.YAMLError: Si el parseo del YAML falla.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        raise
    
    # Get base directory (repo root)
    base_dir = config_path.parent.parent
    
    return Config(
        data_raw=base_dir / cfg["paths"]["raw"],
        data_processed=base_dir / cfg["paths"]["processed"],
        data_reports=base_dir / cfg["paths"]["reports"],
        data_figures=base_dir / cfg["paths"]["figures"],
        cell_size=cfg["grid"]["cell_size"],
        dataset=cfg.get("dataset", {}),
        features=FeatureConfig(**cfg["features"]),
        clustering=ClusteringConfig(**cfg["clustering"]),
        ca=CAConfig(**cfg["ca"]),
        ga=GAConfig(**cfg["ga"]),
        evaluation=EvaluationConfig(**cfg["evaluation"]),
        visualization=VisualizationConfig(**cfg["visualization"]),
    )


def create_directories(config: Config) -> None:
    """Crea directorios necesarios si no existen.

    Argumentos:
        config: Configuración que contiene las rutas a crear.
    """
    directories = [
        config.data_raw,
        config.data_processed,
        config.data_reports,
        config.data_figures,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")