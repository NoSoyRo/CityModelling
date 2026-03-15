"""
Módulo para análisis de transiciones históricas urbanas.

Este módulo extrae y analiza las transiciones urbanas reales del histórico
de mapas clasificados para calibrar el modelo de Autómata Celular.
"""

from .extract_transitions import (
    HistoricalTransitionExtractor,
    extract_transitions_from_directory,
    load_historical_maps
)

from .spatial_variables import (
    SpatialVariableCalculator,
    calculate_neighborhood_density,
    calculate_distance_to_urban,
    calculate_urban_edge_distance
)

__all__ = [
    'HistoricalTransitionExtractor',
    'extract_transitions_from_directory',
    'load_historical_maps',
    'SpatialVariableCalculator',
    'calculate_neighborhood_density',
    'calculate_distance_to_urban',
    'calculate_urban_edge_distance',
]
