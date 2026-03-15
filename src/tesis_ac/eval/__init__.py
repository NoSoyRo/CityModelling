"""Utilidades de evaluación y métricas para modelado de crecimiento urbano.

Este subpaquete expone métricas de validación para modelos (p. ej. Autómatas
Celulares), incluyendo aproximaciones tradicionales pixel-a-pixel y métricas
avanzadas basadas en comportamiento comparables con la literatura.
"""

from .metrics import (
    calculate_spatial_metrics,
    calculate_iou,
    calculate_figure_of_merit,
    calculate_spatial_patterns,
    calculate_confusion_matrix_metrics,
    calculate_temporal_consistency,
    evaluate_model_performance
)

from .advanced_metrics import (
    calculate_multiple_resolution_validation,
    calculate_landscape_metrics,
    calculate_quantity_vs_allocation_disagreement,
    calculate_growth_pattern_similarity,
    calculate_comprehensive_validation
)

__all__ = [
    # Traditional metrics
    'calculate_spatial_metrics',
    'calculate_iou',
    'calculate_figure_of_merit',
    'calculate_spatial_patterns',
    'calculate_confusion_matrix_metrics',
    'calculate_temporal_consistency',
    'evaluate_model_performance',
    
    # Advanced metrics (literature-comparable)
    'calculate_multiple_resolution_validation',
    'calculate_landscape_metrics',
    'calculate_quantity_vs_allocation_disagreement',
    'calculate_growth_pattern_similarity',
    'calculate_comprehensive_validation'
]