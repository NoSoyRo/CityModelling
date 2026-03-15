"""
Métricas de evaluación espacial para modelado urbano.

Este módulo implementa métricas estándar para validación de modelos de
crecimiento urbano, incluyendo precisión espacial y análisis de patrones.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import warnings


def calculate_spatial_metrics(
    predicted_grid: np.ndarray,
    observed_grid: np.ndarray,
    ignore_value: Optional[int] = None
) -> Dict[str, float]:
    """Calcula métricas espaciales entre un grid predicho y uno observado.

    Argumentos:
        predicted_grid: Grid predicho por el modelo.
        observed_grid: Grid observado (ground truth).
        ignore_value: Valor a ignorar en el cálculo (p. ej. zonas restringidas).

    Retorna:
        Diccionario con métricas calculadas. Si no hay datos válidos (por
        ejemplo, todo fue ignorado), retorna un dict vacío.
    """
    if predicted_grid.shape != observed_grid.shape:
        raise ValueError(f"Shapes inconsistentes: {predicted_grid.shape} vs {observed_grid.shape}")
        
    # Aplanar grids para análisis
    pred_flat = predicted_grid.flatten()
    obs_flat = observed_grid.flatten()
    
    # Aplicar máscara si se especifica valor a ignorar
    if ignore_value is not None:
        mask = (obs_flat != ignore_value) & (pred_flat != ignore_value)
        pred_flat = pred_flat[mask]
        obs_flat = obs_flat[mask]
        
    if len(pred_flat) == 0:
        warnings.warn("No hay datos válidos para calcular métricas")
        return {}
        
    metrics = {}
    
    # Métricas básicas de clasificación
    try:
        metrics['accuracy'] = accuracy_score(obs_flat, pred_flat)
        metrics['kappa'] = cohen_kappa_score(obs_flat, pred_flat)
    except Exception as e:
        warnings.warn(f"Error calculando métricas básicas: {e}")
        
    # Métricas para clase urbana (asumiendo que 1 = urbano)
    try:
        if len(np.unique(obs_flat)) > 1:  # Verificar que hay variabilidad
            metrics['precision'] = precision_score(obs_flat, pred_flat, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(obs_flat, pred_flat, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(obs_flat, pred_flat, average='weighted', zero_division=0)
    except Exception as e:
        warnings.warn(f"Error calculando métricas de clasificación: {e}")
        
    # Intersection over Union (IoU)
    try:
        metrics['iou'] = calculate_iou(predicted_grid, observed_grid, ignore_value)
    except Exception as e:
        warnings.warn(f"Error calculando IoU: {e}")
        
    # Figure of Merit (FoM) - específica para cambio de uso de suelo
    try:
        metrics['fom'] = calculate_figure_of_merit(predicted_grid, observed_grid, ignore_value)
    except Exception as e:
        warnings.warn(f"Error calculando FoM: {e}")
        
    # Métricas espaciales avanzadas
    try:
        spatial_metrics = calculate_spatial_patterns(predicted_grid, observed_grid)
        metrics.update(spatial_metrics)
    except Exception as e:
        warnings.warn(f"Error calculando métricas espaciales: {e}")
        
    return metrics


def calculate_iou(
    predicted_grid: np.ndarray,
    observed_grid: np.ndarray,
    ignore_value: Optional[int] = None
) -> float:
    """Calcula Intersection over Union (IoU) para la clase urbana.

    Definición:
        $$IoU = \frac{|A \cap B|}{|A \cup B|}$$

    Argumentos:
        predicted_grid: Grid predicho.
        observed_grid: Grid observado.
        ignore_value: Valor a ignorar (opcional).

    Retorna:
        Valor IoU en $[0,1]$.
    """
    # Crear máscaras binarias para clase urbana (asumiendo 1 = urbano)
    pred_urban = (predicted_grid == 1)
    obs_urban = (observed_grid == 1)
    
    # Aplicar máscara de valores a ignorar si se especifica
    if ignore_value is not None:
        valid_mask = (observed_grid != ignore_value) & (predicted_grid != ignore_value)
        pred_urban = pred_urban & valid_mask
        obs_urban = obs_urban & valid_mask
        
    # Calcular intersección y unión
    intersection = np.sum(pred_urban & obs_urban)
    union = np.sum(pred_urban | obs_urban)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
        
    return float(intersection / union)


def calculate_figure_of_merit(
    predicted_grid: np.ndarray,
    observed_grid: np.ndarray,
    ignore_value: Optional[int] = None,
    initial_grid: Optional[np.ndarray] = None
) -> float:
    """Calcula Figure of Merit (FoM) para evaluación de cambio de uso de suelo.

    Si no se proporciona ``initial_grid``, cae a una aproximación simple usando
    :func:`calculate_iou`.

    Argumentos:
        predicted_grid: Grid predicho.
        observed_grid: Grid observado.
        ignore_value: Valor a ignorar (opcional).
        initial_grid: Grid inicial (opcional) para identificar *cambio* (0→1).

    Retorna:
        Valor FoM en $[0,1]$.
    """
    if initial_grid is None:
        # Si no se proporciona grid inicial, usar análisis simple
        return calculate_iou(predicted_grid, observed_grid, ignore_value)
        
    # Identificar cambios
    obs_change = (observed_grid == 1) & (initial_grid == 0)
    pred_change = (predicted_grid == 1) & (initial_grid == 0)
    
    # Aplicar máscara si se especifica
    if ignore_value is not None:
        valid_mask = (observed_grid != ignore_value) & (predicted_grid != ignore_value)
        obs_change = obs_change & valid_mask
        pred_change = pred_change & valid_mask
        
    # Calcular componentes FoM
    hits = np.sum(obs_change & pred_change)  # A: cambio correcto
    misses = np.sum(obs_change & ~pred_change)  # B: cambio perdido
    false_alarms = np.sum(~obs_change & pred_change)  # C: falsa alarma
    
    # D (wrong hits) es más complejo de calcular, simplificamos
    total_error = hits + misses + false_alarms
    
    if total_error == 0:
        return 1.0
        
    return float(hits / total_error)


def calculate_spatial_patterns(
    predicted_grid: np.ndarray,
    observed_grid: np.ndarray
) -> Dict[str, float]:
    """Calcula métricas simples de patrones espaciales para la clase urbana.

    Incluye conteo de parches (componentes conectados), tamaño promedio y un
    índice de fragmentación básico (parches / área urbana).

    Argumentos:
        predicted_grid: Grid predicho.
        observed_grid: Grid observado.

    Retorna:
        Diccionario con métricas de parches para predicho y observado.
    """
    metrics = {}
    
    try:
        from scipy.ndimage import label
        
        # Análisis de fragmentación para clase urbana
        pred_urban = (predicted_grid == 1)
        obs_urban = (observed_grid == 1)
        
        # Componentes conectados
        pred_labeled, pred_n_components = label(pred_urban)
        obs_labeled, obs_n_components = label(obs_urban)
        
        metrics['predicted_patches'] = float(pred_n_components)
        metrics['observed_patches'] = float(obs_n_components)
        
        # Tamaño promedio de parche
        pred_urban_cells = np.sum(pred_urban)
        obs_urban_cells = np.sum(obs_urban)
        
        if pred_n_components > 0:
            metrics['predicted_avg_patch_size'] = float(pred_urban_cells / pred_n_components)
        else:
            metrics['predicted_avg_patch_size'] = 0.0
            
        if obs_n_components > 0:
            metrics['observed_avg_patch_size'] = float(obs_urban_cells / obs_n_components)
        else:
            metrics['observed_avg_patch_size'] = 0.0
            
        # Índice de fragmentación (número de parches / área urbana)
        if pred_urban_cells > 0:
            metrics['predicted_fragmentation'] = float(pred_n_components / pred_urban_cells)
        else:
            metrics['predicted_fragmentation'] = 0.0
            
        if obs_urban_cells > 0:
            metrics['observed_fragmentation'] = float(obs_n_components / obs_urban_cells)
        else:
            metrics['observed_fragmentation'] = 0.0
            
    except ImportError:
        warnings.warn("scipy no disponible para análisis de patrones espaciales")
    except Exception as e:
        warnings.warn(f"Error en análisis de patrones espaciales: {e}")
        
    return metrics


def calculate_confusion_matrix_metrics(
    predicted_grid: np.ndarray,
    observed_grid: np.ndarray,
    labels: Optional[list] = None
) -> Dict[str, any]:
    """Calcula matriz de confusión y métricas derivadas (por clase si aplica).

    Argumentos:
        predicted_grid: Grid predicho.
        observed_grid: Grid observado.
        labels: Lista opcional de etiquetas/clases para fijar el orden.

    Retorna:
        Diccionario con la matriz de confusión (cruda y normalizada) y, si hay
        más de 2 clases, precisión/recall/F1 por clase.
    """
    pred_flat = predicted_grid.flatten()
    obs_flat = observed_grid.flatten()
    
    # Calcular matriz de confusión
    cm = confusion_matrix(obs_flat, pred_flat, labels=labels)
    
    results = {
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).tolist()
    }
    
    # Métricas por clase si hay más de 2 clases
    if cm.shape[0] > 2:
        try:
            precision_per_class = precision_score(obs_flat, pred_flat, average=None, zero_division=0)
            recall_per_class = recall_score(obs_flat, pred_flat, average=None, zero_division=0)
            f1_per_class = f1_score(obs_flat, pred_flat, average=None, zero_division=0)
            
            results['precision_per_class'] = precision_per_class.tolist()
            results['recall_per_class'] = recall_per_class.tolist()
            results['f1_per_class'] = f1_per_class.tolist()
            
        except Exception as e:
            warnings.warn(f"Error calculando métricas por clase: {e}")
            
    return results


def calculate_temporal_consistency(
    grid_sequence: list,
    reference_sequence: list
) -> Dict[str, float]:
    """Calcula consistencia temporal entre dos secuencias de grids.

    Argumentos:
        grid_sequence: Secuencia de grids predichos.
        reference_sequence: Secuencia de grids de referencia.

    Retorna:
        Diccionario con métricas por paso y estadísticas agregadas (media, std
        y tendencia lineal de IoU/accuracy/kappa).

    Lanza:
        ValueError: Si las secuencias tienen distinta longitud.
    """
    if len(grid_sequence) != len(reference_sequence):
        raise ValueError("Secuencias de diferente longitud")
        
    metrics = {
        'temporal_iou': [],
        'temporal_accuracy': [],
        'temporal_kappa': []
    }
    
    # Calcular métricas para cada paso temporal
    for pred_grid, ref_grid in zip(grid_sequence, reference_sequence):
        step_metrics = calculate_spatial_metrics(pred_grid, ref_grid)
        
        metrics['temporal_iou'].append(step_metrics.get('iou', np.nan))
        metrics['temporal_accuracy'].append(step_metrics.get('accuracy', np.nan))
        metrics['temporal_kappa'].append(step_metrics.get('kappa', np.nan))
        
    # Estadísticas de consistencia
    for metric_name in ['temporal_iou', 'temporal_accuracy', 'temporal_kappa']:
        values = np.array(metrics[metric_name])
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) > 0:
            metrics[f'{metric_name}_mean'] = float(np.mean(valid_values))
            metrics[f'{metric_name}_std'] = float(np.std(valid_values))
            metrics[f'{metric_name}_trend'] = float(np.polyfit(range(len(valid_values)), valid_values, 1)[0])
        else:
            metrics[f'{metric_name}_mean'] = np.nan
            metrics[f'{metric_name}_std'] = np.nan
            metrics[f'{metric_name}_trend'] = np.nan
            
    return metrics


def evaluate_model_performance(
    simulation_results: list,
    reference_grids: list,
    initial_grid: np.ndarray
) -> Dict[str, any]:
    """Realiza una evaluación completa del rendimiento del modelo.

    Argumentos:
        simulation_results: Lista de grids simulados.
        reference_grids: Lista de grids de referencia.
        initial_grid: Grid inicial (actualmente se conserva para análisis de
            cambio en variantes del pipeline).

    Retorna:
        Diccionario con métricas espaciales por paso, consistencia temporal y
        un resumen de desempeño general.
    """
    evaluation = {
        'spatial_metrics': [],
        'temporal_consistency': {},
        'overall_performance': {}
    }
    
    # Métricas espaciales por paso
    for sim_grid, ref_grid in zip(simulation_results, reference_grids):
        spatial_metrics = calculate_spatial_metrics(sim_grid, ref_grid)
        evaluation['spatial_metrics'].append(spatial_metrics)
        
    # Consistencia temporal
    evaluation['temporal_consistency'] = calculate_temporal_consistency(
        simulation_results, reference_grids
    )
    
    # Rendimiento general
    if evaluation['spatial_metrics']:
        final_metrics = evaluation['spatial_metrics'][-1]
        evaluation['overall_performance'] = {
            'final_iou': final_metrics.get('iou', np.nan),
            'final_accuracy': final_metrics.get('accuracy', np.nan),
            'final_kappa': final_metrics.get('kappa', np.nan),
            'avg_temporal_iou': evaluation['temporal_consistency'].get('temporal_iou_mean', np.nan),
            'temporal_stability': 1.0 - evaluation['temporal_consistency'].get('temporal_iou_std', 1.0)
        }
        
    return evaluation