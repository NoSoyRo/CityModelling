"""
Métricas avanzadas para validación de modelos de crecimiento urbano.

Este módulo implementa métricas de validación global basadas en la literatura
de modelado de Autómatas Celulares, específicamente diseñadas para comparar
comportamientos espaciales agregados en lugar de comparación pixel-a-pixel.

Referencias:
-----------
1. Pontius et al. (2008): "Comparing the input, output, and validation maps 
   for several models of land change"
2. White et al. (1997): "The use of constrained cellular automata for 
   high-resolution modelling of urban land-use dynamics"
3. van Vliet et al. (2011): "A review of current calibration and validation 
   practices in land-change modeling"
4. Mas et al. (2014): "Inductive pattern-based land use/cover change models: 
   A comparison of four software packages"
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy import ndimage
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
import warnings


def calculate_multiple_resolution_validation(
    predicted_grid: np.ndarray,
    observed_grid: np.ndarray,
    window_sizes: List[int] = [3, 5, 9, 15, 25]
) -> Dict[str, any]:
    """Calcula validación multi-resolución (Pontius et al.).

    En lugar de comparar pixel-a-pixel, compara patrones a diferentes escalas
    espaciales usando ventanas móviles (fuzzy sets). Esta métrica suele ser
    robusta ante pequeños desplazamientos espaciales.

    Argumentos:
        predicted_grid: Grid predicho (0=no urbano, 1=urbano).
        observed_grid: Grid observado.
        window_sizes: Tamaños de ventana para el análisis multi-escala.

    Retorna:
        Diccionario con métricas por escala y agregadas.
    """
    results = {
        'fuzzy_kappa_by_scale': {},
        'agreement_by_scale': {},
        'overall_fuzzy_kappa': 0.0
    }
    
    for window_size in window_sizes:
        # Aplicar filtro de media móvil (fuzzy set)
        pred_fuzzy = ndimage.uniform_filter(
            predicted_grid.astype(float), 
            size=window_size, 
            mode='constant'
        )
        obs_fuzzy = ndimage.uniform_filter(
            observed_grid.astype(float), 
            size=window_size, 
            mode='constant'
        )
        
        # Calcular agreement a esta escala
        # Agreement = 1 - |diferencia|
        diff = np.abs(pred_fuzzy - obs_fuzzy)
        agreement = 1.0 - diff
        
        # Fuzzy Kappa (Hagen 2003)
        observed_agreement = np.mean(agreement)
        
        # Expected agreement bajo independencia
        pred_mean = np.mean(pred_fuzzy)
        obs_mean = np.mean(obs_fuzzy)
        expected_agreement = 1.0 - np.abs(pred_mean - obs_mean)
        
        # Fuzzy Kappa
        if expected_agreement < 1.0:
            fuzzy_kappa = (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
        else:
            fuzzy_kappa = 1.0
            
        results['fuzzy_kappa_by_scale'][window_size] = float(fuzzy_kappa)
        results['agreement_by_scale'][window_size] = float(observed_agreement)
    
    # Overall fuzzy kappa (promedio pesado)
    if results['fuzzy_kappa_by_scale']:
        results['overall_fuzzy_kappa'] = float(
            np.mean(list(results['fuzzy_kappa_by_scale'].values()))
        )
    
    return results


def calculate_landscape_metrics(
    urban_grid: np.ndarray,
    cell_size: float = 30.0
) -> Dict[str, float]:
    """Calcula métricas de paisaje (landscape metrics) tipo FRAGSTATS.

    Implementación optimizada para grids grandes. Estas métricas caracterizan
    la estructura espacial del patrón urbano sin requerir comparación directa
    con un mapa observado.

    Argumentos:
        urban_grid: Grid binario (0=no urbano, 1=urbano).
        cell_size: Tamaño del pixel en metros (por defecto 30m para Landsat).

    Retorna:
        Diccionario con métricas de paisaje.
    """
    metrics = {}
    
    # Convertir a binario
    urban_binary = (urban_grid > 0).astype(int)
    
    # 1. Porcentaje de paisaje urbano (PLAND)
    total_cells = urban_binary.size
    urban_cells = np.sum(urban_binary)
    metrics['pland'] = float(urban_cells / total_cells * 100)
    
    # 2. Número de parches urbanos (NP) - OPTIMIZADO
    labeled_array, num_patches = ndimage.label(urban_binary)
    metrics['num_patches'] = int(num_patches)
    
    if num_patches == 0:
        # Sin áreas urbanas
        metrics['largest_patch_index'] = 0.0
        metrics['mean_patch_size'] = 0.0
        metrics['patch_density'] = 0.0
        metrics['edge_density'] = 0.0
        metrics['shape_index_mean'] = 0.0
        metrics['aggregation_index'] = 0.0
        return metrics
    
    # 3. Tamaño de parches - OPTIMIZADO con bincount
    patch_sizes_pixels = np.bincount(labeled_array.flatten())[1:]  # Excluir 0 (background)
    patch_sizes_ha = patch_sizes_pixels * (cell_size ** 2) / 10000  # hectáreas
    
    # Largest Patch Index (LPI)
    total_urban_area = np.sum(patch_sizes_ha)
    if total_urban_area > 0:
        metrics['largest_patch_index'] = float(np.max(patch_sizes_ha) / total_urban_area * 100)
    else:
        metrics['largest_patch_index'] = 0.0
    
    # Mean Patch Size (MPS)
    metrics['mean_patch_size'] = float(np.mean(patch_sizes_ha))
    
    # Patch Density (PD) - número de parches por 100 hectáreas
    landscape_area = total_cells * (cell_size ** 2) / 10000  # hectáreas
    metrics['patch_density'] = float(num_patches / landscape_area * 100)
    
    # 4. Edge Density (ED) - OPTIMIZADO
    # Detectar bordes usando convolución
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.int32)
    
    # Células urbanas con al menos un vecino no urbano
    convolved = ndimage.convolve(urban_binary.astype(np.int32), kernel, mode='constant', cval=0)
    edge_cells = np.sum((urban_binary == 1) & (convolved < 5))
    
    edge_length = edge_cells * cell_size  # metros
    metrics['edge_density'] = float(edge_length / landscape_area)  # m/ha
    
    # 5. Shape Index (SHAPE) - SIMPLIFICADO para velocidad
    # En lugar de calcular para cada parche, usamos aproximación agregada
    # Perímetro total / área total como proxy
    total_perimeter = edge_cells
    total_area = urban_cells
    
    if total_area > 0:
        # Shape index agregado normalizado
        shape_idx_agg = (0.25 * total_perimeter * cell_size) / np.sqrt(total_area * cell_size ** 2)
        metrics['shape_index_mean'] = float(shape_idx_agg)
    else:
        metrics['shape_index_mean'] = 0.0
    
    # Eliminar el loop lento que calculaba shape por parche individual
    shape_indices = []  # Mantener para compatibilidad pero no calcular
    # Eliminar el loop lento que calculaba shape por parche individual
    shape_indices = []  # Mantener para compatibilidad pero no calcular
    
    if shape_indices:
        metrics['shape_index_mean'] = float(np.mean(shape_indices))
    else:
        # Ya calculado arriba como agregado
        pass
    
    # 6. Aggregation Index (AI) - YA OPTIMIZADO, mantener
    # Mide qué tan agregadas están las células urbanas
    num_adjacencies = 0
    max_adjacencies = 0
    
    # OPTIMIZACIÓN: usar operaciones vectorizadas en lugar de loops
    # Contar adyacencias horizontales
    h_adjacencies = np.sum((urban_binary[:, :-1] == 1) & (urban_binary[:, 1:] == 1))
    # Contar adyacencias verticales  
    v_adjacencies = np.sum((urban_binary[:-1, :] == 1) & (urban_binary[1:, :] == 1))
    
    num_adjacencies = 2 * (h_adjacencies + v_adjacencies)  # x2 porque cada par se cuenta dos veces
    max_adjacencies = urban_cells * 4  # Cada celda puede tener 4 vecinos
    
    if max_adjacencies > 0:
        metrics['aggregation_index'] = float(num_adjacencies / max_adjacencies * 100)
    else:
        metrics['aggregation_index'] = 0.0
    
    return metrics


def calculate_quantity_vs_allocation_disagreement(
    predicted_grid: np.ndarray,
    observed_grid: np.ndarray
) -> Dict[str, float]:
    """Calcula la descomposición Pontius-Millones del error total.

    Separa el desacuerdo total en:
    - *Quantity disagreement*: diferencia en cantidad total de urbanización.
    - *Allocation disagreement*: error en ubicación espacial.

    Argumentos:
        predicted_grid: Grid predicho (binario).
        observed_grid: Grid observado (binario).

    Retorna:
        Diccionario con la descomposición del error y porcentajes relativos.
    """
    pred_flat = predicted_grid.flatten()
    obs_flat = observed_grid.flatten()
    
    # Contar categorías
    n = len(pred_flat)
    n_obs_urban = np.sum(obs_flat == 1)
    n_pred_urban = np.sum(pred_flat == 1)
    
    # Matriz de confusión
    tp = np.sum((pred_flat == 1) & (obs_flat == 1))  # True positives
    fp = np.sum((pred_flat == 1) & (obs_flat == 0))  # False positives
    fn = np.sum((pred_flat == 0) & (obs_flat == 1))  # False negatives
    tn = np.sum((pred_flat == 0) & (obs_flat == 0))  # True negatives
    
    # Quantity disagreement (Pontius & Millones 2011)
    quantity_disagreement = abs(n_pred_urban - n_obs_urban) / n
    
    # Exchange (swap)
    exchange = 2 * min(fp, fn) / n
    
    # Shift (allocation disagreement después de corregir quantity)
    shift = 2 * max(fp - fn if fp > fn else fn - fp, 0) / n
    
    # Allocation disagreement total
    allocation_disagreement = exchange + shift
    
    # Overall disagreement
    overall_disagreement = quantity_disagreement + allocation_disagreement
    
    # Agreement
    agreement = 1.0 - overall_disagreement
    
    return {
        'quantity_disagreement': float(quantity_disagreement),
        'allocation_disagreement': float(allocation_disagreement),
        'exchange': float(exchange),
        'shift': float(shift),
        'overall_disagreement': float(overall_disagreement),
        'agreement': float(agreement),
        'quantity_error_percent': float(quantity_disagreement / overall_disagreement * 100) if overall_disagreement > 0 else 0.0,
        'allocation_error_percent': float(allocation_disagreement / overall_disagreement * 100) if overall_disagreement > 0 else 0.0
    }


def calculate_growth_pattern_similarity(
    predicted_grid: np.ndarray,
    observed_grid: np.ndarray,
    initial_grid: np.ndarray,
    distance_bands: List[int] = [1, 2, 3, 5, 10]
) -> Dict[str, float]:
    """Calcula similitud de patrones de crecimiento desde el estado inicial.

    Evalúa si el crecimiento predicho replica el observado en términos de:
    - Distribución por distancia al urbano existente.
    - Intensidad de densificación vs. expansión.
    - Similaridad direccional (por cuadrantes).

    Argumentos:
        predicted_grid: Grid predicho final.
        observed_grid: Grid observado final.
        initial_grid: Grid inicial (estado base).
        distance_bands: Bandas de distancia para análisis (en pixeles).

    Retorna:
        Diccionario con métricas de similitud y errores asociados.
    """
    # Identificar nuevas áreas urbanas
    pred_new = (predicted_grid == 1) & (initial_grid == 0)
    obs_new = (observed_grid == 1) & (initial_grid == 0)
    
    # Mapa de distancia euclidiana desde áreas urbanas iniciales
    initial_urban = (initial_grid == 1)
    distance_map = ndimage.distance_transform_edt(~initial_urban)
    
    metrics = {}
    
    # 1. Distribución de nuevo crecimiento por bandas de distancia
    pred_distribution = []
    obs_distribution = []
    
    for i in range(len(distance_bands)):
        if i == 0:
            mask = distance_map <= distance_bands[i]
        else:
            mask = (distance_map > distance_bands[i-1]) & (distance_map <= distance_bands[i])
        
        pred_count = np.sum(pred_new & mask)
        obs_count = np.sum(obs_new & mask)
        
        pred_distribution.append(pred_count)
        obs_distribution.append(obs_count)
    
    # Normalizar distribuciones
    pred_total = np.sum(pred_distribution)
    obs_total = np.sum(obs_distribution)
    
    if pred_total > 0:
        pred_distribution = np.array(pred_distribution) / pred_total
    else:
        pred_distribution = np.zeros(len(distance_bands))
        
    if obs_total > 0:
        obs_distribution = np.array(obs_distribution) / obs_total
    else:
        obs_distribution = np.zeros(len(distance_bands))
    
    # Wasserstein distance (Earth Mover's Distance)
    try:
        emd = wasserstein_distance(
            range(len(distance_bands)), 
            range(len(distance_bands)),
            pred_distribution,
            obs_distribution
        )
        metrics['growth_pattern_emd'] = float(emd)
        metrics['growth_pattern_similarity'] = float(1.0 / (1.0 + emd))
    except:
        metrics['growth_pattern_emd'] = np.nan
        metrics['growth_pattern_similarity'] = np.nan
    
    # 2. Comparar cantidad de crecimiento vs densificación
    # Densificación = crecimiento en bandas cercanas (< 3 pixels)
    # Expansión = crecimiento en bandas lejanas (>= 3 pixels)
    
    pred_densification = np.sum(pred_new & (distance_map <= 3))
    pred_expansion = np.sum(pred_new & (distance_map > 3))
    
    obs_densification = np.sum(obs_new & (distance_map <= 3))
    obs_expansion = np.sum(obs_new & (distance_map > 3))
    
    pred_densification_ratio = pred_densification / (pred_densification + pred_expansion) if (pred_densification + pred_expansion) > 0 else 0
    obs_densification_ratio = obs_densification / (obs_densification + obs_expansion) if (obs_densification + obs_expansion) > 0 else 0
    
    metrics['pred_densification_ratio'] = float(pred_densification_ratio)
    metrics['obs_densification_ratio'] = float(obs_densification_ratio)
    metrics['densification_error'] = float(abs(pred_densification_ratio - obs_densification_ratio))
    
    # 3. Dirección de crecimiento (análisis de cuadrantes) - OPTIMIZADO
    center_y, center_x = np.array(initial_urban.shape) / 2
    
    # Crear máscaras de cuadrantes
    y_coords, x_coords = np.mgrid[0:pred_new.shape[0], 0:pred_new.shape[1]]
    
    mask_NE = (y_coords < center_y) & (x_coords >= center_x)
    mask_SE = (y_coords >= center_y) & (x_coords >= center_x)
    mask_SW = (y_coords >= center_y) & (x_coords < center_x)
    mask_NW = (y_coords < center_y) & (x_coords < center_x)
    
    quadrants_pred = {
        'NE': int(np.sum(pred_new & mask_NE)),
        'SE': int(np.sum(pred_new & mask_SE)),
        'SW': int(np.sum(pred_new & mask_SW)),
        'NW': int(np.sum(pred_new & mask_NW))
    }
    
    quadrants_obs = {
        'NE': int(np.sum(obs_new & mask_NE)),
        'SE': int(np.sum(obs_new & mask_SE)),
        'SW': int(np.sum(obs_new & mask_SW)),
        'NW': int(np.sum(obs_new & mask_NW))
    }
    
    # Normalizar
    pred_total_quadrants = sum(quadrants_pred.values())
    obs_total_quadrants = sum(quadrants_obs.values())
    
    if pred_total_quadrants > 0:
        quadrants_pred = {k: v/pred_total_quadrants for k, v in quadrants_pred.items()}
    if obs_total_quadrants > 0:
        quadrants_obs = {k: v/obs_total_quadrants for k, v in quadrants_obs.items()}
    
    # Similitud direccional (1 - diferencia promedio)
    directional_diff = np.mean([abs(quadrants_pred[k] - quadrants_obs[k]) for k in quadrants_pred.keys()])
    metrics['directional_similarity'] = float(1.0 - directional_diff)
    
    return metrics


def calculate_comprehensive_validation(
    predicted_grid: np.ndarray,
    observed_grid: np.ndarray,
    initial_grid: np.ndarray,
    cell_size: float = 30.0
) -> Dict[str, any]:
    """Ejecuta validación comprehensiva combinando métricas globales.

    Integra prácticas comunes de la literatura para evaluar modelos de AC más
    allá de comparación pixel-a-pixel.

    Argumentos:
        predicted_grid: Grid predicho.
        observed_grid: Grid observado.
        initial_grid: Grid inicial.
        cell_size: Tamaño de celda/pixel (metros).

    Retorna:
        Diccionario con métricas organizadas por categoría y un resumen.
    """
    validation = {
        'multi_resolution': {},
        'quantity_allocation': {},
        'landscape_predicted': {},
        'landscape_observed': {},
        'landscape_comparison': {},
        'growth_patterns': {},
        'summary': {}
    }
    
    print("   [1/5] Calculando Fuzzy Kappa multi-resolución...", flush=True)
    # 1. Validación multi-resolución (robusta a desplazamientos)
    validation['multi_resolution'] = calculate_multiple_resolution_validation(
        predicted_grid, observed_grid
    )
    print("   ✓ Fuzzy Kappa completado", flush=True)
    
    print("   [2/5] Calculando descomposición Quantity-Allocation...", flush=True)
    # 2. Descomposición quantity vs allocation
    validation['quantity_allocation'] = calculate_quantity_vs_allocation_disagreement(
        predicted_grid, observed_grid
    )
    print("   ✓ Quantity-Allocation completado", flush=True)
    
    print("   [3/5] Calculando métricas de paisaje (FRAGSTATS)...", flush=True)
    # 3. Métricas de paisaje para predicho y observado
    validation['landscape_predicted'] = calculate_landscape_metrics(predicted_grid, cell_size)
    validation['landscape_observed'] = calculate_landscape_metrics(observed_grid, cell_size)
    print("   ✓ Landscape metrics completado", flush=True)
    
    print("   [4/5] Comparando métricas de paisaje...", flush=True)
    # 4. Comparación de métricas de paisaje
    landscape_diffs = {}
    for key in validation['landscape_predicted'].keys():
        if key in validation['landscape_observed']:
            pred_val = validation['landscape_predicted'][key]
            obs_val = validation['landscape_observed'][key]
            
            if obs_val != 0:
                relative_error = abs(pred_val - obs_val) / obs_val
            else:
                relative_error = 0.0 if pred_val == 0 else 1.0
                
            landscape_diffs[f'{key}_relative_error'] = float(relative_error)
            landscape_diffs[f'{key}_absolute_diff'] = float(abs(pred_val - obs_val))
    
    validation['landscape_comparison'] = landscape_diffs
    print("   ✓ Comparación completada", flush=True)
    
    print("   [5/5] Analizando patrones de crecimiento...", flush=True)
    # 5. Patrones de crecimiento
    validation['growth_patterns'] = calculate_growth_pattern_similarity(
        predicted_grid, observed_grid, initial_grid
    )
    print("   ✓ Growth patterns completado", flush=True)
    
    # 6. Resumen ejecutivo
    validation['summary'] = {
        'fuzzy_kappa_overall': validation['multi_resolution']['overall_fuzzy_kappa'],
        'quantity_agreement': 1.0 - validation['quantity_allocation']['quantity_disagreement'],
        'allocation_agreement': 1.0 - validation['quantity_allocation']['allocation_disagreement'],
        'landscape_similarity': 1.0 - np.mean(list(landscape_diffs.values())) if landscape_diffs else 0.0,
        'growth_pattern_similarity': validation['growth_patterns'].get('growth_pattern_similarity', 0.0),
        'overall_model_quality': 0.0  # Se calculará como promedio ponderado
    }
    
    # Calcular métrica global de calidad del modelo
    weights = {
        'fuzzy_kappa_overall': 0.3,
        'allocation_agreement': 0.25,
        'landscape_similarity': 0.25,
        'growth_pattern_similarity': 0.2
    }
    
    quality_score = sum(
        validation['summary'][k] * weights[k] 
        for k in weights.keys()
        if not np.isnan(validation['summary'].get(k, np.nan))
    )
    
    validation['summary']['overall_model_quality'] = float(quality_score)
    
    return validation
