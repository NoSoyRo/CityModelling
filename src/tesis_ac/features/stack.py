"""
Combinación y apilamiento de características extraídas.

Este módulo combina diferentes tipos de características (RGB, LBP, Sobel) 
en un vector de características unificado para cada píxel, preparándolo
para clustering y análisis posterior.

Funciones principales:
- stack_features: Combina RGB + LBP + Sobel en un solo array
- normalize_stacked_features: Normalización específica por tipo de feature  
- get_feature_statistics: Estadísticas del vector combinado
- prepare_for_clustering: Prepara datos para algoritmos de clustering
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings


def stack_features(
    rgb_image: np.ndarray,
    lbp_features: np.ndarray,
    sobel_mag: np.ndarray,
    sobel_dir: np.ndarray,
    include_rgb: bool = True,
    include_lbp: bool = True,
    include_sobel: bool = True,
    sobel_combine_method: str = 'magnitude_only'
) -> np.ndarray:
    """
    Combina características RGB, LBP y Sobel en un vector unificado.
    
    Parameters:
    -----------
    rgb_image : np.ndarray
        Imagen RGB original (H, W, 3)
    lbp_features : np.ndarray  
        Características LBP (H, W, C) donde C depende de los parámetros LBP
    sobel_mag : np.ndarray
        Magnitudes Sobel (H, W, 3) para cada canal RGB
    sobel_dir : np.ndarray
        Direcciones Sobel (H, W, 3) para cada canal RGB
    include_rgb : bool, default=True
        Si incluir valores RGB originales
    include_lbp : bool, default=True
        Si incluir características LBP
    include_sobel : bool, default=True
        Si incluir características Sobel
    sobel_combine_method : str, default='magnitude_only'
        Método para combinar Sobel:
        - 'magnitude_only': Solo magnitudes
        - 'direction_only': Solo direcciones  
        - 'both': Magnitudes y direcciones
        - 'magnitude_mean': Promedio de magnitudes RGB
        
    Returns:
    --------
    np.ndarray
        Array combinado con shape (H, W, N) donde N es el número total de features
        
    Notes:
    ------
    Vector típico resultante (ejemplo con P=8, R=1.0):
    - RGB: 3 features (R, G, B)
    - LBP: 3 features (LBP por canal RGB)  
    - Sobel: 3 o 6 features (dependiendo del método)
    Total: 9-12 features por píxel
    """
    # Validaciones básicas
    h, w = rgb_image.shape[:2]
    if lbp_features.shape[:2] != (h, w):
        raise ValueError(f"LBP shape {lbp_features.shape[:2]} doesn't match RGB {(h, w)}")
    if sobel_mag.shape[:2] != (h, w):
        raise ValueError(f"Sobel magnitude shape {sobel_mag.shape[:2]} doesn't match RGB {(h, w)}")
    if sobel_dir.shape[:2] != (h, w):
        raise ValueError(f"Sobel direction shape {sobel_dir.shape[:2]} doesn't match RGB {(h, w)}")
    
    feature_arrays = []
    feature_names = []
    
    # 1. Características RGB
    if include_rgb:
        feature_arrays.append(rgb_image)
        feature_names.extend(['R', 'G', 'B'])
    
    # 2. Características LBP
    if include_lbp:
        feature_arrays.append(lbp_features)
        n_lbp = lbp_features.shape[2]
        feature_names.extend([f'LBP_{i}' for i in range(n_lbp)])
    
    # 3. Características Sobel
    if include_sobel:
        if sobel_combine_method == 'magnitude_only':
            feature_arrays.append(sobel_mag)
            feature_names.extend(['Sobel_mag_R', 'Sobel_mag_G', 'Sobel_mag_B'])
            
        elif sobel_combine_method == 'direction_only':
            feature_arrays.append(sobel_dir)
            feature_names.extend(['Sobel_dir_R', 'Sobel_dir_G', 'Sobel_dir_B'])
            
        elif sobel_combine_method == 'both':
            feature_arrays.extend([sobel_mag, sobel_dir])
            feature_names.extend(['Sobel_mag_R', 'Sobel_mag_G', 'Sobel_mag_B'])
            feature_names.extend(['Sobel_dir_R', 'Sobel_dir_G', 'Sobel_dir_B'])
            
        elif sobel_combine_method == 'magnitude_mean':
            sobel_mean = np.mean(sobel_mag, axis=2, keepdims=True)
            feature_arrays.append(sobel_mean)
            feature_names.extend(['Sobel_mag_mean'])
        
        else:
            raise ValueError(f"Unknown sobel_combine_method: {sobel_combine_method}")
    
    # Combinar todos los arrays
    if not feature_arrays:
        raise ValueError("At least one feature type must be included")
    
    stacked = np.concatenate(feature_arrays, axis=2)
    
    # Agregar metadatos como atributos (para debugging)
    stacked = stacked.astype(np.float32)
    
    return stacked


def normalize_stacked_features(
    stacked_features: np.ndarray,
    method: str = 'standardize',
    feature_groups: Optional[Dict[str, List[int]]] = None
) -> np.ndarray:
    """
    Normaliza características apiladas con métodos específicos por tipo.
    
    Parameters:
    -----------
    stacked_features : np.ndarray
        Características combinadas (H, W, N)
    method : str, default='standardize'
        Método de normalización:
        - 'standardize': Z-score (media=0, std=1)
        - 'minmax': Escala a [0,1]
        - 'robust': Usar mediana y percentiles (robusto a outliers)
        - 'by_group': Normalizar cada grupo de features por separado
    feature_groups : Dict[str, List[int]], optional
        Grupos de features para normalización separada
        Ej: {'rgb': [0,1,2], 'lbp': [3,4,5], 'sobel': [6,7,8]}
        
    Returns:
    --------
    np.ndarray
        Características normalizadas con misma shape
    """
    if stacked_features.size == 0:
        raise ValueError("Empty stacked_features array")
    
    h, w, n_features = stacked_features.shape
    result = stacked_features.copy().astype(np.float32)
    
    # Reshape para operaciones vectorizadas
    features_flat = result.reshape(-1, n_features)  # (H*W, N)
    
    if method == 'standardize':
        # Z-score normalization
        mean_vals = np.mean(features_flat, axis=0)
        std_vals = np.std(features_flat, axis=0)
        std_vals[std_vals == 0] = 1  # Evitar división por cero
        features_flat = (features_flat - mean_vals) / std_vals
        
    elif method == 'minmax':
        # Min-Max scaling to [0,1]
        min_vals = np.min(features_flat, axis=0)
        max_vals = np.max(features_flat, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1  # Evitar división por cero
        features_flat = (features_flat - min_vals) / ranges
        
    elif method == 'robust':
        # Robust scaling usando mediana y IQR
        median_vals = np.median(features_flat, axis=0)
        q75 = np.percentile(features_flat, 75, axis=0)
        q25 = np.percentile(features_flat, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1  # Evitar división por cero
        features_flat = (features_flat - median_vals) / iqr
        
    elif method == 'by_group':
        if feature_groups is None:
            raise ValueError("feature_groups required for 'by_group' method")
        
        # Normalizar cada grupo por separado
        for group_name, indices in feature_groups.items():
            if indices:
                group_features = features_flat[:, indices]
                mean_vals = np.mean(group_features, axis=0)
                std_vals = np.std(group_features, axis=0)
                std_vals[std_vals == 0] = 1
                features_flat[:, indices] = (group_features - mean_vals) / std_vals
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Reshape back to original dimensions
    result = features_flat.reshape(h, w, n_features)
    
    return result


def get_feature_statistics(stacked_features: np.ndarray) -> Dict[str, Union[float, int, np.ndarray]]:
    """
    Computa estadísticas descriptivas de características apiladas.
    
    Parameters:
    -----------
    stacked_features : np.ndarray
        Características combinadas (H, W, N)
        
    Returns:
    --------
    Dict[str, Union[float, int, np.ndarray]]
        Estadísticas por feature y globales
    """
    h, w, n_features = stacked_features.shape
    features_flat = stacked_features.reshape(-1, n_features)
    
    stats = {
        'shape': stacked_features.shape,
        'n_features': n_features,
        'n_pixels': h * w,
        
        # Estadísticas globales
        'global_mean': float(np.mean(stacked_features)),
        'global_std': float(np.std(stacked_features)),
        'global_min': float(np.min(stacked_features)),
        'global_max': float(np.max(stacked_features)),
        
        # Estadísticas por feature
        'feature_means': np.mean(features_flat, axis=0),
        'feature_stds': np.std(features_flat, axis=0),
        'feature_mins': np.min(features_flat, axis=0),
        'feature_maxs': np.max(features_flat, axis=0),
        
        # Métricas de correlación
        'feature_correlations': np.corrcoef(features_flat.T),
        
        # Métricas de diversidad
        'mean_feature_variance': float(np.mean(np.var(features_flat, axis=0))),
        'total_variance': float(np.var(stacked_features))
    }
    
    return stats


def prepare_for_clustering(
    stacked_features: np.ndarray,
    normalize: bool = True,
    remove_outliers: bool = False,
    outlier_percentile: float = 99.0,
    sample_fraction: Optional[float] = None,
    random_seed: int = 42
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Prepara características apiladas para algoritmos de clustering.
    
    Parameters:
    -----------
    stacked_features : np.ndarray
        Características combinadas (H, W, N)
    normalize : bool, default=True
        Si aplicar normalización estándar
    remove_outliers : bool, default=False
        Si remover píxeles outliers extremos
    outlier_percentile : float, default=99.0
        Percentil para definir outliers (solo si remove_outliers=True)
    sample_fraction : float, optional
        Fracción de píxeles a samplear (para datasets grandes)
        Ej: 0.1 = usar solo 10% de píxeles para clustering
    random_seed : int, default=42
        Semilla para sampling reproducible
        
    Returns:
    --------
    Tuple[np.ndarray, Dict[str, any]]
        - Array preparado para clustering (N_samples, N_features)
        - Diccionario con metadatos de la preparación
    """
    h, w, n_features = stacked_features.shape
    total_pixels = h * w
    
    # Reshape to (N_pixels, N_features)
    features_flat = stacked_features.reshape(-1, n_features).astype(np.float32)
    
    metadata = {
        'original_shape': (h, w, n_features),
        'total_pixels': total_pixels,
        'n_features': n_features
    }
    
    # 1. Remover outliers si se solicita
    if remove_outliers:
        # Calcular distancia de cada píxel al centro
        center = np.mean(features_flat, axis=0)
        distances = np.linalg.norm(features_flat - center, axis=1)
        threshold = np.percentile(distances, outlier_percentile)
        
        outlier_mask = distances <= threshold
        features_flat = features_flat[outlier_mask]
        
        metadata['outliers_removed'] = total_pixels - len(features_flat)
        metadata['outlier_threshold'] = float(threshold)
    else:
        metadata['outliers_removed'] = 0
    
    # 2. Normalización
    if normalize:
        mean_vals = np.mean(features_flat, axis=0)
        std_vals = np.std(features_flat, axis=0)
        std_vals[std_vals == 0] = 1  # Evitar división por cero
        features_flat = (features_flat - mean_vals) / std_vals
        
        metadata['normalization'] = {
            'method': 'standardize',
            'means': mean_vals,
            'stds': std_vals
        }
    else:
        metadata['normalization'] = None
    
    # 3. Sampling para datasets grandes
    if sample_fraction is not None:
        np.random.seed(random_seed)
        n_samples = int(len(features_flat) * sample_fraction)
        sample_indices = np.random.choice(len(features_flat), n_samples, replace=False)
        features_flat = features_flat[sample_indices]
        
        metadata['sampling'] = {
            'fraction': sample_fraction,
            'n_samples': n_samples,
            'random_seed': random_seed
        }
    else:
        metadata['sampling'] = None
    
    metadata['final_shape'] = features_flat.shape
    
    return features_flat, metadata


def reconstruct_spatial_features(
    clustered_labels: np.ndarray,
    original_shape: Tuple[int, int],
    sample_indices: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Reconstruye etiquetas de clustering de vuelta a la estructura espacial original.
    
    Parameters:
    -----------
    clustered_labels : np.ndarray
        Etiquetas de clustering (N_samples,)
    original_shape : Tuple[int, int]
        Shape espacial original (H, W)
    sample_indices : np.ndarray, optional
        Índices de sampling si se usó sampling
        
    Returns:
    --------
    np.ndarray
        Etiquetas en estructura espacial (H, W)
    """
    h, w = original_shape
    
    if sample_indices is not None:
        # Reconstruir desde sampling
        full_labels = np.full(h * w, -1, dtype=clustered_labels.dtype)
        full_labels[sample_indices] = clustered_labels
        spatial_labels = full_labels.reshape(h, w)
    else:
        # Reconstrucción directa
        if len(clustered_labels) != h * w:
            raise ValueError(f"Labels length {len(clustered_labels)} doesn't match spatial size {h*w}")
        spatial_labels = clustered_labels.reshape(h, w)
    
    return spatial_labels