"""
Algoritmos de clustering para clasificación no supervisada de píxeles.

Este módulo implementa clustering K-Means para clasificar píxeles en categorías
como urbano, rural, carreteras, agua, etc. basado en las características
RGB, LBP y Sobel extraídas.

Funciones principales:
- cluster_kmeans: Clustering K-Means con características stacked
- assign_semantic_labels: Asignación de etiquetas semánticas a clusters
- evaluate_clustering: Métricas de calidad del clustering
- visualize_clusters: Visualización de resultados de clustering
"""

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Dict, List, Optional, Tuple, Union
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def cluster_kmeans(
    features: np.ndarray,
    n_clusters: int = 3,
    method: str = 'kmeans',
    random_state: int = 42,
    max_iter: int = 300,
    batch_size: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, object]:
    """
    Aplica clustering K-Means a características de píxeles.
    
    Parameters:
    -----------
    features : np.ndarray
        Características preparadas para clustering (N_samples, N_features)
        Típicamente salida de prepare_for_clustering()
    n_clusters : int, default=3
        Número de clusters (ej: 3 para urbano/rural/carreteras)
    method : str, default='kmeans'
        Algoritmo de clustering:
        - 'kmeans': K-Means estándar
        - 'minibatch': Mini-Batch K-Means (más rápido para datasets grandes)
    random_state : int, default=42
        Semilla para reproducibilidad
    max_iter : int, default=300
        Máximo número de iteraciones
    batch_size : int, optional
        Tamaño de batch para MiniBatch K-Means (auto si None)
    verbose : bool, default=True
        Si imprimir información del progreso
        
    Returns:
    --------
    Tuple[np.ndarray, object]
        - Etiquetas de cluster para cada muestra (N_samples,)
        - Objeto KMeans entrenado
        
    Examples:
    ---------
    >>> features = prepare_for_clustering(stacked_features)[0]
    >>> labels, model = cluster_kmeans(features, n_clusters=3)
    >>> print(f"Found {len(np.unique(labels))} clusters")
    """
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D (N_samples, N_features), got {features.shape}")
    
    n_samples, n_features = features.shape
    
    if verbose:
        print(f"🔍 Clustering {n_samples:,} samples with {n_features} features")
        print(f"🎯 Target clusters: {n_clusters}")
        print(f"⚙️ Method: {method}")
    
    # Configurar algoritmo
    if method == 'kmeans':
        if batch_size is not None:
            warnings.warn("batch_size ignored for standard K-Means")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            n_init=10
        )
        
    elif method == 'minibatch':
        if batch_size is None:
            batch_size = min(1000, n_samples // 10)
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            batch_size=batch_size,
            n_init=10
        )
        
        if verbose:
            print(f"📦 Batch size: {batch_size}")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Entrenar modelo
    if verbose:
        print("⚡ Training clustering model...")
    
    labels = kmeans.fit_predict(features)
    
    if verbose:
        unique_labels = np.unique(labels)
        print(f"✅ Clustering completed!")
        print(f"📊 Found clusters: {unique_labels}")
        print(f"📊 Cluster sizes: {[np.sum(labels == i) for i in unique_labels]}")
        print(f"🎯 Inertia: {kmeans.inertia_:.2f}")
    
    return labels, kmeans


def assign_semantic_labels(
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
    feature_names: List[str] = None,
    urban_threshold: float = 0.5,
    edge_threshold: float = 0.3
) -> Dict[int, str]:
    """
    Asigna etiquetas semánticas a clusters basado en características de centros.
    
    Parameters:
    -----------
    cluster_labels : np.ndarray
        Etiquetas de cluster de cluster_kmeans()
    cluster_centers : np.ndarray
        Centros de clusters (n_clusters, n_features)
    feature_names : List[str], optional
        Nombres de features ['R','G','B','LBP1',...,'Sobel1',...]
    urban_threshold : float, default=0.5
        Umbral para clasificar como urbano (basado en LBP promedio normalizado)
    edge_threshold : float, default=0.3
        Umbral para clasificar como carretera (basado en Sobel promedio)
        
    Returns:
    --------
    Dict[int, str]
        Mapeo de cluster_id -> etiqueta semántica
        
    Notes:
    ------
    Heurísticas de clasificación:
    - Alto LBP + Alto Sobel = "Urbano Denso" 
    - Alto LBP + Bajo Sobel = "Urbano Residencial"
    - Bajo LBP + Alto Sobel = "Carreteras"
    - Bajo LBP + Bajo Sobel = "Rural/Natural"
    - Valores RGB específicos pueden indicar "Agua" o "Vegetación"
    """
    n_clusters, n_features = cluster_centers.shape
    
    # Nombres por defecto si no se proporcionan
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Identificar índices de diferentes tipos de features
    rgb_indices = [i for i, name in enumerate(feature_names) if name.startswith(('R', 'G', 'B'))]
    lbp_indices = [i for i, name in enumerate(feature_names) if 'LBP' in name or 'lbp' in name]
    sobel_indices = [i for i, name in enumerate(feature_names) if 'Sobel' in name or 'sobel' in name]
    
    semantic_labels = {}
    
    for cluster_id in range(n_clusters):
        center = cluster_centers[cluster_id]
        
        # Calcular características promedio por tipo
        rgb_mean = np.mean([center[i] for i in rgb_indices]) if rgb_indices else 0
        lbp_mean = np.mean([center[i] for i in lbp_indices]) if lbp_indices else 0
        sobel_mean = np.mean([center[i] for i in sobel_indices]) if sobel_indices else 0
        
        # Heurísticas de clasificación
        if lbp_mean > urban_threshold and sobel_mean > edge_threshold:
            semantic_labels[cluster_id] = "Urbano Denso"
        elif lbp_mean > urban_threshold and sobel_mean <= edge_threshold:
            semantic_labels[cluster_id] = "Urbano Residencial"
        elif lbp_mean <= urban_threshold and sobel_mean > edge_threshold:
            semantic_labels[cluster_id] = "Carreteras/Infraestructura"
        elif lbp_mean <= urban_threshold and sobel_mean <= edge_threshold:
            # Analizar RGB para sub-clasificar
            if len(rgb_indices) >= 3:
                r_val = center[rgb_indices[0]] if len(rgb_indices) > 0 else 0
                g_val = center[rgb_indices[1]] if len(rgb_indices) > 1 else 0
                b_val = center[rgb_indices[2]] if len(rgb_indices) > 2 else 0
                
                # Heurística básica para vegetación (más verde)
                if g_val > r_val and g_val > b_val:
                    semantic_labels[cluster_id] = "Vegetación/Rural"
                # Heurística básica para agua/sombras (valores bajos)
                elif rgb_mean < -0.5:  # Asumiendo características normalizadas
                    semantic_labels[cluster_id] = "Agua/Sombras"
                else:
                    semantic_labels[cluster_id] = "Rural/Natural"
            else:
                semantic_labels[cluster_id] = "Rural/Natural"
        else:
            semantic_labels[cluster_id] = f"Cluster_{cluster_id}"
    
    return semantic_labels


def evaluate_clustering(
    features: np.ndarray,
    labels: np.ndarray,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evalúa calidad del clustering usando métricas estándar.
    
    Parameters:
    -----------
    features : np.ndarray
        Características usadas para clustering (N_samples, N_features)
    labels : np.ndarray
        Etiquetas de cluster (N_samples,)
    verbose : bool, default=True
        Si imprimir métricas
        
    Returns:
    --------
    Dict[str, float]
        Métricas de evaluación:
        - silhouette_score: [-1, 1], mayor es mejor
        - calinski_harabasz_score: [0, inf], mayor es mejor  
        - davies_bouldin_score: [0, inf], menor es mejor
        - n_clusters: número de clusters únicos
        
    Notes:
    ------
    - Silhouette Score: Mide qué tan similar es un punto a su cluster vs otros
    - Calinski-Harabasz: Ratio de dispersión entre/dentro clusters
    - Davies-Bouldin: Promedio de similitud entre clusters
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        raise ValueError("Need at least 2 clusters for evaluation")
    
    metrics = {'n_clusters': n_clusters}
    
    # Silhouette Score
    try:
        sil_score = silhouette_score(features, labels)
        metrics['silhouette_score'] = sil_score
    except Exception as e:
        warnings.warn(f"Could not compute silhouette score: {e}")
        metrics['silhouette_score'] = np.nan
    
    # Calinski-Harabasz Score
    try:
        ch_score = calinski_harabasz_score(features, labels)
        metrics['calinski_harabasz_score'] = ch_score
    except Exception as e:
        warnings.warn(f"Could not compute Calinski-Harabasz score: {e}")
        metrics['calinski_harabasz_score'] = np.nan
    
    # Davies-Bouldin Score
    try:
        db_score = davies_bouldin_score(features, labels)
        metrics['davies_bouldin_score'] = db_score
    except Exception as e:
        warnings.warn(f"Could not compute Davies-Bouldin score: {e}")
        metrics['davies_bouldin_score'] = np.nan
    
    if verbose:
        print(f"\n📊 CLUSTERING EVALUATION:")
        print(f"🔍 Number of clusters: {n_clusters}")
        print(f"📈 Silhouette Score: {metrics['silhouette_score']:.4f} (higher is better)")
        print(f"📈 Calinski-Harabasz: {metrics['calinski_harabasz_score']:.2f} (higher is better)")
        print(f"📈 Davies-Bouldin: {metrics['davies_bouldin_score']:.4f} (lower is better)")
        
        # Interpretación básica
        sil = metrics['silhouette_score']
        if not np.isnan(sil):
            if sil > 0.7:
                print("✅ Excellent clustering quality!")
            elif sil > 0.5:
                print("✅ Good clustering quality")
            elif sil > 0.25:
                print("⚠️ Fair clustering quality")
            else:
                print("❌ Poor clustering quality")
    
    return metrics


def visualize_clusters(
    spatial_labels: np.ndarray,
    semantic_labels: Dict[int, str] = None,
    original_image: np.ndarray = None,
    figsize: Tuple[int, int] = (15, 10),
    colors: List[str] = None
) -> None:
    """
    Visualiza resultados de clustering en estructura espacial.
    
    Parameters:
    -----------
    spatial_labels : np.ndarray
        Etiquetas de cluster en estructura espacial (H, W)
    semantic_labels : Dict[int, str], optional
        Mapeo de cluster_id -> nombre semántico
    original_image : np.ndarray, optional
        Imagen original para comparación (H, W, 3)
    figsize : Tuple[int, int], default=(15, 10)
        Tamaño de figura
    colors : List[str], optional
        Colores específicos para clusters
    """
    unique_labels = np.unique(spatial_labels)
    n_clusters = len(unique_labels)
    
    # Colores por defecto
    if colors is None:
        default_colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 
                         'orange', 'purple', 'brown', 'pink']
        colors = default_colors[:n_clusters]
    
    # Configurar subplot
    n_plots = 2 if original_image is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    # Plot 1: Imagen original (si disponible)
    if original_image is not None:
        axes[0].imshow(original_image)
        axes[0].set_title('🖼️ Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plot_idx = 1
    else:
        plot_idx = 0
    
    # Plot 2: Clusters
    # Crear colormap personalizado
    cmap = ListedColormap(colors[:n_clusters])
    
    im = axes[plot_idx].imshow(spatial_labels, cmap=cmap, vmin=0, vmax=n_clusters-1)
    axes[plot_idx].set_title('🎯 Clustering Results', fontsize=14, fontweight='bold')
    axes[plot_idx].axis('off')
    
    # Crear leyenda
    legend_elements = []
    for i, label_id in enumerate(unique_labels):
        if semantic_labels and label_id in semantic_labels:
            label_name = semantic_labels[label_id]
        else:
            label_name = f'Cluster {label_id}'
        
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=colors[i], 
                                           label=f'{label_id}: {label_name}'))
    
    axes[plot_idx].legend(handles=legend_elements, loc='center left', 
                         bbox_to_anchor=(1.05, 0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas de clusters
    print(f"\n📊 CLUSTER STATISTICS:")
    total_pixels = spatial_labels.size
    for label_id in unique_labels:
        count = np.sum(spatial_labels == label_id)
        percentage = 100 * count / total_pixels
        cluster_name = semantic_labels.get(label_id, f'Cluster {label_id}') if semantic_labels else f'Cluster {label_id}'
        print(f"🎯 {cluster_name}: {count:,} pixels ({percentage:.1f}%)")


def optimize_n_clusters(
    features: np.ndarray,
    max_clusters: int = 10,
    min_clusters: int = 2,
    method: str = 'silhouette',
    random_state: int = 42
) -> Tuple[int, Dict[int, float]]:
    """
    Encuentra el número óptimo de clusters usando métricas de evaluación.
    
    Parameters:
    -----------
    features : np.ndarray
        Características para clustering (N_samples, N_features)
    max_clusters : int, default=10
        Máximo número de clusters a probar
    min_clusters : int, default=2
        Mínimo número de clusters a probar
    method : str, default='silhouette'
        Métrica para optimización:
        - 'silhouette': Usar Silhouette Score
        - 'calinski_harabasz': Usar Calinski-Harabasz Score
        - 'davies_bouldin': Usar Davies-Bouldin Score (minimizar)
    random_state : int, default=42
        Semilla para reproducibilidad
        
    Returns:
    --------
    Tuple[int, Dict[int, float]]
        - Número óptimo de clusters
        - Diccionario con scores para cada número de clusters
    """
    scores = {}
    
    print(f"🔍 Testing {min_clusters}-{max_clusters} clusters...")
    
    for n in range(min_clusters, max_clusters + 1):
        print(f"Testing {n} clusters...", end=' ')
        
        try:
            labels, _ = cluster_kmeans(features, n_clusters=n, 
                                    random_state=random_state, verbose=False)
            metrics = evaluate_clustering(features, labels, verbose=False)
            
            if method == 'silhouette':
                scores[n] = metrics['silhouette_score']
            elif method == 'calinski_harabasz':
                scores[n] = metrics['calinski_harabasz_score']
            elif method == 'davies_bouldin':
                scores[n] = metrics['davies_bouldin_score']
            
            print(f"Score: {scores[n]:.4f}")
            
        except Exception as e:
            print(f"Failed: {e}")
            scores[n] = np.nan
    
    # Encontrar óptimo
    valid_scores = {k: v for k, v in scores.items() if not np.isnan(v)}
    
    if not valid_scores:
        raise ValueError("No valid scores computed")
    
    if method == 'davies_bouldin':
        # Menor es mejor para Davies-Bouldin
        optimal_n = min(valid_scores.keys(), key=lambda k: valid_scores[k])
    else:
        # Mayor es mejor para Silhouette y Calinski-Harabasz
        optimal_n = max(valid_scores.keys(), key=lambda k: valid_scores[k])
    
    print(f"\n🎯 Optimal number of clusters: {optimal_n}")
    print(f"📈 Best {method} score: {scores[optimal_n]:.4f}")
    
    return optimal_n, scores