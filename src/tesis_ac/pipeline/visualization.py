"""
Visualization utilities for satellite image classification results.

This module provides functions to visualize clustering and classification results,
create comparison plots, and generate publication-ready figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


def create_comprehensive_comparison(image: np.ndarray, 
                                   prediction_maps: Dict[str, np.ndarray],
                                   metrics: Dict,
                                   save_path: Optional[str] = None,
                                   year: str = "",
                                   figsize: Tuple[int, int] = (20, 15)) -> None:
    """
    Create comprehensive comparison visualization matching notebook style.
    
    Args:
        image: Original RGB image (H, W, 3)
        prediction_maps: Dictionary with prediction maps
        metrics: Performance metrics
        save_path: Path to save figure
        year: Year identifier for title
        figsize: Figure size
    """
    # Separate K-means and SVM results
    kmeans_maps = {}
    svm_maps = {}
    
    for name, pred_map in prediction_maps.items():
        if 'kmeans' in name:
            n_clusters = name.split('_')[1]
            kmeans_maps[int(n_clusters)] = pred_map
        elif 'svm' in name:
            n_classes = name.split('_')[1]
            svm_maps[int(n_classes)] = pred_map
    
    # Create figure with 3 rows, 4 columns
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    
    # --- ROW 1: IMAGEN REAL + K-MEANS ---
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f'🌍 IMAGEN REAL\n(Google Earth {year})', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # K-means results (2, 3, 4 clusters)
    cluster_counts = sorted(kmeans_maps.keys())
    for i, n_clusters in enumerate(cluster_counts[:3]):
        col = i + 1
        if col < 4:
            pred_map = kmeans_maps[n_clusters]
            im = axes[0, col].imshow(pred_map, cmap='viridis')
            
            # Get inertia if available
            inertia = metrics.get('clustering', {}).get(n_clusters, {}).get('inertia', 0)
            
            axes[0, col].set_title(f'🔍 K-MEANS\n{n_clusters} clusters', 
                                 fontsize=12, fontweight='bold')
            axes[0, col].axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[0, col], shrink=0.8)
            cbar.set_label('Cluster ID', rotation=270, labelpad=15)
    
    # --- ROW 2: IMAGEN REAL + SVM ---
    # Original image (reference)
    axes[1, 0].imshow(image)
    axes[1, 0].set_title('🌍 IMAGEN REAL\n(Referencia)', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # SVM results (2, 3, 4 classes)
    class_counts = sorted(svm_maps.keys())
    for i, n_classes in enumerate(class_counts[:3]):
        col = i + 1
        if col < 4:
            pred_map = svm_maps[n_classes]
            im = axes[1, col].imshow(pred_map, cmap='plasma')
            
            # Get accuracy if available
            accuracy = metrics.get('classification', {}).get(n_classes, {}).get('accuracy', 0)
            
            axes[1, col].set_title(f'🤖 SVM\n{n_classes} clases (Acc: {accuracy:.2f})', 
                                 fontsize=12, fontweight='bold')
            axes[1, col].axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[1, col], shrink=0.8)
            cbar.set_label('Clase SVM', rotation=270, labelpad=15)
    
    # --- ROW 3: ANÁLISIS COMPARATIVO ---
    # Original in grayscale for comparison
    import cv2
    gray_original = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    axes[2, 0].imshow(gray_original, cmap='gray')
    axes[2, 0].set_title('🌍 ORIGINAL\n(Escala de grises)', 
                        fontsize=12, fontweight='bold')
    axes[2, 0].axis('off')
    
    # Best K-means (highest number of clusters)
    if kmeans_maps:
        best_kmeans_n = max(kmeans_maps.keys())
        best_kmeans_map = kmeans_maps[best_kmeans_n]
        
        # Normalize for comparison
        best_kmeans_norm = (best_kmeans_map - best_kmeans_map.min()) / (
            best_kmeans_map.max() - best_kmeans_map.min()) * 255
        
        im = axes[2, 1].imshow(best_kmeans_norm, cmap='gray')
        axes[2, 1].set_title(f'🔍 MEJOR K-MEANS\n{best_kmeans_n} clusters (norm)', 
                           fontsize=12, fontweight='bold')
        axes[2, 1].axis('off')
    
    # Best SVM (highest accuracy)
    if svm_maps:
        # Find best SVM by accuracy
        best_svm_n = max(svm_maps.keys(), 
                        key=lambda x: metrics.get('classification', {}).get(x, {}).get('accuracy', 0))
        best_svm_map = svm_maps[best_svm_n]
        best_accuracy = metrics.get('classification', {}).get(best_svm_n, {}).get('accuracy', 0)
        
        # Normalize for comparison
        best_svm_norm = (best_svm_map - best_svm_map.min()) / (
            best_svm_map.max() - best_svm_map.min()) * 255
        
        im = axes[2, 2].imshow(best_svm_norm, cmap='gray')
        axes[2, 2].set_title(f'🤖 MEJOR SVM\n{best_svm_n} clases (Acc: {best_accuracy:.2f})', 
                           fontsize=12, fontweight='bold')
        axes[2, 2].axis('off')
        
        # Difference between SVM and K-means
        if kmeans_maps:
            diff_image = np.abs(best_svm_norm.astype(float) - best_kmeans_norm.astype(float))
            im = axes[2, 3].imshow(diff_image, cmap='hot')
            axes[2, 3].set_title('🔥 DIFERENCIA\nSVM vs K-means', 
                                fontsize=12, fontweight='bold')
            axes[2, 3].axis('off')
            plt.colorbar(im, ax=axes[2, 3], shrink=0.8)
    
    # Add overall title
    plt.suptitle(f'🔍 COMPARACIÓN: IMAGEN REAL vs TRANSFORMACIONES DE MODELOS ML - {year}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comprehensive comparison saved to: {save_path}")
    
    plt.show()


def visualize_single_image_results(image: np.ndarray, 
                                 prediction_maps: Dict[str, np.ndarray],
                                 metrics: Dict,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (20, 12)) -> None:
    """
    Visualize results for a single image (simple version).
    
    Args:
        image: Original RGB image
        prediction_maps: Dictionary with prediction maps
        metrics: Performance metrics
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    # Separate K-means and SVM results
    kmeans_maps = {k: v for k, v in prediction_maps.items() if 'kmeans' in k}
    svm_maps = {k: v for k, v in prediction_maps.items() if 'svm' in k}
    
    n_kmeans = len(kmeans_maps)
    n_svm = len(svm_maps)
    n_cols = max(4, max(n_kmeans + 1, n_svm + 1))
    
    fig, axes = plt.subplots(2, n_cols, figsize=figsize)
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # K-means results
    col = 1
    for name, pred_map in kmeans_maps.items():
        if col < n_cols:
            n_clusters = name.split('_')[1]
            inertia = metrics.get('clustering', {}).get(int(n_clusters), {}).get('inertia', 0)
            
            im = axes[0, col].imshow(pred_map, cmap='viridis')
            axes[0, col].set_title(f'K-Means {n_clusters} clusters\nInertia: {inertia:.1f}', 
                                 fontsize=10)
            axes[0, col].axis('off')
            plt.colorbar(im, ax=axes[0, col], shrink=0.8)
            col += 1
    
    # Hide unused subplots in first row
    for i in range(col, n_cols):
        axes[0, i].axis('off')
    
    # SVM results
    axes[1, 0].imshow(image)
    axes[1, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    col = 1
    for name, pred_map in svm_maps.items():
        if col < n_cols:
            n_classes = name.split('_')[1]
            accuracy = metrics.get('classification', {}).get(int(n_classes), {}).get('accuracy', 0)
            
            im = axes[1, col].imshow(pred_map, cmap='plasma')
            axes[1, col].set_title(f'SVM {n_classes} classes\nAccuracy: {accuracy:.3f}', 
                                 fontsize=10)
            axes[1, col].axis('off')
            plt.colorbar(im, ax=axes[1, col], shrink=0.8)
            col += 1
    
    # Hide unused subplots in second row
    for i in range(col, n_cols):
        axes[1, i].axis('off')
    
    plt.suptitle('Satellite Image Classification Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to: {save_path}")
    
    plt.show()


def plot_metrics_comparison(metrics: Dict, 
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (15, 6)) -> None:
    """
    Plot comparison of clustering and classification metrics.
    
    Args:
        metrics: Performance metrics dictionary
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Clustering inertia
    clustering_metrics = metrics.get('clustering', {})
    if clustering_metrics:
        n_clusters = list(clustering_metrics.keys())
        inertias = [clustering_metrics[n]['inertia'] for n in n_clusters]
        
        axes[0].plot(n_clusters, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('K-Means Inertia by Number of Clusters')
        axes[0].grid(True, alpha=0.3)
    
    # Classification accuracy
    classification_metrics = metrics.get('classification', {})
    valid_classifications = {k: v for k, v in classification_metrics.items() 
                           if 'accuracy' in v}
    
    if valid_classifications:
        n_classes = list(valid_classifications.keys())
        accuracies = [valid_classifications[n]['accuracy'] for n in n_classes]
        
        axes[1].bar(n_classes, accuracies, color='skyblue', alpha=0.7)
        axes[1].set_xlabel('Number of Classes')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('SVM Accuracy by Number of Classes')
        axes[1].set_ylim([0, 1])
        
        # Add value labels on bars
        for i, (n, acc) in enumerate(zip(n_classes, accuracies)):
            axes[1].text(n, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    # Cluster distribution for best clustering
    if clustering_metrics:
        best_n_clusters = min(clustering_metrics.keys(), 
                            key=lambda x: clustering_metrics[x]['inertia'])
        distribution = clustering_metrics[best_n_clusters]['distribution']
        
        cluster_labels = [f'Cluster {i}' for i in range(len(distribution))]
        colors = sns.color_palette("husl", len(distribution))
        
        wedges, texts, autotexts = axes[2].pie(distribution, labels=cluster_labels, 
                                             autopct='%1.1f%%', colors=colors)
        axes[2].set_title(f'Cluster Distribution\n(Best: {best_n_clusters} clusters)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics plot saved to: {save_path}")
    
    plt.show()


def plot_temporal_comparison(batch_results: Dict[str, Dict],
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (16, 10)) -> None:
    """
    Plot temporal comparison for multiple years of satellite imagery.
    
    Args:
        batch_results: Results from processing multiple images
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    # Extract years and sort
    years = sorted([year for year in batch_results.keys() if 'error' not in batch_results[year]])
    
    if len(years) < 2:
        logger.warning("Need at least 2 years for temporal comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Clustering inertia evolution
    for n_clusters in [2, 3, 4]:
        inertias = []
        valid_years = []
        
        for year in years:
            metrics = batch_results[year].get('metrics', {})
            clustering = metrics.get('clustering', {})
            if n_clusters in clustering:
                inertias.append(clustering[n_clusters]['inertia'])
                valid_years.append(year)
        
        if inertias:
            axes[0, 0].plot(valid_years, inertias, 'o-', 
                          label=f'{n_clusters} clusters', linewidth=2, markersize=6)
    
    axes[0, 0].set_title('K-Means Inertia Evolution')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: SVM accuracy evolution
    for n_classes in [2, 3, 4]:
        accuracies = []
        valid_years = []
        
        for year in years:
            metrics = batch_results[year].get('metrics', {})
            classification = metrics.get('classification', {})
            if n_classes in classification and 'accuracy' in classification[n_classes]:
                accuracies.append(classification[n_classes]['accuracy'])
                valid_years.append(year)
        
        if accuracies:
            axes[0, 1].plot(valid_years, accuracies, 's-', 
                          label=f'{n_classes} classes', linewidth=2, markersize=6)
    
    axes[0, 1].set_title('SVM Accuracy Evolution')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0.8, 1.0])
    
    # Plot 3: Feature extraction metrics
    n_features = [batch_results[year].get('n_features', 0) for year in years]
    pca_variance = [batch_results[year].get('pca_variance_explained', 0) for year in years]
    
    ax3_twin = axes[1, 0].twinx()
    
    line1 = axes[1, 0].bar([f"{year}\n(Features)" for year in years], n_features, 
                          alpha=0.7, color='lightblue', label='# Features')
    line2 = ax3_twin.plot(years, pca_variance, 'ro-', linewidth=2, 
                         markersize=8, label='PCA Variance')
    
    axes[1, 0].set_title('Feature Extraction Metrics')
    axes[1, 0].set_ylabel('Number of Features', color='blue')
    ax3_twin.set_ylabel('PCA Variance Explained', color='red')
    axes[1, 0].tick_params(axis='y', labelcolor='blue')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    
    # Plot 4: Best model performance summary
    best_clustering_inertias = []
    best_svm_accuracies = []
    
    for year in years:
        metrics = batch_results[year].get('metrics', {})
        
        # Best clustering (lowest inertia)
        clustering = metrics.get('clustering', {})
        if clustering:
            best_inertia = min(data['inertia'] for data in clustering.values())
            best_clustering_inertias.append(best_inertia)
        else:
            best_clustering_inertias.append(0)
        
        # Best SVM (highest accuracy)
        classification = metrics.get('classification', {})
        valid_class = {k: v for k, v in classification.items() if 'accuracy' in v}
        if valid_class:
            best_accuracy = max(data['accuracy'] for data in valid_class.values())
            best_svm_accuracies.append(best_accuracy)
        else:
            best_svm_accuracies.append(0)
    
    ax4_twin = axes[1, 1].twinx()
    
    x_pos = np.arange(len(years))
    width = 0.35
    
    bars1 = axes[1, 1].bar(x_pos - width/2, best_clustering_inertias, width, 
                          alpha=0.7, color='green', label='Best K-Means')
    bars2 = ax4_twin.bar(x_pos + width/2, best_svm_accuracies, width, 
                        alpha=0.7, color='orange', label='Best SVM')
    
    axes[1, 1].set_title('Best Model Performance by Year')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Best Inertia (K-Means)', color='green')
    ax4_twin.set_ylabel('Best Accuracy (SVM)', color='orange')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(years)
    axes[1, 1].tick_params(axis='y', labelcolor='green')
    ax4_twin.tick_params(axis='y', labelcolor='orange')
    
    plt.suptitle('Temporal Analysis: Satellite Image Classification', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Temporal comparison saved to: {save_path}")
    
    plt.show()


def create_prediction_map_grid(prediction_maps: Dict[str, np.ndarray],
                             original_image: Optional[np.ndarray] = None,
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (16, 12)) -> None:
    """
    Create a grid visualization of all prediction maps.
    
    Args:
        prediction_maps: Dictionary with prediction maps
        original_image: Original RGB image (optional)
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    n_maps = len(prediction_maps)
    n_total = n_maps + (1 if original_image is not None else 0)
    
    # Calculate grid dimensions
    n_cols = min(4, n_total)
    n_rows = (n_total + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    plot_idx = 0
    
    # Plot original image if provided
    if original_image is not None:
        row, col = plot_idx // n_cols, plot_idx % n_cols
        axes[row, col].imshow(original_image)
        axes[row, col].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
        plot_idx += 1
    
    # Plot prediction maps
    for name, pred_map in prediction_maps.items():
        row, col = plot_idx // n_cols, plot_idx % n_cols
        
        # Choose colormap based on map type
        cmap = 'viridis' if 'kmeans' in name else 'plasma'
        
        im = axes[row, col].imshow(pred_map, cmap=cmap)
        
        # Clean up title
        title = name.replace('_', ' ').title()
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[row, col], shrink=0.8)
        
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle('All Prediction Maps', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction maps grid saved to: {save_path}")
    
    plt.show()


def plot_feature_importance(feature_names: List[str], 
                          feature_importance: np.ndarray,
                          top_n: int = 15,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot feature importance from PCA analysis.
    
    Args:
        feature_names: List of feature names
        feature_importance: Feature importance scores
        top_n: Number of top features to show
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    # Get top features
    top_indices = np.argsort(feature_importance)[-top_n:]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = feature_importance[top_indices]
    
    # Create plot
    plt.figure(figsize=figsize)
    
    y_pos = np.arange(len(top_features))
    colors = sns.color_palette("viridis", len(top_features))
    
    bars = plt.barh(y_pos, top_scores, color=colors)
    
    plt.yticks(y_pos, top_features)
    plt.xlabel('Feature Importance (PCA)')
    plt.title(f'Top {top_n} Most Important Features')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, top_scores):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to: {save_path}")
    
    plt.show()