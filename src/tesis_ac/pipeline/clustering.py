"""
Clustering and Classification Pipeline for Satellite Image Analysis

This module provides classes for K-means clustering and SVM classification
of satellite image features.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)


class ImageClusterer:
    """
    K-means clustering for satellite image segmentation.
    """
    
    def __init__(self, n_clusters_list: List[int] = [2, 3, 4], random_state: int = 42):
        """
        Initialize clusterer.
        
        Args:
            n_clusters_list: List of cluster numbers to try
            random_state: Random state for reproducibility
        """
        self.n_clusters_list = n_clusters_list
        self.random_state = random_state
        self.clustering_results = {}
        
    def fit_predict_all(self, X: np.ndarray) -> Dict[int, Dict]:
        """
        Apply K-means clustering with different numbers of clusters.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Dictionary with clustering results for each n_clusters
        """
        logger.info(f"Starting clustering with {self.n_clusters_list} clusters")
        
        self.clustering_results = {}
        
        for n_clusters in self.n_clusters_list:
            logger.info(f"Clustering with {n_clusters} clusters...")
            
            # Apply K-means
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            labels = kmeans.fit_predict(X)
            
            # Calculate metrics
            inertia = kmeans.inertia_
            
            # Get cluster distribution
            distribution = np.bincount(labels)
            
            self.clustering_results[n_clusters] = {
                'model': kmeans,
                'labels': labels,
                'inertia': inertia,
                'centers': kmeans.cluster_centers_,
                'distribution': distribution
            }
            
            logger.info(f"Clustering {n_clusters}: inertia={inertia:.2f}, distribution={distribution}")
        
        return self.clustering_results
    
    def get_best_clustering(self, criterion: str = 'inertia') -> Tuple[int, Dict]:
        """
        Get best clustering based on specified criterion.
        
        Args:
            criterion: 'inertia' (lower is better) or 'balanced' (more balanced distribution)
            
        Returns:
            Tuple of (best_n_clusters, best_result)
        """
        if not self.clustering_results:
            raise ValueError("No clustering results available. Run fit_predict_all first.")
        
        if criterion == 'inertia':
            # Lower inertia is better
            best_n_clusters = min(self.clustering_results.keys(), 
                                key=lambda x: self.clustering_results[x]['inertia'])
        elif criterion == 'balanced':
            # More balanced distribution is better
            def balance_score(n_clusters):
                distribution = self.clustering_results[n_clusters]['distribution']
                # Calculate coefficient of variation (lower is more balanced)
                return np.std(distribution) / np.mean(distribution)
            
            best_n_clusters = min(self.clustering_results.keys(), key=balance_score)
        else:
            raise ValueError("Criterion must be 'inertia' or 'balanced'")
        
        return best_n_clusters, self.clustering_results[best_n_clusters]
    
    def create_binary_masks(self, original_shape: Tuple[int, int], 
                          clean_masks: bool = True) -> Dict[str, np.ndarray]:
        """
        Create binary masks from clustering results.
        
        Args:
            original_shape: Original image shape (H, W)
            clean_masks: Whether to apply morphological cleaning
            
        Returns:
            Dictionary with binary masks for each cluster
        """
        logger.info("Creating binary masks from clustering results")
        
        binary_masks = {}
        
        for n_clusters, result in self.clustering_results.items():
            labels = result['labels']
            
            # Create masks for each cluster
            for cluster_id in range(n_clusters):
                mask = (labels == cluster_id)
                mask_image = mask.reshape(original_shape).astype(np.uint8)
                
                if clean_masks:
                    # Apply morphological operations to clean the mask
                    kernel = np.ones((3, 3), np.uint8)
                    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel)
                    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel)
                
                mask_name = f'cluster_{n_clusters}_mask_{cluster_id}'
                binary_masks[mask_name] = mask_image
        
        logger.info(f"Created {len(binary_masks)} binary masks")
        return binary_masks


class ImageClassifier:
    """
    SVM classifier for satellite image classification using clustering labels.
    """
    
    def __init__(self, kernel: str = 'linear', C: float = 1.0, random_state: int = 42):
        """
        Initialize classifier.
        
        Args:
            kernel: SVM kernel type
            C: Regularization parameter
            random_state: Random state for reproducibility
        """
        self.kernel = kernel
        self.C = C
        self.random_state = random_state
        self.classification_results = {}
        
    def train_on_clustering_results(self, X: np.ndarray, clustering_results: Dict[int, Dict],
                                  test_size: float = 0.2, sample_size: Optional[int] = 5000) -> Dict[int, Dict]:
        """
        Train SVM classifiers using clustering results as ground truth.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            clustering_results: Results from ImageClusterer
            test_size: Fraction of data to use for testing
            sample_size: Maximum number of samples to use (for speed)
            
        Returns:
            Dictionary with classification results for each n_clusters
        """
        logger.info("Training SVM classifiers on clustering results")
        
        self.classification_results = {}
        
        for n_clusters, cluster_result in clustering_results.items():
            logger.info(f"Training SVM for {n_clusters} clusters...")
            
            labels = cluster_result['labels']
            
            try:
                # Sample data if needed for speed
                if sample_size and len(X) > sample_size:
                    sample_idx = np.random.choice(len(X), sample_size, replace=False)
                    X_sample = X[sample_idx]
                    y_sample = labels[sample_idx]
                else:
                    X_sample = X
                    y_sample = labels
                
                # Split into train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X_sample, y_sample, test_size=test_size, 
                    random_state=self.random_state, stratify=y_sample
                )
                
                # Train SVM
                svm_model = SVC(
                    kernel=self.kernel,
                    C=self.C,
                    random_state=self.random_state
                )
                svm_model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = svm_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                self.classification_results[n_clusters] = {
                    'model': svm_model,
                    'accuracy': accuracy,
                    'report': report,
                    'n_train_samples': len(X_train),
                    'n_test_samples': len(X_test)
                }
                
                logger.info(f"SVM {n_clusters} clusters: accuracy={accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"Error training SVM for {n_clusters} clusters: {str(e)}")
                self.classification_results[n_clusters] = {
                    'model': None,
                    'accuracy': 0.0,
                    'error': str(e)
                }
        
        return self.classification_results
    
    def get_best_classifier(self) -> Tuple[int, Dict]:
        """
        Get best classifier based on accuracy.
        
        Returns:
            Tuple of (best_n_clusters, best_result)
        """
        if not self.classification_results:
            raise ValueError("No classification results available. Train classifiers first.")
        
        # Filter out failed results
        valid_results = {k: v for k, v in self.classification_results.items() 
                        if v['model'] is not None}
        
        if not valid_results:
            raise ValueError("No valid classification results available.")
        
        best_n_clusters = max(valid_results.keys(), 
                            key=lambda x: valid_results[x]['accuracy'])
        
        return best_n_clusters, valid_results[best_n_clusters]
    
    def predict_image(self, X: np.ndarray, n_clusters: int, 
                     original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Predict labels for entire image using trained classifier.
        
        Args:
            X: Feature matrix for entire image
            n_clusters: Number of clusters to use
            original_shape: Original image shape (H, W)
            
        Returns:
            Predicted labels reshaped to image dimensions
        """
        if n_clusters not in self.classification_results:
            raise ValueError(f"No classifier available for {n_clusters} clusters")
        
        model = self.classification_results[n_clusters]['model']
        if model is None:
            raise ValueError(f"Classifier for {n_clusters} clusters failed during training")
        
        predictions = model.predict(X)
        return predictions.reshape(original_shape)


class SatelliteImageProcessor:
    """
    Complete pipeline for satellite image processing: clustering + classification.
    """
    
    def __init__(self, n_clusters_list: List[int] = [2, 3, 4], 
                 svm_kernel: str = 'linear', random_state: int = 42):
        """
        Initialize complete processor.
        
        Args:
            n_clusters_list: List of cluster numbers to try
            svm_kernel: SVM kernel type
            random_state: Random state for reproducibility
        """
        self.n_clusters_list = n_clusters_list
        self.random_state = random_state
        
        self.clusterer = ImageClusterer(
            n_clusters_list=n_clusters_list,
            random_state=random_state
        )
        
        self.classifier = ImageClassifier(
            kernel=svm_kernel,
            random_state=random_state
        )
        
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, sample_size: Optional[int] = 5000) -> 'SatelliteImageProcessor':
        """
        Fit clustering and classification models.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            sample_size: Maximum samples for SVM training
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting satellite image processor...")
        
        # Perform clustering
        clustering_results = self.clusterer.fit_predict_all(X)
        
        # Train classifiers
        classification_results = self.classifier.train_on_clustering_results(
            X, clustering_results, sample_size=sample_size
        )
        
        self.is_fitted = True
        logger.info("Satellite image processor fitted successfully")
        
        return self
    
    def transform(self, X: np.ndarray, original_shape: Tuple[int, int], 
                 method: str = 'best') -> Dict[str, np.ndarray]:
        """
        Transform features to cluster/class predictions.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            original_shape: Original image shape (H, W)
            method: 'best', 'all', or specific number of clusters
            
        Returns:
            Dictionary with prediction maps
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transform")
        
        results = {}
        
        if method == 'best':
            # Use best classifier
            best_n_clusters, _ = self.classifier.get_best_classifier()
            prediction_map = self.classifier.predict_image(X, best_n_clusters, original_shape)
            results[f'svm_{best_n_clusters}_classes'] = prediction_map
            
            # Also include best clustering
            best_kmeans_n, best_kmeans_result = self.clusterer.get_best_clustering()
            kmeans_map = best_kmeans_result['labels'].reshape(original_shape)
            results[f'kmeans_{best_kmeans_n}_clusters'] = kmeans_map
            
        elif method == 'all':
            # Generate all predictions
            for n_clusters in self.n_clusters_list:
                # K-means
                if n_clusters in self.clusterer.clustering_results:
                    kmeans_map = self.clusterer.clustering_results[n_clusters]['labels'].reshape(original_shape)
                    results[f'kmeans_{n_clusters}_clusters'] = kmeans_map
                
                # SVM
                if (n_clusters in self.classifier.classification_results and 
                    self.classifier.classification_results[n_clusters]['model'] is not None):
                    svm_map = self.classifier.predict_image(X, n_clusters, original_shape)
                    results[f'svm_{n_clusters}_classes'] = svm_map
                    
        elif isinstance(method, int):
            # Specific number of clusters
            n_clusters = method
            if n_clusters in self.clusterer.clustering_results:
                kmeans_map = self.clusterer.clustering_results[n_clusters]['labels'].reshape(original_shape)
                results[f'kmeans_{n_clusters}_clusters'] = kmeans_map
            
            if (n_clusters in self.classifier.classification_results and 
                self.classifier.classification_results[n_clusters]['model'] is not None):
                svm_map = self.classifier.predict_image(X, n_clusters, original_shape)
                results[f'svm_{n_clusters}_classes'] = svm_map
        
        return results
    
    def get_metrics(self) -> Dict:
        """
        Get performance metrics for all models.
        
        Returns:
            Dictionary with metrics for clustering and classification
        """
        metrics = {
            'clustering': {},
            'classification': {}
        }
        
        # Clustering metrics
        for n_clusters, result in self.clusterer.clustering_results.items():
            metrics['clustering'][n_clusters] = {
                'inertia': float(result['inertia']),
                'distribution': result['distribution'].tolist()
            }
        
        # Classification metrics
        for n_clusters, result in self.classifier.classification_results.items():
            if result['model'] is not None:
                metrics['classification'][n_clusters] = {
                    'accuracy': float(result['accuracy']),
                    'n_train_samples': result['n_train_samples'],
                    'n_test_samples': result['n_test_samples']
                }
            else:
                metrics['classification'][n_clusters] = {
                    'error': result.get('error', 'Unknown error')
                }
        
        return metrics