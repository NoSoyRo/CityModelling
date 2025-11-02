"""
Complete Pipeline for Satellite Image Classification

This module provides the main pipeline class that orchestrates feature extraction,
clustering, and classification for satellite image analysis.
"""

import numpy as np
import os
import json
import pickle
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union

from .feature_extraction import FeatureExtractor, FeaturePreprocessor, load_image
from .clustering import SatelliteImageProcessor

logger = logging.getLogger(__name__)


class SatelliteImagePipeline:
    """
    Complete pipeline for satellite image classification.
    
    This class orchestrates the entire process:
    1. Load image from path
    2. Extract features (RGB, texture, spectral)
    3. Preprocess features (normalize, PCA)
    4. Apply clustering (K-means)
    5. Train classification (SVM)
    6. Generate prediction maps
    7. Save results
    """
    
    def __init__(self, 
                 n_clusters_list: List[int] = [2, 3, 4],
                 n_pca_components: int = 8,
                 extract_full_features: bool = True,
                 svm_kernel: str = 'linear',
                 random_state: int = 42):
        """
        Initialize pipeline.
        
        Args:
            n_clusters_list: List of cluster numbers to try
            n_pca_components: Number of PCA components
            extract_full_features: Whether to extract full feature set
            svm_kernel: SVM kernel type
            random_state: Random state for reproducibility
        """
        self.n_clusters_list = n_clusters_list
        self.n_pca_components = n_pca_components
        self.extract_full_features = extract_full_features
        self.svm_kernel = svm_kernel
        self.random_state = random_state
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(extract_full=extract_full_features)
        self.preprocessor = FeaturePreprocessor(n_components=n_pca_components)
        self.processor = SatelliteImageProcessor(
            n_clusters_list=n_clusters_list,
            svm_kernel=svm_kernel,
            random_state=random_state
        )
        
        # Storage for results
        self.image = None
        self.X_features = None
        self.X_pca = None
        self.feature_names = None
        self.original_shape = None
        self.prediction_maps = None
        self.metrics = None
        
        self.is_fitted = False
        
    def process_image(self, image_path: str, sample_size: Optional[int] = 5000) -> 'SatelliteImagePipeline':
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to the image file
            sample_size: Maximum samples for SVM training
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Processing image: {image_path}")
        
        # 1. Load image
        self.image = load_image(image_path)
        
        # 2. Extract features
        logger.info("Extracting features...")
        self.X_features, self.feature_names, self.original_shape = self.feature_extractor.extract_all_features(self.image)
        
        # 3. Preprocess features
        logger.info("Preprocessing features...")
        self.X_pca = self.preprocessor.fit_transform(self.X_features)
        
        # 4. Fit clustering and classification
        logger.info("Fitting clustering and classification models...")
        self.processor.fit(self.X_pca, sample_size=sample_size)
        
        # 5. Generate predictions
        logger.info("Generating prediction maps...")
        self.prediction_maps = self.processor.transform(self.X_pca, self.original_shape, method='all')
        
        # 6. Get metrics
        self.metrics = self.processor.get_metrics()
        
        self.is_fitted = True
        logger.info("Image processing completed successfully")
        
        return self
    
    def visualize_results(self, save_path: Optional[str] = None, 
                         style: str = 'comprehensive',
                         year: str = "") -> None:
        """
        Create visualization of pipeline results.
        
        Args:
            save_path: Path to save the visualization
            style: 'comprehensive' (notebook-style) or 'simple'
            year: Year identifier for the image
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before visualization")
        
        from .visualization import visualize_single_image_results, create_comprehensive_comparison
        
        if style == 'comprehensive':
            create_comprehensive_comparison(
                image=self.image,
                prediction_maps=self.prediction_maps,
                metrics=self.metrics,
                save_path=save_path,
                year=year
            )
        else:
            visualize_single_image_results(
                image=self.image,
                prediction_maps=self.prediction_maps,
                metrics=self.metrics,
                save_path=save_path
            )
    
    def process_multiple_images(self, image_paths: Dict[str, str], 
                              sample_size: Optional[int] = 5000,
                              visualize: bool = False,
                              visualization_style: str = 'comprehensive') -> Dict[str, Dict]:
        """
        Process multiple images through the pipeline.
        
        Args:
            image_paths: Dictionary mapping names to image paths
            sample_size: Maximum samples for SVM training per image
            visualize: Whether to create visualizations for each image
            visualization_style: 'comprehensive' or 'simple'
            
        Returns:
            Dictionary with results for each image
        """
        logger.info(f"Processing {len(image_paths)} images")
        
        all_results = {}
        
        for name, path in image_paths.items():
            logger.info(f"Processing {name}: {path}")
            
            try:
                # Create new pipeline instance for each image
                pipeline = SatelliteImagePipeline(
                    n_clusters_list=self.n_clusters_list,
                    n_pca_components=self.n_pca_components,
                    extract_full_features=self.extract_full_features,
                    svm_kernel=self.svm_kernel,
                    random_state=self.random_state
                )
                
                # Process image
                pipeline.process_image(path, sample_size=sample_size)
                
                # Create visualization if requested
                if visualize:
                    viz_path = f"comparison_{name}.png"
                    pipeline.visualize_results(
                        save_path=viz_path,
                        style=visualization_style,
                        year=name
                    )
                
                # Store results with prediction maps list for JSON compatibility
                prediction_map_names = list(pipeline.prediction_maps.keys())
                
                all_results[name] = {
                    'image_shape': list(pipeline.original_shape),
                    'n_features': len(pipeline.feature_names),
                    'feature_names': pipeline.feature_names,
                    'prediction_maps': prediction_map_names,  # Just the names for JSON
                    'metrics': pipeline.metrics,
                    'pca_variance_explained': float(pipeline.preprocessor.get_variance_explained())
                }
                
                logger.info(f"Successfully processed {name}")
                
            except Exception as e:
                logger.error(f"Error processing {name}: {str(e)}")
                all_results[name] = {'error': str(e)}
        
        return all_results
    
    def save_results(self, output_dir: str, save_models: bool = True, 
                    save_features: bool = False) -> str:
        """
        Save pipeline results to disk.
        
        Args:
            output_dir: Directory to save results
            save_models: Whether to save trained models
            save_features: Whether to save feature matrices (can be large)
            
        Returns:
            Path to the saved results directory
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving results")
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_output_dir = Path(output_dir) / f"satellite_classification_{timestamp}"
        full_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to: {full_output_dir}")
        
        # Save prediction maps
        maps_dir = full_output_dir / "prediction_maps"
        maps_dir.mkdir(exist_ok=True)
        
        for name, prediction_map in self.prediction_maps.items():
            map_path = maps_dir / f"{name}.npy"
            np.save(map_path, prediction_map)
        
        # Save metrics and metadata
        metadata = {
            'timestamp': timestamp,
            'image_shape': self.original_shape,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'n_pca_components': self.n_pca_components,
            'pca_variance_explained': float(self.preprocessor.get_variance_explained()),
            'n_clusters_list': self.n_clusters_list,
            'svm_kernel': self.svm_kernel,
            'extract_full_features': self.extract_full_features,
            'metrics': self.metrics
        }
        
        with open(full_output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save models if requested
        if save_models:
            models_dir = full_output_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Save preprocessor
            with open(models_dir / "preprocessor.pkl", 'wb') as f:
                pickle.dump(self.preprocessor, f)
            
            # Save clustering models
            clustering_models = {}
            for n_clusters, result in self.processor.clusterer.clustering_results.items():
                clustering_models[n_clusters] = {
                    'model': result['model'],
                    'inertia': result['inertia']
                }
            
            with open(models_dir / "clustering_models.pkl", 'wb') as f:
                pickle.dump(clustering_models, f)
            
            # Save classification models
            classification_models = {}
            for n_clusters, result in self.processor.classifier.classification_results.items():
                if result['model'] is not None:
                    classification_models[n_clusters] = {
                        'model': result['model'],
                        'accuracy': result['accuracy']
                    }
            
            with open(models_dir / "classification_models.pkl", 'wb') as f:
                pickle.dump(classification_models, f)
        
        # Save features if requested
        if save_features:
            features_dir = full_output_dir / "features"
            features_dir.mkdir(exist_ok=True)
            
            np.save(features_dir / "X_features.npy", self.X_features)
            np.save(features_dir / "X_pca.npy", self.X_pca)
        
        logger.info(f"Results saved successfully to: {full_output_dir}")
        return str(full_output_dir)
    
    @classmethod
    def load_results(cls, results_dir: str) -> 'SatelliteImagePipeline':
        """
        Load a previously saved pipeline.
        
        Args:
            results_dir: Directory containing saved results
            
        Returns:
            Loaded pipeline instance
        """
        results_path = Path(results_dir)
        
        # Load metadata
        with open(results_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create pipeline instance
        pipeline = cls(
            n_clusters_list=metadata['n_clusters_list'],
            n_pca_components=metadata['n_pca_components'],
            extract_full_features=metadata['extract_full_features'],
            svm_kernel=metadata['svm_kernel']
        )
        
        # Load basic properties
        pipeline.feature_names = metadata['feature_names']
        pipeline.original_shape = tuple(metadata['image_shape'])
        pipeline.metrics = metadata['metrics']
        
        # Load prediction maps
        maps_dir = results_path / "prediction_maps"
        if maps_dir.exists():
            pipeline.prediction_maps = {}
            for map_file in maps_dir.glob("*.npy"):
                map_name = map_file.stem
                pipeline.prediction_maps[map_name] = np.load(map_file)
        
        # Load models if available
        models_dir = results_path / "models"
        if models_dir.exists():
            try:
                # Load preprocessor
                with open(models_dir / "preprocessor.pkl", 'rb') as f:
                    pipeline.preprocessor = pickle.load(f)
                
                # Load clustering models
                with open(models_dir / "clustering_models.pkl", 'rb') as f:
                    clustering_models = pickle.load(f)
                
                # Load classification models
                with open(models_dir / "classification_models.pkl", 'rb') as f:
                    classification_models = pickle.load(f)
                
                # Restore models in processor
                pipeline.processor.clusterer.clustering_results = {}
                for n_clusters, model_data in clustering_models.items():
                    pipeline.processor.clusterer.clustering_results[n_clusters] = model_data
                
                pipeline.processor.classifier.classification_results = {}
                for n_clusters, model_data in classification_models.items():
                    pipeline.processor.classifier.classification_results[n_clusters] = model_data
                
                pipeline.is_fitted = True
                
            except Exception as e:
                logger.warning(f"Could not load models: {str(e)}")
        
        # Load features if available
        features_dir = results_path / "features"
        if features_dir.exists():
            try:
                pipeline.X_features = np.load(features_dir / "X_features.npy")
                pipeline.X_pca = np.load(features_dir / "X_pca.npy")
            except Exception as e:
                logger.warning(f"Could not load features: {str(e)}")
        
        logger.info(f"Pipeline loaded from: {results_dir}")
        return pipeline
    
    def get_summary(self) -> Dict:
        """
        Get summary of pipeline results.
        
        Returns:
            Dictionary with summary information
        """
        if not self.is_fitted:
            return {"error": "Pipeline not fitted"}
        
        summary = {
            'image_shape': self.original_shape,
            'n_features_extracted': len(self.feature_names),
            'n_pca_components': self.n_pca_components,
            'pca_variance_explained': float(self.preprocessor.get_variance_explained()),
            'n_clusters_tested': len(self.n_clusters_list),
            'prediction_maps_generated': len(self.prediction_maps) if self.prediction_maps else 0
        }
        
        # Add best performing models
        if self.metrics:
            # Best clustering (lowest inertia)
            clustering_metrics = self.metrics.get('clustering', {})
            if clustering_metrics:
                best_clustering = min(clustering_metrics.items(), 
                                    key=lambda x: x[1]['inertia'])
                summary['best_clustering'] = {
                    'n_clusters': best_clustering[0],
                    'inertia': best_clustering[1]['inertia']
                }
            
            # Best classification (highest accuracy)
            classification_metrics = self.metrics.get('classification', {})
            valid_classifications = {k: v for k, v in classification_metrics.items() 
                                   if 'accuracy' in v}
            if valid_classifications:
                best_classification = max(valid_classifications.items(), 
                                        key=lambda x: x[1]['accuracy'])
                summary['best_classification'] = {
                    'n_classes': best_classification[0],
                    'accuracy': best_classification[1]['accuracy']
                }
        
        return summary


def process_image_batch(image_paths: Dict[str, str], 
                       output_dir: str,
                       n_clusters_list: List[int] = [2, 3, 4],
                       n_pca_components: int = 8,
                       sample_size: Optional[int] = 5000) -> str:
    """
    Convenience function to process multiple images in batch.
    
    Args:
        image_paths: Dictionary mapping names to image paths
        output_dir: Directory to save results
        n_clusters_list: List of cluster numbers to try
        n_pca_components: Number of PCA components
        sample_size: Maximum samples for SVM training per image
        
    Returns:
        Path to the saved results directory
    """
    # Create pipeline
    pipeline = SatelliteImagePipeline(
        n_clusters_list=n_clusters_list,
        n_pca_components=n_pca_components
    )
    
    # Process all images
    results = pipeline.process_multiple_images(image_paths, sample_size=sample_size)
    
    # Save batch results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_output_dir = Path(output_dir) / f"batch_processing_{timestamp}"
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save consolidated results
    with open(batch_output_dir / "batch_results.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for name, result in results.items():
            if 'prediction_maps' in result:
                # Save prediction maps as separate files
                maps_dir = batch_output_dir / name / "prediction_maps"
                maps_dir.mkdir(parents=True, exist_ok=True)
                
                for map_name, map_data in result['prediction_maps'].items():
                    np.save(maps_dir / f"{map_name}.npy", map_data)
                
                # Remove from JSON data
                result_copy = result.copy()
                result_copy['prediction_maps'] = list(result['prediction_maps'].keys())
                json_results[name] = result_copy
            else:
                json_results[name] = result
        
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Batch processing completed. Results saved to: {batch_output_dir}")
    return str(batch_output_dir)