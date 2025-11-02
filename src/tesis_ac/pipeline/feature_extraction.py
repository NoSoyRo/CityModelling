"""
Feature Extraction Pipeline for Satellite Image Analysis

This module provides classes and functions for extracting features from satellite imagery
including RGB characteristics, texture features, and spectral indices.
"""

import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage import feature, filters, color
import logging
from typing import Dict, Tuple, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract various features from RGB satellite images.
    
    Features extracted:
    - RGB channel statistics
    - Texture features (LBP, Sobel, gradients)
    - Spectral indices (NDVI approximation, excess green/red)
    - Color space transformations (LAB, HSV)
    """
    
    def __init__(self, extract_full: bool = True):
        """
        Initialize feature extractor.
        
        Args:
            extract_full: If True, extract all features. If False, extract only basic features.
        """
        self.extract_full = extract_full
        self.feature_names = []
        
    def extract_basic_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract basic RGB features.
        
        Args:
            image: RGB image array of shape (H, W, 3)
            
        Returns:
            Dictionary with basic feature arrays
        """
        logger.info("Extracting basic RGB features...")
        
        r_channel = image[:, :, 0].astype(float)
        g_channel = image[:, :, 1].astype(float)
        b_channel = image[:, :, 2].astype(float)
        
        features = {
            'red': r_channel,
            'green': g_channel,
            'blue': b_channel,
            'intensity': np.mean(image, axis=2),
            'brightness': 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
        }
        
        if self.extract_full:
            # Additional statistical features
            features.update({
                'red_std': filters.rank.mean(r_channel.astype(np.uint8), np.ones((5, 5))),
                'green_std': filters.rank.mean(g_channel.astype(np.uint8), np.ones((5, 5))),
                'blue_std': filters.rank.mean(b_channel.astype(np.uint8), np.ones((5, 5)))
            })
        
        logger.info(f"Basic features extracted: {len(features)} variables")
        return features
    
    def extract_texture_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract texture features using LBP, Sobel, and other operators.
        
        Args:
            image: RGB image array of shape (H, W, 3)
            
        Returns:
            Dictionary with texture feature arrays
        """
        logger.info("Extracting texture features...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Sobel gradients
        sobel_h = filters.sobel_h(gray)
        sobel_v = filters.sobel_v(gray)
        gradient_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
        
        features = {
            'sobel_horizontal': sobel_h,
            'sobel_vertical': sobel_v,
            'gradient_magnitude': gradient_magnitude
        }
        
        if self.extract_full:
            # Local Binary Patterns
            radius = 3
            n_points = 8 * radius
            lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Entropy
            entropy = filters.rank.entropy(gray, np.ones((9, 9)))
            
            features.update({
                'lbp': lbp,
                'entropy': entropy,
                'contrast': filters.rank.mean(gray, np.ones((5, 5)))
            })
        
        logger.info(f"Texture features extracted: {len(features)} variables")
        return features
    
    def extract_spectral_indices(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate spectral indices and color space transformations.
        
        Args:
            image: RGB image array of shape (H, W, 3)
            
        Returns:
            Dictionary with spectral feature arrays
        """
        logger.info("Extracting spectral indices...")
        
        r = image[:, :, 0].astype(float)
        g = image[:, :, 1].astype(float)
        b = image[:, :, 2].astype(float)
        
        # Avoid division by zero
        epsilon = 1e-8
        
        # Basic spectral indices
        features = {
            'ndvi_approx': (g - r) / (g + r + epsilon),
            'excess_green': 2 * g - r - b,
            'excess_red': 1.4 * r - g
        }
        
        if self.extract_full:
            # Color space transformations
            lab_image = color.rgb2lab(image)
            hsv_image = color.rgb2hsv(image)
            
            features.update({
                'lab_l': lab_image[:, :, 0],  # Luminance
                'lab_a': lab_image[:, :, 1],  # Green-Red
                'lab_b': lab_image[:, :, 2],  # Blue-Yellow
                'hsv_h': hsv_image[:, :, 0],  # Hue
                'hsv_s': hsv_image[:, :, 1],  # Saturation
                'hsv_v': hsv_image[:, :, 2],  # Value
            })
        
        logger.info(f"Spectral indices extracted: {len(features)} variables")
        return features
    
    def consolidate_features(self, basic_feat: Dict, texture_feat: Dict, 
                           spectral_feat: Dict) -> Tuple[np.ndarray, List[str], Tuple[int, int]]:
        """
        Consolidate all features into a single feature matrix.
        
        Args:
            basic_feat: Basic RGB features
            texture_feat: Texture features
            spectral_feat: Spectral features
            
        Returns:
            Tuple of (feature_matrix, feature_names, original_shape)
        """
        logger.info("Consolidating features...")
        
        h, w = list(basic_feat.values())[0].shape
        
        # Combine all feature dictionaries
        all_features = {**basic_feat, **texture_feat, **spectral_feat}
        
        # Create feature matrix (pixels x features)
        feature_matrix = []
        feature_names = []
        
        for name, data in all_features.items():
            flattened = data.flatten()
            feature_matrix.append(flattened)
            feature_names.append(name)
        
        # Transpose to have (n_pixels, n_features)
        feature_matrix = np.array(feature_matrix).T
        
        self.feature_names = feature_names
        
        logger.info(f"Feature matrix consolidated: {feature_matrix.shape}")
        logger.info(f"Features: {feature_names}")
        
        return feature_matrix, feature_names, (h, w)
    
    def extract_all_features(self, image: np.ndarray) -> Tuple[np.ndarray, List[str], Tuple[int, int]]:
        """
        Extract all features from an image in one call.
        
        Args:
            image: RGB image array of shape (H, W, 3)
            
        Returns:
            Tuple of (feature_matrix, feature_names, original_shape)
        """
        logger.info(f"Starting feature extraction for image: {image.shape}")
        
        # Extract all feature types
        basic_features = self.extract_basic_features(image)
        texture_features = self.extract_texture_features(image)
        spectral_features = self.extract_spectral_indices(image)
        
        # Consolidate into matrix
        feature_matrix, feature_names, original_shape = self.consolidate_features(
            basic_features, texture_features, spectral_features
        )
        
        logger.info(f"Feature extraction completed: {len(feature_names)} features")
        return feature_matrix, feature_names, original_shape


class FeaturePreprocessor:
    """
    Preprocess features using normalization and dimensionality reduction.
    """
    
    def __init__(self, n_components: int = 8):
        """
        Initialize preprocessor.
        
        Args:
            n_components: Number of PCA components to keep
        """
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit preprocessor and transform features.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Transformed feature matrix
        """
        logger.info(f"Fitting preprocessor on data: {X.shape}")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        X_pca = self.pca.fit_transform(X_scaled)
        
        self.is_fitted = True
        
        logger.info(f"Preprocessing completed: {X_pca.shape}")
        logger.info(f"Variance explained: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return X_pca
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted preprocessor.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        return X_pca
    
    def get_variance_explained(self) -> float:
        """Get total variance explained by PCA components."""
        if not self.is_fitted:
            return 0.0
        return self.pca.explained_variance_ratio_.sum()
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from PCA components."""
        if not self.is_fitted:
            return np.array([])
        return np.abs(self.pca.components_).mean(axis=0)


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from path and convert to RGB.
    
    Args:
        image_path: Path to image file
        
    Returns:
        RGB image array
    """
    logger.info(f"Loading image: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    logger.info(f"Image loaded: {img_rgb.shape}")
    return img_rgb


def extract_features_from_path(image_path: str, extract_full: bool = True, 
                             n_pca_components: int = 8) -> Tuple[np.ndarray, np.ndarray, List[str], Tuple[int, int]]:
    """
    Complete feature extraction pipeline from image path.
    
    Args:
        image_path: Path to image file
        extract_full: Whether to extract full feature set
        n_pca_components: Number of PCA components
        
    Returns:
        Tuple of (X_features, X_pca, feature_names, original_shape)
    """
    # Load image
    image = load_image(image_path)
    
    # Extract features
    extractor = FeatureExtractor(extract_full=extract_full)
    X_features, feature_names, original_shape = extractor.extract_all_features(image)
    
    # Preprocess features
    preprocessor = FeaturePreprocessor(n_components=n_pca_components)
    X_pca = preprocessor.fit_transform(X_features)
    
    logger.info("Feature extraction pipeline completed")
    
    return X_features, X_pca, feature_names, original_shape