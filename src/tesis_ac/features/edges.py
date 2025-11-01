"""Edge detection using Sobel and gradient operators."""

import logging
from typing import Tuple

import numpy as np
from skimage.filters import sobel, sobel_h, sobel_v
from skimage import filters

logger = logging.getLogger(__name__)


def compute_sobel_mag_and_dir(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Sobel magnitude and direction for an image.
    
    Args:
        image: Input image array with shape (H, W, C) or (H, W).
        
    Returns:
        Tuple of (magnitude, direction) arrays with same spatial dimensions.
        For RGB input (H, W, C) -> (H, W, C), (H, W, C)
        For grayscale (H, W) -> (H, W), (H, W)
        
    Example:
        >>> import numpy as np
        >>> img = np.random.rand(100, 100, 3)
        >>> mag, dir = compute_sobel_mag_and_dir(img)
        >>> print(mag.shape, dir.shape)  # (100, 100, 3) (100, 100, 3)
    """
    if image.ndim == 2:
        # Grayscale image
        sobel_x = sobel_h(image)
        sobel_y = sobel_v(image)
        
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        direction = np.arctan2(sobel_y, sobel_x)
        
        return magnitude.astype(np.float32), direction.astype(np.float32)
    
    elif image.ndim == 3:
        # RGB image - compute Sobel for each channel
        height, width, channels = image.shape
        magnitude_channels = np.zeros((height, width, channels), dtype=np.float32)
        direction_channels = np.zeros((height, width, channels), dtype=np.float32)
        
        for c in range(channels):
            channel = image[:, :, c]
            
            sobel_x = sobel_h(channel)
            sobel_y = sobel_v(channel)
            
            magnitude_channels[:, :, c] = np.sqrt(sobel_x**2 + sobel_y**2)
            direction_channels[:, :, c] = np.arctan2(sobel_y, sobel_x)
        
        return magnitude_channels, direction_channels
    
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using Sobel operator.
    
    Args:
        image: Input image array with shape (H, W, C) or (H, W).
        
    Returns:
        Gradient magnitude with same spatial dimensions as input.
    """
    if image.ndim == 2:
        return sobel(image).astype(np.float32)
    
    elif image.ndim == 3:
        height, width, channels = image.shape
        gradient_channels = np.zeros((height, width, channels), dtype=np.float32)
        
        for c in range(channels):
            gradient_channels[:, :, c] = sobel(image[:, :, c])
        
        return gradient_channels
    
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def compute_edge_density(
    magnitude: np.ndarray, 
    threshold: float = 0.1,
    window_size: int = 5
) -> np.ndarray:
    """Compute local edge density from gradient magnitude.
    
    Args:
        magnitude: Gradient magnitude array.
        threshold: Threshold for edge detection.
        window_size: Size of local window for density computation.
        
    Returns:
        Edge density map.
    """
    # Threshold to get binary edge map
    edges = magnitude > threshold
    
    # Compute local density using convolution
    from scipy.ndimage import uniform_filter
    
    if edges.ndim == 3:
        # For multichannel, compute density for each channel
        density = np.zeros_like(edges, dtype=np.float32)
        for c in range(edges.shape[2]):
            density[:, :, c] = uniform_filter(
                edges[:, :, c].astype(np.float32), 
                size=window_size
            )
    else:
        density = uniform_filter(
            edges.astype(np.float32), 
            size=window_size
        )
    
    return density


def normalize_gradients(magnitude: np.ndarray, direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize gradient magnitude and direction.
    
    Args:
        magnitude: Gradient magnitude array.
        direction: Gradient direction array (in radians).
        
    Returns:
        Tuple of normalized (magnitude, direction).
    """
    # Normalize magnitude to [0, 1]
    mag_min, mag_max = magnitude.min(), magnitude.max()
    if mag_max > mag_min:
        norm_magnitude = (magnitude - mag_min) / (mag_max - mag_min)
    else:
        norm_magnitude = np.zeros_like(magnitude)
    
    # Normalize direction to [0, 1] (from [-π, π])
    norm_direction = (direction + np.pi) / (2 * np.pi)
    
    return norm_magnitude.astype(np.float32), norm_direction.astype(np.float32)


def compute_edge_statistics(magnitude: np.ndarray, direction: np.ndarray) -> dict:
    """Compute statistics of edge features.
    
    Args:
        magnitude: Gradient magnitude array.
        direction: Gradient direction array.
        
    Returns:
        Dictionary with edge statistics.
    """
    stats = {
        'magnitude': {
            'mean': float(np.mean(magnitude)),
            'std': float(np.std(magnitude)),
            'min': float(np.min(magnitude)),
            'max': float(np.max(magnitude)),
        },
        'direction': {
            'mean': float(np.mean(direction)),
            'std': float(np.std(direction)),
            'min': float(np.min(direction)),
            'max': float(np.max(direction)),
        },
        'shape': magnitude.shape,
        'strong_edges': float(np.sum(magnitude > np.mean(magnitude) + np.std(magnitude))),
        'weak_edges': float(np.sum(magnitude > np.mean(magnitude))),
    }
    
    return stats