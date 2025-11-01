"""Local Binary Pattern (LBP) feature extraction."""

import logging
from typing import Optional

import numpy as np
from skimage.feature import local_binary_pattern

logger = logging.getLogger(__name__)


def compute_lbp(
    image: np.ndarray,
    P: int = 8,
    R: float = 1.0,
    method: str = 'uniform'
) -> np.ndarray:
    """Compute Local Binary Pattern features for an image.
    
    Args:
        image: Input image array with shape (H, W, C) or (H, W).
        P: Number of sample points on circle.
        R: Radius of sample circle.
        method: LBP method ('uniform', 'default', 'ror', 'var').
        
    Returns:
        LBP feature map with same spatial dimensions as input.
        For RGB input (H, W, C) -> (H, W, C)
        For grayscale (H, W) -> (H, W)
        
    Example:
        >>> import numpy as np
        >>> img = np.random.rand(100, 100, 3)
        >>> lbp_features = compute_lbp(img, P=8, R=1.0)
        >>> print(lbp_features.shape)  # (100, 100, 3)
    """
    if image.ndim == 2:
        # Grayscale image
        lbp = local_binary_pattern(image, P, R, method=method)
        return lbp.astype(np.float32)
    
    elif image.ndim == 3:
        # RGB image - compute LBP for each channel
        height, width, channels = image.shape
        lbp_channels = np.zeros((height, width, channels), dtype=np.float32)
        
        for c in range(channels):
            channel = image[:, :, c]
            lbp_channels[:, :, c] = local_binary_pattern(channel, P, R, method=method)
        
        return lbp_channels
    
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def compute_lbp_histogram(
    lbp_image: np.ndarray,
    n_bins: Optional[int] = None,
    normalize: bool = True
) -> np.ndarray:
    """Compute histogram of LBP values.
    
    Args:
        lbp_image: LBP feature map from compute_lbp().
        n_bins: Number of histogram bins. If None, auto-determined.
        normalize: Whether to normalize histogram to sum to 1.
        
    Returns:
        Histogram array.
    """
    if n_bins is None:
        # For uniform LBP, number of patterns is P + 2
        # Estimate from unique values in image
        n_bins = len(np.unique(lbp_image))
    
    # Flatten image for histogram
    if lbp_image.ndim == 3:
        # For multi-channel, concatenate all channels
        flat_lbp = lbp_image.flatten()
    else:
        flat_lbp = lbp_image.flatten()
    
    hist, _ = np.histogram(flat_lbp, bins=n_bins, density=normalize)
    
    return hist.astype(np.float32)


def compute_lbp_statistics(lbp_image: np.ndarray) -> dict:
    """Compute statistics of LBP features.
    
    Args:
        lbp_image: LBP feature map from compute_lbp().
        
    Returns:
        Dictionary with LBP statistics.
    """
    stats = {
        'mean': float(np.mean(lbp_image)),
        'std': float(np.std(lbp_image)),
        'min': float(np.min(lbp_image)),
        'max': float(np.max(lbp_image)),
        'unique_patterns': len(np.unique(lbp_image)),
        'shape': lbp_image.shape
    }
    
    return stats


def normalize_lbp(lbp_image: np.ndarray) -> np.ndarray:
    """Normalize LBP values to [0, 1] range.
    
    Args:
        lbp_image: LBP feature map.
        
    Returns:
        Normalized LBP features.
    """
    min_val = np.min(lbp_image)
    max_val = np.max(lbp_image)
    
    if max_val > min_val:
        normalized = (lbp_image - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(lbp_image)
    
    return normalized.astype(np.float32)