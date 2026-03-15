"""Extracción de características LBP (Local Binary Pattern).

El LBP es un descriptor de textura basado en comparar un píxel central con sus
vecinos en una circunferencia de radio $R$ y $P$ puntos de muestreo.
"""

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
    """Computa el mapa de características LBP para una imagen.

    Argumentos:
        image: Imagen de entrada con shape ``(H, W, C)`` o ``(H, W)``.
        P: Número de puntos de muestreo sobre la circunferencia.
        R: Radio de la circunferencia de muestreo.
        method: Método LBP (p. ej. ``'uniform'``, ``'default'``, ``'ror'``, ``'var'``).

    Retorna:
        Mapa LBP con las mismas dimensiones espaciales que la entrada.

        - Entrada RGB ``(H, W, C)`` -> salida ``(H, W, C)``.
        - Entrada en escala de grises ``(H, W)`` -> salida ``(H, W)``.

    Ejemplo:
        >>> import numpy as np
        >>> img = np.random.rand(100, 100, 3)
        >>> lbp_features = compute_lbp(img, P=8, R=1.0)
        >>> lbp_features.shape  # doctest: +SKIP
        (100, 100, 3)
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
    """Computa el histograma de valores LBP.

    Argumentos:
        lbp_image: Mapa LBP producido por :func:`compute_lbp`.
        n_bins: Número de bins. Si es ``None``, se estima con valores únicos.
        normalize: Si ``True``, normaliza el histograma (densidad) para que sume 1.

    Retorna:
        Histograma 1D como ``np.ndarray``.
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
    """Calcula estadísticas descriptivas de un mapa LBP.

    Argumentos:
        lbp_image: Mapa LBP producido por :func:`compute_lbp`.

    Retorna:
        Diccionario con métricas (media, desviación, min/max, patrones únicos, etc.).
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
    """Normaliza los valores LBP al rango $[0, 1]$.

    Argumentos:
        lbp_image: Mapa LBP.

    Retorna:
        Mapa LBP normalizado como ``float32``.
    """
    min_val = np.min(lbp_image)
    max_val = np.max(lbp_image)
    
    if max_val > min_val:
        normalized = (lbp_image - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(lbp_image)
    
    return normalized.astype(np.float32)