"""Utilidades de entrada/salida para cargar y preprocesar imágenes.

Este módulo concentra helpers para:

- Cargar series de imágenes por patrón de nombres (p. ej. un año en el nombre).
- Normalizar/desnormalizar valores de píxel.
- Validar consistencia básica (dimensiones, continuidad temporal).

Nota:
    La lógica aquí es intencionalmente ligera; el objetivo es proveer I/O
    reproducible para el pipeline de extracción de características.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_image_series(
    data_path: Union[str, Path], 
    pattern: str = "qrtro_12_*.jpg",
    normalize: bool = True
) -> Dict[int, np.ndarray]:
    """Carga una serie de imágenes que hacen match con un patrón y extrae el año.

    El año se extrae buscando un token de 4 dígitos en el nombre del archivo.

    Argumentos:
        data_path: Ruta al directorio que contiene las imágenes.
        pattern: Patrón tipo glob para hacer match con archivos de imagen.
        normalize: Si ``True``, normaliza los valores de píxel a $[0, 1]$.

    Retorna:
        Diccionario ``{año: imagen}`` donde la imagen es un ``np.ndarray`` con
        shape ``(H, W, C)``.

    Ejemplo:
        >>> images = load_image_series("data/raw", pattern="qrtro_12_*.jpg")
        >>> sorted(images.keys())[:3]  # doctest: +SKIP
        [1984, 1985, 1986]
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # Find all matching files
    image_files = list(data_path.glob(pattern))
    
    if not image_files:
        raise ValueError(f"No images found with pattern '{pattern}' in {data_path}")
    
    logger.info(f"Found {len(image_files)} images matching pattern '{pattern}'")
    
    # Extract years from filenames and load images
    images = {}
    year_pattern = r"(\d{4})"  # Extract 4-digit year
    
    for img_path in sorted(image_files):
        # Extract year from filename
        year_match = re.search(year_pattern, img_path.stem)
        if not year_match:
            logger.warning(f"Could not extract year from filename: {img_path.name}")
            continue
            
        year = int(year_match.group(1))
        
        # Load image
        try:
            img_array = load_single_image(img_path, normalize=normalize)
            images[year] = img_array
            logger.debug(f"Loaded {img_path.name} -> year {year}, shape {img_array.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(images)} images from years {min(images.keys())}-{max(images.keys())}")
    return images


def load_single_image(
    img_path: Union[str, Path], 
    normalize: bool = True,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Carga una sola imagen desde disco y la devuelve como array.

    Argumentos:
        img_path: Ruta al archivo de imagen.
        normalize: Si ``True``, normaliza los valores de píxel a $[0, 1]$.
        target_size: Tamaño objetivo opcional ``(width, height)`` para reescalar.

    Retorna:
        Array de imagen con shape ``(H, W, C)`` (RGB) o ``(H, W)`` (escala de grises).
    """
    img_path = Path(img_path)
    
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")
    
    # Load image with PIL
    with Image.open(img_path) as img:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize if requested
        if target_size:
            img = img.resize(target_size, Image.LANCZOS)
            
        # Convert to numpy array
        img_array = np.array(img)
    
    # Normalize to [0, 1] if requested
    if normalize:
        img_array = img_array.astype(np.float32) / 255.0
    
    return img_array


def save_processed_image(
    img_array: np.ndarray,
    output_path: Union[str, Path],
    denormalize: bool = True
) -> None:
    """Guarda un array de imagen procesada a disco.

    Argumentos:
        img_array: Array de imagen a guardar.
        output_path: Ruta destino.
        denormalize: Si ``True`` y el array es ``float32``, asume $[0,1]$ y
            desnormaliza a $[0,255]$ (``uint8``).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare array for saving
    if denormalize and img_array.dtype == np.float32:
        img_array = (img_array * 255).astype(np.uint8)
    
    # Convert to PIL and save
    if img_array.ndim == 3:  # RGB
        img = Image.fromarray(img_array)
    elif img_array.ndim == 2:  # Grayscale
        img = Image.fromarray(img_array, mode='L')
    else:
        raise ValueError(f"Unsupported image shape: {img_array.shape}")
    
    img.save(output_path)
    logger.info(f"Saved processed image to {output_path}")


def get_image_stats(images: Dict[int, np.ndarray]) -> Dict[str, any]:
    """Calcula estadísticas descriptivas de una serie de imágenes.

    Argumentos:
        images: Diccionario ``{año: imagen}``.

    Retorna:
        Diccionario con estadísticas (rango de años, shape, dtype, etc.).
    """
    if not images:
        return {}
    
    # Get dimensions from first image
    first_img = next(iter(images.values()))
    height, width = first_img.shape[:2]
    channels = first_img.shape[2] if first_img.ndim == 3 else 1
    
    # Check consistency
    inconsistent_shapes = []
    for year, img in images.items():
        if img.shape[:2] != (height, width):
            inconsistent_shapes.append(year)
    
    stats = {
        'total_images': len(images),
        'years': sorted(images.keys()),
        'year_range': (min(images.keys()), max(images.keys())),
        'image_shape': (height, width, channels),
        'inconsistent_shapes': inconsistent_shapes,
        'dtype': str(first_img.dtype),
        'value_range': (float(first_img.min()), float(first_img.max()))
    }
    
    return stats


def validate_image_series(images: Dict[int, np.ndarray]) -> bool:
    """Valida que una serie de imágenes sea consistente para análisis temporal.

    Chequea que existan imágenes, que las dimensiones sean consistentes, y que
    haya suficientes años para considerarse serie temporal.

    Argumentos:
        images: Diccionario ``{año: imagen}``.

    Retorna:
        ``True`` si la serie parece válida; ``False`` si se detecta un problema.
    """
    if not images:
        logger.error("Empty image series")
        return False
    
    stats = get_image_stats(images)
    
    # Check for inconsistent shapes
    if stats['inconsistent_shapes']:
        logger.error(f"Inconsistent image shapes in years: {stats['inconsistent_shapes']}")
        return False
    
    # Check for reasonable time series
    years = stats['years']
    if len(years) < 2:
        logger.error("Need at least 2 images for time series analysis")
        return False
    
    # Check for large gaps
    year_diffs = [years[i+1] - years[i] for i in range(len(years)-1)]
    max_gap = max(year_diffs)
    if max_gap > 5:
        logger.warning(f"Large gap detected in time series: {max_gap} years")
    
    logger.info(f"Image series validation passed: {len(images)} images, {stats['year_range']}")
    return True