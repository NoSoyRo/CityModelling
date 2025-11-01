"""Tests for input/output utilities."""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import os

from tesis_ac.utils.io import (
    load_image_series, 
    load_single_image, 
    save_processed_image,
    get_image_stats,
    validate_image_series
)


@pytest.fixture
def temp_images():
    """Create temporary test images."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test images with different years
        test_images = {}
        for year in [1984, 1990, 2000, 2020]:
            # Create a simple RGB image
            img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            img_path = temp_path / f"qrtro_12_{year}.jpg"
            img.save(img_path)
            test_images[year] = img_array
        
        yield temp_path, test_images


def test_load_image_series(temp_images):
    """Test loading a series of images."""
    temp_path, expected_images = temp_images
    
    # Load images
    loaded_images = load_image_series(temp_path, "qrtro_12_*.jpg", normalize=False)
    
    # Check that all years are loaded
    assert set(loaded_images.keys()) == {1984, 1990, 2000, 2020}
    
    # Check shapes
    for year, img in loaded_images.items():
        assert img.shape == (100, 100, 3)
        assert img.dtype == np.uint8


def test_load_image_series_normalized(temp_images):
    """Test loading images with normalization."""
    temp_path, _ = temp_images
    
    images = load_image_series(temp_path, normalize=True)
    
    for img in images.values():
        assert img.dtype == np.float32
        assert 0.0 <= img.min() <= img.max() <= 1.0


def test_load_single_image(temp_images):
    """Test loading a single image."""
    temp_path, _ = temp_images
    
    img_path = temp_path / "qrtro_12_1984.jpg"
    img = load_single_image(img_path, normalize=True)
    
    assert img.shape == (100, 100, 3)
    assert img.dtype == np.float32
    assert 0.0 <= img.min() <= img.max() <= 1.0


def test_save_processed_image():
    """Test saving a processed image."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test image
        img_array = np.random.rand(50, 50, 3).astype(np.float32)
        
        output_path = Path(temp_dir) / "test_output.png"
        save_processed_image(img_array, output_path, denormalize=True)
        
        # Check file was created
        assert output_path.exists()
        
        # Load and check
        loaded = load_single_image(output_path, normalize=False)
        assert loaded.shape == (50, 50, 3)
        assert loaded.dtype == np.uint8


def test_get_image_stats(temp_images):
    """Test getting image statistics."""
    temp_path, _ = temp_images
    
    images = load_image_series(temp_path)
    stats = get_image_stats(images)
    
    assert stats['total_images'] == 4
    assert stats['years'] == [1984, 1990, 2000, 2020]
    assert stats['year_range'] == (1984, 2020)
    assert stats['image_shape'] == (100, 100, 3)
    assert stats['inconsistent_shapes'] == []


def test_validate_image_series(temp_images):
    """Test image series validation."""
    temp_path, _ = temp_images
    
    images = load_image_series(temp_path)
    assert validate_image_series(images) == True


def test_validate_empty_series():
    """Test validation of empty series."""
    assert validate_image_series({}) == False


def test_load_nonexistent_directory():
    """Test error handling for nonexistent directory."""
    with pytest.raises(FileNotFoundError):
        load_image_series("/nonexistent/path")


def test_load_no_matching_files():
    """Test error handling when no files match pattern."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError):
            load_image_series(temp_dir, "nonexistent_*.jpg")