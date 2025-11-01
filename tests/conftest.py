"""Test configuration and utilities."""
import pytest
from pathlib import Path

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def config_path():
    """Path to test configuration file."""
    return Path(__file__).parent.parent / "configs" / "default.yaml"


@pytest.fixture
def sample_image():
    """Sample RGB image for testing."""
    import numpy as np
    # Create a simple 100x100x3 test image
    return np.random.rand(100, 100, 3).astype(np.float32)


@pytest.fixture
def sample_grid():
    """Sample grid for testing."""
    import numpy as np
    # Create a simple 50x50 grid with urban/non-urban cells
    grid = np.zeros((50, 50), dtype=np.int8)
    grid[20:30, 20:30] = 1  # Urban center
    return grid