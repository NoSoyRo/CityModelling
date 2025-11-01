"""Test configuration loading."""
import pytest
from pathlib import Path
from tesis_ac.config import load_config, create_directories


def test_load_config(config_path):
    """Test loading configuration from YAML file."""
    config = load_config(config_path)
    
    assert config.cell_size == 30
    assert config.clustering.clusters == 3
    assert config.ca.neighborhood == "moore"
    assert config.ga.pop_size == 50


def test_config_paths_exist_after_creation(config_path, tmp_path):
    """Test that directories are created correctly."""
    # Load config with temporary base path
    config = load_config(config_path)
    
    # Override paths to use temp directory
    config.data_raw = tmp_path / "data" / "raw"
    config.data_processed = tmp_path / "data" / "processed"
    config.data_reports = tmp_path / "reports"
    config.data_figures = tmp_path / "figs"
    
    create_directories(config)
    
    assert config.data_raw.exists()
    assert config.data_processed.exists()
    assert config.data_reports.exists()
    assert config.data_figures.exists()


def test_config_file_not_found():
    """Test error handling for missing config file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")