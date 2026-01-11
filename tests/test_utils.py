"""Tests for utility functions."""

import sys
from pathlib import Path

import pytest

# Add notebooks directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))


class TestConfigLoading:
    """Test configuration loading functionality."""

    def test_config_file_exists(self):
        """Test that run.yml configuration file exists."""
        config_path = Path(__file__).parent.parent / "notebooks" / "run.yml"
        assert config_path.exists(), "run.yml configuration file not found"

    def test_get_config_function_exists(self):
        """Test that get_config function can be imported."""
        try:
            from utils import get_config

            assert callable(get_config)
        except ImportError as e:
            pytest.fail(f"Could not import get_config: {e}")

    def test_get_config_returns_dict(self):
        """Test that get_config returns a dictionary for valid cities."""
        from utils import get_config

        # Test with Oslo
        config = get_config("Oslo")
        assert isinstance(config, dict), "Config should return a dictionary"
        assert "POIs" in config or "collect_area" in config, "Config should have expected keys"


class TestImports:
    """Test that critical imports work."""

    def test_geospatial_imports(self):
        """Test that geospatial libraries can be imported."""
        import geopandas
        import h3
        import osmnx
        import shapely

        assert geopandas.__version__
        assert h3.__version__
        assert osmnx.__version__
        assert shapely.__version__

    def test_data_science_imports(self):
        """Test that data science libraries can be imported."""
        import numpy
        import pandas
        import scipy
        import sklearn
        import tensorly

        assert numpy.__version__
        assert pandas.__version__
        assert scipy.__version__
        assert sklearn.__version__
        assert tensorly.__version__
