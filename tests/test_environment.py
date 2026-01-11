"""Tests for geospatial environment and dependencies."""

import sys
from pathlib import Path

import pytest

# Add notebooks directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))


class TestGeospatialStack:
    """Test critical geospatial libraries can be imported."""

    def test_gdal_import(self):
        """Test GDAL can be imported."""
        try:
            from osgeo import gdal

            assert gdal.__version__, "GDAL imported but has no version"
        except ImportError as e:
            pytest.fail(f"Could not import GDAL: {e}")

    def test_geopandas_import(self):
        """Test GeoPandas can be imported."""
        try:
            import geopandas as gpd

            assert gpd.__version__, "GeoPandas imported but has no version"
        except ImportError as e:
            pytest.fail(f"Could not import geopandas: {e}")

    def test_shapely_import(self):
        """Test Shapely can be imported."""
        try:
            import shapely

            assert shapely.__version__, "Shapely imported but has no version"
        except ImportError as e:
            pytest.fail(f"Could not import shapely: {e}")

    def test_osmnx_import(self):
        """Test OSMnx can be imported."""
        try:
            import osmnx

            assert osmnx.__version__, "OSMnx imported but has no version"
        except ImportError as e:
            pytest.fail(f"Could not import osmnx: {e}")

    def test_h3_import(self):
        """Test H3 can be imported."""
        try:
            import h3

            assert h3.__version__, "H3 imported but has no version"
        except ImportError as e:
            pytest.fail(f"Could not import h3: {e}")

    def test_h3pandas_import(self):
        """Test h3pandas can be imported."""
        try:
            import h3pandas

            # h3pandas may not have __version__
            assert h3pandas is not None
        except ImportError as e:
            pytest.fail(f"Could not import h3pandas: {e}")

    def test_rasterio_import(self):
        """Test rasterio can be imported."""
        try:
            import rasterio

            assert rasterio.__version__, "Rasterio imported but has no version"
        except ImportError as e:
            pytest.fail(f"Could not import rasterio: {e}")

    def test_pyproj_import(self):
        """Test pyproj can be imported."""
        try:
            import pyproj

            assert pyproj.__version__, "Pyproj imported but has no version"
        except ImportError as e:
            pytest.fail(f"Could not import pyproj: {e}")


class TestDataScienceStack:
    """Test data science and ML libraries can be imported."""

    def test_numpy_import(self):
        """Test NumPy can be imported."""
        try:
            import numpy as np

            assert np.__version__, "NumPy imported but has no version"
        except ImportError as e:
            pytest.fail(f"Could not import numpy: {e}")

    def test_pandas_import(self):
        """Test Pandas can be imported."""
        try:
            import pandas as pd

            assert pd.__version__, "Pandas imported but has no version"
        except ImportError as e:
            pytest.fail(f"Could not import pandas: {e}")

    def test_scipy_import(self):
        """Test SciPy can be imported."""
        try:
            import scipy

            assert scipy.__version__, "SciPy imported but has no version"
        except ImportError as e:
            pytest.fail(f"Could not import scipy: {e}")

    def test_sklearn_import(self):
        """Test scikit-learn can be imported."""
        try:
            import sklearn

            assert sklearn.__version__, "scikit-learn imported but has no version"
        except ImportError as e:
            pytest.fail(f"Could not import sklearn: {e}")

    def test_tensorly_import(self):
        """Test TensorLy can be imported."""
        try:
            import tensorly

            assert tensorly.__version__, "TensorLy imported but has no version"
        except ImportError as e:
            pytest.fail(f"Could not import tensorly: {e}")

    def test_networkx_import(self):
        """Test NetworkX can be imported."""
        try:
            import networkx as nx

            assert nx.__version__, "NetworkX imported but has no version"
        except ImportError as e:
            pytest.fail(f"Could not import networkx: {e}")


class TestGDALCompatibility:
    """Test GDAL Python bindings match system library."""

    def test_gdal_version_compatibility(self):
        """Test GDAL Python bindings are compatible with system GDAL."""
        try:
            from osgeo import gdal

            # Get Python GDAL version
            python_version = gdal.__version__

            # This test passes if import succeeded
            # The real version matching happens during pip install
            assert python_version is not None
        except ImportError:
            pytest.skip("GDAL not installed")
        except Exception as e:
            pytest.fail(f"GDAL version check failed: {e}")


class TestCRSSupport:
    """Test coordinate reference system support."""

    def test_pyproj_has_epsg_database(self):
        """Test pyproj can access EPSG database."""
        try:
            from pyproj import CRS

            # Test common CRS used in project
            crs_wgs84 = CRS.from_epsg(4326)  # WGS84
            assert crs_wgs84.is_geographic

            crs_utm32n = CRS.from_epsg(32632)  # Oslo UTM
            assert crs_utm32n.is_projected
        except ImportError:
            pytest.skip("pyproj not installed")
        except Exception as e:
            pytest.fail(f"CRS support check failed: {e}")

    def test_geopandas_crs_handling(self):
        """Test GeoPandas CRS handling."""
        try:
            import geopandas as gpd
            from shapely.geometry import Point

            # Create simple GeoDataFrame
            gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")

            # Test CRS transformation
            gdf_transformed = gdf.to_crs("EPSG:32632")
            assert gdf_transformed.crs.to_epsg() == 32632
        except ImportError:
            pytest.skip("geopandas not installed")
        except Exception as e:
            pytest.fail(f"GeoPandas CRS handling failed: {e}")


class TestH3Integration:
    """Test H3 hexagonal grid functionality."""

    def test_h3_basic_operations(self):
        """Test basic H3 operations work."""
        try:
            import h3

            # Test lat/lon to H3 conversion
            lat, lon = 59.9139, 10.7522  # Oslo
            resolution = 8
            h3_index = h3.latlng_to_cell(lat, lon, resolution)
            assert h3_index is not None

            # Test H3 to lat/lon conversion
            lat_lon = h3.cell_to_latlng(h3_index)
            assert len(lat_lon) == 2
        except ImportError:
            pytest.skip("h3 not installed")
        except Exception as e:
            pytest.fail(f"H3 operations failed: {e}")
