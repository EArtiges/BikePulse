"""Pytest configuration and fixtures for BikePulse tests."""

import sys
from pathlib import Path

import pytest

# Add notebooks directory to Python path
NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
sys.path.insert(0, str(NOTEBOOKS_DIR))


@pytest.fixture
def notebooks_dir():
    """Return the path to the notebooks directory."""
    return NOTEBOOKS_DIR


@pytest.fixture
def project_root():
    """Return the path to the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_config():
    """Return a sample configuration dictionary."""
    return {
        "collect_area": {
            "name": "Oslo, Norway",
            "filename": "data/Oslo/oslo_area.geoparquet",
        },
        "grid": {
            "resolution": 8,
            "full_grid_filename": "data/Oslo/full_hex_grid.geoparquet",
            "clipped_grid_filename": "data/Oslo/hex_grid.geoparquet",
        },
        "POIs": {
            "crs": 32632,
            "folder": "data/Oslo/POIs/",
        },
    }
