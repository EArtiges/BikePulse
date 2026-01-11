"""Tests for geospatial utility functions."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add notebooks directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))

from geo_utils import get_boundaries, get_spread, get_window, window_filter


class TestGeoBoundaries:
    """Test geographic boundary utility functions."""

    def test_get_boundaries(self):
        """Test get_boundaries returns min and max."""
        arr = np.array([1, 2, 3, 4, 5])
        min_val, max_val = get_boundaries(arr)
        assert min_val == 1
        assert max_val == 5

    def test_get_boundaries_negative(self):
        """Test get_boundaries with negative values."""
        arr = np.array([-5, -2, 0, 3, 7])
        min_val, max_val = get_boundaries(arr)
        assert min_val == -5
        assert max_val == 7

    def test_get_spread(self):
        """Test get_spread returns range."""
        boundaries = (10, 50)
        spread = get_spread(boundaries)
        assert spread == 40  # 50 - 10

    def test_get_window_default_buffer(self):
        """Test get_window with default 5% buffer."""
        center = 50
        spread = 100
        buffer = 0.05
        window = get_window(center, spread, buffer)
        # With 5% buffer: center=50, spread=100, buffer=0.05*100/2=2.5
        # window = (50 - 100/2 - 2.5, 50 + 100/2 + 2.5) = (-2.5, 102.5)
        assert window[0] == pytest.approx(-2.5)
        assert window[1] == pytest.approx(102.5)

    def test_get_window_custom_buffer(self):
        """Test get_window with custom buffer."""
        center = 50
        spread = 100
        buffer = 0.10
        window = get_window(center, spread, buffer)
        # With 10% buffer: center=50, spread=100, buffer=0.10*100/2=5
        # window = (50 - 100/2 - 5, 50 + 100/2 + 5) = (-5, 105)
        assert window[0] == pytest.approx(-5.0)
        assert window[1] == pytest.approx(105.0)

    def test_window_filter(self):
        """Test window_filter filters Series correctly."""
        import pandas as pd

        series = pd.Series([1, 5, 10, 15, 20, 25, 30])
        window = (10, 20)
        mask = window_filter(series, window)

        # mask should be a boolean array
        assert isinstance(mask, pd.Series)
        assert mask.sum() == 3  # 10, 15, 20 should pass filter
        filtered_values = series[mask]
        assert filtered_values.min() == 10
        assert filtered_values.max() == 20
