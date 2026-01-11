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
        arr = np.array([10, 20, 30, 40, 50])
        spread = get_spread(arr)
        assert spread == 40  # 50 - 10

    def test_get_window_default_buffer(self):
        """Test get_window with default 5% buffer."""
        arr = np.array([0, 100])
        window = get_window(arr)
        # With 5% buffer: -5 to 105
        assert window[0] == pytest.approx(-5.0)
        assert window[1] == pytest.approx(105.0)

    def test_get_window_custom_buffer(self):
        """Test get_window with custom buffer."""
        arr = np.array([0, 100])
        window = get_window(arr, buffer_percent=0.10)
        # With 10% buffer: -10 to 110
        assert window[0] == pytest.approx(-10.0)
        assert window[1] == pytest.approx(110.0)

    def test_window_filter(self):
        """Test window_filter filters DataFrame correctly."""
        import pandas as pd

        df = pd.DataFrame({"value": [1, 5, 10, 15, 20, 25, 30]})
        window = (10, 20)
        filtered = window_filter(df, "value", window)

        assert len(filtered) == 3  # 10, 15, 20
        assert filtered["value"].min() == 10
        assert filtered["value"].max() == 20
