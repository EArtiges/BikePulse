"""Tests for factorization module."""
import numpy as np
import pandas as pd
import pytest
import tensorly as tl

from analysis.factorization.decomposition import (
    extract_spatial_factors,
    factorize_tensor,
    prepare_od_tensor,
    reconstruct_tensor,
)
from analysis.factorization.evaluation import (
    compute_reconstruction_metrics,
    evaluate_rmse_by_rank,
)


@pytest.fixture
def sample_od_matrix():
    """Create a sample OD matrix for testing."""
    np.random.seed(42)

    # Create synthetic data
    years = [2020, 2021]
    months = [1, 2, 3]
    weekdays = [0, 1, 2, 5, 6]  # Mix of weekdays and weekends
    hours = list(range(24))
    cells = [f"cell_{i}" for i in range(10)]

    data = []
    for year in years:
        for month in months:
            for weekday in weekdays:
                for hour in hours:
                    for start_cell in cells[:5]:  # Subset for speed
                        for end_cell in cells[:5]:
                            trips = max(0, np.random.poisson(10))
                            data.append(
                                {
                                    "year": year,
                                    "month": month,
                                    "weekday": weekday,
                                    "hour": hour,
                                    "start_cell": start_cell,
                                    "end_cell": end_cell,
                                    "trip": trips,
                                }
                            )

    df = pd.DataFrame(data)
    df = df.set_index(["year", "month", "weekday", "hour", "start_cell", "end_cell"])
    return df["trip"]


def test_prepare_od_tensor(sample_od_matrix):
    """Test OD tensor preparation."""
    tensor, hours, cells = prepare_od_tensor(
        sample_od_matrix, weekdays_only=True, month_range=(1, 3), normalize_by_year=True
    )

    # Check tensor shape
    assert len(tensor.shape) == 3
    assert tensor.shape[0] == 24  # 24 hours
    assert tensor.shape[1] == tensor.shape[2]  # Square spatial matrix

    # Check hours
    assert hours == list(range(24))

    # Check cells
    assert len(cells) > 0
    assert all(isinstance(c, str) for c in cells)

    # Check normalization (should sum to 100)
    assert np.abs(tensor.sum() - 100) < 0.01


def test_prepare_od_tensor_weekends(sample_od_matrix):
    """Test OD tensor preparation for weekends."""
    tensor_weekday, _, _ = prepare_od_tensor(
        sample_od_matrix, weekdays_only=True, month_range=(1, 3)
    )
    tensor_weekend, _, _ = prepare_od_tensor(
        sample_od_matrix, weekdays_only=False, month_range=(1, 3)
    )

    # Tensors should be different
    assert not np.allclose(tensor_weekday, tensor_weekend)


def test_factorize_tensor_tucker():
    """Test Tucker factorization."""
    # Create simple tensor
    np.random.seed(42)
    tensor = tl.tensor(np.random.rand(10, 15, 15).astype("float64"))

    # Factorize
    result = factorize_tensor(
        tensor, method="tucker", rank=(3, 5, 5), random_seed=42
    )

    # Check result structure
    assert result["method"] == "tucker"
    assert "core" in result
    assert "factors" in result
    assert result["rank"] == (3, 5, 5)
    assert len(result["factors"]) == 3


def test_factorize_tensor_parafac():
    """Test PARAFAC factorization."""
    # Create simple tensor
    np.random.seed(42)
    tensor = tl.tensor(np.random.rand(10, 15, 15).astype("float64"))

    # Factorize
    result = factorize_tensor(tensor, method="parafac", rank=5, random_seed=42)

    # Check result structure
    assert result["method"] == "parafac"
    assert "factors" in result
    assert "weights" in result
    assert result["rank"] == 5
    assert len(result["factors"]) == 3


def test_factorize_tensor_invalid_method():
    """Test that invalid method raises error."""
    tensor = tl.tensor(np.random.rand(10, 15, 15).astype("float64"))

    with pytest.raises(ValueError):
        factorize_tensor(tensor, method="invalid")


def test_reconstruct_tensor():
    """Test tensor reconstruction."""
    # Create tensor
    np.random.seed(42)
    original = tl.tensor(np.random.rand(10, 15, 15).astype("float64"))

    # Factorize
    result = factorize_tensor(
        original, method="tucker", rank=(5, 10, 10), random_seed=42
    )

    # Reconstruct
    reconstructed = reconstruct_tensor(result)

    # Check shape matches
    assert reconstructed.shape == original.shape

    # Check reconstruction is close (with high rank should be very close)
    assert np.allclose(original, reconstructed, rtol=0.5)


def test_extract_spatial_factors():
    """Test spatial factor extraction."""
    # Create tensor and factorize
    np.random.seed(42)
    tensor = tl.tensor(np.random.rand(10, 15, 15).astype("float64"))
    result = factorize_tensor(tensor, method="parafac", rank=3, random_seed=42)

    # Extract spatial factors
    cells = [f"cell_{i}" for i in range(15)]
    factors = extract_spatial_factors(result, cells)

    # Check structure
    assert isinstance(factors, pd.DataFrame)
    assert set(factors.columns) == {"start_cell", "end_cell", "component", "trips"}

    # Check components
    assert set(factors["component"].unique()) == {0, 1, 2, "full"}

    # Check cells
    assert set(factors["start_cell"].unique()).issubset(set(cells))
    assert set(factors["end_cell"].unique()).issubset(set(cells))


def test_compute_reconstruction_metrics():
    """Test reconstruction metrics computation."""
    # Create tensors
    original = tl.tensor(np.random.rand(10, 15, 15).astype("float64"))
    reconstructed = original + tl.tensor(
        np.random.randn(10, 15, 15).astype("float64") * 0.1
    )

    # Compute metrics
    metrics = compute_reconstruction_metrics(original, reconstructed)

    # Check metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r_squared" in metrics
    assert "relative_error_mean" in metrics
    assert "relative_error_std" in metrics

    # Check values are reasonable
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert 0 <= metrics["r_squared"] <= 1


def test_evaluate_rmse_by_rank():
    """Test RMSE evaluation across ranks."""
    # Create tensor
    np.random.seed(42)
    tensor = tl.tensor(np.random.rand(10, 15, 15).astype("float64"))

    # Evaluate
    rank_list, errors, error_stds = evaluate_rmse_by_rank(
        tensor, method="parafac", rank_range=range(2, 5), n_runs=3, init="svd"
    )

    # Check results
    assert rank_list == [2, 3, 4]
    assert len(errors) == 3
    assert len(error_stds) == 3

    # Errors should be positive
    assert all(e > 0 for e in errors)
    assert all(std >= 0 for std in error_stds)

    # Higher rank should generally have lower error
    # (not always true with random data, but usually)


def test_factorization_reproducibility():
    """Test that factorization is reproducible with same random seed."""
    tensor = tl.tensor(np.random.rand(10, 15, 15).astype("float64"))

    result1 = factorize_tensor(tensor, method="parafac", rank=3, random_seed=42)
    result2 = factorize_tensor(tensor, method="parafac", rank=3, random_seed=42)

    # Factors should be very similar (may have small numerical differences)
    for f1, f2 in zip(result1["factors"], result2["factors"]):
        assert np.allclose(f1, f2, rtol=1e-3)
