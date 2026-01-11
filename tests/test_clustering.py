"""Tests for clustering module."""
import numpy as np
import pandas as pd
import pytest

from analysis.clustering.classifiers import (
    cluster_cells,
    create_clustering_pipeline,
    evaluate_cluster_stability,
)


def test_create_clustering_pipeline():
    """Test pipeline creation with different configurations."""
    # Test with default parameters
    pipeline = create_clustering_pipeline()
    assert len(pipeline.steps) == 2
    assert pipeline.named_steps["cluster"].n_clusters == 5

    # Test with custom parameters
    pipeline = create_clustering_pipeline(n_clusters=3, normalization="l2")
    assert pipeline.named_steps["cluster"].n_clusters == 3

    # Test with invalid normalization
    with pytest.raises(ValueError):
        create_clustering_pipeline(normalization="invalid")


def test_cluster_cells():
    """Test clustering with sample data."""
    # Create sample data
    features = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 10, 11],
            "feature2": [5, 4, 3, 15, 14],
            "feature3": [2, 3, 4, 12, 13],
        },
        index=pd.Index(["cell1", "cell2", "cell3", "cell4", "cell5"], name="cell_id"),
    )

    # Perform clustering
    labels, centroids = cluster_cells(features, n_clusters=2, random_seed=42)

    # Check labels
    assert len(labels) == 5
    assert labels.nunique() == 2
    assert labels.index.name == "cell_id"
    assert labels.name == "cluster"

    # Check centroids
    assert centroids.shape == (3, 2)  # 3 features, 2 clusters
    assert list(centroids.index) == ["feature1", "feature2", "feature3"]


def test_cluster_cells_with_missing_values():
    """Test clustering handles missing values."""
    features = pd.DataFrame(
        {
            "feature1": [1, 2, np.nan, 10, 11],
            "feature2": [5, np.nan, 3, 15, 14],
            "feature3": [2, 3, 4, np.nan, 13],
        }
    )

    # Should not raise an error
    labels, centroids = cluster_cells(features, n_clusters=2, random_seed=42)

    assert len(labels) == 5
    assert labels.nunique() <= 2


def test_evaluate_cluster_stability():
    """Test cluster stability evaluation."""
    # Create sample data with clear clusters
    np.random.seed(42)
    cluster1 = np.random.randn(20, 3) + np.array([0, 0, 0])
    cluster2 = np.random.randn(20, 3) + np.array([10, 10, 10])
    data = np.vstack([cluster1, cluster2])

    # Evaluate stability
    metrics = evaluate_cluster_stability(
        data=data,
        cluster_range=range(2, 5),
        n_iterations=5,
        random_seed=42,
    )

    # Check metrics structure
    assert set(metrics.keys()) == {
        "n_clusters",
        "silhouette_mean",
        "silhouette_std",
        "calinski_harabasz_mean",
        "calinski_harabasz_std",
    }

    # Check metrics values
    assert len(metrics["n_clusters"]) == 3
    assert all(isinstance(x, (int, np.integer)) for x in metrics["n_clusters"])
    assert all(isinstance(x, float) for x in metrics["silhouette_mean"])
    assert all(0 <= x <= 1 for x in metrics["silhouette_mean"])


def test_cluster_reproducibility():
    """Test that clustering is reproducible with same random seed."""
    features = pd.DataFrame(
        np.random.randn(50, 5),
        columns=[f"feature_{i}" for i in range(5)],
    )

    labels1, _ = cluster_cells(features, n_clusters=3, random_seed=42)
    labels2, _ = cluster_cells(features, n_clusters=3, random_seed=42)

    # Should produce identical results
    pd.testing.assert_series_equal(labels1, labels2)


def test_cluster_different_algorithms():
    """Test clustering with different normalizations."""
    features = pd.DataFrame(np.random.randn(30, 4))

    for normalization in ["l1", "l2", "standard"]:
        labels, centroids = cluster_cells(
            features, n_clusters=3, normalization=normalization, random_seed=42
        )
        assert len(labels) == 30
        assert labels.nunique() <= 3
