"""Cell clustering algorithms."""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler


def create_clustering_pipeline(
    algorithm: str = "kmeans",
    normalization: str = "l1",
    n_clusters: int = 5,
    random_seed: int = 42,
) -> Pipeline:
    """
    Create sklearn pipeline for clustering.

    Parameters
    ----------
    algorithm : str
        Clustering algorithm (currently only 'kmeans' supported)
    normalization : str
        Normalization method: 'l1', 'l2', or 'standard'
    n_clusters : int
        Number of clusters
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    Pipeline
        Configured sklearn pipeline
    """
    normalizers = {
        "l1": Normalizer("l1"),
        "l2": Normalizer("l2"),
        "standard": StandardScaler(),
    }

    if normalization not in normalizers:
        raise ValueError(
            f"Unknown normalization: {normalization}. "
            f"Must be one of {list(normalizers.keys())}"
        )

    return Pipeline(
        [
            ("normalize", normalizers[normalization]),
            ("cluster", KMeans(n_clusters=n_clusters, random_state=random_seed)),
        ]
    )


def evaluate_cluster_stability(
    data: np.ndarray,
    cluster_range: range,
    n_iterations: int = 50,
    **pipeline_kwargs,
) -> Dict[str, List[float]]:
    """
    Evaluate clustering stability across different k values.

    Parameters
    ----------
    data : np.ndarray
        Feature matrix (n_samples, n_features)
    cluster_range : range
        Range of cluster numbers to evaluate
    n_iterations : int
        Number of iterations per k value
    **pipeline_kwargs
        Additional arguments for create_clustering_pipeline

    Returns
    -------
    Dict[str, List[float]]
        Dictionary with evaluation metrics
    """
    results = {
        "n_clusters": [],
        "silhouette_mean": [],
        "silhouette_std": [],
        "calinski_harabasz_mean": [],
        "calinski_harabasz_std": [],
    }

    for n_clusters in cluster_range:
        sil_scores = []
        ch_scores = []

        for _ in range(n_iterations):
            pipeline = create_clustering_pipeline(
                n_clusters=n_clusters, **pipeline_kwargs
            )
            labels = pipeline.fit_predict(data)

            sil_scores.append(silhouette_score(data, labels, metric="euclidean"))
            ch_scores.append(calinski_harabasz_score(data, labels))

        results["n_clusters"].append(n_clusters)
        results["silhouette_mean"].append(np.mean(sil_scores))
        results["silhouette_std"].append(np.std(sil_scores))
        results["calinski_harabasz_mean"].append(np.mean(ch_scores))
        results["calinski_harabasz_std"].append(np.std(ch_scores))

    return results


def cluster_cells(
    cell_features: pd.DataFrame, n_clusters: int = 5, **pipeline_kwargs
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Cluster cells based on features.

    Parameters
    ----------
    cell_features : pd.DataFrame
        Cell features (index: cell_id, columns: features)
    n_clusters : int
        Number of clusters
    **pipeline_kwargs
        Additional arguments for create_clustering_pipeline

    Returns
    -------
    Tuple[pd.Series, pd.DataFrame]
        - Cluster labels (index: cell_id)
        - Cluster centroids (index: feature_name, columns: cluster_id)
    """
    # Handle missing values
    data = cell_features.fillna(0).values

    # Create and fit pipeline
    pipeline = create_clustering_pipeline(n_clusters=n_clusters, **pipeline_kwargs)
    labels = pipeline.fit_predict(data)

    # Extract centroids
    kmeans = pipeline.named_steps["cluster"]
    centroids = pd.DataFrame(
        kmeans.cluster_centers_, columns=cell_features.columns
    ).T

    # Create labels series
    labels_series = pd.Series(labels, index=cell_features.index, name="cluster")

    return labels_series, centroids
