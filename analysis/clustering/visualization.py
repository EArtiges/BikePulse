"""Visualization functions for clustering results."""
from pathlib import Path
from typing import Optional

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


def plot_cluster_metrics(
    metrics: dict,
    output_dir: Path,
    plot_format: str = "png",
    dpi: int = 300,
) -> Path:
    """
    Plot silhouette and Calinski-Harabasz scores.

    Parameters
    ----------
    metrics : dict
        Dictionary with 'n_clusters', 'silhouette_mean', 'silhouette_std',
        'calinski_harabasz_mean', 'calinski_harabasz_std'
    output_dir : Path
        Directory to save plots
    plot_format : str
        Format for saved plots (png, pdf, svg)
    dpi : int
        Resolution for saved plots

    Returns
    -------
    Path
        Path to saved plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Silhouette scores
    axes[0].errorbar(
        metrics["n_clusters"],
        metrics["silhouette_mean"],
        yerr=metrics["silhouette_std"],
        marker="o",
        capsize=5,
    )
    axes[0].set_xlabel("Number of Clusters")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title("Clustering Quality: Silhouette Score")
    axes[0].grid(True, alpha=0.3)

    # Calinski-Harabasz scores
    axes[1].errorbar(
        metrics["n_clusters"],
        metrics["calinski_harabasz_mean"],
        yerr=metrics["calinski_harabasz_std"],
        marker="o",
        capsize=5,
    )
    axes[1].set_xlabel("Number of Clusters")
    axes[1].set_ylabel("Calinski-Harabasz Score")
    axes[1].set_title("Clustering Quality: Calinski-Harabasz Score")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"cluster_metrics.{plot_format}"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return output_path


def plot_cluster_maps(
    hex_grid: gpd.GeoDataFrame,
    labels: pd.Series,
    output_dir: Path,
    plot_format: str = "png",
    dpi: int = 300,
) -> Path:
    """
    Plot cluster assignments on map.

    Parameters
    ----------
    hex_grid : gpd.GeoDataFrame
        Hexagonal grid with geometry
    labels : pd.Series
        Cluster labels (index: cell_id)
    output_dir : Path
        Directory to save plots
    plot_format : str
        Format for saved plots (png, pdf, svg)
    dpi : int
        Resolution for saved plots

    Returns
    -------
    Path
        Path to saved plot
    """
    gdf = hex_grid.copy()
    gdf["cluster"] = labels

    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.plot(
        column="cluster",
        categorical=True,
        legend=True,
        alpha=0.5,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
        cmap="tab10",
    )
    ctx.add_basemap(ax=ax, crs=gdf.crs, attribution="")
    ax.set_title(f"Cell Clusters (n={labels.nunique()})")
    ax.set_axis_off()

    plt.tight_layout()
    output_path = output_dir / f"cluster_map.{plot_format}"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return output_path


def plot_feature_distributions(
    hex_grid: gpd.GeoDataFrame,
    cell_features: pd.DataFrame,
    top_n: int = 20,
    output_dir: Optional[Path] = None,
    plot_format: str = "png",
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot spatial distribution of top features.

    Parameters
    ----------
    hex_grid : gpd.GeoDataFrame
        Hexagonal grid with geometry
    cell_features : pd.DataFrame
        Cell features (index: cell_id, columns: features)
    top_n : int
        Number of features to plot (features with least missing values)
    output_dir : Path, optional
        Directory to save plots (if None, displays instead)
    plot_format : str
        Format for saved plots (png, pdf, svg)
    dpi : int
        Resolution for saved plots

    Returns
    -------
    Path or None
        Path to saved plot if output_dir provided, else None
    """
    # Select features with least missing values
    columns = cell_features.isna().mean().sort_values()[:top_n].index

    ncols = 4
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    gdf = hex_grid.copy()

    for ax, col in zip(axes, columns):
        gdf["feature"] = cell_features[col].fillna(0)
        gdf.plot(
            column="feature",
            alpha=0.5,
            edgecolor="white",
            linewidth=0.3,
            ax=ax,
            cmap="Reds",
            legend=True,
        )
        ctx.add_basemap(ax=ax, crs=gdf.crs, attribution="")
        ax.set_title(col, fontsize=10)
        ax.set_axis_off()

    # Hide unused subplots
    for ax in axes[len(columns) :]:
        ax.set_visible(False)

    plt.tight_layout()

    if output_dir:
        output_path = output_dir / f"feature_distributions.{plot_format}"
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return output_path
    else:
        plt.show()
        return None
