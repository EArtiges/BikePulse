"""Visualization functions for tensor factorization results."""
from itertools import cycle
from pathlib import Path
from typing import List, Optional

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorly as tl


def plot_rmse_errors(
    rank_range: List[int],
    errors: List[float],
    error_stds: List[float],
    output_dir: Optional[Path] = None,
    plot_format: str = "png",
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot RMSE errors across ranks.

    Parameters
    ----------
    rank_range : List[int]
        List of ranks
    errors : List[float]
        Mean RMSE errors
    error_stds : List[float]
        Standard deviations of RMSE
    output_dir : Path, optional
        Directory to save plot
    plot_format : str
        Plot format (png, pdf, svg)
    dpi : int
        Resolution for saved plots

    Returns
    -------
    Path or None
        Path to saved plot if output_dir provided
    """
    plt.figure(figsize=(10, 6))
    plt.errorbar(rank_range, errors, yerr=error_stds, marker="o", capsize=5)
    plt.xlabel("Rank")
    plt.ylabel("RMSE")
    plt.title("Reconstruction Error vs. Rank")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_dir:
        output_path = output_dir / f"rmse_by_rank.{plot_format}"
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def plot_ccc_scores(
    rank_range: List[int],
    ccc_h: List[float],
    ccc_w: List[float],
    output_dir: Optional[Path] = None,
    plot_format: str = "png",
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot Consensus Clustering Coefficient scores.

    Parameters
    ----------
    rank_range : List[int]
        List of ranks
    ccc_h : List[float]
        CCC scores for H (destination) matrix
    ccc_w : List[float]
        CCC scores for W (source) matrix
    output_dir : Path, optional
        Directory to save plot
    plot_format : str
        Plot format
    dpi : int
        Resolution

    Returns
    -------
    Path or None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rank_range, ccc_h, marker="o", label="H (Destination)")
    plt.plot(rank_range, ccc_w, marker="o", label="W (Source)")
    plt.xlabel("Rank")
    plt.ylabel("Consensus Clustering Coefficient")
    plt.title("Clustering Stability vs. Rank")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_dir:
        output_path = output_dir / f"ccc_by_rank.{plot_format}"
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def plot_error_tradeoff(
    normalized_rmse: List[float],
    ccc_h: List[float],
    ccc_w: List[float],
    output_dir: Optional[Path] = None,
    plot_format: str = "png",
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot RMSE vs CCC tradeoff.

    Parameters
    ----------
    normalized_rmse : List[float]
        Normalized RMSE errors (0-1)
    ccc_h : List[float]
        CCC scores for H matrix
    ccc_w : List[float]
        CCC scores for W matrix
    output_dir : Path, optional
        Directory to save plot
    plot_format : str
        Plot format
    dpi : int
        Resolution

    Returns
    -------
    Path or None
    """
    plt.figure(figsize=(8, 8))

    for color, ccc, label in zip(["b", "r"], [ccc_h, ccc_w], ["H", "W"]):
        plt.scatter(normalized_rmse, ccc, color=color, label=label)
        for i, (x, y) in enumerate(zip(normalized_rmse, ccc)):
            plt.annotate(i + 1, (x, y), color=color, fontsize=8)

    plt.xlabel("Normalized RMSE")
    plt.ylabel("Consensus Clustering Coefficient")
    plt.title("Error-Stability Tradeoff")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_dir:
        output_path = output_dir / f"error_tradeoff.{plot_format}"
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def plot_reconstruction_quality(
    tensor: tl.tensor,
    reconstructed: tl.tensor,
    output_dir: Optional[Path] = None,
    plot_format: str = "png",
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot reconstruction quality (scatter plot and error histogram).

    Parameters
    ----------
    tensor : tl.tensor
        Original tensor
    reconstructed : tl.tensor
        Reconstructed tensor
    output_dir : Path, optional
        Directory to save plot
    plot_format : str
        Plot format
    dpi : int
        Resolution

    Returns
    -------
    Path or None
    """
    error = tensor - reconstructed

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(tensor.flatten(), reconstructed.flatten(), alpha=0.2, s=1)
    max_val = max(tensor.max(), reconstructed.max())
    axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=2)
    axes[0].set_xlabel("Original Values")
    axes[0].set_ylabel("Reconstructed Values")
    axes[0].set_title("Reconstruction Quality")
    axes[0].grid(True, alpha=0.3)

    # Error histogram
    relative_error = (error / tensor.mean()).flatten()
    axes[1].hist(relative_error, bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Relative Error")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Relative Errors")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = output_dir / f"reconstruction_quality.{plot_format}"
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def plot_temporal_factors(
    temporal_factors: np.ndarray,
    trips_in_components: List[float],
    hours: List[int],
    output_dir: Optional[Path] = None,
    plot_format: str = "png",
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot temporal factors (hourly patterns).

    Parameters
    ----------
    temporal_factors : np.ndarray
        Temporal factor matrix (hours × components)
    trips_in_components : List[float]
        Percentage of trips in each component
    hours : List[int]
        Hour labels
    output_dir : Path, optional
        Directory to save plot
    plot_format : str
        Plot format
    dpi : int
        Resolution

    Returns
    -------
    Path or None
    """
    n_components = temporal_factors.shape[1]
    n_rows = int(np.ceil(np.sqrt(n_components))) + 1
    n_cols = int(np.ceil(n_components / n_rows))

    if n_rows * n_cols == n_components:
        n_rows += 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 1.8 * n_rows))
    axes = axes.flatten()

    colors = cycle(["Blue", "Green", "Purple", "Orange", "Red", "Grey"])

    for idx, (ax, temporal_pattern, n_trips, color) in enumerate(
        zip(axes, temporal_factors.T, trips_in_components, colors)
    ):
        ax.plot(hours, temporal_pattern, color=color, linewidth=2)
        ax.set_title(f"Component {idx}: {int(n_trips)}% of trips")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Factor Value")
        ax.grid(True, alpha=0.3)

        # Add to summary plot (last subplot)
        if idx != 0:
            axes[-1].plot(hours, temporal_pattern, color=color, alpha=0.7, label=f"C{idx}")

    # Configure summary plot
    axes[-1].set_title("All Components")
    axes[-1].set_xlabel("Hour")
    axes[-1].set_ylabel("Factor Value")
    axes[-1].legend(fontsize=8, ncol=2)
    axes[-1].grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes[n_components + 1 : -1]:
        ax.set_visible(False)

    plt.tight_layout()

    if output_dir:
        output_path = output_dir / f"temporal_factors.{plot_format}"
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def plot_spatial_factors(
    factors: pd.DataFrame,
    hex_grid: gpd.GeoDataFrame,
    component: int | str,
    output_dir: Optional[Path] = None,
    plot_format: str = "png",
    dpi: int = 300,
) -> Optional[Path]:
    """
    Plot spatial factor (flow maps: departures, arrivals, net flow).

    Parameters
    ----------
    factors : pd.DataFrame
        DataFrame with columns: start_cell, end_cell, component, trips
    hex_grid : gpd.GeoDataFrame
        Hexagonal grid with geometry
    component : int or str
        Component number to plot (or 'full' for sum)
    output_dir : Path, optional
        Directory to save plot
    plot_format : str
        Plot format
    dpi : int
        Resolution

    Returns
    -------
    Path or None
    """
    factor = factors[factors["component"] == component].copy()

    # Aggregate by start and end cells
    departures = factor.groupby("start_cell")["trips"].sum().rename("departures")
    arrivals = factor.groupby("end_cell")["trips"].sum().rename("arrivals")

    # Join with geometry
    gdf = pd.concat([departures, arrivals, hex_grid["geometry"]], axis=1).dropna()
    gdf = gpd.GeoDataFrame(gdf, crs=hex_grid.crs)
    gdf["net_flow"] = gdf["arrivals"] - gdf["departures"]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    gdf.plot(column="net_flow", cmap="RdBu_r", alpha=0.5, ax=axes[0], legend=True)
    gdf.plot(column="departures", cmap="Blues", alpha=0.5, ax=axes[1], legend=True)
    gdf.plot(column="arrivals", cmap="Reds", alpha=0.5, ax=axes[2], legend=True)

    axes[0].set_title(f"Component {component}: Net Flow")
    axes[1].set_title(f"Component {component}: Departures")
    axes[2].set_title(f"Component {component}: Arrivals")

    for ax in axes:
        ctx.add_basemap(ax=ax, crs=hex_grid.crs, attribution="")
        ax.axis("off")

    plt.tight_layout()

    if output_dir:
        output_path = output_dir / f"spatial_factor_component_{component}.{plot_format}"
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return output_path
    else:
        plt.show()
        return None
