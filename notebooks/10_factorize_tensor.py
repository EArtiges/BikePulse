"""
Tensor Factorization Pipeline

Applies tensor decomposition to OD matrices to discover latent mobility patterns.
Evaluates optimal rank and visualizes results.
"""
import json
import logging
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

# Add parent directory to path for analysis module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
from analysis.factorization.visualization import (
    plot_reconstruction_quality,
    plot_rmse_errors,
    plot_spatial_factors,
    plot_temporal_factors,
)
from utils import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
city = "Oslo"
config = get_config(city)
analysis_config = config.get("analysis", {})
factorization_config = analysis_config.get("factorization", {})
output_config = analysis_config.get("output", {})

# Load data
logger.info(f"Loading OD matrix for {city}")
od_matrix = pd.read_pickle(f"../data/{city}/cell_OD.pkl")
hex_grid = gpd.read_parquet(f"../data/{city}/hex_grid.geoparquet")

logger.info(f"OD matrix shape: {od_matrix.shape}")

# Create output directory
results_dir = Path(
    output_config.get("results_dir", f"results/{city}").format(city=city)
)
factorization_dir = results_dir / "factorization"
factors_dir = factorization_dir / "factors"
plots_dir = factorization_dir / "plots"
factors_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

# Prepare tensor
logger.info("Preparing OD tensor")
tensor, hours, cells = prepare_od_tensor(
    od_matrix=od_matrix,
    weekdays_only=factorization_config.get("weekdays_only", True),
    month_range=tuple(factorization_config.get("month_range", [1, 12])),
    normalize_by_year=True,
)

logger.info(f"Tensor shape: {tensor.shape}")
logger.info(f"  Hours: {len(hours)} (from {min(hours)} to {max(hours)})")
logger.info(f"  Cells: {len(cells)}")

# Determine method and rank
method = factorization_config.get("method", "tucker")
rank_config = factorization_config.get("rank", [3, 5, 5])

# Optionally evaluate RMSE across ranks to find optimal
evaluate_rank = factorization_config.get("evaluate_rank", False)
if evaluate_rank:
    logger.info("Evaluating RMSE across different ranks")
    rank_range = range(1, 31)
    rank_list, errors, error_stds = evaluate_rmse_by_rank(
        tensor=tensor,
        method=method,
        rank_range=rank_range,
        n_runs=10,
        init="svd",
    )

    # Plot RMSE errors
    plot_rmse_errors(
        rank_range=rank_list,
        errors=errors,
        error_stds=error_stds,
        output_dir=plots_dir,
        plot_format=output_config.get("plot_format", "png"),
        dpi=output_config.get("plot_dpi", 300),
    )

    # Save RMSE results
    rmse_results = {
        "rank": rank_list,
        "errors": errors,
        "error_stds": error_stds,
    }
    with open(factorization_dir / "rmse_evaluation.json", "w") as f:
        json.dump(rmse_results, f, indent=2)

# Perform factorization
logger.info(f"Performing {method} factorization with rank {rank_config}")
result = factorize_tensor(
    tensor=tensor,
    method=method,
    rank=tuple(rank_config) if isinstance(rank_config, list) else rank_config,
    random_seed=factorization_config.get("random_seed", 42),
    init="svd",
)

logger.info("Factorization complete")

# Reconstruct tensor
logger.info("Reconstructing tensor")
reconstructed = reconstruct_tensor(result)

# Compute reconstruction metrics
logger.info("Computing reconstruction metrics")
metrics = compute_reconstruction_metrics(tensor, reconstructed)
logger.info(f"  RMSE: {metrics['rmse']:.4f}")
logger.info(f"  MAE: {metrics['mae']:.4f}")
logger.info(f"  R²: {metrics['r_squared']:.4f}")

# Save metrics
with open(factorization_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save factorization results
logger.info("Saving factorization results")
with open(factors_dir / "factorization_result.pkl", "wb") as f:
    pickle.dump(result, f)

# Plot reconstruction quality
logger.info("Plotting reconstruction quality")
plot_reconstruction_quality(
    tensor=tensor,
    reconstructed=reconstructed,
    output_dir=plots_dir,
    plot_format=output_config.get("plot_format", "png"),
    dpi=output_config.get("plot_dpi", 300),
)

# Extract spatial factors
logger.info("Extracting spatial factors")
spatial_factors = extract_spatial_factors(result, cells)

# Save spatial factors
spatial_factors.to_parquet(factors_dir / "spatial_factors.parquet")

# Join with geometry for visualization
spatial_factors_geo = spatial_factors.copy()
spatial_factors_geo = spatial_factors_geo.merge(
    hex_grid[["geometry"]].rename_axis("start_cell").reset_index(),
    on="start_cell",
    how="left",
    suffixes=("", "_start"),
)
spatial_factors_geo = spatial_factors_geo.merge(
    hex_grid[["geometry"]].rename_axis("end_cell").reset_index(),
    on="end_cell",
    how="left",
    suffixes=("", "_end"),
)
spatial_factors_geo = gpd.GeoDataFrame(spatial_factors_geo)

# Extract temporal factors
temporal_factors = result["factors"][0]  # First dimension is time

# Compute trips in components
if method == "parafac":
    W, H = result["factors"][1], result["factors"][2]
    trips_in_components = [
        np.outer(np.outer(w, h), t).sum()
        for w, h, t in zip(W.T, H.T, temporal_factors.T)
    ]
else:  # tucker
    W, H = result["factors"][1], result["factors"][2]
    trips_in_components = [
        np.outer(np.outer(w, h), t).sum()
        for w, h, t in zip(W.T, H.T, temporal_factors.T)
    ]

# Plot temporal factors
logger.info("Plotting temporal factors")
plot_temporal_factors(
    temporal_factors=temporal_factors,
    trips_in_components=trips_in_components,
    hours=hours,
    output_dir=plots_dir,
    plot_format=output_config.get("plot_format", "png"),
    dpi=output_config.get("plot_dpi", 300),
)

# Plot spatial factors for each component
logger.info("Plotting spatial factors")
n_components = (
    result["rank"] if isinstance(result["rank"], int) else result["rank"][0]
)
for component in range(min(n_components, 10)):  # Limit to first 10 components
    logger.info(f"  Component {component}")
    plot_spatial_factors(
        factors=spatial_factors_geo,
        hex_grid=hex_grid,
        component=component,
        output_dir=plots_dir,
        plot_format=output_config.get("plot_format", "png"),
        dpi=output_config.get("plot_dpi", 300),
    )

# Plot full factor (sum of all components)
logger.info("Plotting full factor (sum of all components)")
plot_spatial_factors(
    factors=spatial_factors_geo,
    hex_grid=hex_grid,
    component="full",
    output_dir=plots_dir,
    plot_format=output_config.get("plot_format", "png"),
    dpi=output_config.get("plot_dpi", 300),
)

logger.info(f"Tensor factorization complete! Results saved to {factorization_dir}")
logger.info(f"  - Metrics: {factorization_dir / 'metrics.json'}")
logger.info(f"  - Factors: {factors_dir}")
logger.info(f"  - Plots: {plots_dir}")
