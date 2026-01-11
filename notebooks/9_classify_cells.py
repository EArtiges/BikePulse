"""
Cell Classification Pipeline

Clusters cells based on engineered features using unsupervised learning.
Evaluates clustering quality and saves results.
"""
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Add parent directory to path for analysis module imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.clustering.classifiers import cluster_cells, evaluate_cluster_stability
from analysis.clustering.visualization import (
    plot_cluster_maps,
    plot_cluster_metrics,
    plot_feature_distributions,
)
from utils import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
city = "Edinburgh"
config = get_config(city)
analysis_config = config.get("analysis", {})
clustering_config = analysis_config.get("clustering", {})
output_config = analysis_config.get("output", {})

# Load data
logger.info(f"Loading data for {city}")
cell_features = pd.read_parquet(f"../data/{city}/cell_features.parquet")
cell_features.index.set_names("cell_id", inplace=True)
hex_grid = gpd.read_parquet(f"../data/{city}/hex_grid.geoparquet")

logger.info(f"Loaded {len(cell_features)} cells with {len(cell_features.columns)} features")

# Create output directory
results_dir = Path(
    output_config.get("results_dir", f"results/{city}").format(city=city)
)
clustering_dir = results_dir / "clustering"
plots_dir = clustering_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

logger.info("Plotting feature distributions")
plot_feature_distributions(
    hex_grid=hex_grid,
    cell_features=cell_features,
    output_dir=plots_dir,
    plot_format=output_config.get("plot_format", "png"),
    dpi=output_config.get("plot_dpi", 300),
)

# Evaluate cluster stability
logger.info("Evaluating cluster stability")
cluster_range_config = clustering_config.get("cluster_range", [2, 20])
cluster_range = range(cluster_range_config[0], cluster_range_config[1])

metrics = evaluate_cluster_stability(
    data=cell_features.fillna(0).values,
    cluster_range=cluster_range,
    n_iterations=clustering_config.get("n_iterations", 50),
    algorithm=clustering_config.get("algorithm", "kmeans"),
    normalization=clustering_config.get("normalization", "l1"),
    random_seed=clustering_config.get("random_seed", 42),
)

# Save metrics
logger.info("Saving evaluation metrics")
metrics_path = clustering_dir / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

# Plot metrics
logger.info("Plotting cluster quality metrics")
plot_cluster_metrics(
    metrics=metrics,
    output_dir=plots_dir,
    plot_format=output_config.get("plot_format", "png"),
    dpi=output_config.get("plot_dpi", 300),
)

# Perform final clustering
logger.info("Clustering cells")
n_clusters = clustering_config.get("n_clusters", 5)
labels, centroids = cluster_cells(
    cell_features=cell_features,
    n_clusters=n_clusters,
    algorithm=clustering_config.get("algorithm", "kmeans"),
    normalization=clustering_config.get("normalization", "l1"),
    random_seed=clustering_config.get("random_seed", 42),
)

logger.info(f"Created {n_clusters} clusters")

# Save cluster labels and centroids
logger.info("Saving cluster labels and centroids")
labels_path = clustering_dir / "labels.parquet"
labels.to_frame().to_parquet(labels_path)

centroids_path = clustering_dir / "centroids.parquet"
centroids.to_parquet(centroids_path)

# Also save as pickle for compatibility
labels.to_pickle(clustering_dir / "cell_clusters.pkl")
centroids.to_pickle(clustering_dir / "cluster_centers.pkl")

# Print top features for each cluster
logger.info("Top features per cluster:")
for c in range(n_clusters):
    top_features = centroids[c].sort_values(ascending=False).head(10)
    logger.info(f"  Cluster {c}: {top_features.to_dict()}")

# Plot cluster maps
logger.info("Plotting cluster maps")
plot_cluster_maps(
    hex_grid=hex_grid,
    labels=labels,
    output_dir=plots_dir,
    plot_format=output_config.get("plot_format", "png"),
    dpi=output_config.get("plot_dpi", 300),
)

logger.info(f"Clustering complete! Results saved to {clustering_dir}")
logger.info(f"  - Metrics: {metrics_path}")
logger.info(f"  - Labels: {labels_path}")
logger.info(f"  - Centroids: {centroids_path}")
logger.info(f"  - Plots: {plots_dir}")
