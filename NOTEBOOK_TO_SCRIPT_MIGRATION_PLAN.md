# Notebook to Script Migration Plan

**Date:** 2026-01-11
**Status:** Planning Phase
**Goal:** Convert remaining Jupyter notebooks to production-ready Python scripts

---

## Executive Summary

The BikePulse pipeline is **already 90% script-based**. Only 2 analysis notebooks remain:
- `cell_classifier.ipynb` - Cell clustering and classification
- `factors.ipynb` - Tensor factorization analysis

This plan outlines a modular, testable, and maintainable approach to complete the migration.

---

## Current State Analysis

### Already Script-Based ✅
- **Pipeline scripts (1a-8):** All data collection and feature engineering
- **Library modules:** `oslo_lib.py`, `geo_utils.py`, `CCC.py`, `utils.py`
- **Tests:** Comprehensive test suite with CI/CD

### Needs Conversion 🔄
- **cell_classifier.ipynb** (6 cells, ~227 lines)
  - Feature visualization
  - Clustering with KMeans
  - Silhouette and Calinski-Harabasz scoring
  - Result visualization on maps

- **factors.ipynb** (25 cells, ~1156 lines)
  - OD matrix preparation
  - Tensor factorization (Tucker, PARAFAC)
  - Consensus clustering validation
  - Factor visualization and interpretation

---

## Design Principles

### 1. **Separation of Concerns**
- **Data processing** → Pure functions, no side effects
- **Visualization** → Separate module with save-to-file functions
- **Configuration** → YAML-based parameters
- **Execution** → CLI scripts that orchestrate

### 2. **Testability**
- Unit tests for pure functions
- Integration tests for full pipeline
- Test fixtures for sample data

### 3. **Reproducibility**
- Deterministic results (set random seeds)
- Version-controlled configuration
- Logged parameters and outputs

### 4. **Modularity**
- Reusable components
- Clear function boundaries
- Minimal coupling

---

## Proposed Architecture

### Directory Structure

```
BikePulse/
├── analysis/                          # NEW: Analysis module
│   ├── __init__.py
│   ├── clustering/                    # Cell classification
│   │   ├── __init__.py
│   │   ├── classifiers.py            # Clustering algorithms
│   │   ├── metrics.py                # Evaluation metrics
│   │   └── visualization.py          # Plotting functions
│   ├── factorization/                 # Tensor analysis
│   │   ├── __init__.py
│   │   ├── decomposition.py          # Tucker, PARAFAC
│   │   ├── consensus.py              # Bootstrap validation
│   │   └── visualization.py          # Factor plots
│   └── config.py                      # Configuration schemas
│
├── notebooks/                         # Pipeline scripts
│   ├── 1a_collect_bike_trips.py
│   ├── ...
│   ├── 8_compute_cell_features.py
│   ├── 9_classify_cells.py           # NEW: Clustering script
│   ├── 10_factorize_tensor.py        # NEW: Factorization script
│   ├── oslo_lib.py
│   ├── geo_utils.py
│   ├── CCC.py
│   ├── utils.py
│   └── run.yml                        # Extended with analysis params
│
├── results/                           # NEW: Output directory (gitignored)
│   └── {city}/
│       ├── clustering/
│       │   ├── plots/                # Saved visualizations
│       │   ├── metrics.json          # Evaluation scores
│       │   └── labels.parquet        # Cluster assignments
│       └── factorization/
│           ├── plots/                # Factor visualizations
│           ├── factors/              # Saved tensor factors
│           ├── metrics.json          # RMSE, consensus scores
│           └── summary.md            # Human-readable report
│
├── examples/                          # NEW: Optional viewing notebooks
│   ├── view_clustering_results.ipynb
│   └── view_factorization_results.ipynb
│
├── tests/
│   ├── test_clustering.py            # NEW: Clustering tests
│   └── test_factorization.py         # NEW: Factorization tests
│
├── cell_classifier.ipynb             # ARCHIVE (move to old_notebooks/)
└── factors.ipynb                     # ARCHIVE (move to old_notebooks/)
```

---

## Implementation Plan

### Phase 1: Setup & Infrastructure (Day 1)

#### 1.1 Create Module Structure
```bash
mkdir -p analysis/clustering analysis/factorization
touch analysis/__init__.py
touch analysis/clustering/__init__.py
touch analysis/factorization/__init__.py
mkdir -p results examples
```

#### 1.2 Extend Configuration
Add to `notebooks/run.yml`:
```yaml
Oslo:
  # ... existing config ...

  analysis:
    clustering:
      algorithm: kmeans
      n_clusters: 5
      cluster_range: [2, 20]
      n_iterations: 50  # For stability scoring
      normalization: l1  # l1, l2, or standard
      random_seed: 42

    factorization:
      method: tucker  # tucker or parafac
      rank: [3, 5, 5]  # [temporal, spatial_source, spatial_dest]
      weekdays_only: true
      month_range: [1, 12]
      n_bootstraps: 100
      consensus_threshold: 0.5
      random_seed: 42

    output:
      results_dir: results/{city}
      save_plots: true
      plot_format: png
      plot_dpi: 300
```

#### 1.3 Update .gitignore
```
# Add to .gitignore
results/
!results/.gitkeep
```

---

### Phase 2: Extract Cell Clustering (Days 2-3)

#### 2.1 Create `analysis/clustering/classifiers.py`

```python
"""Cell clustering algorithms."""
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def create_clustering_pipeline(
    algorithm: str = "kmeans",
    normalization: str = "l1",
    n_clusters: int = 5,
    random_seed: int = 42
) -> Pipeline:
    """Create sklearn pipeline for clustering."""
    normalizers = {
        "l1": Normalizer("l1"),
        "l2": Normalizer("l2"),
        "standard": StandardScaler(),
    }

    return Pipeline([
        ("normalize", normalizers[normalization]),
        ("cluster", KMeans(n_clusters=n_clusters, random_state=random_seed)),
    ])


def evaluate_cluster_stability(
    data: np.ndarray,
    cluster_range: range,
    n_iterations: int = 50,
    **pipeline_kwargs
) -> Dict[str, List[float]]:
    """Evaluate clustering stability across different k values."""
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

            sil_scores.append(silhouette_score(data, labels))
            ch_scores.append(calinski_harabasz_score(data, labels))

        results["n_clusters"].append(n_clusters)
        results["silhouette_mean"].append(np.mean(sil_scores))
        results["silhouette_std"].append(np.std(sil_scores))
        results["calinski_harabasz_mean"].append(np.mean(ch_scores))
        results["calinski_harabasz_std"].append(np.std(ch_scores))

    return results


def cluster_cells(
    cell_features: pd.DataFrame,
    n_clusters: int = 5,
    **pipeline_kwargs
) -> pd.Series:
    """Cluster cells based on features."""
    # Handle missing values
    data = cell_features.fillna(0).values

    # Create and fit pipeline
    pipeline = create_clustering_pipeline(n_clusters=n_clusters, **pipeline_kwargs)
    labels = pipeline.fit_predict(data)

    return pd.Series(labels, index=cell_features.index, name="cluster")
```

#### 2.2 Create `analysis/clustering/visualization.py`

```python
"""Visualization functions for clustering results."""
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
import pandas as pd


def plot_cluster_metrics(
    metrics: dict,
    output_dir: Path,
    plot_format: str = "png",
    dpi: int = 300
):
    """Plot silhouette and Calinski-Harabasz scores."""
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
    dpi: int = 300
):
    """Plot cluster assignments on map."""
    gdf = hex_grid.copy()
    gdf["cluster"] = labels

    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.plot(
        column="cluster",
        categorical=True,
        legend=True,
        alpha=0.7,
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
    output_dir: Path,
    plot_format: str = "png",
    dpi: int = 300
):
    """Plot spatial distribution of top features."""
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
            alpha=0.7,
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
    for ax in axes[len(columns):]:
        ax.set_visible(False)

    plt.tight_layout()
    output_path = output_dir / f"feature_distributions.{plot_format}"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return output_path
```

#### 2.3 Create `notebooks/9_classify_cells.py`

```python
"""
Cell Classification Pipeline

Clusters cells based on engineered features using unsupervised learning.
Evaluates clustering quality and saves results.
"""
import json
import logging
from pathlib import Path
import pandas as pd
import geopandas as gpd
from utils import get_config

# Import analysis modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.clustering.classifiers import (
    evaluate_cluster_stability,
    cluster_cells,
)
from analysis.clustering.visualization import (
    plot_cluster_metrics,
    plot_cluster_maps,
    plot_feature_distributions,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
cell_features = pd.read_parquet(f"data/{city}/cell_features.parquet")
cell_features.index.set_names("cell_id", inplace=True)
hex_grid = gpd.read_parquet(f"data/{city}/hex_grid.geoparquet")

# Create output directory
results_dir = Path(output_config.get("results_dir", f"results/{city}").format(city=city))
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
cluster_range = range(*clustering_config.get("cluster_range", [2, 20]))
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
plot_cluster_metrics(
    metrics=metrics,
    output_dir=plots_dir,
    plot_format=output_config.get("plot_format", "png"),
    dpi=output_config.get("plot_dpi", 300),
)

# Perform final clustering
logger.info("Clustering cells")
n_clusters = clustering_config.get("n_clusters", 5)
labels = cluster_cells(
    cell_features=cell_features,
    n_clusters=n_clusters,
    algorithm=clustering_config.get("algorithm", "kmeans"),
    normalization=clustering_config.get("normalization", "l1"),
    random_seed=clustering_config.get("random_seed", 42),
)

# Save cluster labels
logger.info("Saving cluster labels")
labels_path = clustering_dir / "labels.parquet"
labels.to_frame().to_parquet(labels_path)

# Plot cluster maps
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
logger.info(f"  - Plots: {plots_dir}")
```

---

### Phase 3: Extract Tensor Factorization (Days 4-6)

#### 3.1 Create `analysis/factorization/decomposition.py`

```python
"""Tensor decomposition algorithms."""
import tensorly as tl
import numpy as np
import pandas as pd
from typing import Tuple, Literal


def prepare_od_tensor(
    od_matrix: pd.DataFrame,
    weekdays_only: bool = True,
    month_range: Tuple[int, int] = (1, 12),
) -> Tuple[tl.tensor, list]:
    """Prepare OD matrix as tensor for factorization."""
    # Filter by weekday
    if weekdays_only:
        od_matrix = od_matrix.query("weekday < 5")
    else:
        od_matrix = od_matrix.query("weekday >= 5")

    # Filter by month
    min_month, max_month = month_range
    od_matrix = od_matrix.query(f"month >= {min_month} & month <= {max_month}")

    # Normalize and aggregate
    od_matrix = od_matrix.drop("weekday", axis=1, errors="ignore")
    dataset = (
        od_matrix.groupby(["hour", "start_cell", "end_cell"])
        .trip.mean()
        .reset_index()
    )

    # Pivot to matrix form
    dataset = dataset.pivot(
        index=["hour", "start_cell"],
        columns="end_cell",
        values="trip"
    ).fillna(0)

    # Normalize to percentage
    dataset = 100 * dataset / dataset.sum().sum()

    # Convert to tensor
    hours = sorted(dataset.index.get_level_values(0).unique())
    time_slices = [dataset.loc[h] for h in hours]
    tensor_array = np.array([
        ts.reindex(ts.columns).fillna(0) for ts in time_slices
    ])

    tensor = tl.tensor(data=tensor_array.astype("float64"))

    return tensor, hours


def factorize_tensor(
    tensor: tl.tensor,
    method: Literal["tucker", "parafac"] = "tucker",
    rank: Tuple[int, int, int] = (3, 5, 5),
    random_seed: int = 42,
) -> dict:
    """Perform tensor factorization."""
    tl.set_backend("numpy")
    np.random.seed(random_seed)

    if method == "tucker":
        from tensorly.decomposition import tucker
        core, factors = tucker(tensor, rank=rank, random_state=random_seed)
        return {
            "method": "tucker",
            "core": core,
            "factors": factors,
            "rank": rank,
        }
    elif method == "parafac":
        from tensorly.decomposition import parafac
        factors = parafac(tensor, rank=rank[0], random_state=random_seed)
        return {
            "method": "parafac",
            "factors": factors.factors,
            "weights": factors.weights,
            "rank": rank[0],
        }
    else:
        raise ValueError(f"Unknown method: {method}")
```

#### 3.2 Create `notebooks/10_factorize_tensor.py`

```python
"""
Tensor Factorization Pipeline

Applies tensor decomposition to OD matrices to discover latent mobility patterns.
"""
import json
import logging
from pathlib import Path
import pandas as pd
import geopandas as gpd
from utils import get_config

# Import analysis modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from analysis.factorization.decomposition import (
    prepare_od_tensor,
    factorize_tensor,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
od_matrix = pd.read_pickle(f"data/{city}/cell_OD.pkl")
hex_grid = gpd.read_parquet(f"data/{city}/hex_grid.geoparquet")

# Create output directory
results_dir = Path(output_config.get("results_dir", f"results/{city}").format(city=city))
factorization_dir = results_dir / "factorization"
factors_dir = factorization_dir / "factors"
plots_dir = factorization_dir / "plots"
factors_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

# Prepare tensor
logger.info("Preparing OD tensor")
tensor, hours = prepare_od_tensor(
    od_matrix=od_matrix,
    weekdays_only=factorization_config.get("weekdays_only", True),
    month_range=tuple(factorization_config.get("month_range", [1, 12])),
)

logger.info(f"Tensor shape: {tensor.shape}")

# Perform factorization
logger.info("Performing tensor factorization")
result = factorize_tensor(
    tensor=tensor,
    method=factorization_config.get("method", "tucker"),
    rank=tuple(factorization_config.get("rank", [3, 5, 5])),
    random_seed=factorization_config.get("random_seed", 42),
)

# Save factors
logger.info("Saving factorization results")
import pickle
with open(factors_dir / "factorization_result.pkl", "wb") as f:
    pickle.dump(result, f)

logger.info(f"Factorization complete! Results saved to {factorization_dir}")
```

---

### Phase 4: Testing (Day 7)

Create comprehensive tests:

```python
# tests/test_clustering.py
def test_create_clustering_pipeline():
    """Test pipeline creation."""
    from analysis.clustering.classifiers import create_clustering_pipeline

    pipeline = create_clustering_pipeline(n_clusters=3)
    assert len(pipeline.steps) == 2
    assert pipeline.named_steps["cluster"].n_clusters == 3


def test_cluster_cells():
    """Test clustering with sample data."""
    import pandas as pd
    from analysis.clustering.classifiers import cluster_cells

    # Create sample data
    features = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
    })

    labels = cluster_cells(features, n_clusters=2)

    assert len(labels) == 5
    assert labels.nunique() <= 2
```

---

### Phase 5: Documentation Updates (Day 8)

Update `CLAUDE.md`:

```markdown
## Analysis Scripts

### 9. Cell Classification (`9_classify_cells.py`)
- **Input:** `data/{city}/cell_features.parquet`, `data/{city}/hex_grid.geoparquet`
- **Process:**
  - Evaluates clustering stability across different k values
  - Performs final clustering with optimal k
  - Generates feature distribution maps
  - Saves cluster assignments and visualizations
- **Output:** `results/{city}/clustering/`
- **Configuration:** `run.yml` → `analysis.clustering`

### 10. Tensor Factorization (`10_factorize_tensor.py`)
- **Input:** `data/{city}/cell_OD.pkl`, `data/{city}/hex_grid.geoparquet`
- **Process:**
  - Prepares OD tensor with filtering
  - Applies Tucker or PARAFAC decomposition
  - Validates with consensus clustering
  - Visualizes factors and patterns
- **Output:** `results/{city}/factorization/`
- **Configuration:** `run.yml` → `analysis.factorization`
```

---

## Migration Checklist

### Pre-Migration
- [ ] Back up existing notebooks
- [ ] Review notebook content thoroughly
- [ ] Document all parameters and assumptions

### Phase 1: Setup
- [ ] Create `analysis/` module structure
- [ ] Extend `run.yml` with analysis configuration
- [ ] Update `.gitignore` for `results/`
- [ ] Create `results/.gitkeep` files

### Phase 2: Cell Clustering
- [ ] Extract clustering logic to `analysis/clustering/`
- [ ] Create `9_classify_cells.py` script
- [ ] Test on sample data
- [ ] Validate output matches notebook
- [ ] Add unit tests

### Phase 3: Tensor Factorization
- [ ] Extract factorization logic to `analysis/factorization/`
- [ ] Create `10_factorize_tensor.py` script
- [ ] Test on sample data
- [ ] Validate output matches notebook
- [ ] Add unit tests

### Phase 4: Testing & Validation
- [ ] Run full pipeline on all cities
- [ ] Compare outputs with notebook outputs
- [ ] Add integration tests
- [ ] Performance testing

### Phase 5: Documentation
- [ ] Update CLAUDE.md
- [ ] Update README.md
- [ ] Create usage examples
- [ ] Document configuration options

### Post-Migration
- [ ] Move old notebooks to `old_notebooks/`
- [ ] Create optional viewing notebooks in `examples/`
- [ ] Update CI/CD for new scripts
- [ ] Team review and feedback

---

## Benefits of Script-Based Approach

### Version Control
- ✅ Clean diffs (no notebook JSON noise)
- ✅ Easy code review
- ✅ Better merge conflict resolution

### Reproducibility
- ✅ Deterministic execution
- ✅ Version-controlled configuration
- ✅ Logged parameters and outputs

### Testing
- ✅ Unit testable functions
- ✅ Integration tests
- ✅ CI/CD compatible

### Modularity
- ✅ Reusable components
- ✅ Clear separation of concerns
- ✅ Easier refactoring

### Production Ready
- ✅ CLI execution
- ✅ Logging and monitoring
- ✅ Error handling
- ✅ Scalable architecture

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 1. Setup | 1 day | - |
| 2. Cell Clustering | 2-3 days | Phase 1 |
| 3. Tensor Factorization | 3-4 days | Phase 1 |
| 4. Testing | 1-2 days | Phases 2-3 |
| 5. Documentation | 1 day | All phases |
| **Total** | **8-11 days** | - |

---

## Next Steps

1. Review this plan
2. Get stakeholder approval
3. Create feature branch: `feature/notebook-to-script-migration`
4. Begin Phase 1
5. Iterate with code reviews after each phase

---

**Questions? Concerns?** This is a living document - update as needed during implementation.
