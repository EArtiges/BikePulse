# CLAUDE.md - BikePulse Codebase Guide for AI Assistants

**Last Updated:** 2026-01-11
**Project:** BikePulse - Urban Bike-Sharing Network Analysis
**Status:** Active Development

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Codebase Structure](#codebase-structure)
3. [Technology Stack](#technology-stack)
4. [Development Setup](#development-setup)
5. [CI/CD & Code Quality](#cicd--code-quality)
6. [Data Pipeline Workflow](#data-pipeline-workflow)
7. [Configuration Management](#configuration-management)
8. [Code Conventions](#code-conventions)
9. [Key Modules Reference](#key-modules-reference)
10. [Common Tasks](#common-tasks)
11. [Known Issues & Gotchas](#known-issues--gotchas)
12. [AI Assistant Guidelines](#ai-assistant-guidelines)

---

## Project Overview

### What is BikePulse?

BikePulse is a **data science pipeline** for analyzing urban bike-sharing systems across multiple cities. It's not a web or mobile application, but rather a computational research tool for urban mobility analysis.

**Core Capabilities:**
- Collect bike trip data from Urban Sharing APIs
- Perform geospatial feature engineering using H3 hexagonal grids
- Analyze origin-destination patterns and temporal dynamics
- Apply machine learning for cell classification and trip prediction
- Use tensor factorization to discover latent mobility patterns

**Supported Cities:** Oslo (Norway), Milan (Italy), Edinburgh (Scotland)

**Research Focus:**
- Understanding urban mobility patterns
- Identifying distinct urban typologies through clustering
- Predicting bike-sharing demand based on urban features
- Discovering interpretable factors in travel behavior

---

## Codebase Structure

```
BikePulse/
├── notebooks/                          # Main pipeline scripts (run in sequence)
│   ├── 1a_collect_bike_trips.py       # Collect raw trip data from APIs
│   ├── 1b_collect_stations.py         # Extract station locations from trips
│   ├── 2_collect_area.py              # Define geographic area boundaries (OSM)
│   ├── 3_build_grid.py                # Create H3 hexagonal grid system
│   ├── 4_preprocess_population.py     # Prepare population raster data
│   ├── 5_collect_population.py        # Assign population to grid cells
│   ├── 6_collect_POIs.py              # Collect Points of Interest (OSM)
│   ├── 7_compute_OD_matrix.py         # Compute origin-destination trip matrix
│   ├── 8_compute_cell_features.py     # Engineer features for cells
│   ├── oslo_lib.py                    # Core data processing library (182 lines)
│   ├── geo_utils.py                   # Geospatial utility functions (44 lines)
│   ├── CCC.py                         # Consensus Clustering Coefficient (81 lines)
│   ├── utils.py                       # Configuration loader (7 lines)
│   └── run.yml                        # City-specific configurations
│
├── cell_classifier.ipynb              # ML model for cell classification
├── factors.ipynb                      # Tensor factorization analysis (main notebook)
│
├── old_notebooks/                     # Legacy Jupyter notebooks (for reference)
│   ├── 1_Collect.ipynb
│   ├── 2_Explore.ipynb
│   ├── 3_Landmarks.ipynb
│   ├── 4_Connect.ipynb
│   └── 5_Model.ipynb
│
├── data/                              # Data storage (gitignored)
│   ├── {city}/                        # City-specific data directories
│   │   ├── trips/                     # Trip data (pkl, geoparquet)
│   │   ├── POIs/                      # Points of Interest
│   │   └── *.geoparquet               # Grids, features, etc.
│   └── global/                        # Global datasets (population)
│       └── GHS_POP/                   # Population rasters
│
├── cache/                             # Cached results (gitignored, 17 JSON files)
├── .venv/                             # Python virtual environment (gitignored)
├── requirements.txt                   # Python dependencies (84 packages)
├── README.md                          # High-level project plan
├── .gitignore                         # Excludes: .venv/, cache/, data/
└── CLAUDE.md                          # This file
```

### Recent Restructuring (Dec 2025)

The project recently underwent a major refactoring:
- **Before:** Jupyter notebook-based workflow in `app/` directory
- **After:** Script-based pipeline in `notebooks/` directory
- **Rationale:** Improved reproducibility and version control
- **Legacy Code:** Retained in `old_notebooks/` for reference

---

## Technology Stack

### Core Dependencies

#### Geospatial Analysis (Primary Focus)
```
geopandas==1.1.0          # Spatial dataframes
shapely==2.1.1            # Geometric operations
osmnx==2.0.3              # OpenStreetMap data extraction
h3==4.2.2                 # H3 hexagonal grid library
h3pandas==0.3.0           # H3 integration with pandas
rasterio==1.4.3           # Raster data I/O
rioxarray==0.19.0         # Raster arrays with xarray
pyproj==3.7.1             # Coordinate system transformations
GDAL==3.11.0              # Geospatial Data Abstraction Library
geopy==2.4.1              # Geocoding services
contextily==1.6.2         # Basemap tiles for visualization
mercantile==1.2.1         # Web mercator tile utilities
```

#### Data Processing
```
pandas==2.3.0             # Tabular data manipulation
numpy==2.3.0              # Numerical computing
xarray==2025.4.0          # Multi-dimensional labeled arrays
pyarrow==20.0.0           # Columnar data format (parquet)
```

#### Machine Learning & Math
```
scikit-learn==1.7.0       # ML algorithms (classification, clustering)
tensorly==0.9.0           # Tensor decomposition (Tucker, PARAFAC)
scipy==1.15.3             # Scientific computing
networkx==3.5             # Graph analysis
scikit-image==0.25.2      # Image processing
```

#### Visualization
```
matplotlib==3.10.3        # Static plotting
```

#### Development & Utilities
```
jupyter ecosystem         # Interactive notebooks
PyYAML==6.0.2            # Configuration files
requests==2.32.3         # HTTP API calls
```

### System Dependencies

**Critical:** This project requires **GDAL** and **PROJ** system libraries. These must be installed at the OS level before `pip install`.

**Platform:** Developed on Linux (4.4.0), but should work on macOS with appropriate GDAL installation.

---

## Development Setup

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Directory Structure

Ensure the following directories exist (created automatically by scripts):

```bash
data/
├── Oslo/
│   ├── trips/
│   └── POIs/
├── Milan/
│   ├── trips/
│   └── POIs/
├── Edinburgh/
│   ├── trips/
│   └── POIs/
└── global/
    └── GHS_POP/
```

### 3. Configuration

City configurations are defined in `notebooks/run.yml`. Each city requires:
- Area boundary file path
- H3 grid resolution (typically 8)
- Coordinate Reference System (CRS/EPSG code)
- Population data paths

### 4. Verify Setup

```bash
cd notebooks
python -c "import geopandas, osmnx, h3, tensorly; print('All imports successful')"
```

---

## CI/CD & Code Quality

The project uses automated code quality checks and continuous integration to maintain high standards.

### Pre-commit Hooks

**Setup:**

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

**Hooks configured:**
- **black**: Automatic code formatting (100 char line length)
- **ruff**: Fast Python linting (replaces flake8)
- **isort**: Import statement sorting
- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure files end with newline
- **check-yaml**: Validate YAML syntax
- **check-added-large-files**: Prevent files >5MB
- **nbstripout**: Clear Jupyter notebook outputs

**Running manually:**

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Update hook versions
pre-commit autoupdate
```

### Code Quality Tools

All tools are configured in `pyproject.toml`:

#### Black (Code Formatting)
```bash
# Format all Python files
black notebooks/ *.py

# Check without modifying
black --check notebooks/
```

#### Ruff (Linting)
```bash
# Lint and auto-fix
ruff check --fix notebooks/ *.py

# Check only
ruff check notebooks/
```

#### isort (Import Sorting)
```bash
# Sort imports
isort notebooks/ *.py

# Check only
isort --check-only notebooks/
```

**Configuration highlights:**
- Line length: 100 characters
- Python target: 3.9+
- Excludes: `.venv/`, `cache/`, `data/`, `old_notebooks/`
- Mathematical notation exceptions: Allows uppercase variables (T for tensors, N for counts)

### Testing

**Run tests:**

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=notebooks --cov-report=term

# Run specific test file
pytest tests/test_utils.py

# Run tests in parallel
pytest -n auto
```

**Test structure:**
```
tests/
├── __init__.py
├── conftest.py           # Pytest configuration and fixtures
├── test_utils.py         # Configuration and utility tests
└── test_geo_utils.py     # Geospatial utility tests
```

**Test categories:**
- `@pytest.mark.unit`: Fast unit tests
- `@pytest.mark.integration`: Integration tests (slower)
- `@pytest.mark.slow`: Long-running tests

**Run specific categories:**
```bash
pytest -m unit              # Only unit tests
pytest -m "not slow"        # Skip slow tests
```

### GitHub Actions CI

**Workflows:** `.github/workflows/ci.yml`

**Triggered on:**
- Pull requests to `main` or `develop`
- Pushes to `main` or `develop`

**Jobs:**

1. **lint** - Code Quality & Linting
   - Runs black, ruff, isort
   - Fast feedback on code style
   - No external dependencies needed

2. **pipeline-validation** - Pipeline Validation (Smoke Tests)
   - Validates all pipeline scripts exist and are syntactically correct
   - Checks library modules can be imported
   - Verifies pipeline structure and configuration files
   - Fast: no heavy dependencies required

3. **environment-check** - Geospatial Environment Check
   - Verifies GDAL/PROJ installation compatibility
   - Tests all critical imports (geopandas, osmnx, h3, tensorly)
   - Validates CRS support and H3 operations
   - Catches environment setup issues early

4. **test** - Unit Tests
   - Runs unit tests for utilities and geospatial functions
   - Installs GDAL system dependencies
   - Runs pytest with coverage
   - Uploads coverage to Codecov

5. **documentation** - Documentation Validation
   - Validates CLAUDE.md exists and is up-to-date
   - Checks all pipeline scripts are documented
   - Verifies file references in documentation
   - Ensures CI/CD setup is documented

6. **security** - Security Scan
   - Runs `safety` to check for vulnerable dependencies
   - Uses TruffleHog to detect secrets in commits

7. **notebook-quality** - Notebook Quality Check
   - Ensures notebooks have outputs cleared
   - Checks for hardcoded absolute paths

**Status badges:** Add to README.md:
```markdown
![CI](https://github.com/EArtiges/BikePulse/workflows/CI/badge.svg)
```

### Dependabot

**Configuration:** `.github/dependabot.yml`

**Features:**
- Weekly dependency updates (Mondays at 9 AM)
- Groups updates by category:
  - Geospatial: geopandas, shapely, osmnx, h3, rasterio, GDAL
  - Data science: pandas, numpy, scipy, scikit-learn, tensorly
  - Jupyter: jupyter, ipython, ipykernel
- Automatic GitHub Actions updates
- Labels PRs with `dependencies` tag

**Managing Dependabot PRs:**
- Review grouped updates as a batch
- Test critical updates (GDAL, geopandas) carefully
- Major version updates require manual review

### Security Scanning

**safety check:**
```bash
# Check for known vulnerabilities
safety check --file requirements.txt

# Generate detailed report
safety check --file requirements.txt --output json
```

**Common issues:**
- GDAL vulnerabilities: Often system-level, requires OS updates
- Numpy/Pillow: Update if CVE severity is high

### Development Workflow

**Before committing:**

1. Pre-commit hooks run automatically
2. If hooks fail, fix issues and re-stage:
   ```bash
   git add -u
   git commit
   ```

**Before pushing:**

1. Run tests locally:
   ```bash
   pytest
   ```

2. Ensure notebooks have cleared outputs:
   ```bash
   jupyter nbconvert --clear-output --inplace *.ipynb
   ```

**On pull request:**

1. GitHub Actions runs all checks
2. Review CI results before merging
3. Address any failing checks

### Common CI Issues

#### GDAL Installation Failures

**Problem:** GDAL Python package version must match system GDAL library version

**Error Message:**
```
Exception: Python bindings of GDAL 3.11.0 require at least libgdal 3.11.0, but 3.8.4 was found
```

**Root Cause:** The Python GDAL package (specified in `requirements.txt`) must match the system GDAL library version. Ubuntu's apt repositories often have older versions than the latest Python package.

**Solution:** The CI workflow automatically installs the matching GDAL version:
```yaml
- name: Install system dependencies
  run: |
    sudo apt-get update
    sudo apt-get install -y gdal-bin libgdal-dev

- name: Install Python dependencies
  run: |
    python -m pip install --upgrade pip
    # Install GDAL Python bindings matching system version
    pip install GDAL==$(gdal-config --version)
    # Install other dependencies (excluding GDAL)
    grep -v "^GDAL==" requirements.txt > /tmp/requirements-no-gdal.txt
    pip install -r /tmp/requirements-no-gdal.txt
```

**For Local Development:** Check your system GDAL version and install matching Python package:
```bash
gdal-config --version  # Check system version (e.g., 3.8.4)
pip install GDAL==3.8.4  # Install matching version
```

#### Notebook Output Errors

**Problem:** Notebooks committed with outputs

**Solution:**
```bash
# Clear all notebook outputs
find . -name "*.ipynb" ! -path "./old_notebooks/*" -exec jupyter nbconvert --clear-output --inplace {} \;

# Commit the changes
git add *.ipynb
git commit --amend --no-edit
```

#### Import Errors in Tests

**Problem:** Tests can't import modules from `notebooks/`

**Solution:** Already configured in `conftest.py`:
```python
sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))
```

#### Ruff Linting Errors

**Problem:** Mathematical notation flagged (N806: variable should be lowercase)

**Solution:** Already configured in `pyproject.toml` to allow uppercase mathematical variables

---

## Data Pipeline Workflow

### Pipeline Execution Order

The pipeline must be executed **sequentially** from 1a to 8. Some steps (4, 5, 6) can run in parallel.

```
1a_collect_bike_trips.py
    ↓
1b_collect_stations.py
    ↓
2_collect_area.py
    ↓
3_build_grid.py
    ↓
┌───────────┬────────────┬─────────────┐
│           │            │             │
4_preprocess  5_collect   6_collect
_population   _population _POIs
    │           │            │
└───────────┴────────────┴─────────────┘
    ↓
7_compute_OD_matrix.py
    ↓
8_compute_cell_features.py
    ↓
┌─────────────────┬──────────────┐
│                 │              │
cell_classifier   factors
.ipynb            .ipynb
```

### Pipeline Stages

#### Stage 1: Data Collection

**1a. Collect Bike Trips** (`1a_collect_bike_trips.py`)
- **Input:** Urban Sharing API URLs (based on year/month)
- **Process:** Downloads trip JSON files, parses to DataFrame
- **Output:** `data/{city}/trips/{city.lower()}_data.pkl`
- **Key Function:** `oslo_lib.collect_data(years, months, provider)`

**1b. Extract Stations** (`1b_collect_stations.py`)
- **Input:** Trip data pickle
- **Process:** Extracts unique station locations, creates GeoDataFrame
- **Output:** `data/{city}/trips/stations.geoparquet`
- **Transforms:** WGS84 (EPSG:4326) → City CRS

#### Stage 2: Geographic Foundation

**2. Collect Area** (`2_collect_area.py`)
- **Input:** City name (queries OpenStreetMap)
- **Process:** Downloads city boundary polygon
- **Output:** `data/{city}/{city.lower()}_area.geoparquet`
- **Library:** Uses `osmnx` for OSM queries

**3. Build Grid** (`3_build_grid.py`)
- **Input:** Area boundary, station coverage
- **Process:** Creates H3 hexagonal grid at resolution 8
- **Outputs:**
  - `data/{city}/full_hex_grid.geoparquet` (full area)
  - `data/{city}/hex_grid.geoparquet` (clipped to station coverage)
- **Additional:** Includes 1st, 2nd, 3rd neighbor ring cells

#### Stage 3: Contextual Data Collection

**4. Preprocess Population** (`4_preprocess_population.py`)
- **Input:** Global population raster (GHS_POP CSV)
- **Process:** Converts CSV to GeoPackage
- **Output:** `data/global/GHS_POP/R*_C*.gpkg`

**5. Collect Population** (`5_collect_population.py`)
- **Input:** Population raster, H3 grid
- **Process:** Assigns population to grid cells
- **Output:** Grid with population counts per cell

**6. Collect POIs** (`6_collect_POIs.py`)
- **Input:** City boundary, OSM queries
- **Process:** Downloads from OpenStreetMap:
  - Cycleways and bike infrastructure
  - Land use categories
  - Water bodies
  - Public transport stations
- **Output:** `data/{city}/POIs/*.geoparquet`

#### Stage 4: Trip Analysis

**7. Compute OD Matrix** (`7_compute_OD_matrix.py`)
- **Input:** Trip data, H3 grid with station-to-cell mapping
- **Process:** Aggregates trips into origin-destination matrix
- **Index:** `(year, month, weekday, hour, source_cell, dest_cell)`
- **Output:** Multi-index DataFrame with trip counts

**8. Compute Cell Features** (`8_compute_cell_features.py`)
- **Input:** Grid, POIs, population, infrastructure
- **Process:** Engineers features for each cell:
  - POI counts (by category)
  - Infrastructure lengths (cycleways, roads)
  - Building volumes
  - Population density
  - Distance-weighted aggregates
- **Output:** `data/{city}/cell_features.parquet`

#### Stage 5: Modeling & Analysis

**Cell Classifier** (`cell_classifier.ipynb`)
- Supervised/unsupervised classification of cells
- Uses engineered features
- Identifies urban typologies

**Tensor Factorization** (`factors.ipynb`)
- Non-negative Tucker/PARAFAC decomposition
- 3D tensor: (hour × source_cell × dest_cell)
- Discovers latent mobility patterns
- Consensus clustering for validation (CCC.py)

---

## Configuration Management

### YAML Configuration (`notebooks/run.yml`)

Each city has a dedicated configuration section:

```yaml
CityName:
  collect_area:
    name: "City, Country"
    filename: data/City/area.geoparquet

  grid:
    resolution: 8
    full_grid_filename: data/City/full_hex_grid.geoparquet
    clipped_grid_filename: data/City/hex_grid.geoparquet

  POIs:
    crs: EPSG_CODE  # e.g., 32632 for Oslo
    folder: data/City/POIs/

  population:
    input: data/global/GHS_POP/R*_C*.csv
    output: data/global/GHS_POP/R*_C*.gpkg
```

### City-Specific CRS Codes

| City | EPSG Code | Projection |
|------|-----------|------------|
| Oslo | 32632 | UTM Zone 32N |
| Milan | 6875 | Monte Mario / Italy zone 2 |
| Edinburgh | 5653 | British National Grid |

### Loading Configuration

```python
from utils import get_config

city = 'Oslo'
config = get_config(city)
crs = config['POIs']['crs']
```

**WARNING:** Current `utils.py` has hardcoded path `app/run.yml` which should be `notebooks/run.yml` after restructuring. Update this when running from different directories.

---

## Code Conventions

### Naming Conventions

#### Files
- **Pipeline scripts:** Numbered with descriptive names (`1a_collect_bike_trips.py`)
- **Libraries:** Lowercase with underscores (`oslo_lib.py`, `geo_utils.py`)
- **Notebooks:** Descriptive names (`cell_classifier.ipynb`, `factors.ipynb`)

#### Variables
- **DataFrames:** Descriptive names (`trips`, `stations`, `grid`)
- **GeoDataFrames:** Same convention, context-dependent
- **Temporal columns:** `year`, `month`, `weekday`, `hour`, `quarter`
- **Spatial columns:** `geometry`, `latitude`, `longitude`

#### Functions
- **Lowercase with underscores:** `get_processed_trips()`, `collect_data()`
- **Prefix conventions:**
  - `get_*`: Retrieves/computes and returns data
  - `collect_*`: Fetches external data (API, OSM)
  - `retrieve_*`: Downloads raw data

### Data Processing Patterns

#### 1. **Map-Reduce Aggregation**

```python
# Pattern: Aggregate trips by dimensions
trips.groupby(['hour', 'start_cluster', 'end_cluster']).trip.sum()
```

#### 2. **Geospatial Transformations**

```python
# Pattern: Always transform to city CRS for analysis
gdf = gdf.set_crs('epsg:4326')  # WGS84 (GPS coordinates)
gdf = gdf.to_crs(city_crs)      # City projection
```

#### 3. **Temporal Feature Engineering**

```python
# Pattern: Extract temporal dimensions from timestamps
df['year'] = df.started_at.apply(lambda x: x.year)
df['month'] = df.started_at.apply(lambda x: x.month)
df['weekday'] = df.started_at.apply(lambda x: x.weekday())
df['hour'] = df.started_at.apply(lambda x: x.hour)
```

#### 4. **Error Handling for API Calls**

```python
# Pattern: Graceful degradation on API failures
try:
    r = requests.get(url)
    df = pd.DataFrame(r.json())
except:
    df = pd.DataFrame()  # Return empty on failure
```

### File Format Conventions

| Data Type | Format | Extension | Use Case |
|-----------|--------|-----------|----------|
| Raw trips | Pickle | `.pkl` | Fast serialization, preserves types |
| Spatial data | GeoParquet | `.geoparquet` | Compressed spatial dataframes |
| Features | Parquet | `.parquet` | Columnar storage for analytics |
| Population | GeoPackage | `.gpkg` | Raster-to-vector conversion |
| Config | YAML | `.yml` | Human-readable settings |

### Import Order

```python
# Standard library
import requests
from datetime import datetime

# Third-party data
import pandas as pd
import geopandas as gpd
import numpy as np

# Geospatial
from shapely.ops import Point
import osmnx

# ML/Math
import tensorly as tl
from sklearn import ...

# Visualization
import matplotlib.pyplot as plt

# Local modules
from utils import get_config
import oslo_lib
```

---

## Key Modules Reference

### `oslo_lib.py` - Core Data Processing

Main library for trip data collection and tensor analysis.

**Key Functions:**

```python
# Data Collection
obtain_url(y, m, provider)
    # Build API URL for Urban Sharing data
    # Returns: https://data.urbansharing.com/{provider}/trips/v1/{y}/{m}.json

urls(years, months, provider)
    # Generator for all URLs in date range
    # Skips: 2019 months 1-3, 2023 months 8-12

retrieve_dataset(url)
    # Fetch and parse JSON from API
    # Returns: DataFrame with typed columns or empty DataFrame on error

collect_data(years, months, provider='oslobysykkel.no')
    # Main orchestrator: downloads all months, concatenates
    # Prints progress: timestamp, record count per month

# Data Processing
get_processed_trips(trips, station_distances)
    # Engineers temporal features: year, month, weekday, hour, quarter
    # Computes: duration_min, distance_km, speed, speed_kmh
    # Joins with station distance matrix

get_geostations(stations, city_crs)
    # Converts station lat/lon to GeoDataFrame
    # Transforms: EPSG:4326 → city_crs

get_trips_per_station(trips, stations)
    # Aggregates starts + ends per station
    # Classifies stations: 0 (<60k), 1 (60-80k), 2 (80-120k), 3 (>120k)

# Tensor Analysis
get_matrix(data, clusters, sample_size=None)
    # Creates hour × cluster × cluster matrix
    # Makes symmetric: matrix + reverse_matrix
    # Returns: MultiIndex DataFrame

get_T(matrix)
    # Converts matrix to 3D TensorLy tensor
    # Shape: (24, n_clusters, n_clusters)

bootstrap_T(matrix, frac, data, clusters)
    # Samples fraction of data for consensus clustering

get_factorization(use)
    # Returns: (factorization_function, reconstruction_function)
    # Options: 'tucker' or 'parafac'
```

**Usage Example:**

```python
import oslo_lib

# Collect 2020-2022 data
years = [2020, 2021, 2022]
months = range(1, 13)
trips = oslo_lib.collect_data(years, months, 'oslobysykkel.no')

# Process trips
processed = oslo_lib.get_processed_trips(trips, station_distances)

# Create tensor for factorization
matrix = oslo_lib.get_matrix(processed, clusters)
T = oslo_lib.get_T(matrix)
```

### `geo_utils.py` - Geospatial Utilities

Geometric filtering and bounding box operations.

**Key Functions:**

```python
get_boundaries(arr)
    # Returns: (min, max) from array

get_spread(arr)
    # Returns: max - min

get_window(arr, buffer_percent=0.05)
    # Computes spatial window with 5% buffer
    # Returns: (min - buffer, max + buffer)

window_filter(df, col, window)
    # Filters DataFrame rows where col in [window[0], window[1]]

get_city_bbox(city_gdf)
    # Extracts bounding box from city boundary
    # Returns: (lon_min, lon_max, lat_min, lat_max)

geo_filter(df, lon_min, lon_max, lat_min, lat_max)
    # Filters DataFrame by geographic bounds
```

### `CCC.py` - Consensus Clustering

Validates clustering stability through bootstrap sampling.

**Key Functions:**

```python
get_consensus_matrix(data, clusters, n_bootstraps, frac, rank, use='tucker')
    # Runs n_bootstraps factorizations on sampled data
    # Computes consensus: how often cells cluster together
    # Returns: Consensus matrix (0-1 values)

get_adjacency_matrix(factors, threshold=0.5)
    # Converts factor loadings to binary adjacency
    # Cells with correlation > threshold are connected

get_CCC(consensus_matrix)
    # Cophenetic Correlation Coefficient
    # Measures: clustering stability/reproducibility
    # Returns: Float in [0, 1], higher = more stable

compute_rho(data, clusters, n_bootstraps, frac, rank, use='tucker')
    # Full pipeline: consensus → CCC
    # Used to validate optimal rank selection
```

**Purpose:** Determine if tensor factorization produces consistent clusters across data samples.

### `utils.py` - Configuration Loader

```python
get_config(city)
    # Loads notebooks/run.yml (or app/run.yml - see known issues)
    # Returns: Dictionary of city-specific settings
```

---

## Common Tasks

### Adding a New City

1. **Add configuration to `notebooks/run.yml`:**

```yaml
NewCity:
  collect_area:
    name: "City Name, Country"
    filename: data/NewCity/area.geoparquet
  grid:
    resolution: 8
    full_grid_filename: data/NewCity/full_hex_grid.geoparquet
    clipped_grid_filename: data/NewCity/hex_grid.geoparquet
  POIs:
    crs: EPSG_CODE  # Find appropriate UTM zone
    folder: data/NewCity/POIs/
  population:
    input: data/global/GHS_POP/R*_C*.csv
    output: data/global/GHS_POP/R*_C*.gpkg
```

2. **Create data directory:**

```bash
mkdir -p data/NewCity/{trips,POIs}
```

3. **Update scripts:** Change `city = 'NewCity'` in each numbered script

4. **Find bike-sharing API:** Update `provider` in `1a_collect_bike_trips.py`

5. **Determine CRS:** Use appropriate EPSG code for the city's UTM zone

### Running the Full Pipeline

```bash
cd notebooks

# Stage 1: Data collection
python 1a_collect_bike_trips.py  # ~10-60 min depending on date range
python 1b_collect_stations.py   # <1 min

# Stage 2: Geographic foundation
python 2_collect_area.py         # ~1 min
python 3_build_grid.py           # ~2-5 min

# Stage 3: Contextual data (can run in parallel)
python 4_preprocess_population.py  # ~5 min
python 5_collect_population.py     # ~10 min
python 6_collect_POIs.py          # ~5-15 min (depends on OSM)

# Stage 4: Trip analysis
python 7_compute_OD_matrix.py     # ~5-20 min
python 8_compute_cell_features.py # ~10-30 min

# Stage 5: Analysis (Jupyter notebooks)
jupyter notebook cell_classifier.ipynb
jupyter notebook factors.ipynb
```

### Debugging Pipeline Failures

**Check data outputs at each stage:**

```python
import pandas as pd
import geopandas as gpd

# Verify trip data
trips = pd.read_pickle('data/Oslo/trips/oslo_data.pkl')
print(f"Trips: {len(trips):,} records")

# Verify stations
stations = gpd.read_parquet('data/Oslo/trips/stations.geoparquet')
print(f"Stations: {len(stations)}")

# Verify grid
grid = gpd.read_parquet('data/Oslo/hex_grid.geoparquet')
print(f"Grid cells: {len(grid)}")
```

**Common Issues:**
- **Empty DataFrames:** API changed or date range has no data
- **CRS Errors:** Check EPSG code matches city location
- **Missing files:** Ensure previous pipeline stages completed
- **Path errors:** Verify `utils.py` points to correct `run.yml`

### Modifying Feature Engineering

Features are computed in `8_compute_cell_features.py`. To add new features:

1. Load necessary POI or infrastructure data
2. Spatial join with H3 grid
3. Aggregate (count, sum, mean) per cell
4. Consider distance-weighted variants
5. Save to parquet with descriptive column names

**Example pattern:**

```python
# Load POI data
cafes = gpd.read_parquet('data/City/POIs/cafes.geoparquet')

# Spatial join to grid
grid_with_cafes = grid.sjoin(cafes, how='left', predicate='contains')

# Aggregate
cafe_counts = grid_with_cafes.groupby('hex_id').size()
grid['cafe_count'] = cafe_counts.reindex(grid.hex_id).fillna(0)
```

---

## Known Issues & Gotchas

### 1. Hardcoded Path in `utils.py`

**Issue:** `utils.py` line 4 has `config_path = "app/run.yml"` but actual location is `notebooks/run.yml`

**Workaround:**
- Run scripts from project root, or
- Update `utils.py`:

```python
config_path = "notebooks/run.yml"
```

### 2. City Variable Hardcoded in Scripts

**Issue:** Each script has `city = 'Oslo'` (or other city) hardcoded

**Workaround:** Manually edit each script before running, or create a wrapper script:

```python
import sys
city = sys.argv[1] if len(sys.argv) > 1 else 'Oslo'
```

### 3. API Rate Limits & Failures

**Issue:** `oslo_lib.collect_data()` has bare `except` that silently returns empty DataFrame

**Impact:** Missing months may go unnoticed

**Fix:** Check output for expected number of records:

```python
trips = collect_data([2022], range(1, 13))
monthly_counts = trips.groupby(trips.started_at.dt.to_period('M')).size()
print(monthly_counts)  # Should have 12 entries
```

### 4. Date Range Filters in `urls()`

**Issue:** `oslo_lib.urls()` hardcodes skipping 2019 months 1-3 and 2023 months 8+

**Impact:** Cannot collect these months even if they exist

**Fix:** Comment out lines 20-21 in `oslo_lib.py` if needed:

```python
# if (y==2019 and m<4) or (y==2023 and m>7):
#     pass
```

### 5. CRS Confusion

**Issue:** Multiple coordinate systems in play:
- WGS84 (EPSG:4326) - GPS coordinates from APIs
- City-specific (UTM zones) - For metric calculations

**Best Practice:**
- Always check CRS: `gdf.crs`
- Transform before distance calculations
- Store spatial data in city CRS, not WGS84

### 6. Memory Usage

**Issue:** Loading full trip datasets can consume 4-8 GB RAM

**Workaround:**
- Process in chunks if memory-constrained
- Use `sample_size` parameter in `get_matrix()`
- Close notebooks after use to free memory

### 7. H3 Resolution

**Issue:** Resolution 8 is hardcoded (hexagons ~0.5 km²)

**Impact:** Too coarse for small cities, too fine for large regions

**Customization:** Edit `run.yml` `grid.resolution` (range: 5-11)
- 5: ~250 km² (regional)
- 8: ~0.5 km² (neighborhood)
- 11: ~1000 m² (block)

---

## AI Assistant Guidelines

### When Working with This Codebase

#### 1. **Always Check Current State**

Before making changes:
- Verify which pipeline stage has been completed
- Check if data files exist at expected paths
- Read the relevant script/notebook before modifying

#### 2. **Respect Sequential Dependencies**

- Never run stage N without completing stage N-1
- Verify outputs exist before starting dependent stages
- Scripts 4, 5, 6 can run in parallel but all must complete before stage 7

#### 3. **Preserve Data Formats**

- Spatial data → GeoParquet (`.geoparquet`)
- Trip data → Pickle (`.pkl`)
- Features → Parquet (`.parquet`)
- Don't change formats without updating all dependent code

#### 4. **Coordinate System Awareness**

When working with spatial data:
1. Check current CRS: `gdf.crs`
2. Transform to city CRS for metric operations
3. Use WGS84 only for input/output with GPS coordinates
4. Never mix CRS in distance calculations

#### 5. **Configuration-Driven Changes**

- Prefer editing `run.yml` over hardcoding values
- Add new cities by extending configuration, not duplicating code
- Keep city-specific logic in config, not scattered in scripts

#### 6. **Error Handling**

Current code has minimal error handling. When adding:
- Check for empty DataFrames after API calls
- Verify file existence before reading
- Validate CRS transformations succeeded
- Log meaningful messages, not bare `except: pass`

#### 7. **Naming Conventions**

When creating new functions/variables:
- Functions: `lowercase_with_underscores`
- DataFrames: Descriptive nouns (`trips`, `stations`, `features`)
- Booleans: `is_*` or `has_*` prefix
- Files: Match existing patterns (numbered scripts, descriptive notebooks)

#### 8. **Documentation**

When adding new features:
- Update this `CLAUDE.md` file
- Add docstrings to functions with parameters/returns
- Comment non-obvious operations (e.g., tensor manipulations)
- Update `README.md` if pipeline workflow changes

#### 9. **Jupyter Notebook Usage**

- Use notebooks for **exploration and visualization** only
- Move reproducible pipelines to **scripts**
- Clear outputs before committing: `jupyter nbconvert --clear-output`
- Keep notebooks focused (don't create mega-notebooks)

#### 10. **Testing Changes**

Before committing:
- Test on small subset of data first
- Verify output file formats and sizes
- Check that downstream stages still work
- Run a full pipeline on one city if major changes

#### 11. **Git Practices**

- Don't commit `data/`, `cache/`, `.venv/` (already in `.gitignore`)
- Commit incremental changes, not massive refactors
- Use descriptive commit messages referencing pipeline stages
- Keep branches short-lived

#### 12. **Performance Considerations**

This is a data-intensive pipeline:
- Scripts may run 5-60 minutes depending on data size
- Consider adding progress bars for long operations
- Cache intermediate results when appropriate
- Don't load entire datasets if aggregates suffice

### Common AI Assistant Tasks

#### **"Add a new feature to cells"**

1. Identify data source (POI, infrastructure, demographics)
2. Load or collect source data in stage 6
3. Modify `8_compute_cell_features.py`
4. Spatial join to grid, aggregate
5. Save with descriptive column name
6. Update downstream notebooks to use new feature

#### **"Fix a bug in data collection"**

1. Read current implementation
2. Identify the issue (API change? date filter? parsing?)
3. Test fix on small sample (single month)
4. Verify DataFrame schema matches expected
5. Re-run affected pipeline stages
6. Update documentation if behavior changed

#### **"Optimize a slow script"**

1. Profile to identify bottleneck (I/O? computation?)
2. Consider vectorization (avoid `.apply()` with lambda)
3. Use appropriate file formats (Parquet > CSV for large data)
4. Parallelize if independent operations
5. Cache expensive computations
6. Measure improvement

#### **"Explain how X works"**

1. Trace the data flow from source to output
2. Read relevant functions in `oslo_lib.py` or other modules
3. Identify mathematical operations (especially tensor decomposition)
4. Explain in terms of urban mobility domain
5. Provide code examples with sample data

### Red Flags (Don't Do This)

- ❌ Running scripts out of order
- ❌ Mixing coordinate systems in distance calculations
- ❌ Hardcoding file paths instead of using configuration
- ❌ Adding dependencies without updating `requirements.txt`
- ❌ Committing large data files
- ❌ Modifying legacy notebooks in `old_notebooks/`
- ❌ Changing data formats without updating all consumers
- ❌ Skipping stages "because they look optional"
- ❌ Using `print()` instead of proper logging in production scripts
- ❌ Creating circular dependencies between modules

### Green Lights (Best Practices)

- ✅ Reading files before editing
- ✅ Checking data exists before processing
- ✅ Adding new features via configuration
- ✅ Writing modular, reusable functions
- ✅ Testing on small samples first
- ✅ Documenting assumptions and design decisions
- ✅ Using appropriate geospatial libraries
- ✅ Following existing code patterns
- ✅ Updating this CLAUDE.md when adding features
- ✅ Asking for clarification when unsure

---

## Quick Reference

### File Paths

```python
# Configuration
config = "notebooks/run.yml"

# City data
trips = f"data/{city}/trips/{city.lower()}_data.pkl"
stations = f"data/{city}/trips/stations.geoparquet"
area = f"data/{city}/{city.lower()}_area.geoparquet"
grid = f"data/{city}/hex_grid.geoparquet"
features = f"data/{city}/cell_features.parquet"

# Global data
population = "data/global/GHS_POP/R*_C*.csv"
```

### Common Code Snippets

```python
# Load configuration
from utils import get_config
config = get_config('Oslo')

# Load spatial data
import geopandas as gpd
grid = gpd.read_parquet('data/Oslo/hex_grid.geoparquet')

# Load trips
import pandas as pd
trips = pd.read_pickle('data/Oslo/trips/oslo_data.pkl')

# Transform CRS
gdf = gdf.to_crs('EPSG:32632')  # Oslo UTM

# Spatial join
result = grid.sjoin(pois, how='left', predicate='contains')

# Aggregate by cell
counts = result.groupby('hex_id').size()
```

### Useful Terminal Commands

```bash
# Check data sizes
du -sh data/*/

# Count trip records
python -c "import pandas as pd; print(len(pd.read_pickle('data/Oslo/trips/oslo_data.pkl')))"

# List grid cells
python -c "import geopandas as gpd; print(len(gpd.read_parquet('data/Oslo/hex_grid.geoparquet')))"

# Check Python environment
which python  # Should be .venv/bin/python

# Install new package
pip install package_name && pip freeze > requirements.txt
```

---

## Contact & Resources

**Project Repository:** (Add Git repo URL here)

**Key Papers:**
- H3 Hexagonal Grid: https://h3geo.org/
- OpenStreetMap: https://www.openstreetmap.org/
- Tensor Factorization: TensorLy documentation

**Data Sources:**
- Urban Sharing API: https://data.urbansharing.com/
- GHS Population: https://ghsl.jrc.ec.europa.eu/
- OpenStreetMap via OSMNX

**Maintainer:** (Add contact info)

---

**End of CLAUDE.md**

*This documentation is maintained for AI assistants working with the BikePulse codebase. Keep it updated as the project evolves.*
