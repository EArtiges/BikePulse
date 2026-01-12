# BikePulse

**Urban Bike-Sharing Network Analysis & Mobility Pattern Discovery**

[![CI](https://github.com/EArtiges/BikePulse/workflows/CI/badge.svg)](https://github.com/EArtiges/BikePulse/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

BikePulse is a **data science pipeline** for analyzing urban bike-sharing systems across multiple cities. It combines geospatial analysis, feature engineering, and machine learning to understand urban mobility patterns and discover latent factors in travel behavior.

**This is a computational research tool, not a web or mobile application.**

### Key Capabilities

- 🚴 **Data Collection**: Automated extraction from Urban Sharing APIs
- 🗺️ **Geospatial Analysis**: H3 hexagonal grid system for spatial aggregation
- 🏙️ **Urban Context**: Integration of POIs, population, infrastructure data
- 🤖 **Machine Learning**: Cell classification and trip prediction models
- 📊 **Tensor Factorization**: Discovery of interpretable mobility patterns
- 🔬 **Validation**: Consensus clustering for robust factor analysis

### Supported Cities

- **Oslo** (Norway)
- **Milan** (Italy)
- **Edinburgh** (Scotland)

---

## Quick Start

### Prerequisites

- **Python 3.9+**
- **GDAL** and **PROJ** system libraries (required for geospatial operations)

**Install GDAL (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev
```

**Install GDAL (macOS):**
```bash
brew install gdal
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/EArtiges/BikePulse.git
cd BikePulse
```

2. **Create and activate virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

3. **Install Python dependencies:**
```bash
# Install GDAL matching system version
pip install GDAL==$(gdal-config --version)

# Install other dependencies
grep -v "^GDAL==" requirements.txt > /tmp/requirements-no-gdal.txt
pip install -r /tmp/requirements-no-gdal.txt
```

4. **Install pre-commit hooks (for development):**
```bash
pip install -r requirements-dev.txt
pre-commit install
```

### Running the Pipeline

```bash
cd notebooks

# Stage 1: Data Collection
python 1a_collect_bike_trips.py    # Collect trip data from API
python 1b_collect_stations.py      # Extract station locations

# Stage 2: Geographic Foundation
python 2_collect_area.py           # Download city boundaries
python 3_build_grid.py             # Create H3 hexagonal grid

# Stage 3: Contextual Data
python 4_preprocess_population.py  # Prepare population data
python 5_collect_population.py     # Assign population to cells
python 6_collect_POIs.py           # Collect Points of Interest

# Stage 4: Trip Analysis
python 7_compute_OD_matrix.py      # Origin-destination matrix
python 8_compute_cell_features.py  # Engineer cell features

# Stage 5: Modeling & Analysis
jupyter notebook cell_classifier.ipynb  # Cell classification
jupyter notebook factors.ipynb          # Tensor factorization
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Data Collection                                   │
│  ├─ 1a: Collect bike trips from Urban Sharing API          │
│  └─ 1b: Extract station locations                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Geographic Foundation                             │
│  ├─ 2: Download city boundary from OpenStreetMap           │
│  └─ 3: Build H3 hexagonal grid system                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Contextual Data (parallel)                        │
│  ├─ 4+5: Process and assign population data                │
│  └─ 6: Collect POIs (cycleways, landuse, transport)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Trip Analysis                                     │
│  ├─ 7: Compute origin-destination trip matrix              │
│  └─ 8: Engineer features for each grid cell                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 5: Modeling & Discovery                              │
│  ├─ Cell Classifier: Identify urban typologies             │
│  └─ Tensor Factorization: Discover mobility patterns       │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Geospatial Analysis
- **GeoPandas** - Spatial dataframes
- **H3** - Uber's hexagonal grid system
- **OSMnx** - OpenStreetMap data extraction
- **Rasterio** - Raster data processing
- **Shapely** - Geometric operations

### Data Processing
- **Pandas** - Tabular data manipulation
- **NumPy** - Numerical computing
- **PyArrow** - Parquet file format

### Machine Learning
- **Scikit-learn** - Classification and clustering
- **TensorLy** - Tensor decomposition (Tucker, PARAFAC)
- **NetworkX** - Graph analysis

### Visualization
- **Matplotlib** - Static plotting
- **Contextily** - Basemap tiles

---

## Project Structure

```
BikePulse/
├── notebooks/              # Pipeline scripts (run sequentially)
│   ├── 1a_collect_bike_trips.py
│   ├── 1b_collect_stations.py
│   ├── 2_collect_area.py
│   ├── 3_build_grid.py
│   ├── 4_preprocess_population.py
│   ├── 5_collect_population.py
│   ├── 6_collect_POIs.py
│   ├── 7_compute_OD_matrix.py
│   ├── 8_compute_cell_features.py
│   ├── oslo_lib.py        # Core data processing library
│   ├── geo_utils.py       # Geospatial utilities
│   ├── CCC.py             # Consensus clustering
│   ├── utils.py           # Configuration loader
│   └── run.yml            # City-specific configurations
│
├── cell_classifier.ipynb  # ML model for cell classification
├── factors.ipynb          # Tensor factorization analysis
│
├── data/                  # Data storage (gitignored)
│   ├── {city}/
│   │   ├── trips/
│   │   └── POIs/
│   └── global/
│       └── GHS_POP/
│
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
├── requirements-dev.txt   # Development dependencies
├── .pre-commit-config.yaml
├── pyproject.toml         # Code quality configuration
├── CLAUDE.md              # Detailed codebase documentation
└── README.md              # This file
```

---

## Key Features

### 1. Hexagonal Grid System (H3)

BikePulse uses Uber's H3 hexagonal grid for spatial analysis:
- **Resolution 8**: ~0.5 km² per hexagon (neighborhood scale)
- Uniform cell sizes for consistent analysis
- Efficient neighbor operations
- Hierarchical aggregation support

### 2. Geospatial Feature Engineering

Each grid cell is enriched with:
- **POI counts**: Restaurants, shops, transit stations, etc.
- **Infrastructure**: Cycleway length, road network density
- **Demographics**: Population density from GHS_POP
- **Building metrics**: Volume and density
- **Distance-weighted aggregates**: Influence from neighboring cells

### 3. Origin-Destination Matrix

Multi-dimensional trip aggregation:
- **Dimensions**: (year, month, weekday, hour, source_cell, dest_cell)
- Captures temporal patterns and spatial flows
- Supports tensor factorization for pattern discovery

### 4. Tensor Factorization

Discovers latent mobility patterns using:
- **Tucker Decomposition**: Separable temporal and spatial factors
- **PARAFAC**: Polyadic decomposition for interpretable factors
- **Consensus Clustering**: Bootstrap validation (CCC metric)

---

## Configuration

City-specific settings are defined in `notebooks/run.yml`:

```yaml
Oslo:
  collect_area:
    name: "Oslo, Norway"
    filename: data/Oslo/oslo_area.geoparquet
  grid:
    resolution: 8
    full_grid_filename: data/Oslo/full_hex_grid.geoparquet
    clipped_grid_filename: data/Oslo/hex_grid.geoparquet
  POIs:
    crs: 32632  # UTM Zone 32N
    folder: data/Oslo/POIs/
  population:
    input: data/global/GHS_POP/R*_C*.csv
    output: data/global/GHS_POP/R*_C*.gpkg
```

---

## Development

### Code Quality

The project uses automated code quality tools:

- **Black**: Code formatting (100 char line length)
- **Ruff**: Fast linting
- **isort**: Import sorting
- **pytest**: Unit testing
- **pre-commit**: Automated checks

**Run checks manually:**
```bash
# Format code
black notebooks/ *.py

# Lint
ruff check --fix notebooks/ *.py

# Run tests
pytest

# Run all pre-commit hooks
pre-commit run --all-files
```

### Continuous Integration

GitHub Actions workflows:
- **Linting**: Code quality checks
- **Pipeline Validation**: Syntax and import checks
- **Environment Check**: GDAL/PROJ compatibility
- **Tests**: Unit tests with coverage
- **Documentation**: Validates docs are up-to-date
- **Security**: Dependency vulnerability scanning

### Adding a New City

1. Add configuration to `notebooks/run.yml`
2. Determine appropriate EPSG code (UTM zone)
3. Create data directory: `mkdir -p data/NewCity/{trips,POIs}`
4. Update `city` variable in pipeline scripts
5. Find bike-sharing API endpoint
6. Run pipeline stages 1-8

---

## Data Sources

- **Bike Trips**: [Urban Sharing API](https://data.urbansharing.com/)
- **Population**: [GHS Population Grid](https://ghsl.jrc.ec.europa.eu/)
- **POIs & Infrastructure**: [OpenStreetMap](https://www.openstreetmap.org/) via OSMnx
- **City Boundaries**: OpenStreetMap administrative boundaries

---

## Research Applications

BikePulse enables research on:

- **Urban typologies**: Identifying distinct neighborhood types
- **Mobility patterns**: Understanding how people move through cities
- **Demand prediction**: Forecasting bike-sharing usage
- **Infrastructure planning**: Optimizing station placement
- **Policy evaluation**: Assessing impact of urban interventions
- **Cross-city comparison**: Universal mobility patterns vs. local characteristics

---

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Comprehensive codebase guide for developers and AI assistants
- **Inline docstrings**: Function-level documentation
- **Configuration examples**: `notebooks/run.yml`

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install pre-commit hooks: `pre-commit install`
4. Make your changes and commit: `git commit -m "Add amazing feature"`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

**Code standards:**
- Follow existing code style (enforced by Black/Ruff)
- Add tests for new functionality
- Update CLAUDE.md for significant changes
- Clear notebook outputs before committing

---

## Known Issues

1. **GDAL version mismatch**: Python GDAL package must match system library version
   - Solution: `pip install GDAL==$(gdal-config --version)`

2. **Path configuration**: `utils.py` may need path updates after directory changes

3. **API rate limits**: Urban Sharing API may throttle requests during bulk collection

See [CLAUDE.md - Known Issues](CLAUDE.md#known-issues--gotchas) for full list and workarounds.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use BikePulse in your research, please cite:

```bibtex
@software{bikepulse2026,
  title = {BikePulse: Urban Bike-Sharing Network Analysis},
  author = {Artiges, E.},
  year = {2026},
  url = {https://github.com/EArtiges/BikePulse}
}
```

---

## Acknowledgments

- **H3**: Uber Technologies for the hexagonal grid system
- **OpenStreetMap**: Community-contributed geographic data
- **Urban Sharing**: Bike-sharing data APIs
- **GHS**: Global Human Settlement Layer for population data

---

## Contact

For questions, issues, or collaboration opportunities:
- **Issues**: [GitHub Issues](https://github.com/EArtiges/BikePulse/issues)
- **Discussions**: [GitHub Discussions](https://github.com/EArtiges/BikePulse/discussions)

---

**Last Updated**: January 2026
