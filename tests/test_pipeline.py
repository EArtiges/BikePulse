"""Tests for pipeline script validation (smoke tests)."""

import ast
import sys
from pathlib import Path

import pytest

# Add notebooks directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "notebooks"))

# Pipeline scripts in execution order
PIPELINE_SCRIPTS = [
    "1a_collect_bike_trips.py",
    "1b_collect_stations.py",
    "2_collect_area.py",
    "3_build_grid.py",
    "4_preprocess_population.py",
    "5_collect_population.py",
    "6_collect_POIs.py",
    "7_compute_OD_matrix.py",
    "8_compute_cell_features.py",
]

LIBRARY_MODULES = [
    "oslo_lib.py",
    "geo_utils.py",
    "CCC.py",
    "utils.py",
]


class TestPipelineScripts:
    """Validate pipeline scripts are syntactically correct."""

    @pytest.mark.parametrize("script_name", PIPELINE_SCRIPTS)
    def test_script_exists(self, script_name):
        """Test that pipeline script exists."""
        script_path = Path(__file__).parent.parent / "notebooks" / script_name
        assert script_path.exists(), f"Pipeline script {script_name} not found"

    @pytest.mark.parametrize("script_name", PIPELINE_SCRIPTS)
    def test_script_syntax(self, script_name):
        """Test that pipeline script has valid Python syntax."""
        script_path = Path(__file__).parent.parent / "notebooks" / script_name
        with open(script_path, "r") as f:
            source = f.read()

        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {script_name}: {e}")

    @pytest.mark.parametrize("script_name", PIPELINE_SCRIPTS)
    def test_script_has_imports(self, script_name):
        """Test that pipeline script has import statements."""
        script_path = Path(__file__).parent.parent / "notebooks" / script_name
        with open(script_path, "r") as f:
            source = f.read()

        tree = ast.parse(source)
        imports = [
            node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert len(imports) > 0, f"{script_name} has no import statements"


class TestLibraryModules:
    """Validate library modules can be imported."""

    @pytest.mark.parametrize("module_name", LIBRARY_MODULES)
    def test_module_exists(self, module_name):
        """Test that library module exists."""
        module_path = Path(__file__).parent.parent / "notebooks" / module_name
        assert module_path.exists(), f"Library module {module_name} not found"

    @pytest.mark.parametrize("module_name", LIBRARY_MODULES)
    def test_module_can_be_imported(self, module_name):
        """Test that library module can be imported."""
        module_base = module_name.replace(".py", "")
        try:
            __import__(module_base)
        except ImportError as e:
            pytest.fail(f"Could not import {module_base}: {e}")

    @pytest.mark.parametrize("module_name", LIBRARY_MODULES)
    def test_module_syntax(self, module_name):
        """Test that library module has valid Python syntax."""
        module_path = Path(__file__).parent.parent / "notebooks" / module_name
        with open(module_path, "r") as f:
            source = f.read()

        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {module_name}: {e}")


class TestPipelineStructure:
    """Validate pipeline structure and dependencies."""

    def test_all_pipeline_scripts_numbered(self):
        """Test that all pipeline scripts follow numbering convention."""
        notebooks_dir = Path(__file__).parent.parent / "notebooks"
        pipeline_files = [f.name for f in notebooks_dir.glob("[0-9]*.py")]

        assert len(pipeline_files) >= len(
            PIPELINE_SCRIPTS
        ), "Missing pipeline scripts in notebooks/"

    def test_config_file_exists(self):
        """Test that run.yml configuration exists."""
        config_path = Path(__file__).parent.parent / "notebooks" / "run.yml"
        assert config_path.exists(), "run.yml configuration file not found"

    def test_notebooks_exist(self):
        """Test that analysis notebooks exist."""
        notebooks = ["cell_classifier.ipynb", "factors.ipynb"]
        for notebook in notebooks:
            notebook_path = Path(__file__).parent.parent / notebook
            assert notebook_path.exists(), f"Analysis notebook {notebook} not found"
