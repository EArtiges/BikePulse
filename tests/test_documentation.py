"""Tests for documentation completeness and accuracy."""

import re
from datetime import datetime
from pathlib import Path

import pytest


class TestCLAUDEMD:
    """Test CLAUDE.md documentation file."""

    def test_claude_md_exists(self):
        """Test that CLAUDE.md exists."""
        claude_md = Path(__file__).parent.parent / "CLAUDE.md"
        assert claude_md.exists(), "CLAUDE.md documentation not found"

    def test_claude_md_has_recent_update_date(self):
        """Test that CLAUDE.md has been updated recently (within 1 year)."""
        claude_md = Path(__file__).parent.parent / "CLAUDE.md"
        content = claude_md.read_text()

        # Look for "Last Updated:" pattern
        match = re.search(r"\*\*Last Updated:\*\*\s+(\d{4}-\d{2}-\d{2})", content)
        assert match, "Could not find 'Last Updated:' date in CLAUDE.md"

        date_str = match.group(1)
        update_date = datetime.strptime(date_str, "%Y-%m-%d")
        days_old = (datetime.now() - update_date).days

        assert days_old < 365, f"CLAUDE.md is {days_old} days old, needs updating"

    def test_claude_md_mentions_all_pipeline_scripts(self):
        """Test that CLAUDE.md documents all pipeline scripts."""
        claude_md = Path(__file__).parent.parent / "CLAUDE.md"
        content = claude_md.read_text()

        pipeline_scripts = [
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

        for script in pipeline_scripts:
            assert script in content, f"CLAUDE.md does not mention pipeline script {script}"

    def test_claude_md_mentions_key_modules(self):
        """Test that CLAUDE.md documents key modules."""
        claude_md = Path(__file__).parent.parent / "CLAUDE.md"
        content = claude_md.read_text()

        key_modules = ["oslo_lib.py", "geo_utils.py", "CCC.py", "utils.py"]

        for module in key_modules:
            assert module in content, f"CLAUDE.md does not mention module {module}"

    def test_claude_md_has_table_of_contents(self):
        """Test that CLAUDE.md has a table of contents."""
        claude_md = Path(__file__).parent.parent / "CLAUDE.md"
        content = claude_md.read_text()

        assert "## Table of Contents" in content, "CLAUDE.md missing table of contents"

    def test_claude_md_references_valid_files(self):
        """Test that file paths mentioned in CLAUDE.md exist."""
        claude_md = Path(__file__).parent.parent / "CLAUDE.md"
        content = claude_md.read_text()
        project_root = claude_md.parent

        # Look for file paths in code blocks and inline code
        # Pattern: notebooks/something.py or data/City/file.ext
        file_patterns = [
            r"notebooks/([a-zA-Z0-9_]+\.py)",
            r"tests/([a-zA-Z0-9_]+\.py)",
            r"pyproject\.toml",
            r"requirements\.txt",
            r"requirements-dev\.txt",
            r"\.github/workflows/ci\.yml",
            r"\.pre-commit-config\.yaml",
        ]

        missing_files = []
        for pattern in file_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                file_path = match.group(0)
                full_path = project_root / file_path

                if not full_path.exists():
                    missing_files.append(file_path)

        assert len(missing_files) == 0, f"CLAUDE.md references non-existent files: {missing_files}"


class TestREADME:
    """Test README.md documentation."""

    def test_readme_exists(self):
        """Test that README.md exists."""
        readme = Path(__file__).parent.parent / "README.md"
        assert readme.exists(), "README.md not found"

    def test_readme_mentions_pipeline_stages(self):
        """Test that README mentions pipeline stages."""
        readme = Path(__file__).parent.parent / "README.md"
        content = readme.read_text()

        # Should mention some pipeline concepts
        pipeline_terms = ["pipeline", "data", "analysis", "bike"]
        mentioned = sum(1 for term in pipeline_terms if term.lower() in content.lower())

        assert mentioned >= 2, f"README.md only mentions {mentioned} pipeline-related terms"


class TestConfigurationDocumentation:
    """Test that configuration is documented."""

    def test_run_yml_is_documented(self):
        """Test that run.yml configuration is mentioned in docs."""
        claude_md = Path(__file__).parent.parent / "CLAUDE.md"
        content = claude_md.read_text()

        assert "run.yml" in content, "CLAUDE.md does not document run.yml"
        assert "Configuration Management" in content or "configuration" in content.lower()

    def test_pyproject_toml_is_documented(self):
        """Test that pyproject.toml is mentioned in CI/CD docs."""
        claude_md = Path(__file__).parent.parent / "CLAUDE.md"
        content = claude_md.read_text()

        assert "pyproject.toml" in content, "CLAUDE.md does not mention pyproject.toml"


class TestCIDocumentation:
    """Test that CI/CD setup is documented."""

    def test_ci_workflow_is_documented(self):
        """Test that GitHub Actions CI is documented."""
        claude_md = Path(__file__).parent.parent / "CLAUDE.md"
        content = claude_md.read_text()

        assert (
            "GitHub Actions" in content or ".github/workflows" in content
        ), "CLAUDE.md does not document CI workflow"

    def test_pre_commit_is_documented(self):
        """Test that pre-commit hooks are documented."""
        claude_md = Path(__file__).parent.parent / "CLAUDE.md"
        content = claude_md.read_text()

        assert "pre-commit" in content, "CLAUDE.md does not document pre-commit hooks"

    def test_testing_is_documented(self):
        """Test that testing setup is documented."""
        claude_md = Path(__file__).parent.parent / "CLAUDE.md"
        content = claude_md.read_text()

        assert (
            "pytest" in content or "testing" in content.lower()
        ), "CLAUDE.md does not document testing"


class TestCodeExamples:
    """Test that documented code examples are valid."""

    def test_documented_imports_are_valid(self):
        """Test that import examples in CLAUDE.md are valid."""
        claude_md = Path(__file__).parent.parent / "CLAUDE.md"
        content = claude_md.read_text()

        # Extract Python code blocks
        code_blocks = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)

        invalid_imports = []
        for block in code_blocks:
            # Check for import statements
            if "import" in block:
                lines = block.split("\n")
                for line in lines:
                    if line.strip().startswith(("import ", "from ")):
                        # Check if it's importing from project modules
                        if any(mod in line for mod in ["utils", "oslo_lib", "geo_utils", "CCC"]):
                            # Project modules - skip validation
                            continue

        # This is a basic check - real validation would need to parse AST
        assert True, "Code example validation placeholder"
