"""Packaging validation — verify the package is release-ready.

Checks that pyproject.toml, module structure, version, extras,
and entry points are consistent and complete.

These tests run without building a wheel — they validate the
configuration and module structure directly.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


# ---------------------------------------------------------------------------
# pyproject.toml structure
# ---------------------------------------------------------------------------


class TestPyprojectStructure:
    """Verify pyproject.toml is well-formed and complete."""

    def test_pyproject_exists(self):
        assert PYPROJECT.exists()

    def test_has_build_system(self):
        text = PYPROJECT.read_text()
        assert "[build-system]" in text

    def test_has_project_metadata(self):
        text = PYPROJECT.read_text()
        assert 'name = "neptune-ais"' in text
        assert "version =" in text
        assert "description =" in text
        assert "license =" in text
        assert "requires-python =" in text

    def test_has_dependencies(self):
        text = PYPROJECT.read_text()
        assert "dependencies =" in text
        assert '"polars' in text
        assert '"pydantic' in text
        assert '"httpx' in text

    def test_has_optional_dependencies(self):
        text = PYPROJECT.read_text()
        for extra in ("sql", "parquet", "geo", "stream", "cli", "dev", "all"):
            assert f"{extra} =" in text or f'{extra} =' in text, (
                f"Missing optional dependency group: {extra}"
            )

    def test_has_entry_point(self):
        text = PYPROJECT.read_text()
        assert "[project.scripts]" in text
        assert "neptune =" in text

    def test_has_wheel_config(self):
        text = PYPROJECT.read_text()
        assert "[tool.hatch.build.targets.wheel]" in text
        assert 'packages = ["neptune_ais"]' in text


# ---------------------------------------------------------------------------
# Version consistency
# ---------------------------------------------------------------------------


class TestVersionConsistency:
    """Version is defined and accessible."""

    def test_version_in_init(self):
        from neptune_ais import __version__
        assert __version__
        assert isinstance(__version__, str)

    def test_version_in_pyproject(self):
        text = PYPROJECT.read_text()
        assert "version =" in text

    def test_version_format(self):
        from neptune_ais import __version__
        # Should be semver-ish: X.Y.Z or X.Y.ZdevN or X.Y.Z.rcN
        parts = __version__.replace("dev", ".").replace("rc", ".").split(".")
        assert len(parts) >= 3, f"Version {__version__!r} doesn't look like semver"


# ---------------------------------------------------------------------------
# Module structure — all expected modules are importable
# ---------------------------------------------------------------------------


class TestModuleStructure:
    """All package modules are importable without optional deps."""

    @pytest.mark.parametrize("module", [
        "neptune_ais",
        "neptune_ais.api",
        "neptune_ais.catalog",
        "neptune_ais.storage",
        "neptune_ais.stream",
        "neptune_ais.sinks",
        "neptune_ais.qc",
        "neptune_ais.fusion",
        "neptune_ais.helpers",
        "neptune_ais.sql",
        "neptune_ais.datasets",
        "neptune_ais.datasets.positions",
        "neptune_ais.datasets.vessels",
        "neptune_ais.datasets.tracks",
        "neptune_ais.datasets.events",
        "neptune_ais.adapters",
        "neptune_ais.adapters.base",
        "neptune_ais.adapters.registry",
    ])
    def test_core_module_importable(self, module: str):
        """Core modules import without requiring optional deps."""
        importlib.import_module(module)

    @pytest.mark.parametrize("module", [
        "neptune_ais.adapters.noaa",
        "neptune_ais.adapters.dma",
        "neptune_ais.adapters.gfw",
        "neptune_ais.adapters.finland",
        "neptune_ais.adapters.aishub",
        "neptune_ais.adapters.aisstream",
    ])
    def test_adapter_module_importable(self, module: str):
        """All adapter modules import successfully."""
        importlib.import_module(module)

    def test_no_stale_pyc_only_modules(self):
        """No .pyc files without corresponding .py source."""
        pkg_dir = REPO_ROOT / "neptune_ais"
        for pyc in pkg_dir.rglob("*.pyc"):
            # pyc files are in __pycache__, source is one level up.
            source_name = pyc.stem.split(".")[0] + ".py"
            source = pyc.parent.parent / source_name
            assert source.exists(), f"Stale .pyc without source: {pyc}"


# ---------------------------------------------------------------------------
# Extras isolation — optional deps are actually optional
# ---------------------------------------------------------------------------


class TestExtrasIsolation:
    """Optional dependencies are lazily imported, not at module level."""

    def test_duckdb_not_imported_at_module_level(self):
        """duckdb is only imported inside functions, not at module level."""
        api_source = (REPO_ROOT / "neptune_ais" / "api.py").read_text()
        sinks_source = (REPO_ROOT / "neptune_ais" / "sinks.py").read_text()

        # Check that 'import duckdb' only appears inside function bodies,
        # not at the top of the file.
        for name, source in [("api.py", api_source), ("sinks.py", sinks_source)]:
            lines = source.split("\n")
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped in ("import duckdb", "from duckdb import"):
                    # Must be indented (inside a function), not at column 0.
                    assert line[0] == " ", (
                        f"{name}:{i+1}: 'import duckdb' at module level — "
                        "should be inside a function for lazy loading"
                    )

    def test_websockets_not_imported_at_module_level(self):
        """websockets is only imported inside functions."""
        aisstream_source = (REPO_ROOT / "neptune_ais" / "adapters" / "aisstream.py").read_text()
        lines = aisstream_source.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "import websockets" in stripped:
                assert line[0] == " ", (
                    f"aisstream.py:{i+1}: websockets at module level"
                )

    def test_click_not_imported_at_module_level_in_core(self):
        """click is only imported in cli/, not in core modules."""
        core_modules = [
            "api.py", "catalog.py", "storage.py", "stream.py",
            "sinks.py", "qc.py", "fusion.py", "helpers.py",
        ]
        for mod in core_modules:
            source = (REPO_ROOT / "neptune_ais" / mod).read_text()
            assert "import click" not in source, (
                f"{mod} imports click — should be cli-only"
            )


# ---------------------------------------------------------------------------
# Entry points and plugin system
# ---------------------------------------------------------------------------


class TestEntryPoints:
    """Plugin entry point group is declared and functional."""

    def test_plugin_entry_point_group_defined(self):
        from neptune_ais.adapters.registry import ENTRY_POINT_GROUP
        assert ENTRY_POINT_GROUP == "neptune_ais.adapters"

    def test_load_all_adapters_works(self):
        from neptune_ais.adapters.registry import load_all_adapters
        sources = load_all_adapters()
        assert len(sources) >= 6

    def test_cli_entry_point_callable(self):
        """The neptune CLI entry point is importable and callable."""
        from neptune_ais.cli.main import cli
        assert callable(cli)
