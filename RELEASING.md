# Releasing Neptune AIS

Steps to cut a release. Follow in order.

## Pre-release checks

```bash
# 1. All tests pass
python -m pytest tests/ -q

# 2. Benchmarks run without regression
python benchmarks/bench_core.py --rows 100000

# 3. Adapter certification passes
python -m pytest tests/test_adapter_certification.py -v

# 4. Reproducibility tests pass
python -m pytest tests/test_reproducibility.py -v

# 5. Packaging tests pass
python -m pytest tests/test_packaging.py -v
```

## Version bump

Edit `pyproject.toml` and `neptune_ais/__init__.py`:

```python
# pyproject.toml
version = "0.1.0"  # remove dev0 suffix

# neptune_ais/__init__.py
__version__ = "0.1.0"
```

## Build

```bash
pip install build
python -m build
```

This produces:
- `dist/neptune_ais-0.1.0.tar.gz` (source)
- `dist/neptune_ais-0.1.0-py3-none-any.whl` (wheel)

## Verify the wheel

```bash
# Install in a clean venv and verify core import
python -m venv /tmp/neptune-test
/tmp/neptune-test/bin/pip install dist/neptune_ais-0.1.0-py3-none-any.whl
/tmp/neptune-test/bin/python -c "from neptune_ais import __version__; print(__version__)"

# Verify extras install correctly
/tmp/neptune-test/bin/pip install "dist/neptune_ais-0.1.0-py3-none-any.whl[sql]"
/tmp/neptune-test/bin/python -c "import duckdb; print('sql extra works')"

/tmp/neptune-test/bin/pip install "dist/neptune_ais-0.1.0-py3-none-any.whl[cli]"
/tmp/neptune-test/bin/python -c "import click; print('cli extra works')"
```

## Publish

```bash
# Push the release commit and tag. The GitHub Actions workflow
# .github/workflows/publish-pypi.yml will build and publish to PyPI
# via trusted publishing.
git push
git tag v0.1.0
git push origin v0.1.0
```

## Post-release

1. Confirm the `Publish to PyPI` GitHub Actions workflow succeeded
2. Bump version to next dev: `0.2.0dev0`
3. Update benchmark baseline if needed

## Extras reference

| Extra | Deps | Used by |
|-------|------|---------|
| `[sql]` | duckdb | `Neptune.sql()`, `DuckDBSink` |
| `[parquet]` | pyarrow | Full Parquet write options |
| `[geo]` | shapely, geopandas, movingpandas, lonboard, h3 | `geometry/`, `viz.py` |
| `[stream]` | websockets | `NeptuneStream`, AISStream adapter |
| base install | click | `neptune` console script |
| `[cli]` | rich | Richer terminal output |
| `[all]` | all of the above | Full-featured install |
| `[dev]` | pytest, pytest-asyncio, ruff, mypy, coverage | Development only |
