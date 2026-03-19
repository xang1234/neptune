# Release Candidate Checklist — Neptune AIS v0.1.0

**Date:** 2026-03-19
**RC Status:** PASS — all gates clear

## Checklist Results

| # | Gate | Result | Details |
|---|------|--------|---------|
| 1 | Full test suite | PASS | 768 passed, 1 skipped, 0 failed |
| 2 | Performance benchmarks | PASS | 17 benchmarks, no regressions vs baseline |
| 3 | Adapter certification | PASS | 89/89 tests, all 6 sources certified |
| 4 | Reproducibility | PASS | 20/20 tests, deterministic writes confirmed |
| 5 | Packaging validation | PASS | 41/41 tests, extras isolation verified |
| 6 | Documentation audit | PASS | 5 drift items found and fixed |
| 7 | Open blockers | PASS | 0 blocking issues (3 open are epics being closed) |

## Test Coverage Summary

| Test Suite | Tests | Purpose |
|------------|-------|---------|
| test_stream.py | 77 | Stream lifecycle, backpressure, dedup, health |
| test_sinks.py | 27 | ParquetSink, DuckDBSink, promote_landing |
| test_streaming_soak.py | 16 | E2E streaming, sustained load, stats invariants |
| test_gfw_adapter.py | 27 | GFW normalization, sentinels, vessel extraction |
| test_finland_adapter.py | 32 | Finland scaling, epoch timestamps, coord swap |
| test_aishub_adapter.py | 33 | AISHub quality, dimensions, response formats |
| test_dma_adapter.py | (existing) | DMA normalization |
| test_adapter_certification.py | 89 | Cross-adapter certification harness |
| test_reproducibility.py | 20 | Schema migration, rebuild, cache invalidation |
| test_packaging.py | 41 | pyproject, modules, extras isolation |
| test_plugin_registry.py | 18 | Plugin loading, find_sources, discovery |
| (other existing tests) | ~388 | Positions, tracks, events, viz, boundaries, etc. |
| **Total** | **768** | |

## Platform Surface

| Subsystem | Components |
|-----------|------------|
| Sources | 6 adapters: NOAA, DMA, GFW, Finland, AISHub, AISStream |
| Storage | 3-layer store (raw/canonical/derived), PartitionWriter, manifests |
| Catalog | Schema versioning, health checks, provenance tracking |
| Query | Polars LazyFrame, DuckDB SQL views, fusion (union/best/prefer) |
| Events | 4 detectors: port_call, eez_crossing, encounter, loitering |
| Streaming | NeptuneStream, backpressure, dedup, compaction, health |
| Sinks | ParquetSink, DuckDBSink, promote_landing |
| CLI | 10 commands: download, inventory, qc, sql, health, sources, fusion, events, provenance, promote |
| Plugins | Entry point discovery, load_all_adapters, find_sources |
| Examples | 6 narrative examples |

## Known Follow-Up Work (Post-Release)

These are not release blockers. They are explicit follow-up items
for after v0.1.0 ships.

### Architecture
- **Extract shared `_cast_and_normalize` helper** — The column-iteration
  pattern in adapter normalization is duplicated across 5 adapters.
  A shared helper in `base.py` accepting per-adapter config (timestamp
  format, sentinels, scaling) would eliminate ~100 lines of duplication.
  Deferred because each adapter works correctly and the duplication is
  structural, not buggy.

- **Norway adapter** — Deferred with documented rationale and reopen
  criteria in `adapters/norway.py`. Requires NMEA parser + institutional
  access.

### Testing
- **CLI integration tests** — CLI commands are tested indirectly via
  their underlying functions but not via `click.testing.CliRunner`.
  Adding CliRunner tests would catch argument parsing edge cases.

- **Streaming tests with real WebSocket** — Current soak tests use
  synthetic ingest. A real AISStream connection test (skippable without
  API key) would validate the full WebSocket→normalize→ingest path.

### Packaging
- **`conftest.py` extraction** — `_run()`, `_sample_message()`, and
  skip guards (`HAS_PYDANTIC`, `HAS_DUCKDB`) are duplicated across
  3+ test files. Extracting to `tests/conftest.py` would reduce ~50
  lines of test boilerplate.

- **Register `soak` pytest mark** — `pytest.mark.soak` triggers a
  warning about unregistered marks. Adding `[tool.pytest.ini_options]`
  to `pyproject.toml` would silence it.

### Documentation
- **API reference docs** — Module docstrings are comprehensive but
  there is no rendered API reference (Sphinx/MkDocs). This is a
  post-release documentation enhancement.

## Readiness Statement

Neptune AIS v0.1.0 is ready for release. All 7 RC gates pass.
The platform delivers a complete AIS data pipeline from source
discovery through archival ingest, multi-source fusion, event
detection, streaming with backpressure, and canonical promotion.
6 sources are certified, 768 tests pass, and performance baselines
are established.

Follow-up work is documented above and tracked as explicit items,
not hidden gaps.
