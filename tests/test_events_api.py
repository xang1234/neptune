"""Tests for events API accessor and CLI surface.

Validates that Neptune.events() returns the correct LazyFrame with
filtering by event_type, min_confidence, and MMSI, and that the
CLI events command is wired correctly.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from neptune_ais.api import Neptune
from neptune_ais.datasets.events import Col, SCHEMA, SCHEMA_VERSION

try:
    import duckdb  # noqa: F401
    _has_duckdb = True
except ImportError:
    _has_duckdb = False

try:
    import click  # noqa: F401
    _has_click = True
except ImportError:
    _has_click = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_events_partition(store_root: Path) -> pl.DataFrame:
    """Write synthetic events to the derived store and return the DataFrame."""
    from neptune_ais.catalog import (
        BBox,
        Manifest,
        QCSummary,
        WriteStatus,
        current_schema_version,
    )
    from neptune_ais.storage import PartitionWriter, RawPolicy, shard_filename

    events_df = pl.DataFrame({
        Col.EVENT_ID: ["evt_aaa", "evt_bbb", "evt_ccc", "evt_ddd"],
        Col.EVENT_TYPE: ["port_call", "port_call", "encounter", "loitering"],
        Col.MMSI: pl.Series([111, 222, 111, 333], dtype=pl.Int64),
        Col.OTHER_MMSI: pl.Series([None, None, 222, None], dtype=pl.Int64),
        Col.START_TIME: pl.Series([
            datetime(2024, 6, 15, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 15, 2, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 15, 4, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 15, 6, 0, tzinfo=timezone.utc),
        ], dtype=pl.Datetime("us", "UTC")),
        Col.END_TIME: pl.Series([
            datetime(2024, 6, 15, 1, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 15, 3, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 15, 5, 0, tzinfo=timezone.utc),
            datetime(2024, 6, 15, 7, 0, tzinfo=timezone.utc),
        ], dtype=pl.Datetime("us", "UTC")),
        Col.LAT: [40.0, 55.0, 42.0, 30.0],
        Col.LON: [-74.0, 12.0, -70.0, -80.0],
        Col.GEOMETRY_WKB: pl.Series([None] * 4, dtype=pl.Binary),
        Col.CONFIDENCE_SCORE: [0.9, 0.5, 0.8, 0.2],
        Col.SOURCE: ["noaa", "noaa", "noaa", "noaa"],
        Col.RECORD_PROVENANCE: [
            "noaa:port_call_detector/0.1.0[tracks]",
            "noaa:port_call_detector/0.1.0[tracks]",
            "noaa:encounter_detector/0.1.0[tracks]",
            "noaa:loitering_detector/0.1.0[tracks]",
        ],
    })

    writer = PartitionWriter(store_root, "events", "noaa", "2024-06-15")
    staging = writer.prepare()
    events_df.write_parquet(staging / shard_filename(0))
    writer.validate(expected_files=[shard_filename(0)])

    manifest = Manifest(
        dataset="events",
        source="noaa",
        date="2024-06-15",
        schema_version=SCHEMA_VERSION,
        adapter_version="events/derived",
        transform_version="detectors/0.1.0",
        files=[shard_filename(0)],
        raw_policy=RawPolicy.METADATA,
        record_count=len(events_df),
        distinct_mmsi_count=events_df[Col.MMSI].n_unique(),
        min_timestamp=events_df[Col.START_TIME].min(),
        max_timestamp=events_df[Col.END_TIME].max(),
        bbox=BBox(
            west=events_df[Col.LON].min(),
            south=events_df[Col.LAT].min(),
            east=events_df[Col.LON].max(),
            north=events_df[Col.LAT].max(),
        ),
        qc_summary=QCSummary(
            total_rows=len(events_df),
            rows_ok=len(events_df),
            rows_warning=0,
            rows_error=0,
        ),
        write_status=WriteStatus.COMMITTED,
    )
    writer.commit(manifest_json=manifest.model_dump_json(indent=2))

    return events_df


# ---------------------------------------------------------------------------
# Neptune.events() accessor
# ---------------------------------------------------------------------------


class TestEventsAccessor:
    """Test Neptune.events() with synthetic stored events."""

    def _setup(self, tmp: str) -> tuple[Neptune, pl.DataFrame]:
        store = Path(tmp)
        events_df = _write_events_partition(store)
        n = Neptune("2024-06-15", sources=["noaa"], cache_dir=store)
        return n, events_df

    def test_events_returns_lazyframe(self):
        with tempfile.TemporaryDirectory() as tmp:
            n, _ = self._setup(tmp)
            result = n.events()
            assert isinstance(result, pl.LazyFrame)

    def test_events_returns_all_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            n, events_df = self._setup(tmp)
            result = n.events().collect()
            assert len(result) == len(events_df)

    def test_events_filter_by_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            n, _ = self._setup(tmp)
            result = n.events(kind="port_call").collect()
            assert len(result) == 2
            assert (result[Col.EVENT_TYPE] == "port_call").all()

    def test_events_filter_by_encounter(self):
        with tempfile.TemporaryDirectory() as tmp:
            n, _ = self._setup(tmp)
            result = n.events(kind="encounter").collect()
            assert len(result) == 1
            assert result[Col.OTHER_MMSI][0] == 222

    def test_events_filter_by_min_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            n, _ = self._setup(tmp)
            result = n.events(min_confidence=0.7).collect()
            assert len(result) == 2  # 0.9 and 0.8
            assert (result[Col.CONFIDENCE_SCORE] >= 0.7).all()

    def test_events_filter_by_kind_and_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            n, _ = self._setup(tmp)
            result = n.events(kind="port_call", min_confidence=0.7).collect()
            assert len(result) == 1  # only the 0.9 port_call
            assert result[Col.CONFIDENCE_SCORE][0] == 0.9

    def test_events_filter_by_mmsi(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            _write_events_partition(store)
            n = Neptune(
                "2024-06-15", sources=["noaa"], mmsi=[111], cache_dir=store
            )
            result = n.events().collect()
            assert len(result) == 2  # port_call + encounter for mmsi 111
            assert (result[Col.MMSI] == 111).all()

    def test_events_empty_when_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            n = Neptune("2024-06-15", sources=["noaa"], cache_dir=tmp)
            result = n.events().collect()
            assert len(result) == 0
            # Should have the correct schema.
            for col in SCHEMA:
                assert col in result.columns

    def test_events_nonexistent_kind_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            n, _ = self._setup(tmp)
            result = n.events(kind="fishing").collect()
            assert len(result) == 0


# ---------------------------------------------------------------------------
# DuckDB integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_duckdb, reason="duckdb not installed")
class TestEventsDuckDB:
    def test_events_view_in_duckdb(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            _write_events_partition(store)
            n = Neptune("2024-06-15", sources=["noaa"], cache_dir=store)
            con = n.duckdb()
            count = con.sql("SELECT count(*) FROM events").fetchone()[0]
            assert count == 4

    def test_events_sql_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp)
            _write_events_partition(store)
            n = Neptune("2024-06-15", sources=["noaa"], cache_dir=store)
            result = n.sql(
                "SELECT event_type, count(*) as cnt "
                "FROM events GROUP BY event_type ORDER BY event_type"
            ).fetchall()
            assert ("encounter", 1) in result
            assert ("port_call", 2) in result


# ---------------------------------------------------------------------------
# CLI events command
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_click, reason="click not installed")
@pytest.mark.skipif(not _has_duckdb, reason="duckdb not installed")
class TestEventsCLI:
    def test_events_command_exists(self):
        from click.testing import CliRunner
        from neptune_ais.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["events", "--help"])
        assert result.exit_code == 0
        assert "Query stored events" in result.output

    def test_events_command_with_data(self):
        from click.testing import CliRunner
        from neptune_ais.cli.main import cli

        with tempfile.TemporaryDirectory() as tmp:
            _write_events_partition(Path(tmp))
            runner = CliRunner()
            result = runner.invoke(cli, [
                "events",
                "--date", "2024-06-15",
                "--source", "noaa",
                "--cache-dir", tmp,
            ])
            assert result.exit_code == 0
            assert "4 row(s)" in result.output

    def test_events_command_with_kind_filter(self):
        from click.testing import CliRunner
        from neptune_ais.cli.main import cli

        with tempfile.TemporaryDirectory() as tmp:
            _write_events_partition(Path(tmp))
            runner = CliRunner()
            result = runner.invoke(cli, [
                "events",
                "--date", "2024-06-15",
                "--source", "noaa",
                "--kind", "port_call",
                "--cache-dir", tmp,
            ])
            assert result.exit_code == 0
            assert "2 row(s)" in result.output

    def test_events_command_no_data(self):
        from click.testing import CliRunner
        from neptune_ais.cli.main import cli

        with tempfile.TemporaryDirectory() as tmp:
            runner = CliRunner()
            result = runner.invoke(cli, [
                "events",
                "--date", "2024-06-15",
                "--source", "noaa",
                "--cache-dir", tmp,
            ])
            assert result.exit_code == 0
            assert "No events found" in result.output
