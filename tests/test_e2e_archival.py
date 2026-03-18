"""End-to-end archival validation for the Phase 1 NOAA path.

This test exercises the full pipeline using synthetic data:
  adapter normalize → pipeline columns → sort → shard → write → catalog →
  Polars query → DuckDB query → inventory → QC report → provenance

It does NOT require network access or a real NOAA download.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

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

from neptune_ais.adapters.base import RawArtifact
from neptune_ais.adapters.noaa import NOAAAdapter
from neptune_ais.api import Neptune
from neptune_ais.catalog import (
    CatalogRegistry,
    WriteStatus,
    current_schema_version,
)
from neptune_ais.datasets.positions import SCHEMA_VERSION
from neptune_ais.storage import (
    PartitionWriter,
    shard_filename,
)


def _create_synthetic_noaa_parquet(path: Path) -> None:
    """Write a synthetic NOAA-format Parquet file."""
    df = pl.DataFrame(
        {
            "MMSI": [123456789, 123456789, 123456789, 987654321, 987654321],
            "BaseDateTime": [
                "2024-06-15T00:00:00",
                "2024-06-15T00:05:00",
                "2024-06-15T00:10:00",
                "2024-06-15T00:00:00",
                "2024-06-15T00:05:00",
            ],
            "LAT": [40.0, 40.01, 40.02, 55.0, 55.01],
            "LON": [-74.0, -74.01, -74.02, 12.0, 12.01],
            "SOG": [10.5, 11.0, 10.8, 0.0, 0.1],
            "COG": [180.0, 185.0, 182.0, 0.0, 5.0],
            "Heading": [180.0, 511.0, 185.0, 270.0, 511.0],
            "VesselName": ["VESSEL A"] * 3 + ["VESSEL B"] * 2,
            "IMO": ["IMO1234567", "IMO0000000", "IMO1234567", "IMO9876543", "IMO9876543"],
            "CallSign": ["CALL1"] * 3 + ["CALL2"] * 2,
            "VesselType": [70, 70, 70, 30, 30],
            "Status": ["Under way"] * 3 + ["At anchor"] * 2,
            "Length": [100.0] * 3 + [50.0] * 2,
            "Width": [20.0] * 3 + [10.0] * 2,
            "Draft": [5.0] * 3 + [3.0] * 2,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def test_noaa_normalize_positions():
    """Test that NOAA normalization produces valid canonical positions."""
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "test.parquet"
        _create_synthetic_noaa_parquet(raw_path)

        artifact = RawArtifact(
            source_url="test://",
            filename="test.parquet",
            local_path=str(raw_path),
            content_hash="test",
            size_bytes=raw_path.stat().st_size,
            fetch_timestamp=datetime.now(timezone.utc),
        )

        adapter = NOAAAdapter()
        df = adapter.normalize_positions([artifact])

        # Check row count.
        assert len(df) == 5

        # Check sentinel normalization.
        headings = df["heading"].to_list()
        assert headings[1] is None, "Heading 511 should be null"

        imos = df["imo"].to_list()
        assert imos[1] is None, "IMO0000000 should be null"

        # Check types.
        assert df["mmsi"].dtype == pl.Int64
        assert df["lat"].dtype == pl.Float64
        assert df["source"].to_list() == ["noaa"] * 5


def test_noaa_normalize_vessels():
    """Test vessel extraction from NOAA position data."""
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "test.parquet"
        _create_synthetic_noaa_parquet(raw_path)

        artifact = RawArtifact(
            source_url="test://",
            filename="test.parquet",
            local_path=str(raw_path),
            content_hash="test",
            size_bytes=raw_path.stat().st_size,
            fetch_timestamp=datetime.now(timezone.utc),
        )

        adapter = NOAAAdapter()
        vessels = adapter.normalize_vessels([artifact])

        assert vessels is not None
        assert len(vessels) == 2  # Two distinct MMSIs
        assert "first_seen" in vessels.columns
        assert "last_seen" in vessels.columns
        assert "record_provenance" in vessels.columns


@pytest.mark.skipif(not _has_duckdb, reason="duckdb not installed")
def test_end_to_end_archival_pipeline():
    """Full pipeline: write synthetic data → catalog → query → inspect."""
    with tempfile.TemporaryDirectory() as tmp:
        store_root = Path(tmp)

        # --- Step 1: Write synthetic NOAA data through the pipeline ---
        raw_path = store_root / "raw" / "noaa" / "2024" / "06" / "15" / "test.parquet"
        _create_synthetic_noaa_parquet(raw_path)

        adapter = NOAAAdapter()
        artifact = RawArtifact(
            source_url="test://",
            filename="test.parquet",
            local_path=str(raw_path),
            content_hash="testhash",
            size_bytes=raw_path.stat().st_size,
            fetch_timestamp=datetime.now(timezone.utc),
        )

        # Normalize.
        positions_df = adapter.normalize_positions([artifact])

        # Add pipeline-generated columns.
        positions_df = positions_df.with_columns(
            pl.lit("test.parquet").alias("source_file"),
            pl.lit("test-batch-001").alias("ingest_id"),
            pl.lit("ok").alias("qc_severity"),
            pl.lit("noaa:direct").alias("record_provenance"),
        )

        # Sort.
        positions_df = positions_df.sort(["mmsi", "timestamp"])

        # Write via PartitionWriter.
        writer = PartitionWriter(store_root, "positions", "noaa", "2024-06-15")
        staging = writer.prepare()
        positions_df.write_parquet(staging / shard_filename(0))
        writer.validate(expected_files=[shard_filename(0)])

        # Build and write manifest.
        from neptune_ais.catalog import Manifest, QCSummary, BBox
        from neptune_ais.storage import RawPolicy

        manifest = Manifest(
            dataset="positions",
            source="noaa",
            date="2024-06-15",
            schema_version=current_schema_version("positions"),
            adapter_version="noaa/0.1.0",
            transform_version="normalize/0.1.0",
            files=[shard_filename(0)],
            raw_policy=RawPolicy.METADATA,
            record_count=len(positions_df),
            distinct_mmsi_count=positions_df["mmsi"].n_unique(),
            min_timestamp=positions_df["timestamp"].min(),
            max_timestamp=positions_df["timestamp"].max(),
            bbox=BBox(
                west=positions_df["lon"].min(),
                south=positions_df["lat"].min(),
                east=positions_df["lon"].max(),
                north=positions_df["lat"].max(),
            ),
            qc_summary=QCSummary(
                total_rows=len(positions_df),
                rows_ok=len(positions_df),
                rows_warning=0,
                rows_error=0,
            ),
            write_status=WriteStatus.COMMITTED,
        )
        writer.commit(manifest_json=manifest.model_dump_json(indent=2))

        # --- Step 2: Verify catalog ---
        registry = CatalogRegistry(store_root)
        count = registry.scan()
        assert count == 1, f"Expected 1 manifest, got {count}"

        m = registry.get_manifest("positions", "noaa", "2024-06-15")
        assert m is not None
        assert m.write_status == WriteStatus.COMMITTED
        assert m.record_count == 5
        assert m.schema_version == SCHEMA_VERSION

        # --- Step 3: Query via Neptune ---
        n = Neptune("2024-06-15", sources=["noaa"], cache_dir=store_root)

        # Polars positions.
        pos = n.positions().collect()
        assert len(pos) == 5
        assert pos["mmsi"].n_unique() == 2

        # Polars with bbox filter.
        n_bbox = Neptune(
            "2024-06-15",
            sources=["noaa"],
            cache_dir=store_root,
            bbox=(-75.0, 39.0, -73.0, 41.0),
        )
        pos_filtered = n_bbox.positions().collect()
        assert len(pos_filtered) == 3  # Only vessel A's positions

        # Polars with MMSI filter.
        n_mmsi = Neptune(
            "2024-06-15",
            sources=["noaa"],
            cache_dir=store_root,
            mmsi=[987654321],
        )
        pos_mmsi = n_mmsi.positions().collect()
        assert len(pos_mmsi) == 2

        # DuckDB SQL.
        result = n.sql(
            "SELECT mmsi, count(*) as cnt "
            "FROM positions GROUP BY mmsi ORDER BY mmsi"
        ).fetchall()
        assert result == [(123456789, 3), (987654321, 2)]

        # --- Step 4: Inspection surfaces ---
        inv = n.inventory()
        assert len(inv) == 1
        assert inv[0].dataset == "positions"
        assert inv[0].total_rows == 5

        qr = n.quality_report()
        assert qr.total_rows == 5
        assert qr.ok_rate == 1.0

        prov = n.provenance()
        assert prov.schema_versions == ["positions/v1"]

        health = registry.check_health()
        assert health == [], f"Unexpected health issues: {health}"


@pytest.mark.skipif(not _has_click, reason="click not installed (cli extra)")
@pytest.mark.skipif(not _has_duckdb, reason="duckdb not installed")
def test_cli_commands():
    """Test CLI commands with --cache-dir pointing to synthetic data."""
    from click.testing import CliRunner
    from neptune_ais.cli.main import cli

    with tempfile.TemporaryDirectory() as tmp:
        store_root = Path(tmp)

        # Write synthetic data.
        raw_path = store_root / "raw" / "test.parquet"
        _create_synthetic_noaa_parquet(raw_path)
        adapter = NOAAAdapter()
        artifact = RawArtifact(
            source_url="test://", filename="test.parquet",
            local_path=str(raw_path), content_hash="test",
            size_bytes=raw_path.stat().st_size,
            fetch_timestamp=datetime.now(timezone.utc),
        )
        positions_df = adapter.normalize_positions([artifact])
        positions_df = positions_df.with_columns(
            pl.lit("test.parquet").alias("source_file"),
            pl.lit("batch").alias("ingest_id"),
            pl.lit("ok").alias("qc_severity"),
            pl.lit("noaa:direct").alias("record_provenance"),
        ).sort(["mmsi", "timestamp"])

        writer = PartitionWriter(store_root, "positions", "noaa", "2024-06-15")
        writer.prepare()
        positions_df.write_parquet(writer.shard_path(0))
        writer.validate(expected_files=[shard_filename(0)])

        from neptune_ais.catalog import Manifest, QCSummary, BBox
        from neptune_ais.storage import RawPolicy

        manifest = Manifest(
            dataset="positions", source="noaa", date="2024-06-15",
            schema_version=current_schema_version("positions"),
            adapter_version="noaa/0.1.0", transform_version="normalize/0.1.0",
            files=[shard_filename(0)], raw_policy=RawPolicy.METADATA,
            record_count=5, distinct_mmsi_count=2,
            min_timestamp=positions_df["timestamp"].min(),
            max_timestamp=positions_df["timestamp"].max(),
            bbox=BBox(west=-74.02, south=40.0, east=12.01, north=55.01),
            qc_summary=QCSummary(total_rows=5, rows_ok=5, rows_warning=0, rows_error=0),
            write_status=WriteStatus.COMMITTED,
        )
        writer.commit(manifest_json=manifest.model_dump_json(indent=2))

        runner = CliRunner()

        # neptune --version
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "neptune" in result.output

        # neptune inventory
        result = runner.invoke(cli, ["inventory", "--cache-dir", str(store_root)])
        assert result.exit_code == 0
        assert "positions" in result.output
        assert "noaa" in result.output

        # neptune qc
        result = runner.invoke(cli, [
            "qc", "--dataset", "positions", "--date", "2024-06-15",
            "--cache-dir", str(store_root),
        ])
        assert result.exit_code == 0
        assert "Rows OK" in result.output

        # neptune health
        result = runner.invoke(cli, ["health", "--cache-dir", str(store_root)])
        assert result.exit_code == 0
        assert "No issues found" in result.output

        # neptune sql
        result = runner.invoke(cli, [
            "sql", "SELECT count(*) FROM positions",
            "--date", "2024-06-15",
            "--cache-dir", str(store_root),
        ])
        assert result.exit_code == 0


if __name__ == "__main__":
    print("Running E2E tests...")
    test_noaa_normalize_positions()
    print("  normalize_positions ✓")
    test_noaa_normalize_vessels()
    print("  normalize_vessels ✓")
    test_end_to_end_archival_pipeline()
    print("  end_to_end_archival_pipeline ✓")
    test_cli_commands()
    print("  cli_commands ✓")
    print("\nAll E2E tests passed ✓")
