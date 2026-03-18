"""DMA adapter fixtures and compatibility checks.

Exercises the DMA adapter against the shared contract, verifies schema
conformance, provenance, and compatibility with NOAA data for future
fusion tests.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from neptune_ais.adapters.base import RawArtifact, SourceAdapter
from neptune_ais.adapters.dma import DMAAdapter, SOURCE_ID
from neptune_ais.adapters.noaa import NOAAAdapter
from neptune_ais.api import Neptune
from neptune_ais.catalog import (
    CatalogRegistry,
    Manifest,
    QCSummary,
    BBox,
    WriteStatus,
    current_schema_version,
)
from neptune_ais.storage import (
    PartitionWriter,
    RawPolicy,
    shard_filename,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic DMA raw data
# ---------------------------------------------------------------------------

# Vessel 219000001 appears in both DMA and NOAA fixtures (for fusion tests).
# Vessel 219000002 is DMA-only.
SHARED_MMSI = 219000001
DMA_ONLY_MMSI = 219000002


def _create_dma_csv(path: Path) -> None:
    """Write a synthetic DMA-format semicolon-delimited CSV."""
    lines = [
        "# Timestamp;Type of mobile;MMSI;Latitude;Longitude;Navigational status;"
        "ROT;SOG;COG;Heading;IMO;Callsign;Name;Ship type;Cargo type;Width;Length;"
        "Type of position fixing device;Draught;Destination;ETA;Data source type",
        # Shared vessel — also in NOAA fixtures
        f"15/06/2024 00:00:00;Class A;{SHARED_MMSI};55.6761;12.5683;"
        "Under way using engine;0;10.5;180.0;180;9876543;OXYZ;SHARED VESSEL;70;"
        "0;20;100;GPS;5.0;COPENHAGEN;15/06/2024 12:00:00;AIS",
        f"15/06/2024 00:05:00;Class A;{SHARED_MMSI};55.6800;12.5700;"
        "Under way using engine;0;11.0;185.0;511;9876543;SHARED VESSEL;70;"
        "0;20;100;GPS;5.0;COPENHAGEN;;AIS",
        f"15/06/2024 00:10:00;Class A;{SHARED_MMSI};55.6850;12.5750;"
        "Under way using engine;0;10.8;182.0;183;9876543;SHARED VESSEL;70;"
        "0;20;100;GPS;5.0;COPENHAGEN;;AIS",
        # DMA-only vessel
        f"15/06/2024 00:00:00;Class A;{DMA_ONLY_MMSI};56.0000;10.0000;"
        "At anchor;0;0.0;0.0;270;0;OABC;DMA ONLY VESSEL;30;"
        "0;10;50;GPS;3.0;AARHUS;;AIS",
        f"15/06/2024 00:05:00;Class A;{DMA_ONLY_MMSI};56.0001;10.0001;"
        "At anchor;0;0.1;5.0;511;;OABC;DMA ONLY VESSEL;30;"
        "0;10;50;GPS;3.0;AARHUS;;AIS",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _create_noaa_parquet_with_shared_vessel(path: Path) -> None:
    """Write synthetic NOAA data with the shared vessel for fusion tests."""
    df = pl.DataFrame({
        "MMSI": [SHARED_MMSI, SHARED_MMSI, 123456789],
        "BaseDateTime": [
            "2024-06-15T00:00:00",
            "2024-06-15T00:05:00",
            "2024-06-15T00:00:00",
        ],
        "LAT": [55.6761, 55.6800, 40.0],
        "LON": [12.5683, 12.5700, -74.0],
        "SOG": [10.5, 11.0, 5.0],
        "COG": [180.0, 185.0, 90.0],
        "Heading": [180.0, 511.0, 90.0],
        "VesselName": ["SHARED VESSEL", "SHARED VESSEL", "US VESSEL"],
        "IMO": ["IMO9876543", "IMO9876543", "IMO1111111"],
        "CallSign": ["OXYZ", "OXYZ", "WCALL"],
        "VesselType": [70, 70, 80],
        "Status": ["Under way", "Under way", "Under way"],
        "Length": [100.0, 100.0, 200.0],
        "Width": [20.0, 20.0, 30.0],
        "Draft": [5.0, 5.0, 8.0],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dma_protocol_conformance():
    """DMA adapter satisfies the SourceAdapter protocol."""
    adapter = DMAAdapter()
    assert isinstance(adapter, SourceAdapter)
    assert adapter.source_id == "dma"
    assert adapter.capabilities.supports_backfill is True
    assert adapter.capabilities.supports_streaming is False


def test_dma_normalize_positions_schema():
    """DMA normalization produces a DataFrame conforming to positions schema."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "test.csv"
        _create_dma_csv(csv_path)

        art = RawArtifact(
            source_url="test://", filename="test.csv",
            local_path=str(csv_path), content_hash="test",
            size_bytes=csv_path.stat().st_size,
            fetch_timestamp=datetime.now(timezone.utc),
        )

        adapter = DMAAdapter()
        df = adapter.normalize_positions([art])

        # Row count.
        assert len(df) == 5

        # Types match canonical schema for present columns.
        assert df["mmsi"].dtype == pl.Int64
        assert df["timestamp"].dtype == pl.Datetime("us", "UTC")
        assert df["lat"].dtype == pl.Float64
        assert df["lon"].dtype == pl.Float64
        assert df["sog"].dtype == pl.Float64
        assert df["source"].dtype == pl.String

        # Source provenance.
        assert all(s == "dma" for s in df["source"].to_list())


def test_dma_sentinel_normalization():
    """DMA sentinel values are correctly normalized to null."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "test.csv"
        _create_dma_csv(csv_path)

        art = RawArtifact(
            source_url="test://", filename="test.csv",
            local_path=str(csv_path), content_hash="test",
            size_bytes=csv_path.stat().st_size,
            fetch_timestamp=datetime.now(timezone.utc),
        )

        adapter = DMAAdapter()
        df = adapter.normalize_positions([art])

        # Heading=511 → null (rows 1 and 4 have 511).
        headings = df["heading"].to_list()
        assert headings[1] is None, f"Heading 511 should be null, got {headings[1]}"
        assert headings[4] is None, f"Heading 511 should be null, got {headings[4]}"
        assert headings[0] == 180.0, "Non-sentinel heading should be preserved"

        # IMO="0" or "" → null (rows 3 and 4).
        imos = df["imo"].to_list()
        assert imos[3] is None, f"IMO '0' should be null, got {imos[3]}"
        assert imos[4] is None, f"IMO '' should be null, got {imos[4]}"
        assert imos[0] is not None, "Valid IMO should be preserved"


def test_dma_timestamp_format():
    """DMA DD/MM/YYYY HH:MM:SS timestamps parse correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "test.csv"
        _create_dma_csv(csv_path)

        art = RawArtifact(
            source_url="test://", filename="test.csv",
            local_path=str(csv_path), content_hash="test",
            size_bytes=csv_path.stat().st_size,
            fetch_timestamp=datetime.now(timezone.utc),
        )

        adapter = DMAAdapter()
        df = adapter.normalize_positions([art])

        ts = df["timestamp"][0]
        assert ts.year == 2024
        assert ts.month == 6
        assert ts.day == 15
        assert ts.hour == 0


def test_dma_normalize_vessels():
    """DMA vessel extraction produces valid vessel records."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "test.csv"
        _create_dma_csv(csv_path)

        art = RawArtifact(
            source_url="test://", filename="test.csv",
            local_path=str(csv_path), content_hash="test",
            size_bytes=csv_path.stat().st_size,
            fetch_timestamp=datetime.now(timezone.utc),
        )

        adapter = DMAAdapter()
        vessels = adapter.normalize_vessels([art])

        assert vessels is not None
        assert len(vessels) == 2
        assert "first_seen" in vessels.columns
        assert "last_seen" in vessels.columns
        assert "source" in vessels.columns
        assert "record_provenance" in vessels.columns
        assert all(s == "dma" for s in vessels["source"].to_list())


def test_dma_noaa_schema_compatibility():
    """DMA and NOAA produce DataFrames with compatible column types.

    This is critical for fusion — both sources must normalize to the
    same canonical schema so they can be concatenated or deduplicated.
    """
    with tempfile.TemporaryDirectory() as tmp:
        # DMA data.
        dma_csv = Path(tmp) / "dma.csv"
        _create_dma_csv(dma_csv)
        dma_art = RawArtifact(
            source_url="test://", filename="dma.csv",
            local_path=str(dma_csv), content_hash="dma",
            size_bytes=dma_csv.stat().st_size,
            fetch_timestamp=datetime.now(timezone.utc),
        )
        dma_df = DMAAdapter().normalize_positions([dma_art])

        # NOAA data.
        noaa_pq = Path(tmp) / "noaa.parquet"
        _create_noaa_parquet_with_shared_vessel(noaa_pq)
        noaa_art = RawArtifact(
            source_url="test://", filename="noaa.parquet",
            local_path=str(noaa_pq), content_hash="noaa",
            size_bytes=noaa_pq.stat().st_size,
            fetch_timestamp=datetime.now(timezone.utc),
        )
        noaa_df = NOAAAdapter().normalize_positions([noaa_art])

        # Both should have the same core columns with the same types.
        shared_cols = set(dma_df.columns) & set(noaa_df.columns)
        assert "mmsi" in shared_cols
        assert "timestamp" in shared_cols
        assert "lat" in shared_cols
        assert "lon" in shared_cols
        assert "source" in shared_cols

        for col in shared_cols:
            assert dma_df[col].dtype == noaa_df[col].dtype, (
                f"Column {col!r}: DMA={dma_df[col].dtype}, NOAA={noaa_df[col].dtype}"
            )

        # The shared vessel appears in both.
        dma_mmsis = set(dma_df["mmsi"].to_list())
        noaa_mmsis = set(noaa_df["mmsi"].to_list())
        assert SHARED_MMSI in dma_mmsis
        assert SHARED_MMSI in noaa_mmsis

        # Can concatenate without error (proves schema compatibility).
        common = sorted(shared_cols)
        combined = pl.concat([dma_df.select(common), noaa_df.select(common)])
        assert len(combined) == len(dma_df) + len(noaa_df)


def test_dma_multi_source_catalog():
    """DMA and NOAA partitions coexist in the same catalog."""
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp)

        # Write DMA partition.
        dma_csv = store / "raw" / "dma.csv"
        _create_dma_csv(dma_csv)
        dma_art = RawArtifact(
            source_url="test://", filename="dma.csv",
            local_path=str(dma_csv), content_hash="dma",
            size_bytes=dma_csv.stat().st_size,
            fetch_timestamp=datetime.now(timezone.utc),
        )
        dma_df = DMAAdapter().normalize_positions([dma_art])
        dma_df = dma_df.with_columns(
            pl.lit("dma.csv").alias("source_file"),
            pl.lit("batch-dma").alias("ingest_id"),
            pl.lit("ok").alias("qc_severity"),
            pl.lit("dma:direct").alias("record_provenance"),
        ).sort(["mmsi", "timestamp"])

        w1 = PartitionWriter(store, "positions", "dma", "2024-06-15")
        w1.prepare()
        dma_df.write_parquet(w1.shard_path(0))
        w1.validate(expected_files=[shard_filename(0)])
        m1 = Manifest(
            dataset="positions", source="dma", date="2024-06-15",
            schema_version=current_schema_version("positions"),
            adapter_version="dma/0.1.0", transform_version="normalize/0.1.0",
            files=[shard_filename(0)], raw_policy=RawPolicy.METADATA,
            record_count=len(dma_df), distinct_mmsi_count=dma_df["mmsi"].n_unique(),
            min_timestamp=dma_df["timestamp"].min(),
            max_timestamp=dma_df["timestamp"].max(),
            bbox=BBox(west=dma_df["lon"].min(), south=dma_df["lat"].min(),
                      east=dma_df["lon"].max(), north=dma_df["lat"].max()),
            qc_summary=QCSummary(total_rows=len(dma_df), rows_ok=len(dma_df),
                                 rows_warning=0, rows_error=0),
            write_status=WriteStatus.COMMITTED,
        )
        w1.commit(manifest_json=m1.model_dump_json(indent=2))

        # Write NOAA partition.
        noaa_pq = store / "raw" / "noaa.parquet"
        _create_noaa_parquet_with_shared_vessel(noaa_pq)
        noaa_art = RawArtifact(
            source_url="test://", filename="noaa.parquet",
            local_path=str(noaa_pq), content_hash="noaa",
            size_bytes=noaa_pq.stat().st_size,
            fetch_timestamp=datetime.now(timezone.utc),
        )
        noaa_df = NOAAAdapter().normalize_positions([noaa_art])
        noaa_df = noaa_df.with_columns(
            pl.lit("noaa.parquet").alias("source_file"),
            pl.lit("batch-noaa").alias("ingest_id"),
            pl.lit("ok").alias("qc_severity"),
            pl.lit("noaa:direct").alias("record_provenance"),
        ).sort(["mmsi", "timestamp"])

        w2 = PartitionWriter(store, "positions", "noaa", "2024-06-15")
        w2.prepare()
        noaa_df.write_parquet(w2.shard_path(0))
        w2.validate(expected_files=[shard_filename(0)])
        m2 = Manifest(
            dataset="positions", source="noaa", date="2024-06-15",
            schema_version=current_schema_version("positions"),
            adapter_version="noaa/0.1.0", transform_version="normalize/0.1.0",
            files=[shard_filename(0)], raw_policy=RawPolicy.METADATA,
            record_count=len(noaa_df), distinct_mmsi_count=noaa_df["mmsi"].n_unique(),
            min_timestamp=noaa_df["timestamp"].min(),
            max_timestamp=noaa_df["timestamp"].max(),
            bbox=BBox(west=noaa_df["lon"].min(), south=noaa_df["lat"].min(),
                      east=noaa_df["lon"].max(), north=noaa_df["lat"].max()),
            qc_summary=QCSummary(total_rows=len(noaa_df), rows_ok=len(noaa_df),
                                 rows_warning=0, rows_error=0),
            write_status=WriteStatus.COMMITTED,
        )
        w2.commit(manifest_json=m2.model_dump_json(indent=2))

        # --- Verify catalog sees both ---
        reg = CatalogRegistry(store)
        reg.scan()

        parts = reg.partitions("positions")
        assert len(parts) == 2
        sources = {p.source for p in parts}
        assert sources == {"dma", "noaa"}

        # --- Verify Neptune can query both ---
        # Use union merge to get all rows from both sources.
        n = Neptune("2024-06-15", sources=["noaa", "dma"], merge="union", cache_dir=store)
        pos = n.positions().collect()
        assert len(pos) == len(dma_df) + len(noaa_df)

        # The shared vessel appears from both sources.
        shared = pos.filter(pl.col("mmsi") == SHARED_MMSI)
        shared_sources = set(shared["source"].to_list())
        assert shared_sources == {"dma", "noaa"}

        # DuckDB sees the raw (unfused) data — it doesn't apply fusion.
        count = n.sql("SELECT count(*) FROM positions").fetchone()[0]
        assert count == len(dma_df) + len(noaa_df)

        # Inventory shows both sources.
        inv = n.inventory()
        assert len(inv) == 1  # One dataset: positions
        assert set(inv[0].sources) == {"dma", "noaa"}
        assert inv[0].total_rows == len(dma_df) + len(noaa_df)


if __name__ == "__main__":
    print("Running DMA adapter tests...")
    test_dma_protocol_conformance()
    print("  protocol_conformance ✓")
    test_dma_normalize_positions_schema()
    print("  normalize_positions_schema ✓")
    test_dma_sentinel_normalization()
    print("  sentinel_normalization ✓")
    test_dma_timestamp_format()
    print("  timestamp_format ✓")
    test_dma_normalize_vessels()
    print("  normalize_vessels ✓")
    test_dma_noaa_schema_compatibility()
    print("  noaa_schema_compatibility ✓")
    test_dma_multi_source_catalog()
    print("  multi_source_catalog ✓")
    print("\nAll DMA adapter tests passed ✓")
