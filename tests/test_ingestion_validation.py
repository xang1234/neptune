"""Ingestion validation through the public API.

Drives ``Neptune.download()`` end-to-end against a NOAA-format ZIP placed in
the raw cache (so no network is used) and checks that QC actually runs:
invalid rows are quarantined, counts reconcile, vessels populate, and
per-partition outcomes are explicit.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import polars as pl

from neptune_ais.api import Neptune, PartitionStatus
from neptune_ais.storage import raw_partition_path
from neptune_ais.qc import run_checks

DAY = date(2024, 6, 15)

NOAA_HEADER = (
    "MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,VesselName,IMO,CallSign,"
    "VesselType,Status,Length,Width,Draft,Cargo,TransceiverClass"
)
VALID_ROW = "367000001,2024-06-15T00:00:00,40.0,-74.0,10.0,90.0,90,ALPHA,IMO1234567,WABC,70,0,100,20,5,70,A"
INVALID_ROW = "123,2024-06-15T00:01:00,91.0,-74.0,100.0,90.0,90,BAD,IMO0000000,WBAD,70,0,100,20,5,70,A"
FAST_ROW = "367000002,2024-06-15T00:02:00,41.0,-73.0,100.0,90.0,90,BRAVO,IMO7654321,WDEF,70,0,100,20,5,70,A"


def _seed_noaa_zip(store: Path, rows: list[str]) -> None:
    raw_dir = store / raw_partition_path("noaa", DAY.isoformat())
    raw_dir.mkdir(parents=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("AIS_2024_06_15.csv", "\n".join([NOAA_HEADER, *rows]) + "\n")
    (raw_dir / "AIS_2024_06_15.zip").write_bytes(buf.getvalue())


def test_invalid_rows_are_quarantined_and_counts_reconcile(tmp_path):
    _seed_noaa_zip(tmp_path, [VALID_ROW, INVALID_ROW, FAST_ROW])
    n = Neptune(DAY, sources=["noaa"], cache_dir=tmp_path)

    result = n.download()

    assert result.complete
    [p] = result.partitions
    assert p.status == PartitionStatus.OK
    assert (p.accepted_rows, p.quarantined_rows, p.warning_rows) == (2, 1, 1)

    positions = n.positions().collect().sort("mmsi")
    assert positions["mmsi"].to_list() == [367000001, 367000002]
    assert positions["qc_severity"].to_list() == ["ok", "warning"]
    # High speed is context-dependent: retained, but flagged.
    assert positions["qc_flags"].to_list() == [[], ["speed_plausibility"]]

    quarantined = n.quarantine().collect()
    assert quarantined["mmsi"].to_list() == [123]
    assert set(quarantined["qc_flags"][0]) == {"lat_range", "mmsi_format"}

    report = n.quality_report()
    assert report.total_rows == 3
    assert (report.rows_ok, report.rows_warning, report.rows_error, report.rows_dropped) == (1, 1, 0, 1)
    assert report.rows_written == len(positions)
    assert "lat_range" in report.checks_applied
    flagged = {c.check_name: c.rows_flagged for c in report.check_results}
    assert flagged["lat_range"] == 1
    assert flagged["speed_plausibility"] == 1


def test_vessels_populate_from_accepted_positions(tmp_path):
    _seed_noaa_zip(tmp_path, [VALID_ROW, INVALID_ROW, FAST_ROW])
    n = Neptune(DAY, sources=["noaa"], cache_dir=tmp_path)

    result = n.download()

    assert "vessels/noaa/2024-06-15" in result.written
    vessels = n.vessels().collect().sort("mmsi")
    # The quarantined MMSI 123 must not produce a vessel record.
    assert vessels["mmsi"].to_list() == [367000001, 367000002]
    assert vessels["vessel_name"].to_list() == ["ALPHA", "BRAVO"]


def test_unavailable_provider_is_explicit(tmp_path, monkeypatch):
    from neptune_ais.adapters import noaa

    def boom(*a, **kw):
        raise ConnectionError("provider down")

    monkeypatch.setattr(noaa, "download_and_hash", boom)
    result = Neptune(DAY, sources=["noaa"], cache_dir=tmp_path).download()

    assert not result.complete
    [p] = result.partitions
    assert p.status == PartitionStatus.UNAVAILABLE
    assert "provider down" in p.error
    assert result.failures == [p]


def test_normalization_failure_is_explicit(tmp_path):
    raw_dir = tmp_path / raw_partition_path("noaa", DAY.isoformat())
    raw_dir.mkdir(parents=True)
    (raw_dir / "AIS_2024_06_15.zip").write_bytes(b"not a zip")

    result = Neptune(DAY, sources=["noaa"], cache_dir=tmp_path).download()

    [p] = result.partitions
    assert p.status == PartitionStatus.FAILED
    assert p.error
    assert not result.complete


def test_no_observations_is_explicit(tmp_path):
    _seed_noaa_zip(tmp_path, [])

    result = Neptune(DAY, sources=["noaa"], cache_dir=tmp_path).download()

    [p] = result.partitions
    assert p.status == PartitionStatus.NO_DATA
    assert result.complete
    assert result.written == []


def test_monotonicity_and_stale_checks():
    ts = pl.datetime(2024, 6, 15, 0, pl.col("m"), 0, time_zone="UTC")
    df = pl.DataFrame({"m": [0, 2, 1, 3, 4, 5]}).select(
        pl.lit(367000001, pl.Int64).alias("mmsi"),
        ts.alias("timestamp"),
        pl.lit(40.0).alias("lat"),
        pl.lit(-74.0).alias("lon"),
        pl.lit("test").alias("source"),
    )
    from neptune_ais.qc import StalePositionCheck, TimestampMonotonicityCheck

    out = run_checks(df, [TimestampMonotonicityCheck(), StalePositionCheck(max_repeats=4)])

    flags = dict(zip(out.accepted["timestamp"].dt.minute(), out.accepted["qc_flags"].to_list()))
    # Minute 1 arrived after minute 2 → out of order.
    assert flags[1] == ["timestamp_monotonicity"]
    assert flags[2] == []
    # Rows 5 and 6 of an unmoving run exceed max_repeats=4.
    assert flags[4] == ["stale_position"]
    assert flags[5] == ["stale_position"]
    assert flags[3] == []


def test_promotion_quarantines_invalid_stream_rows(tmp_path):
    from neptune_ais.sinks import promote_landing

    landing = tmp_path / "landing" / "live"
    landing.mkdir(parents=True)
    pl.DataFrame({
        "mmsi": [367000001, 123],
        "timestamp": ["2024-06-15T00:00:00+00:00", "2024-06-15T00:01:00+00:00"],
        "lat": [40.0, 91.0],
        "lon": [-74.0, -74.0],
        "source": ["live", "live"],
    }).write_parquet(landing / "landing-20240615-0000.parquet")

    [r] = promote_landing(tmp_path / "landing", tmp_path, source="live")

    assert (r.record_count, r.quarantined_count) == (1, 1)
    n = Neptune(DAY, sources=["live"], cache_dir=tmp_path)
    assert n.positions().collect()["mmsi"].to_list() == [367000001]
    assert n.quarantine().collect()["mmsi"].to_list() == [123]
    report = n.quality_report()
    assert (report.total_rows, report.rows_ok, report.rows_dropped) == (2, 1, 1)
