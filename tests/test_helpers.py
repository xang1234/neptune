"""Tests for high-level maritime helper APIs.

Covers latest_positions, snapshot, vessel_history, and their
Neptune class wrappers.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from neptune_ais.datasets.positions import Col as PosCol
from neptune_ais.helpers import latest_positions, snapshot, vessel_history


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASE = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)


def _ts(minutes: float) -> datetime:
    return _BASE + timedelta(minutes=minutes)


def _multi_vessel_positions() -> pl.DataFrame:
    """Three vessels with multiple positions each.

    Vessel 111: minutes 0, 10, 20 (latest at 20)
    Vessel 222: minutes 5, 15, 25 (latest at 25)
    Vessel 333: minutes 0, 30     (latest at 30)
    """
    return pl.DataFrame({
        PosCol.MMSI: pl.Series([111, 111, 111, 222, 222, 222, 333, 333], dtype=pl.Int64),
        PosCol.TIMESTAMP: pl.Series([
            _ts(0), _ts(10), _ts(20),
            _ts(5), _ts(15), _ts(25),
            _ts(0), _ts(30),
        ], dtype=pl.Datetime("us", "UTC")),
        PosCol.LAT: [40.0, 40.1, 40.2, 55.0, 55.1, 55.2, 30.0, 30.1],
        PosCol.LON: [-74.0, -74.1, -74.2, 12.0, 12.1, 12.2, -80.0, -80.1],
        "sog": [5.0] * 8,
        "source": ["noaa"] * 8,
        "record_provenance": ["noaa:direct"] * 8,
        "qc_severity": ["ok"] * 8,
    })


# ---------------------------------------------------------------------------
# latest_positions
# ---------------------------------------------------------------------------


class TestLatestPositions:
    def test_one_row_per_vessel(self):
        lf = _multi_vessel_positions().lazy()
        result = latest_positions(lf).collect()
        assert len(result) == 3

    def test_returns_latest_timestamp(self):
        lf = _multi_vessel_positions().lazy()
        result = latest_positions(lf).collect().sort(PosCol.MMSI)
        # Vessel 111 latest at minute 20.
        assert result.filter(pl.col(PosCol.MMSI) == 111)[PosCol.TIMESTAMP][0] == _ts(20)
        # Vessel 222 latest at minute 25.
        assert result.filter(pl.col(PosCol.MMSI) == 222)[PosCol.TIMESTAMP][0] == _ts(25)
        # Vessel 333 latest at minute 30.
        assert result.filter(pl.col(PosCol.MMSI) == 333)[PosCol.TIMESTAMP][0] == _ts(30)

    def test_sorted_by_mmsi(self):
        lf = _multi_vessel_positions().lazy()
        result = latest_positions(lf).collect()
        mmsis = result[PosCol.MMSI].to_list()
        assert mmsis == sorted(mmsis)

    def test_preserves_columns(self):
        lf = _multi_vessel_positions().lazy()
        result = latest_positions(lf).collect()
        assert PosCol.LAT in result.columns
        assert PosCol.LON in result.columns

    def test_empty_input(self):
        from neptune_ais.datasets.positions import SCHEMA
        empty = pl.DataFrame(schema=SCHEMA).lazy()
        result = latest_positions(empty).collect()
        assert len(result) == 0


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_closest_to_target(self):
        lf = _multi_vessel_positions().lazy()
        # Snapshot at minute 12 — closest for each vessel:
        # 111: minute 10 (diff=2), 222: minute 15 (diff=3), 333: minute 0 (diff=12)
        result = snapshot(lf, _ts(12)).collect().sort(PosCol.MMSI)
        assert result.filter(pl.col(PosCol.MMSI) == 111)[PosCol.TIMESTAMP][0] == _ts(10)
        assert result.filter(pl.col(PosCol.MMSI) == 222)[PosCol.TIMESTAMP][0] == _ts(15)
        assert result.filter(pl.col(PosCol.MMSI) == 333)[PosCol.TIMESTAMP][0] == _ts(0)

    def test_one_row_per_vessel(self):
        lf = _multi_vessel_positions().lazy()
        result = snapshot(lf, _ts(10)).collect()
        assert len(result) == 3

    def test_string_timestamp(self):
        lf = _multi_vessel_positions().lazy()
        result = snapshot(lf, "2024-06-15T00:12:00").collect().sort(PosCol.MMSI)
        assert len(result) == 3
        # Same result as datetime-based snapshot at minute 12.
        assert result.filter(pl.col(PosCol.MMSI) == 111)[PosCol.TIMESTAMP][0] == _ts(10)

    def test_sorted_by_mmsi(self):
        lf = _multi_vessel_positions().lazy()
        result = snapshot(lf, _ts(10)).collect()
        mmsis = result[PosCol.MMSI].to_list()
        assert mmsis == sorted(mmsis)

    def test_no_time_diff_column_in_output(self):
        lf = _multi_vessel_positions().lazy()
        result = snapshot(lf, _ts(10)).collect()
        assert "_time_diff" not in result.columns


# ---------------------------------------------------------------------------
# vessel_history
# ---------------------------------------------------------------------------


class TestVesselHistory:
    def test_positions_only(self):
        lf = _multi_vessel_positions().lazy()
        hist = vessel_history(111, positions=lf)
        assert "positions" in hist
        result = hist["positions"].collect()
        assert (result[PosCol.MMSI] == 111).all()
        assert len(result) == 3

    def test_with_tracks(self):
        from neptune_ais.datasets.tracks import Col as TrackCol

        positions = _multi_vessel_positions().lazy()
        tracks = pl.DataFrame({
            TrackCol.TRACK_ID: ["t1", "t2"],
            TrackCol.MMSI: pl.Series([111, 222], dtype=pl.Int64),
        }).lazy()

        hist = vessel_history(111, positions=positions, tracks=tracks)
        assert "tracks" in hist
        tracks_result = hist["tracks"].collect()
        assert len(tracks_result) == 1
        assert tracks_result[TrackCol.MMSI][0] == 111

    def test_with_events_primary_mmsi(self):
        from neptune_ais.datasets.events import Col as EventCol

        positions = _multi_vessel_positions().lazy()
        events = pl.DataFrame({
            EventCol.EVENT_ID: ["e1", "e2"],
            EventCol.MMSI: pl.Series([111, 222], dtype=pl.Int64),
            EventCol.OTHER_MMSI: pl.Series([None, None], dtype=pl.Int64),
        }).lazy()

        hist = vessel_history(111, positions=positions, events=events)
        assert "events" in hist
        events_result = hist["events"].collect()
        assert len(events_result) == 1
        assert events_result[EventCol.MMSI][0] == 111

    def test_with_events_other_mmsi(self):
        """vessel_history includes events where the vessel is OTHER_MMSI."""
        from neptune_ais.datasets.events import Col as EventCol

        positions = _multi_vessel_positions().lazy()
        events = pl.DataFrame({
            EventCol.EVENT_ID: ["e1"],
            EventCol.MMSI: pl.Series([222], dtype=pl.Int64),
            EventCol.OTHER_MMSI: pl.Series([111], dtype=pl.Int64),
        }).lazy()

        hist = vessel_history(111, positions=positions, events=events)
        events_result = hist["events"].collect()
        # Vessel 111 is other_mmsi in this encounter.
        assert len(events_result) == 1

    def test_no_tracks_or_events(self):
        positions = _multi_vessel_positions().lazy()
        hist = vessel_history(111, positions=positions)
        assert "positions" in hist
        assert "tracks" not in hist
        assert "events" not in hist

    def test_unknown_mmsi_returns_empty(self):
        positions = _multi_vessel_positions().lazy()
        hist = vessel_history(999, positions=positions)
        result = hist["positions"].collect()
        assert len(result) == 0
