"""Tests for EEZ-crossing detection.

Validates the EEZ-crossing detector against synthetic positions and
EEZ boundaries, covering: transition detection, gap filtering,
confidence scoring, midpoint location, and provenance.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from neptune_ais.datasets.events import (
    Col as EventCol,
    EVENT_TYPE_EEZ_CROSSING,
    REQUIRED_COLUMNS,
    SCHEMA,
    validate_schema,
)
from neptune_ais.derive.events import (
    EEZCrossingConfig,
    EventProvenance,
    detect_eez_crossings,
    parse_provenance,
)
from neptune_ais.geometry.boundaries import (
    BoundaryDataset,
    BoundaryRegion,
    BoundaryRegistry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASE = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)


def _ts(minutes: float) -> datetime:
    return _BASE + timedelta(minutes=minutes)


def _eez_registry() -> BoundaryRegistry:
    """Registry with two adjacent EEZ zones."""
    reg = BoundaryRegistry()
    reg.register(BoundaryDataset(
        name="eez",
        version="11.0",
        regions=(
            BoundaryRegion(name="NLD_EEZ", bbox=(2.5, 51.0, 7.2, 55.8)),
            BoundaryRegion(name="GBR_EEZ", bbox=(-5.0, 49.0, 2.5, 56.0)),
        ),
    ))
    return reg


def _crossing_positions() -> pl.DataFrame:
    """Vessel 111 sails from NLD_EEZ to GBR_EEZ.

    Points 0-4: in NLD_EEZ (lon 5.0 → 3.0, moving west)
    Points 5-9: in GBR_EEZ (lon 2.0 → -1.0, continuing west)
    The crossing happens between point 4 (lon=3.0, NLD) and point 5 (lon=2.0, GBR).
    """
    rows = []
    for i in range(5):
        rows.append((111, _ts(i * 10), 53.0, 5.0 - i * 0.5, 8.0, "noaa"))
    for i in range(5):
        rows.append((111, _ts(50 + i * 10), 53.0, 2.0 - i * 0.75, 8.0, "noaa"))

    return pl.DataFrame({
        "mmsi": pl.Series([r[0] for r in rows], dtype=pl.Int64),
        "timestamp": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us", "UTC")),
        "lat": [r[2] for r in rows],
        "lon": [r[3] for r in rows],
        "sog": [r[4] for r in rows],
        "source": [r[5] for r in rows],
    })


def _two_vessel_crossing() -> pl.DataFrame:
    """Two vessels crossing between EEZs at different times."""
    rows = []
    # Vessel 111: NLD → GBR at minutes 0-50
    for i in range(3):
        rows.append((111, _ts(i * 10), 53.0, 5.0 - i * 1.0, 8.0, "noaa"))
    for i in range(3):
        rows.append((111, _ts(30 + i * 10), 53.0, 2.0 - i * 1.0, 8.0, "noaa"))
    # Vessel 222: GBR → NLD at minutes 60-110
    for i in range(3):
        rows.append((222, _ts(60 + i * 10), 53.0, 0.0 + i * 1.0, 8.0, "noaa"))
    for i in range(3):
        rows.append((222, _ts(90 + i * 10), 53.0, 3.0 + i * 1.0, 8.0, "noaa"))

    return pl.DataFrame({
        "mmsi": pl.Series([r[0] for r in rows], dtype=pl.Int64),
        "timestamp": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us", "UTC")),
        "lat": [r[2] for r in rows],
        "lon": [r[3] for r in rows],
        "sog": [r[4] for r in rows],
        "source": [r[5] for r in rows],
    }).sort(["mmsi", "timestamp"])


# ---------------------------------------------------------------------------
# EEZCrossingConfig
# ---------------------------------------------------------------------------


class TestEEZCrossingConfig:
    def test_defaults(self):
        config = EEZCrossingConfig()
        assert config.max_gap_s == 7200
        assert config.max_distance_m == 100_000.0

    def test_config_hash_deterministic(self):
        assert EEZCrossingConfig().config_hash() == EEZCrossingConfig().config_hash()

    def test_config_hash_changes(self):
        a = EEZCrossingConfig(max_gap_s=7200).config_hash()
        b = EEZCrossingConfig(max_gap_s=3600).config_hash()
        assert a != b

    def test_config_hash_format(self):
        h = EEZCrossingConfig().config_hash()
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# EEZ-crossing detection
# ---------------------------------------------------------------------------


class TestDetectEEZCrossings:
    def _detect(self, positions=None, config=None):
        if positions is None:
            positions = _crossing_positions()
        reg = _eez_registry()
        eez_regions = reg.lookup_column(positions, "eez")
        return detect_eez_crossings(
            positions, eez_regions, config=config, source="noaa"
        )

    def test_detects_crossing(self):
        events = self._detect()
        assert len(events) == 1
        assert events[EventCol.EVENT_TYPE][0] == EVENT_TYPE_EEZ_CROSSING

    def test_schema_conformance(self):
        events = self._detect()
        errors = validate_schema(events)
        assert errors == [], f"Schema errors: {errors}"

    def test_required_columns_present(self):
        events = self._detect()
        for col in REQUIRED_COLUMNS:
            assert col in events.columns, f"Missing: {col}"

    def test_event_id_deterministic(self):
        a = self._detect()
        b = self._detect()
        assert a[EventCol.EVENT_ID].to_list() == b[EventCol.EVENT_ID].to_list()

    def test_event_id_changes_with_config(self):
        a = self._detect(config=EEZCrossingConfig(max_gap_s=7200))
        b = self._detect(config=EEZCrossingConfig(max_gap_s=3600))
        assert a[EventCol.EVENT_ID][0] != b[EventCol.EVENT_ID][0]

    def test_temporal_bounds(self):
        events = self._detect()
        assert (events[EventCol.START_TIME] <= events[EventCol.END_TIME]).all()
        # Crossing happens between point 4 (minute 40) and point 5 (minute 50).
        assert events[EventCol.START_TIME][0] == _ts(40)
        assert events[EventCol.END_TIME][0] == _ts(50)

    def test_midpoint_location(self):
        events = self._detect()
        # Midpoint between lon=3.0 (NLD) and lon=2.0 (GBR) → ~2.5
        lon = events[EventCol.LON][0]
        assert 2.0 <= lon <= 3.0

    def test_confidence_score(self):
        events = self._detect()
        score = events[EventCol.CONFIDENCE_SCORE][0]
        assert 0.0 <= score <= 1.0
        # 10-minute gap but ~67km distance → 0.7 (gap<=2h, dist<=100km).
        assert score == 0.7

    def test_provenance_token(self):
        events = self._detect()
        token = events[EventCol.RECORD_PROVENANCE][0]
        prov = parse_provenance(token)
        assert prov.source == "noaa"
        assert prov.detector == "eez_crossing_detector"
        assert prov.detector_version == "0.1.0"
        assert "boundaries" in prov.upstream_datasets
        assert "positions" in prov.upstream_datasets

    def test_other_mmsi_is_null(self):
        events = self._detect()
        assert events[EventCol.OTHER_MMSI].is_null().all()

    def test_source_column(self):
        events = self._detect()
        assert (events[EventCol.SOURCE] == "noaa").all()


# ---------------------------------------------------------------------------
# Filtering behavior
# ---------------------------------------------------------------------------


class TestEEZCrossingFiltering:
    def _detect(self, positions, config=None):
        reg = _eez_registry()
        eez_regions = reg.lookup_column(positions, "eez")
        return detect_eez_crossings(
            positions, eez_regions, config=config, source="noaa"
        )

    def test_no_crossing_same_eez(self):
        """Vessel stays in one EEZ → no crossings."""
        rows = [
            (111, _ts(i * 10), 53.0, 5.0 - i * 0.1, 8.0, "noaa")
            for i in range(10)
        ]
        positions = pl.DataFrame({
            "mmsi": pl.Series([r[0] for r in rows], dtype=pl.Int64),
            "timestamp": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us", "UTC")),
            "lat": [r[2] for r in rows],
            "lon": [r[3] for r in rows],
            "sog": [r[4] for r in rows],
            "source": [r[5] for r in rows],
        })
        events = self._detect(positions)
        assert len(events) == 0

    def test_large_gap_excluded(self):
        """Crossing with gap > max_gap_s is excluded."""
        # Two positions separated by 3 hours.
        positions = pl.DataFrame({
            "mmsi": pl.Series([111, 111], dtype=pl.Int64),
            "timestamp": pl.Series(
                [_ts(0), _ts(180)], dtype=pl.Datetime("us", "UTC")
            ),
            "lat": [53.0, 53.0],
            "lon": [5.0, 1.0],  # NLD → GBR
            "sog": [8.0, 8.0],
            "source": ["noaa", "noaa"],
        })
        events = self._detect(positions)
        assert len(events) == 0

    def test_two_vessels_crossing(self):
        """Two vessels each crossing → two events."""
        events = self._detect(_two_vessel_crossing())
        assert len(events) == 2
        mmsis = sorted(events[EventCol.MMSI].to_list())
        assert mmsis == [111, 222]

    def test_entering_from_no_eez_excluded(self):
        """Transition from null EEZ to an EEZ is not a crossing."""
        positions = pl.DataFrame({
            "mmsi": pl.Series([111, 111], dtype=pl.Int64),
            "timestamp": pl.Series(
                [_ts(0), _ts(10)], dtype=pl.Datetime("us", "UTC")
            ),
            "lat": [53.0, 53.0],
            "lon": [10.0, 5.0],  # outside → NLD_EEZ
            "sog": [8.0, 8.0],
            "source": ["noaa", "noaa"],
        })
        events = self._detect(positions)
        assert len(events) == 0

    def test_multi_crossing_a_b_a(self):
        """Vessel crosses NLD → GBR → NLD → two crossing events.

        Uses a generous max_distance_m to avoid filtering out the
        return crossing.
        """
        rows = [
            # In NLD_EEZ (lon 4.0 → 3.0)
            (111, _ts(0), 53.0, 4.0, 8.0, "noaa"),
            (111, _ts(10), 53.0, 3.5, 8.0, "noaa"),
            (111, _ts(20), 53.0, 3.0, 8.0, "noaa"),
            # In GBR_EEZ (lon 2.0 → 1.5)
            (111, _ts(30), 53.0, 2.0, 8.0, "noaa"),
            (111, _ts(40), 53.0, 1.5, 8.0, "noaa"),
            # Back in NLD_EEZ (lon 3.0 → 4.0)
            (111, _ts(50), 53.0, 3.0, 8.0, "noaa"),
            (111, _ts(60), 53.0, 4.0, 8.0, "noaa"),
        ]
        positions = pl.DataFrame({
            "mmsi": pl.Series([r[0] for r in rows], dtype=pl.Int64),
            "timestamp": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us", "UTC")),
            "lat": [r[2] for r in rows],
            "lon": [r[3] for r in rows],
            "sog": [r[4] for r in rows],
            "source": [r[5] for r in rows],
        })
        config = EEZCrossingConfig(max_distance_m=200_000)
        events = self._detect(positions, config=config)
        assert len(events) == 2
        assert (events[EventCol.MMSI] == 111).all()
        # First crossing: NLD → GBR, second: GBR → NLD.
        sorted_events = events.sort(EventCol.START_TIME)
        assert sorted_events[EventCol.START_TIME][1] > sorted_events[EventCol.END_TIME][0]

    def test_empty_positions(self):
        empty = pl.DataFrame(schema={
            "mmsi": pl.Int64, "timestamp": pl.Datetime("us", "UTC"),
            "lat": pl.Float64, "lon": pl.Float64,
            "sog": pl.Float64, "source": pl.String,
        })
        events = self._detect(empty)
        assert len(events) == 0
        for col in SCHEMA:
            assert col in events.columns
