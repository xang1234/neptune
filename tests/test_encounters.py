"""Tests for encounter detection.

Validates the encounter detector against synthetic multi-vessel
positions, covering: proximity detection, time bucketing, gap splitting,
secondary vessel linkage, confidence scoring, and provenance.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from neptune_ais.datasets.events import (
    Col as EventCol,
    EVENT_TYPE_ENCOUNTER,
    REQUIRED_COLUMNS,
    SCHEMA,
    validate_schema,
)
from neptune_ais.derive.events import (
    EncounterConfig,
    detect_encounters,
    parse_provenance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASE = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)


def _ts(minutes: float) -> datetime:
    return _BASE + timedelta(minutes=minutes)


def _encounter_positions() -> pl.DataFrame:
    """Two vessels close together for 30 minutes.

    Vessel 111 and 222 are ~100m apart at lat ~40, lon ~-74
    for 7 observations over 30 minutes (5-min intervals).
    """
    rows = []
    for i in range(7):
        t = _ts(i * 5)
        rows.append((111, t, 40.0, -74.0, 2.0, "noaa"))
        rows.append((222, t, 40.001, -74.001, 1.5, "noaa"))

    return pl.DataFrame({
        "mmsi": pl.Series([r[0] for r in rows], dtype=pl.Int64),
        "timestamp": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us", "UTC")),
        "lat": [r[2] for r in rows],
        "lon": [r[3] for r in rows],
        "sog": [r[4] for r in rows],
        "source": [r[5] for r in rows],
    }).sort(["mmsi", "timestamp"])


def _far_apart_positions() -> pl.DataFrame:
    """Two vessels far apart — no encounter."""
    rows = []
    for i in range(5):
        t = _ts(i * 5)
        rows.append((111, t, 40.0, -74.0, 5.0, "noaa"))
        rows.append((222, t, 55.0, 12.0, 5.0, "noaa"))

    return pl.DataFrame({
        "mmsi": pl.Series([r[0] for r in rows], dtype=pl.Int64),
        "timestamp": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us", "UTC")),
        "lat": [r[2] for r in rows],
        "lon": [r[3] for r in rows],
        "sog": [r[4] for r in rows],
        "source": [r[5] for r in rows],
    }).sort(["mmsi", "timestamp"])


# ---------------------------------------------------------------------------
# EncounterConfig
# ---------------------------------------------------------------------------


class TestEncounterConfig:
    def test_defaults(self):
        config = EncounterConfig()
        assert config.max_distance_m == 500.0
        assert config.min_duration_s == 600
        assert config.min_observations == 2
        assert config.time_bucket_s == 300
        assert config.gap_seconds == 3600

    def test_config_hash_deterministic(self):
        assert EncounterConfig().config_hash() == EncounterConfig().config_hash()

    def test_config_hash_changes(self):
        a = EncounterConfig(max_distance_m=500).config_hash()
        b = EncounterConfig(max_distance_m=1000).config_hash()
        assert a != b

    def test_config_hash_format(self):
        h = EncounterConfig().config_hash()
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class TestDetectEncounters:
    def _detect(self, positions=None, config=None):
        if positions is None:
            positions = _encounter_positions()
        return detect_encounters(positions, config=config, source="noaa")

    def test_detects_encounter(self):
        events = self._detect()
        assert len(events) == 1
        assert events[EventCol.EVENT_TYPE][0] == EVENT_TYPE_ENCOUNTER

    def test_schema_conformance(self):
        errors = validate_schema(self._detect())
        assert errors == []

    def test_required_columns_present(self):
        events = self._detect()
        for col in REQUIRED_COLUMNS:
            assert col in events.columns, f"Missing: {col}"

    def test_other_mmsi_populated(self):
        """Encounters must have other_mmsi set."""
        events = self._detect()
        assert events[EventCol.OTHER_MMSI].is_not_null().all()
        # mmsi < other_mmsi (canonical ordering).
        assert (events[EventCol.MMSI] < events[EventCol.OTHER_MMSI]).all()

    def test_mmsi_pair(self):
        events = self._detect()
        assert events[EventCol.MMSI][0] == 111
        assert events[EventCol.OTHER_MMSI][0] == 222

    def test_event_id_deterministic(self):
        a = self._detect()
        b = self._detect()
        assert a[EventCol.EVENT_ID].to_list() == b[EventCol.EVENT_ID].to_list()

    def test_event_id_changes_with_config(self):
        a = self._detect(config=EncounterConfig(max_distance_m=500))
        b = self._detect(config=EncounterConfig(max_distance_m=1000))
        assert a[EventCol.EVENT_ID][0] != b[EventCol.EVENT_ID][0]

    def test_temporal_bounds(self):
        events = self._detect()
        assert (events[EventCol.START_TIME] <= events[EventCol.END_TIME]).all()
        duration = (
            events[EventCol.END_TIME][0] - events[EventCol.START_TIME][0]
        ).total_seconds()
        # 7 observations at 5-min intervals = 30 min.
        assert duration >= 1500  # at least 25 min (with bucket alignment)

    def test_location_near_encounter(self):
        events = self._detect()
        assert 39.0 < events[EventCol.LAT][0] < 41.0
        assert -75.0 < events[EventCol.LON][0] < -73.0

    def test_confidence_score(self):
        events = self._detect()
        score = events[EventCol.CONFIDENCE_SCORE][0]
        assert 0.0 <= score <= 1.0
        assert score >= 0.5

    def test_provenance_token(self):
        events = self._detect()
        prov = parse_provenance(events[EventCol.RECORD_PROVENANCE][0])
        assert prov.source == "noaa"
        assert prov.detector == "encounter_detector"
        assert prov.detector_version == "0.1.0"
        assert "positions" in prov.upstream_datasets

    def test_source_column(self):
        events = self._detect()
        assert (events[EventCol.SOURCE] == "noaa").all()


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestEncounterFiltering:
    def _detect(self, positions, config=None):
        return detect_encounters(positions, config=config, source="noaa")

    def test_far_apart_excluded(self):
        events = self._detect(_far_apart_positions())
        assert len(events) == 0

    def test_single_vessel_no_encounter(self):
        """Only one vessel → no encounter possible."""
        rows = [(111, _ts(i * 5), 40.0, -74.0, 2.0, "noaa") for i in range(10)]
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

    def test_brief_proximity_excluded(self):
        """Vessels close for < min_duration → no encounter."""
        # Only 1 time bucket of proximity.
        rows = [
            (111, _ts(0), 40.0, -74.0, 2.0, "noaa"),
            (222, _ts(0), 40.001, -74.001, 1.5, "noaa"),
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

    def test_three_vessels_multiple_encounters(self):
        """Three vessels close together → up to 3 pair encounters."""
        rows = []
        for i in range(7):
            t = _ts(i * 5)
            rows.append((111, t, 40.0, -74.0, 1.0, "noaa"))
            rows.append((222, t, 40.001, -74.001, 1.0, "noaa"))
            rows.append((333, t, 40.002, -74.002, 1.0, "noaa"))

        positions = pl.DataFrame({
            "mmsi": pl.Series([r[0] for r in rows], dtype=pl.Int64),
            "timestamp": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us", "UTC")),
            "lat": [r[2] for r in rows],
            "lon": [r[3] for r in rows],
            "sog": [r[4] for r in rows],
            "source": [r[5] for r in rows],
        }).sort(["mmsi", "timestamp"])

        events = self._detect(positions)
        # 3 pairs: (111,222), (111,333), (222,333)
        assert len(events) == 3
        pairs = set(
            (row[EventCol.MMSI], row[EventCol.OTHER_MMSI])
            for row in events.iter_rows(named=True)
        )
        assert pairs == {(111, 222), (111, 333), (222, 333)}

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
