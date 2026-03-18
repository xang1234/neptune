"""Tests for loitering detection and density derivation.

Covers: loitering detector with spatial radius filtering, density
grid derivation with vessel counts.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from neptune_ais.datasets.events import (
    Col as EventCol,
    EVENT_TYPE_LOITERING,
    REQUIRED_COLUMNS,
    SCHEMA as EVENT_SCHEMA,
    validate_schema,
)
from neptune_ais.derive.events import (
    LoiteringConfig,
    detect_loitering,
    parse_provenance,
)
from neptune_ais.derive.density import (
    DensityConfig,
    compute_density,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASE = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)


def _ts(minutes: float) -> datetime:
    return _BASE + timedelta(minutes=minutes)


def _loitering_positions() -> pl.DataFrame:
    """Vessel 111 drifts slowly in a small area for 1 hour.

    10 positions over 60 minutes, all within ~200m of (40.0, -74.0).
    """
    rows = [
        (111, _ts(i * 6), 40.0 + i * 0.0001, -74.0 + i * 0.0001, 0.5, "noaa")
        for i in range(11)  # 0 to 60 min
    ]
    return pl.DataFrame({
        "mmsi": pl.Series([r[0] for r in rows], dtype=pl.Int64),
        "timestamp": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us", "UTC")),
        "lat": [r[2] for r in rows],
        "lon": [r[3] for r in rows],
        "sog": [r[4] for r in rows],
        "source": [r[5] for r in rows],
    })


def _fast_positions() -> pl.DataFrame:
    """Vessel moving fast — no loitering."""
    rows = [
        (111, _ts(i * 6), 40.0 + i * 0.1, -74.0, 12.0, "noaa")
        for i in range(10)
    ]
    return pl.DataFrame({
        "mmsi": pl.Series([r[0] for r in rows], dtype=pl.Int64),
        "timestamp": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us", "UTC")),
        "lat": [r[2] for r in rows],
        "lon": [r[3] for r in rows],
        "sog": [r[4] for r in rows],
        "source": [r[5] for r in rows],
    })


def _density_positions() -> pl.DataFrame:
    """Three vessels in two areas for density testing."""
    rows = []
    # Vessel 111 and 222 near (40.0, -74.0)
    for i in range(5):
        rows.append((111, _ts(i * 10), 40.0 + i * 0.001, -74.0, 5.0, "noaa"))
        rows.append((222, _ts(i * 10), 40.0 - i * 0.001, -74.0, 5.0, "noaa"))
    # Vessel 333 near (55.0, 12.0)
    for i in range(5):
        rows.append((333, _ts(i * 10), 55.0, 12.0 + i * 0.001, 5.0, "noaa"))
    return pl.DataFrame({
        "mmsi": pl.Series([r[0] for r in rows], dtype=pl.Int64),
        "timestamp": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us", "UTC")),
        "lat": [r[2] for r in rows],
        "lon": [r[3] for r in rows],
        "sog": [r[4] for r in rows],
        "source": [r[5] for r in rows],
    })


# ---------------------------------------------------------------------------
# LoiteringConfig
# ---------------------------------------------------------------------------


class TestLoiteringConfig:
    def test_defaults(self):
        config = LoiteringConfig()
        assert config.max_speed_knots == 2.0
        assert config.max_radius_m == 1000.0
        assert config.min_duration_s == 1800
        assert config.min_points == 3
        assert config.gap_seconds == 3600

    def test_config_hash_deterministic(self):
        assert LoiteringConfig().config_hash() == LoiteringConfig().config_hash()

    def test_config_hash_changes(self):
        a = LoiteringConfig(max_speed_knots=2.0).config_hash()
        b = LoiteringConfig(max_speed_knots=1.0).config_hash()
        assert a != b


# ---------------------------------------------------------------------------
# Loitering detection
# ---------------------------------------------------------------------------


class TestDetectLoitering:
    def _detect(self, positions=None, config=None):
        if positions is None:
            positions = _loitering_positions()
        return detect_loitering(positions, config=config, source="noaa")

    def test_detects_loitering(self):
        events = self._detect()
        assert len(events) == 1
        assert events[EventCol.EVENT_TYPE][0] == EVENT_TYPE_LOITERING

    def test_schema_conformance(self):
        errors = validate_schema(self._detect())
        assert errors == []

    def test_required_columns_present(self):
        events = self._detect()
        for col in REQUIRED_COLUMNS:
            assert col in events.columns

    def test_event_id_deterministic(self):
        a = self._detect()
        b = self._detect()
        assert a[EventCol.EVENT_ID].to_list() == b[EventCol.EVENT_ID].to_list()

    def test_temporal_bounds(self):
        events = self._detect()
        assert (events[EventCol.START_TIME] <= events[EventCol.END_TIME]).all()
        duration = (
            events[EventCol.END_TIME][0] - events[EventCol.START_TIME][0]
        ).total_seconds()
        assert duration >= 1800  # at least 30 min

    def test_location_near_loitering_area(self):
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
        assert prov.detector == "loitering_detector"
        assert "positions" in prov.upstream_datasets

    def test_other_mmsi_is_null(self):
        events = self._detect()
        assert events[EventCol.OTHER_MMSI].is_null().all()


# ---------------------------------------------------------------------------
# Loitering filtering
# ---------------------------------------------------------------------------


class TestLoiteringFiltering:
    def _detect(self, positions, config=None):
        return detect_loitering(positions, config=config, source="noaa")

    def test_fast_vessel_excluded(self):
        events = self._detect(_fast_positions())
        assert len(events) == 0

    def test_short_stay_excluded(self):
        """Stay < min_duration excluded."""
        rows = [
            (111, _ts(i * 2), 40.0, -74.0, 0.5, "noaa")
            for i in range(5)  # 8 minutes, below 30 min default
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

    def test_wide_spread_excluded(self):
        """Positions too spread out (> max_radius_m) excluded."""
        rows = [
            (111, _ts(i * 6), 40.0 + i * 0.1, -74.0, 0.5, "noaa")
            for i in range(11)  # ~10 km spread
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

    def test_empty_positions(self):
        empty = pl.DataFrame(schema={
            "mmsi": pl.Int64, "timestamp": pl.Datetime("us", "UTC"),
            "lat": pl.Float64, "lon": pl.Float64,
            "sog": pl.Float64, "source": pl.String,
        })
        events = self._detect(empty)
        assert len(events) == 0
        for col in EVENT_SCHEMA:
            assert col in events.columns


# ---------------------------------------------------------------------------
# DensityConfig
# ---------------------------------------------------------------------------


class TestDensityConfig:
    def test_defaults(self):
        config = DensityConfig()
        assert config.resolution == 4
        assert config.time_bucket_hours == 24

    def test_config_hash_deterministic(self):
        assert DensityConfig().config_hash() == DensityConfig().config_hash()

    def test_config_hash_changes(self):
        a = DensityConfig(resolution=4).config_hash()
        b = DensityConfig(resolution=6).config_hash()
        assert a != b


# ---------------------------------------------------------------------------
# Density derivation
# ---------------------------------------------------------------------------


class TestComputeDensity:
    def test_basic_density(self):
        result = compute_density(_density_positions())
        assert "cell_id" in result.columns
        assert "vessel_count" in result.columns
        assert "observation_count" in result.columns
        assert "center_lat" in result.columns
        assert "center_lon" in result.columns
        assert result["observation_count"].sum() == 15

    def test_vessel_count(self):
        result = compute_density(_density_positions())
        # The cell near (40.0, -74.0) should have 2 vessels (111, 222).
        cell_40 = result.filter(
            (pl.col("center_lat").round(0) == 40.0)
            & (pl.col("center_lon").round(0) == -74.0)
        )
        assert len(cell_40) > 0
        assert cell_40["vessel_count"].max() >= 2

    def test_sorted_by_observation_count(self):
        result = compute_density(_density_positions())
        counts = result["observation_count"].to_list()
        assert counts == sorted(counts, reverse=True)

    def test_empty_positions(self):
        empty = pl.DataFrame(schema={
            "mmsi": pl.Int64, "timestamp": pl.Datetime("us", "UTC"),
            "lat": pl.Float64, "lon": pl.Float64,
        })
        result = compute_density(empty)
        assert len(result) == 0
        assert result.columns == [
            "cell_id", "center_lat", "center_lon",
            "vessel_count", "observation_count",
        ]

    def test_resolution_affects_granularity(self):
        positions = _density_positions()
        coarse = compute_density(positions, config=DensityConfig(resolution=1))
        fine = compute_density(positions, config=DensityConfig(resolution=8))
        # Finer resolution should produce more cells (or equal).
        assert len(fine) >= len(coarse)
