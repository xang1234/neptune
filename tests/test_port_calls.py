"""Tests for port-call detection.

Validates the port-call detector against synthetic positions and port
boundaries, covering the full pipeline: boundary lookup → low-speed
filter → grouping → aggregation → confidence → provenance.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from neptune_ais.datasets.events import (
    Col as EventCol,
    EVENT_TYPE_PORT_CALL,
    REQUIRED_COLUMNS,
    SCHEMA,
    validate_schema,
)
from neptune_ais.derive.events import (
    EventProvenance,
    PortCallConfig,
    detect_port_calls,
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


def _port_registry() -> BoundaryRegistry:
    """Registry with one port: Rotterdam bbox."""
    reg = BoundaryRegistry()
    reg.register(BoundaryDataset(
        name="ports",
        version="2024.1",
        regions=(
            BoundaryRegion(name="Rotterdam", bbox=(3.8, 51.8, 4.5, 52.1)),
        ),
    ))
    return reg


def _port_call_positions() -> pl.DataFrame:
    """Vessel 111 enters Rotterdam, stops for 2 hours, then leaves.

    Minutes 0-5: in transit (high speed, outside port)
    Minutes 10-130: in Rotterdam port (low speed, 2-hour stay)
    Minutes 135-140: departing (high speed, outside port)
    """
    transit_in = [
        (111, _ts(i), 51.5 + i * 0.02, 3.5, 12.0, "noaa")
        for i in range(6)
    ]
    in_port = [
        (111, _ts(10 + i * 10), 51.9, 4.0 + i * 0.001, 0.5, "noaa")
        for i in range(13)  # 13 points over 120 minutes
    ]
    transit_out = [
        (111, _ts(135 + i), 52.2 + i * 0.02, 4.6, 11.0, "noaa")
        for i in range(6)
    ]

    rows = transit_in + in_port + transit_out
    return pl.DataFrame(
        {
            "mmsi": pl.Series([r[0] for r in rows], dtype=pl.Int64),
            "timestamp": pl.Series(
                [r[1] for r in rows], dtype=pl.Datetime("us", "UTC")
            ),
            "lat": [r[2] for r in rows],
            "lon": [r[3] for r in rows],
            "sog": [r[4] for r in rows],
            "source": [r[5] for r in rows],
        }
    )


def _two_vessel_port_positions() -> pl.DataFrame:
    """Two vessels visit Rotterdam at different times.

    Vessel 111: minutes 0-120 (2 hours)
    Vessel 222: minutes 60-180 (2 hours, overlapping)
    """
    v111 = [
        (111, _ts(i * 10), 51.9, 4.0, 0.5, "noaa")
        for i in range(13)  # 0 to 120 min
    ]
    v222 = [
        (222, _ts(60 + i * 10), 51.95, 4.1, 1.0, "noaa")
        for i in range(13)  # 60 to 180 min
    ]

    rows = v111 + v222
    return pl.DataFrame(
        {
            "mmsi": pl.Series([r[0] for r in rows], dtype=pl.Int64),
            "timestamp": pl.Series(
                [r[1] for r in rows], dtype=pl.Datetime("us", "UTC")
            ),
            "lat": [r[2] for r in rows],
            "lon": [r[3] for r in rows],
            "sog": [r[4] for r in rows],
            "source": [r[5] for r in rows],
        }
    ).sort(["mmsi", "timestamp"])


# ---------------------------------------------------------------------------
# PortCallConfig
# ---------------------------------------------------------------------------


class TestPortCallConfig:
    def test_defaults(self):
        config = PortCallConfig()
        assert config.max_speed_knots == 3.0
        assert config.min_duration_s == 3600
        assert config.min_points == 3
        assert config.gap_seconds == 7200

    def test_config_hash_deterministic(self):
        a = PortCallConfig().config_hash()
        b = PortCallConfig().config_hash()
        assert a == b

    def test_config_hash_changes(self):
        a = PortCallConfig(max_speed_knots=3.0).config_hash()
        b = PortCallConfig(max_speed_knots=5.0).config_hash()
        assert a != b

    def test_config_hash_format(self):
        h = PortCallConfig().config_hash()
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Port-call detection
# ---------------------------------------------------------------------------


class TestDetectPortCalls:
    """Test the full port-call detection pipeline."""

    def _detect(
        self,
        positions: pl.DataFrame | None = None,
        config: PortCallConfig | None = None,
    ) -> pl.DataFrame:
        """Helper: run port-call detection with Rotterdam boundary."""
        if positions is None:
            positions = _port_call_positions()
        reg = _port_registry()
        port_regions = reg.lookup_column(positions, "ports")
        return detect_port_calls(
            positions, port_regions, config=config, source="noaa"
        )

    def test_detects_port_call(self):
        """A vessel stopped in Rotterdam for 2 hours is detected."""
        events = self._detect()
        assert len(events) == 1
        assert events[EventCol.EVENT_TYPE][0] == EVENT_TYPE_PORT_CALL

    def test_schema_conformance(self):
        """Output conforms to events/v1 schema."""
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
        a = self._detect(config=PortCallConfig(max_speed_knots=3.0))
        b = self._detect(config=PortCallConfig(max_speed_knots=5.0))
        assert a[EventCol.EVENT_ID][0] != b[EventCol.EVENT_ID][0]

    def test_temporal_bounds(self):
        events = self._detect()
        assert (events[EventCol.START_TIME] <= events[EventCol.END_TIME]).all()
        # Start should be around minute 10, end around minute 130.
        start = events[EventCol.START_TIME][0]
        end = events[EventCol.END_TIME][0]
        duration = (end - start).total_seconds()
        assert duration >= 3600  # at least 1 hour

    def test_representative_location(self):
        events = self._detect()
        # Location should be near Rotterdam (lat ~51.9, lon ~4.0).
        assert 51.0 < events[EventCol.LAT][0] < 53.0
        assert 3.0 < events[EventCol.LON][0] < 5.0

    def test_confidence_score(self):
        events = self._detect()
        score = events[EventCol.CONFIDENCE_SCORE][0]
        assert 0.0 <= score <= 1.0
        # 2-hour stay should be at least medium confidence.
        assert score >= 0.5

    def test_high_confidence_for_long_stay(self):
        """4+ hour stay gets high confidence (0.9)."""
        # Extend the stay to 5 hours (300 minutes).
        long_stay = [
            (111, _ts(10 + i * 10), 51.9, 4.0, 0.5, "noaa")
            for i in range(31)  # 31 points over 300 minutes
        ]
        positions = pl.DataFrame({
            "mmsi": pl.Series([r[0] for r in long_stay], dtype=pl.Int64),
            "timestamp": pl.Series(
                [r[1] for r in long_stay], dtype=pl.Datetime("us", "UTC")
            ),
            "lat": [r[2] for r in long_stay],
            "lon": [r[3] for r in long_stay],
            "sog": [r[4] for r in long_stay],
            "source": [r[5] for r in long_stay],
        })
        events = self._detect(positions=positions)
        assert events[EventCol.CONFIDENCE_SCORE][0] == 0.9

    def test_provenance_token(self):
        events = self._detect()
        token = events[EventCol.RECORD_PROVENANCE][0]
        prov = parse_provenance(token)
        assert prov.source == "noaa"
        assert prov.detector == "port_call_detector"
        assert prov.detector_version == "0.1.0"
        assert "boundaries" in prov.upstream_datasets
        assert "positions" in prov.upstream_datasets

    def test_source_column(self):
        events = self._detect()
        assert (events[EventCol.SOURCE] == "noaa").all()

    def test_other_mmsi_is_null(self):
        """Port calls have no secondary vessel."""
        events = self._detect()
        assert events[EventCol.OTHER_MMSI].is_null().all()


# ---------------------------------------------------------------------------
# Filtering behavior
# ---------------------------------------------------------------------------


class TestPortCallFiltering:
    def _detect(self, positions, config=None):
        reg = _port_registry()
        port_regions = reg.lookup_column(positions, "ports")
        return detect_port_calls(
            positions, port_regions, config=config, source="noaa"
        )

    def test_high_speed_excluded(self):
        """High-speed positions in port are not port calls."""
        fast = pl.DataFrame({
            "mmsi": pl.Series([111] * 10, dtype=pl.Int64),
            "timestamp": pl.Series(
                [_ts(i * 10) for i in range(10)], dtype=pl.Datetime("us", "UTC")
            ),
            "lat": [51.9] * 10,
            "lon": [4.0] * 10,
            "sog": [15.0] * 10,  # too fast
            "source": ["noaa"] * 10,
        })
        events = self._detect(fast)
        assert len(events) == 0

    def test_short_stay_excluded(self):
        """Stay shorter than min_duration is excluded."""
        short = pl.DataFrame({
            "mmsi": pl.Series([111] * 5, dtype=pl.Int64),
            "timestamp": pl.Series(
                [_ts(i * 5) for i in range(5)],  # 20 minutes total
                dtype=pl.Datetime("us", "UTC"),
            ),
            "lat": [51.9] * 5,
            "lon": [4.0] * 5,
            "sog": [0.5] * 5,
            "source": ["noaa"] * 5,
        })
        events = self._detect(short)
        assert len(events) == 0

    def test_outside_port_excluded(self):
        """Low-speed positions outside any port are not port calls."""
        ocean = pl.DataFrame({
            "mmsi": pl.Series([111] * 10, dtype=pl.Int64),
            "timestamp": pl.Series(
                [_ts(i * 10) for i in range(10)], dtype=pl.Datetime("us", "UTC")
            ),
            "lat": [20.0] * 10,  # middle of ocean
            "lon": [-30.0] * 10,
            "sog": [0.5] * 10,
            "source": ["noaa"] * 10,
        })
        events = self._detect(ocean)
        assert len(events) == 0

    def test_two_vessels_detected(self):
        """Two vessels in port at different times → two events."""
        positions = _two_vessel_port_positions()
        events = self._detect(positions)
        assert len(events) == 2
        mmsis = sorted(events[EventCol.MMSI].to_list())
        assert mmsis == [111, 222]

    def test_gap_splits_port_calls(self):
        """A gap > gap_seconds splits into two port calls."""
        # First stay: minutes 0-120, then gap, second stay: 300-420
        stay1 = [
            (111, _ts(i * 10), 51.9, 4.0, 0.5, "noaa")
            for i in range(13)
        ]
        stay2 = [
            (111, _ts(300 + i * 10), 51.9, 4.0, 0.5, "noaa")
            for i in range(13)
        ]
        positions = pl.DataFrame({
            "mmsi": pl.Series([r[0] for r in stay1 + stay2], dtype=pl.Int64),
            "timestamp": pl.Series(
                [r[1] for r in stay1 + stay2], dtype=pl.Datetime("us", "UTC")
            ),
            "lat": [r[2] for r in stay1 + stay2],
            "lon": [r[3] for r in stay1 + stay2],
            "sog": [r[4] for r in stay1 + stay2],
            "source": [r[5] for r in stay1 + stay2],
        })
        events = self._detect(positions)
        assert len(events) == 2

    def test_empty_positions(self):
        """Empty positions → empty events."""
        empty = pl.DataFrame(schema={
            "mmsi": pl.Int64, "timestamp": pl.Datetime("us", "UTC"),
            "lat": pl.Float64, "lon": pl.Float64,
            "sog": pl.Float64, "source": pl.String,
        })
        events = self._detect(empty)
        assert len(events) == 0
        # Schema should still be correct.
        for col in SCHEMA:
            assert col in events.columns
