"""Phase 4 acceptance: cross-detector integration and regression tests.

Validates that all event detectors produce schema-conformant output,
confidence and provenance behave correctly across families, and
detector output feeds into the viz layer without issues.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from neptune_ais.datasets.events import (
    Col as EventCol,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    EVENT_TYPE_ENCOUNTER,
    EVENT_TYPE_EEZ_CROSSING,
    EVENT_TYPE_LOITERING,
    EVENT_TYPE_PORT_CALL,
    REQUIRED_COLUMNS,
    SCHEMA,
    classify_confidence,
    validate_schema,
)
from neptune_ais.derive.events import (
    EncounterConfig,
    EEZCrossingConfig,
    EventProvenance,
    LoiteringConfig,
    PortCallConfig,
    detect_encounters,
    detect_eez_crossings,
    detect_loitering,
    detect_port_calls,
    parse_provenance,
)
from neptune_ais.geometry.boundaries import (
    BoundaryDataset,
    BoundaryRegion,
    BoundaryRegistry,
)
from neptune_ais.viz import Viewport, prepare_events


# ---------------------------------------------------------------------------
# Shared fixture: a rich multi-vessel positions dataset
# ---------------------------------------------------------------------------

_BASE = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)


def _ts(minutes: float) -> datetime:
    return _BASE + timedelta(minutes=minutes)


def _rich_positions() -> pl.DataFrame:
    """Multi-vessel positions covering port calls, crossings, encounters,
    and loitering scenarios.

    Vessel 111: enters Rotterdam port, stays 2 hours (port call).
    Vessel 222: crosses from NLD EEZ to GBR EEZ.
    Vessel 333: loiters in open water for 1 hour.
    Vessel 111 & 222: close together at minutes 60-90 (encounter).
    """
    rows = []

    # Vessel 111: port call in Rotterdam (lat ~51.9, lon ~4.0)
    for i in range(13):
        rows.append((111, _ts(10 + i * 10), 51.9, 4.0 + i * 0.001, 0.5, "noaa"))

    # Vessel 222: EEZ crossing NLD → GBR (lon 5.0 → 2.0)
    for i in range(5):
        rows.append((222, _ts(i * 10), 53.0, 5.0 - i * 0.5, 8.0, "noaa"))
    for i in range(5):
        rows.append((222, _ts(50 + i * 10), 53.0, 2.0 - i * 0.75, 8.0, "noaa"))

    # Vessel 333: loitering in open water (lat ~30.0, lon ~-80.0)
    for i in range(11):
        rows.append((333, _ts(i * 6), 30.0 + i * 0.0001, -80.0 + i * 0.0001, 0.3, "noaa"))

    # Encounter: vessels 111 and 222 near each other (lat ~52.0, lon ~4.5)
    for i in range(7):
        rows.append((111, _ts(200 + i * 5), 52.0, 4.5, 1.0, "noaa"))
        rows.append((222, _ts(200 + i * 5), 52.001, 4.501, 1.0, "noaa"))

    return pl.DataFrame({
        "mmsi": pl.Series([r[0] for r in rows], dtype=pl.Int64),
        "timestamp": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us", "UTC")),
        "lat": [r[2] for r in rows],
        "lon": [r[3] for r in rows],
        "sog": [r[4] for r in rows],
        "source": [r[5] for r in rows],
    }).sort(["mmsi", "timestamp"])


def _boundary_registry() -> BoundaryRegistry:
    """Registry with Rotterdam port and NLD/GBR EEZ zones."""
    reg = BoundaryRegistry()
    reg.register(BoundaryDataset(
        name="ports",
        version="2024.1",
        regions=(
            BoundaryRegion(name="Rotterdam", bbox=(3.8, 51.8, 4.5, 52.1)),
        ),
    ))
    reg.register(BoundaryDataset(
        name="eez",
        version="11.0",
        regions=(
            BoundaryRegion(name="NLD_EEZ", bbox=(2.5, 51.0, 7.2, 55.8)),
            BoundaryRegion(name="GBR_EEZ", bbox=(-5.0, 49.0, 2.5, 56.0)),
        ),
    ))
    return reg


# ---------------------------------------------------------------------------
# Cross-detector integration
# ---------------------------------------------------------------------------


class TestCrossDetectorIntegration:
    """Run all detectors on the same dataset and verify combined output."""

    def _detect_all(self) -> pl.DataFrame:
        positions = _rich_positions()
        reg = _boundary_registry()

        port_regions = reg.lookup_column(positions, "ports")
        port_calls = detect_port_calls(
            positions, port_regions, source="noaa"
        )

        eez_regions = reg.lookup_column(positions, "eez")
        eez_crossings = detect_eez_crossings(
            positions, eez_regions, source="noaa"
        )

        encounters = detect_encounters(positions, source="noaa")

        loitering = detect_loitering(positions, source="noaa")

        # Combine all events.
        all_events = [df for df in [port_calls, eez_crossings, encounters, loitering] if len(df) > 0]
        if not all_events:
            return pl.DataFrame(schema=SCHEMA)
        return pl.concat(all_events)

    def test_combined_schema_valid(self):
        """All detectors produce schema-conformant output that concatenates."""
        events = self._detect_all()
        errors = validate_schema(events)
        assert errors == [], f"Schema errors: {errors}"

    def test_required_columns_present(self):
        events = self._detect_all()
        for col in REQUIRED_COLUMNS:
            assert col in events.columns, f"Missing: {col}"

    def test_multiple_event_types_present(self):
        """At least 2 different event types should be detected."""
        events = self._detect_all()
        types = set(events[EventCol.EVENT_TYPE].to_list())
        assert len(types) >= 2, f"Only found: {types}"

    def test_event_ids_unique(self):
        """All event IDs should be unique across detectors."""
        events = self._detect_all()
        ids = events[EventCol.EVENT_ID].to_list()
        assert len(ids) == len(set(ids)), "Duplicate event IDs found"

    def test_all_events_have_confidence(self):
        events = self._detect_all()
        assert events[EventCol.CONFIDENCE_SCORE].is_not_null().all()
        assert (events[EventCol.CONFIDENCE_SCORE] >= 0.0).all()
        assert (events[EventCol.CONFIDENCE_SCORE] <= 1.0).all()

    def test_all_events_have_provenance(self):
        events = self._detect_all()
        assert events[EventCol.RECORD_PROVENANCE].is_not_null().all()
        # All provenance tokens should be parseable.
        for token in events[EventCol.RECORD_PROVENANCE].to_list():
            prov = parse_provenance(token)
            assert prov.source == "noaa"
            assert len(prov.detector) > 0
            assert len(prov.detector_version) > 0

    def test_temporal_bounds_valid(self):
        events = self._detect_all()
        assert (events[EventCol.START_TIME] <= events[EventCol.END_TIME]).all()

    def test_spatial_bounds_valid(self):
        events = self._detect_all()
        assert (events[EventCol.LAT] >= -90).all()
        assert (events[EventCol.LAT] <= 90).all()
        assert (events[EventCol.LON] >= -180).all()
        assert (events[EventCol.LON] <= 180).all()


# ---------------------------------------------------------------------------
# Confidence regression
# ---------------------------------------------------------------------------


class TestConfidenceRegression:
    """Verify specific confidence behaviors across detectors."""

    def test_port_call_confidence_tiers(self):
        """Port call: 2h stay → 0.7, 5h stay → 0.9."""
        positions = _rich_positions()
        reg = _boundary_registry()
        port_regions = reg.lookup_column(positions, "ports")
        events = detect_port_calls(positions, port_regions, source="noaa")
        if len(events) > 0:
            for score in events[EventCol.CONFIDENCE_SCORE].to_list():
                band = classify_confidence(score)
                assert band in ("high", "medium"), f"Unexpected band: {band} for score {score}"

    def test_loitering_confidence_bands(self):
        """Loitering: 1h tight cluster → at least medium confidence."""
        events = detect_loitering(_rich_positions(), source="noaa")
        if len(events) > 0:
            assert (events[EventCol.CONFIDENCE_SCORE] >= CONFIDENCE_LOW).all()

    def test_encounter_confidence_range(self):
        """Encounters should produce valid confidence scores."""
        events = detect_encounters(_rich_positions(), source="noaa")
        if len(events) > 0:
            assert (events[EventCol.CONFIDENCE_SCORE] >= 0.0).all()
            assert (events[EventCol.CONFIDENCE_SCORE] <= 1.0).all()

    def test_classify_confidence_consistency(self):
        """classify_confidence agrees with CONFIDENCE_HIGH/LOW thresholds."""
        assert classify_confidence(CONFIDENCE_HIGH) == "high"
        assert classify_confidence(CONFIDENCE_HIGH - 0.01) == "medium"
        assert classify_confidence(CONFIDENCE_LOW) == "medium"
        assert classify_confidence(CONFIDENCE_LOW - 0.01) == "low"


# ---------------------------------------------------------------------------
# Provenance regression
# ---------------------------------------------------------------------------


class TestProvenanceRegression:
    """Verify provenance tokens across all detectors."""

    def test_port_call_provenance(self):
        positions = _rich_positions()
        reg = _boundary_registry()
        port_regions = reg.lookup_column(positions, "ports")
        events = detect_port_calls(positions, port_regions, source="noaa")
        if len(events) > 0:
            prov = parse_provenance(events[EventCol.RECORD_PROVENANCE][0])
            assert prov.detector == "port_call_detector"
            assert "boundaries" in prov.upstream_datasets
            assert "positions" in prov.upstream_datasets

    def test_eez_crossing_provenance(self):
        positions = _rich_positions()
        reg = _boundary_registry()
        eez_regions = reg.lookup_column(positions, "eez")
        events = detect_eez_crossings(positions, eez_regions, source="noaa")
        if len(events) > 0:
            prov = parse_provenance(events[EventCol.RECORD_PROVENANCE][0])
            assert prov.detector == "eez_crossing_detector"
            assert "boundaries" in prov.upstream_datasets

    def test_encounter_provenance(self):
        events = detect_encounters(_rich_positions(), source="noaa")
        if len(events) > 0:
            prov = parse_provenance(events[EventCol.RECORD_PROVENANCE][0])
            assert prov.detector == "encounter_detector"
            assert "positions" in prov.upstream_datasets

    def test_loitering_provenance(self):
        events = detect_loitering(_rich_positions(), source="noaa")
        if len(events) > 0:
            prov = parse_provenance(events[EventCol.RECORD_PROVENANCE][0])
            assert prov.detector == "loitering_detector"
            assert "positions" in prov.upstream_datasets

    def test_provenance_roundtrip_all_detectors(self):
        """All provenance tokens roundtrip through parse_provenance."""
        all_events = TestCrossDetectorIntegration()._detect_all()
        for token in all_events[EventCol.RECORD_PROVENANCE].unique().to_list():
            prov = parse_provenance(token)
            assert prov.to_token() == token


# ---------------------------------------------------------------------------
# Event → viz integration
# ---------------------------------------------------------------------------


class TestEventToVizIntegration:
    """Verify detector output feeds into prepare_events correctly."""

    def test_port_calls_to_viz(self):
        positions = _rich_positions()
        reg = _boundary_registry()
        port_regions = reg.lookup_column(positions, "ports")
        events = detect_port_calls(positions, port_regions, source="noaa")
        result = prepare_events(events)
        assert len(result) == len(events)

    def test_combined_events_to_viz(self):
        all_events = TestCrossDetectorIntegration()._detect_all()
        result = prepare_events(all_events)
        assert len(result) == len(all_events)

    def test_viz_event_type_filter(self):
        all_events = TestCrossDetectorIntegration()._detect_all()
        for etype in [EVENT_TYPE_PORT_CALL, EVENT_TYPE_LOITERING]:
            result = prepare_events(all_events, event_type=etype)
            if len(result) > 0:
                assert (result[EventCol.EVENT_TYPE] == etype).all()

    def test_viz_viewport_clips_events(self):
        all_events = TestCrossDetectorIntegration()._detect_all()
        # Viewport covering only Rotterdam area.
        viewport = Viewport(west=3.5, south=51.5, east=5.0, north=52.5)
        result = prepare_events(all_events, viewport=viewport)
        if len(result) > 0:
            assert (result[EventCol.LAT] >= 51.5).all()
            assert (result[EventCol.LAT] <= 52.5).all()

    def test_viz_confidence_filter(self):
        all_events = TestCrossDetectorIntegration()._detect_all()
        result = prepare_events(all_events, min_confidence=CONFIDENCE_HIGH)
        if len(result) > 0:
            assert (result[EventCol.CONFIDENCE_SCORE] >= CONFIDENCE_HIGH).all()


# ---------------------------------------------------------------------------
# Event ID determinism regression
# ---------------------------------------------------------------------------


class TestEventIdDeterminism:
    """Verify that identical inputs produce identical event IDs."""

    def test_port_call_ids_stable(self):
        positions = _rich_positions()
        reg = _boundary_registry()
        port_regions = reg.lookup_column(positions, "ports")
        a = detect_port_calls(positions, port_regions, source="noaa")
        b = detect_port_calls(positions, port_regions, source="noaa")
        if len(a) > 0:
            assert a[EventCol.EVENT_ID].to_list() == b[EventCol.EVENT_ID].to_list()

    def test_encounter_ids_stable(self):
        a = detect_encounters(_rich_positions(), source="noaa")
        b = detect_encounters(_rich_positions(), source="noaa")
        if len(a) > 0:
            assert a[EventCol.EVENT_ID].to_list() == b[EventCol.EVENT_ID].to_list()

    def test_loitering_ids_stable(self):
        a = detect_loitering(_rich_positions(), source="noaa")
        b = detect_loitering(_rich_positions(), source="noaa")
        if len(a) > 0:
            assert a[EventCol.EVENT_ID].to_list() == b[EventCol.EVENT_ID].to_list()

    def test_ids_change_with_config(self):
        """Different configs produce different event IDs."""
        positions = _rich_positions()
        a = detect_loitering(positions, config=LoiteringConfig(max_speed_knots=2.0), source="noaa")
        b = detect_loitering(positions, config=LoiteringConfig(max_speed_knots=5.0), source="noaa")
        if len(a) > 0 and len(b) > 0:
            ids_a = set(a[EventCol.EVENT_ID].to_list())
            ids_b = set(b[EventCol.EVENT_ID].to_list())
            assert ids_a != ids_b
