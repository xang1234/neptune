"""Tests for events schema, event_id generation, and cache-key contract.

Validates that the events/v1 schema is well-formed, the event_id
function is deterministic and sensitive to inputs, and the
EventCacheKey follows the same invalidation patterns as TrackCacheKey.
"""

from __future__ import annotations

import polars as pl
import pytest

from neptune_ais.catalog import current_schema_version, is_compatible
from neptune_ais.datasets.events import (
    Col,
    DATASET_NAME,
    EVENT_TYPE_PORT_CALL,
    EVENT_TYPE_EEZ_CROSSING,
    EVENT_TYPE_ENCOUNTER,
    EVENT_TYPE_LOITERING,
    EVENT_TYPES,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    SCHEMA,
    SCHEMA_VERSION,
    VALID_RANGES,
    make_event_id,
    validate_schema,
)
from neptune_ais.derive.events import EventCacheKey


# ---------------------------------------------------------------------------
# Schema structure
# ---------------------------------------------------------------------------


class TestEventsSchema:
    """Validate the events/v1 schema definition is self-consistent."""

    def test_schema_version_format(self):
        assert SCHEMA_VERSION == "events/v1"

    def test_dataset_name(self):
        assert DATASET_NAME == "events"

    def test_required_columns_are_subset_of_schema(self):
        """Every required column must appear in the schema."""
        for col in REQUIRED_COLUMNS:
            assert col in SCHEMA, f"Required column {col!r} not in SCHEMA"

    def test_optional_columns_are_remainder(self):
        """Optional = schema keys - required."""
        assert OPTIONAL_COLUMNS == frozenset(SCHEMA.keys()) - REQUIRED_COLUMNS

    def test_required_columns_count(self):
        """Exactly 10 required columns."""
        assert len(REQUIRED_COLUMNS) == 10

    def test_optional_columns_content(self):
        """other_mmsi and geometry_wkb are optional."""
        assert Col.OTHER_MMSI in OPTIONAL_COLUMNS
        assert Col.GEOMETRY_WKB in OPTIONAL_COLUMNS

    def test_col_names_match_schema_keys(self):
        """Every Col attribute that's a string should be a schema key."""
        col_values = {
            v for k, v in vars(Col).items()
            if not k.startswith("_") and isinstance(v, str)
        }
        assert col_values == set(SCHEMA.keys())

    def test_temporal_columns_are_datetime(self):
        assert SCHEMA[Col.START_TIME] == pl.Datetime("us", "UTC")
        assert SCHEMA[Col.END_TIME] == pl.Datetime("us", "UTC")

    def test_spatial_columns_are_float64(self):
        assert SCHEMA[Col.LAT] == pl.Float64
        assert SCHEMA[Col.LON] == pl.Float64

    def test_confidence_is_float64(self):
        assert SCHEMA[Col.CONFIDENCE_SCORE] == pl.Float64

    def test_mmsi_columns_are_int64(self):
        assert SCHEMA[Col.MMSI] == pl.Int64
        assert SCHEMA[Col.OTHER_MMSI] == pl.Int64

    def test_geometry_wkb_is_binary(self):
        assert SCHEMA[Col.GEOMETRY_WKB] == pl.Binary

    def test_string_columns(self):
        for col in [Col.EVENT_ID, Col.EVENT_TYPE, Col.SOURCE, Col.RECORD_PROVENANCE]:
            assert SCHEMA[col] == pl.String, f"{col} should be String"


# ---------------------------------------------------------------------------
# Event type vocabulary
# ---------------------------------------------------------------------------


class TestEventTypes:
    def test_known_types(self):
        assert EVENT_TYPE_PORT_CALL == "port_call"
        assert EVENT_TYPE_EEZ_CROSSING == "eez_crossing"
        assert EVENT_TYPE_ENCOUNTER == "encounter"
        assert EVENT_TYPE_LOITERING == "loitering"

    def test_event_types_frozenset(self):
        assert len(EVENT_TYPES) == 4
        assert EVENT_TYPES == {
            "port_call", "eez_crossing", "encounter", "loitering",
        }


# ---------------------------------------------------------------------------
# Valid ranges
# ---------------------------------------------------------------------------


class TestValidRanges:
    def test_lat_range(self):
        assert VALID_RANGES[Col.LAT] == (-90.0, 90.0)

    def test_lon_range(self):
        assert VALID_RANGES[Col.LON] == (-180.0, 180.0)

    def test_confidence_range(self):
        assert VALID_RANGES[Col.CONFIDENCE_SCORE] == (0.0, 1.0)


# ---------------------------------------------------------------------------
# make_event_id
# ---------------------------------------------------------------------------


class TestMakeEventId:
    def test_format(self):
        eid = make_event_id("port_call", 111, 1718409600000000, "noaa", "cfg1")
        assert isinstance(eid, str)
        assert len(eid) == 16
        assert all(c in "0123456789abcdef" for c in eid)

    def test_deterministic(self):
        a = make_event_id("port_call", 111, 1718409600000000, "noaa", "cfg1")
        b = make_event_id("port_call", 111, 1718409600000000, "noaa", "cfg1")
        assert a == b

    def test_differs_by_event_type(self):
        a = make_event_id("port_call", 111, 1718409600000000, "noaa", "cfg1")
        b = make_event_id("encounter", 111, 1718409600000000, "noaa", "cfg1")
        assert a != b

    def test_differs_by_mmsi(self):
        a = make_event_id("port_call", 111, 1718409600000000, "noaa", "cfg1")
        b = make_event_id("port_call", 222, 1718409600000000, "noaa", "cfg1")
        assert a != b

    def test_differs_by_start_time(self):
        a = make_event_id("port_call", 111, 1718409600000000, "noaa", "cfg1")
        b = make_event_id("port_call", 111, 1718409601000000, "noaa", "cfg1")
        assert a != b

    def test_differs_by_source(self):
        a = make_event_id("port_call", 111, 1718409600000000, "noaa", "cfg1")
        b = make_event_id("port_call", 111, 1718409600000000, "dma", "cfg1")
        assert a != b

    def test_differs_by_config_hash(self):
        a = make_event_id("port_call", 111, 1718409600000000, "noaa", "cfg1")
        b = make_event_id("port_call", 111, 1718409600000000, "noaa", "cfg2")
        assert a != b

    def test_empty_config_hash(self):
        """Empty config_hash is valid (for source-native events)."""
        eid = make_event_id("port_call", 111, 1718409600000000, "noaa")
        assert len(eid) == 16


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def _valid_events_df(self) -> pl.DataFrame:
        """Minimal valid events DataFrame."""
        from datetime import datetime, timezone

        return pl.DataFrame({
            Col.EVENT_ID: ["abc123def4567890"],
            Col.EVENT_TYPE: ["port_call"],
            Col.MMSI: pl.Series([111], dtype=pl.Int64),
            Col.OTHER_MMSI: pl.Series([None], dtype=pl.Int64),
            Col.START_TIME: pl.Series(
                [datetime(2024, 6, 15, 0, 0, tzinfo=timezone.utc)],
                dtype=pl.Datetime("us", "UTC"),
            ),
            Col.END_TIME: pl.Series(
                [datetime(2024, 6, 15, 1, 0, tzinfo=timezone.utc)],
                dtype=pl.Datetime("us", "UTC"),
            ),
            Col.LAT: [40.0],
            Col.LON: [-74.0],
            Col.GEOMETRY_WKB: pl.Series([None], dtype=pl.Binary),
            Col.CONFIDENCE_SCORE: [0.85],
            Col.SOURCE: ["noaa"],
            Col.RECORD_PROVENANCE: ["noaa:port_call"],
        })

    def test_valid_df_passes(self):
        errors = validate_schema(self._valid_events_df())
        assert errors == []

    def test_missing_required_column(self):
        df = self._valid_events_df().drop(Col.EVENT_ID)
        errors = validate_schema(df)
        assert any("event_id" in e for e in errors)

    def test_wrong_dtype(self):
        df = self._valid_events_df().with_columns(
            pl.col(Col.MMSI).cast(pl.String)
        )
        errors = validate_schema(df)
        assert any("mmsi" in e for e in errors)

    def test_unexpected_column(self):
        df = self._valid_events_df().with_columns(
            pl.lit("extra").alias("unknown_col")
        )
        errors = validate_schema(df)
        assert any("unknown_col" in e for e in errors)

    def test_lazyframe_validation(self):
        errors = validate_schema(self._valid_events_df().lazy())
        assert errors == []


# ---------------------------------------------------------------------------
# Catalog registration
# ---------------------------------------------------------------------------


class TestCatalogRegistration:
    def test_events_schema_registered(self):
        version = current_schema_version("events")
        assert version == "events/v1"

    def test_events_schema_compatible(self):
        assert is_compatible("events", "events/v1")

    def test_events_schema_future_incompatible(self):
        assert not is_compatible("events", "events/v99")


# ---------------------------------------------------------------------------
# EventCacheKey
# ---------------------------------------------------------------------------


class TestEventCacheKey:
    # Shared baseline kwargs — sensitivity tests override one field at a time.
    _BASE = dict(
        event_type="port_call",
        source="noaa",
        date="2024-06-15",
        config_hash="abc",
        upstream_schema_version="tracks/v1",
        events_schema_version="events/v1",
        upstream_manifest_hash="def",
    )

    def test_cache_key_format(self):
        ck = EventCacheKey(**self._BASE).cache_key()
        assert isinstance(ck, str)
        assert len(ck) == 16
        assert all(c in "0123456789abcdef" for c in ck)

    def test_cache_key_deterministic(self):
        assert (
            EventCacheKey(**self._BASE).cache_key()
            == EventCacheKey(**self._BASE).cache_key()
        )

    def test_cache_key_changes_with_event_type(self):
        a = EventCacheKey(**{**self._BASE, "event_type": "port_call"}).cache_key()
        b = EventCacheKey(**{**self._BASE, "event_type": "encounter"}).cache_key()
        assert a != b

    def test_cache_key_changes_with_config(self):
        a = EventCacheKey(**{**self._BASE, "config_hash": "abc"}).cache_key()
        b = EventCacheKey(**{**self._BASE, "config_hash": "xyz"}).cache_key()
        assert a != b

    def test_cache_key_changes_with_upstream_hash(self):
        a = EventCacheKey(**{**self._BASE, "upstream_manifest_hash": "hash1"}).cache_key()
        b = EventCacheKey(**{**self._BASE, "upstream_manifest_hash": "hash2"}).cache_key()
        assert a != b

    def test_cache_key_changes_with_schema_version(self):
        a = EventCacheKey(**{**self._BASE, "events_schema_version": "events/v1"}).cache_key()
        b = EventCacheKey(**{**self._BASE, "events_schema_version": "events/v2"}).cache_key()
        assert a != b

    def test_from_manifest_factory(self):
        key = EventCacheKey.from_manifest(
            event_type="port_call",
            source="noaa",
            date="2024-06-15",
            config_hash="abc",
            upstream_manifest_hash="def",
        )
        assert key.event_type == "port_call"
        assert key.upstream_schema_version == "tracks/v1"
        assert key.events_schema_version == "events/v1"
        assert len(key.cache_key()) == 16

    def test_from_manifest_custom_versions(self):
        key = EventCacheKey.from_manifest(
            event_type="eez_crossing",
            source="dma",
            date="2024-06-15",
            config_hash="xyz",
            upstream_manifest_hash="ghi",
            upstream_schema_version="positions/v1",
            events_schema_version="events/v1",
        )
        assert key.upstream_schema_version == "positions/v1"
