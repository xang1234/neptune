"""Tests for the GFW (Global Fishing Watch) adapter.

Covers: registration, capabilities, events/vessels/effort normalization,
async bridge, auth fallback, and SourceAdapter protocol conformance.

The GFW adapter uses the official ``gfwapiclient`` package and provides
events, vessels, and fishing effort — NOT individual AIS positions.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from neptune_ais.adapters.gfw import (
    ADAPTER_VERSION,
    GFW_EVENT_TYPE_MAP,
    GFWAdapter,
    SOURCE_ID,
)
from neptune_ais.adapters.base import SourceAdapter, SourceCapabilities


# ---------------------------------------------------------------------------
# Sample GFW data (shaped like gfwapiclient model_dump output)
# ---------------------------------------------------------------------------


def _sample_gfw_events() -> list[dict]:
    """Return sample GFW Events API response records."""
    return [
        {
            "id": "evt-fishing-001",
            "type": "FISHING",
            "start": "2024-06-15T08:00:00Z",
            "end": "2024-06-15T14:00:00Z",
            "position": {"lat": 60.0, "lon": 22.0},
            "vessel": {
                "id": "vessel-abc",
                "ssvid": "230000001",
                "name": "FISHING VESSEL A",
                "flag": "FIN",
                "type": "FISHING",
            },
            "encounter": None,
            "fishing": {
                "totalDistanceKm": 25.0,
                "averageSpeedKnots": 3.2,
            },
            "loitering": None,
            "portVisit": None,
        },
        {
            "id": "evt-encounter-002",
            "type": "ENCOUNTER",
            "start": "2024-06-15T10:00:00Z",
            "end": "2024-06-15T12:00:00Z",
            "position": {"lat": 59.0, "lon": 21.0},
            "vessel": {
                "id": "vessel-abc",
                "ssvid": "230000001",
                "name": "FISHING VESSEL A",
                "flag": "FIN",
                "type": "FISHING",
            },
            "encounter": {
                "vessel": {
                    "id": "vessel-def",
                    "ssvid": "987654321",
                    "name": "CARRIER SHIP B",
                    "flag": "PAN",
                    "type": "CARRIER",
                },
                "medianDistanceKilometers": 0.3,
                "medianSpeedKnots": 1.5,
                "type": "CARRIER-FISHING",
            },
            "fishing": None,
            "loitering": None,
            "portVisit": None,
        },
        {
            "id": "evt-port-003",
            "type": "PORT_VISIT",
            "start": "2024-06-15T16:00:00Z",
            "end": "2024-06-15T20:00:00Z",
            "position": {"lat": 60.16, "lon": 24.94},
            "vessel": {
                "id": "vessel-ghi",
                "ssvid": "111222333",
                "name": "CARGO SHIP C",
                "flag": "NOR",
                "type": "CARGO",
            },
            "encounter": None,
            "fishing": None,
            "loitering": None,
            "portVisit": {
                "visitId": "pv-001",
                "confidence": "4",
                "durationHrs": 4.0,
            },
        },
        {
            "id": "evt-loitering-004",
            "type": "LOITERING",
            "start": "2024-06-15T06:00:00Z",
            "end": "2024-06-15T09:00:00Z",
            "position": {"lat": 58.0, "lon": 20.0},
            "vessel": {
                "id": "vessel-jkl",
                "ssvid": "444555666",
                "name": "REEFER D",
                "flag": "ESP",
                "type": "CARRIER",
            },
            "encounter": None,
            "fishing": None,
            "loitering": {
                "totalTimeHours": 3.0,
                "totalDistanceKm": 2.5,
                "averageSpeedKnots": 0.8,
            },
            "portVisit": None,
        },
    ]


def _sample_gfw_effort() -> list[dict]:
    """Return sample GFW 4Wings effort response records."""
    return [
        {
            "date": "2024-06-15",
            "flag": "ESP",
            "geartype": "TRAWLERS",
            "hours": 26.6,
            "lat": 49.33,
            "lon": 141.15,
            "vesselIDs": 3,
        },
        {
            "date": "2024-06-15",
            "flag": "FRA",
            "geartype": "PURSE_SEINES",
            "hours": 12.4,
            "lat": 48.0,
            "lon": -5.0,
            "vesselIDs": 1,
        },
    ]


def _write_events_json(path: Path, records: list[dict] | None = None) -> str:
    """Write sample events data to a JSON file, return the path."""
    if records is None:
        records = _sample_gfw_events()
    filepath = path / "gfw_events_2024-06-15.json"
    filepath.write_text(json.dumps(records))
    return str(filepath)


def _write_effort_json(path: Path, records: list[dict] | None = None) -> str:
    """Write sample effort data to a JSON file, return the path."""
    if records is None:
        records = _sample_gfw_effort()
    filepath = path / "gfw_effort_2024-06-15.json"
    filepath.write_text(json.dumps(records))
    return str(filepath)


def _make_artifact(path: str) -> "RawArtifact":
    """Create a minimal RawArtifact for testing."""
    from neptune_ais.adapters.base import RawArtifact

    return RawArtifact(
        source_url="https://example.com/gfw_test.json",
        filename=Path(path).name,
        local_path=path,
        content_hash="test_hash",
        size_bytes=1000,
        fetch_timestamp=datetime(2024, 6, 15, tzinfo=timezone.utc),
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# Registration and protocol
# ---------------------------------------------------------------------------


class TestGFWRegistration:
    def test_adapter_registered(self):
        """GFW adapter is auto-registered on import."""
        from neptune_ais.adapters.registry import _ADAPTERS

        assert "gfw" in _ADAPTERS

    def test_source_id(self):
        adapter = GFWAdapter()
        assert adapter.source_id == "gfw"

    def test_adapter_version(self):
        assert ADAPTER_VERSION == "gfw/0.2.0"

    def test_protocol_conformance(self):
        """GFWAdapter satisfies the SourceAdapter protocol."""
        assert isinstance(GFWAdapter(), SourceAdapter)


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestGFWCapabilities:
    def test_capabilities_type(self):
        adapter = GFWAdapter()
        assert isinstance(adapter.capabilities, SourceCapabilities)

    def test_capabilities_values(self):
        caps = GFWAdapter().capabilities
        assert caps.source_id == "gfw"
        assert caps.provider == "Global Fishing Watch"
        assert caps.supports_backfill is True
        assert caps.supports_streaming is False
        assert caps.auth_scheme == "api_key"
        assert caps.history_start == "2020-01-01"
        assert caps.coverage == "Global"

    def test_events_in_datasets(self):
        caps = GFWAdapter().capabilities
        assert "events" in caps.datasets_provided

    def test_vessels_in_datasets(self):
        caps = GFWAdapter().capabilities
        assert "vessels" in caps.datasets_provided

    def test_effort_in_datasets(self):
        caps = GFWAdapter().capabilities
        assert "fishing_effort" in caps.datasets_provided

    def test_positions_not_in_datasets(self):
        caps = GFWAdapter().capabilities
        assert "positions" not in caps.datasets_provided

    def test_capabilities_summary(self):
        summary = GFWAdapter().capabilities.summary()
        assert summary["source"] == "gfw"
        assert summary["backfill"] == "yes"
        assert summary["streaming"] == "no"

    def test_available_dates(self):
        adapter = GFWAdapter()
        start, end = adapter.available_dates()
        assert start == date(2020, 1, 1)
        assert end <= date.today()

    def test_known_quirks(self):
        caps = GFWAdapter().capabilities
        assert len(caps.known_quirks) >= 3


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TestGFWAuth:
    def test_api_key_param(self):
        adapter = GFWAdapter(access_token="tok123")
        assert adapter._api_key == "tok123"

    def test_api_key_backward_compat(self):
        adapter = GFWAdapter(api_key="key456")
        assert adapter._api_key == "key456"

    def test_api_key_takes_precedence(self):
        adapter = GFWAdapter(access_token="token", api_key="key")
        assert adapter._api_key == "token"

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("GFW_API_TOKEN", "env_tok")
        adapter = GFWAdapter()
        assert adapter._api_key == "env_tok"

    def test_env_var_gfw_api_key_compat(self, monkeypatch):
        monkeypatch.setenv("GFW_API_KEY", "env_key")
        adapter = GFWAdapter()
        assert adapter._api_key == "env_key"

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("GFW_API_TOKEN", "env_tok")
        adapter = GFWAdapter(access_token="explicit")
        assert adapter._api_key == "explicit"


# ---------------------------------------------------------------------------
# Normalize positions — raises NotImplementedError
# ---------------------------------------------------------------------------


class TestGFWNormalizePositions:
    def test_raises_not_implemented(self, tmp_path):
        """normalize_positions() raises with a clear error message."""
        events_path = _write_events_json(tmp_path)
        adapter = GFWAdapter()
        with pytest.raises(NotImplementedError, match="does not provide"):
            adapter.normalize_positions([_make_artifact(events_path)])


# ---------------------------------------------------------------------------
# Normalize events
# ---------------------------------------------------------------------------


class TestGFWNormalizeEvents:
    def test_basic_event_normalization(self, tmp_path):
        """Events are normalized with correct columns."""
        events_path = _write_events_json(tmp_path)
        adapter = GFWAdapter()
        df = adapter.normalize_events([_make_artifact(events_path)])

        assert df is not None
        assert len(df) == 4
        assert "event_id" in df.columns
        assert "event_type" in df.columns
        assert "mmsi" in df.columns
        assert "lat" in df.columns
        assert "lon" in df.columns
        assert "source" in df.columns

    def test_event_type_mapping(self, tmp_path):
        """GFW event types are mapped to Neptune event types."""
        events_path = _write_events_json(tmp_path)
        df = GFWAdapter().normalize_events([_make_artifact(events_path)])
        types = set(df["event_type"].to_list())
        assert "fishing" in types
        assert "encounter" in types
        assert "port_call" in types
        assert "loitering" in types

    def test_encounter_has_other_mmsi(self, tmp_path):
        """Encounter events have the other vessel's MMSI."""
        events_path = _write_events_json(tmp_path)
        df = GFWAdapter().normalize_events([_make_artifact(events_path)])
        encounters = df.filter(pl.col("event_type") == "encounter")
        assert len(encounters) == 1
        assert encounters["other_mmsi"][0] == 987654321

    def test_fishing_event_has_no_other_mmsi(self, tmp_path):
        """Fishing events have null other_mmsi."""
        events_path = _write_events_json(tmp_path)
        df = GFWAdapter().normalize_events([_make_artifact(events_path)])
        fishing = df.filter(pl.col("event_type") == "fishing")
        assert fishing["other_mmsi"][0] is None

    def test_confidence_score_set(self, tmp_path):
        """GFW events get confidence_score = 1.0."""
        events_path = _write_events_json(tmp_path)
        df = GFWAdapter().normalize_events([_make_artifact(events_path)])
        assert (df["confidence_score"] == 1.0).all()

    def test_source_field(self, tmp_path):
        """Source is always 'gfw'."""
        events_path = _write_events_json(tmp_path)
        df = GFWAdapter().normalize_events([_make_artifact(events_path)])
        assert (df["source"] == "gfw").all()

    def test_mmsi_parsed_from_ssvid(self, tmp_path):
        """vessel.ssvid string is parsed to Int64 MMSI."""
        events_path = _write_events_json(tmp_path)
        df = GFWAdapter().normalize_events([_make_artifact(events_path)])
        assert df["mmsi"].dtype == pl.Int64
        assert 230000001 in df["mmsi"].to_list()

    def test_timestamps_are_utc_datetimes(self, tmp_path):
        """start_time and end_time are UTC datetimes."""
        events_path = _write_events_json(tmp_path)
        df = GFWAdapter().normalize_events([_make_artifact(events_path)])
        assert df["start_time"].dtype == pl.Datetime("us", "UTC")
        assert df["end_time"].dtype == pl.Datetime("us", "UTC")
        assert df["start_time"].null_count() == 0
        assert df["end_time"].null_count() == 0

    def test_empty_events_returns_empty_df(self, tmp_path):
        """Empty events list returns empty DataFrame with correct schema."""
        events_path = _write_events_json(tmp_path, [])
        df = GFWAdapter().normalize_events([_make_artifact(events_path)])
        assert df is not None
        assert len(df) == 0
        assert "event_id" in df.columns

    def test_unknown_event_type_skipped(self, tmp_path):
        """Unknown event types (e.g. GAP) are skipped."""
        records = [
            {
                "id": "evt-gap-999",
                "type": "GAP",
                "start": "2024-06-15T00:00:00Z",
                "end": "2024-06-15T06:00:00Z",
                "position": {"lat": 50.0, "lon": 10.0},
                "vessel": {"ssvid": "123456789"},
            },
        ]
        events_path = _write_events_json(tmp_path, records)
        df = GFWAdapter().normalize_events([_make_artifact(events_path)])
        assert len(df) == 0

    def test_no_events_artifact_returns_none(self, tmp_path):
        """If no events artifact is in the list, returns None."""
        effort_path = _write_effort_json(tmp_path)
        df = GFWAdapter().normalize_events([_make_artifact(effort_path)])
        assert df is None


# ---------------------------------------------------------------------------
# Normalize vessels
# ---------------------------------------------------------------------------


class TestGFWVessels:
    def test_vessel_fields(self, tmp_path):
        """Vessels extracted from events have correct fields."""
        events_path = _write_events_json(tmp_path)
        vessels = GFWAdapter().normalize_vessels([_make_artifact(events_path)])
        assert vessels is not None
        assert "mmsi" in vessels.columns
        assert "vessel_name" in vessels.columns
        assert "flag" in vessels.columns
        assert "ship_type" in vessels.columns
        assert "source" in vessels.columns

    def test_unique_mmsis(self, tmp_path):
        """Each MMSI appears once in the vessels output."""
        events_path = _write_events_json(tmp_path)
        vessels = GFWAdapter().normalize_vessels([_make_artifact(events_path)])
        assert vessels is not None
        # Sample data has 4 events but only 3 unique main vessel MMSIs
        # (230000001 appears in both fishing and encounter).
        assert vessels["mmsi"].n_unique() == len(vessels)

    def test_source_field(self, tmp_path):
        events_path = _write_events_json(tmp_path)
        vessels = GFWAdapter().normalize_vessels([_make_artifact(events_path)])
        assert (vessels["source"] == "gfw").all()


# ---------------------------------------------------------------------------
# Normalize fishing effort
# ---------------------------------------------------------------------------


class TestGFWNormalizeFishingEffort:
    def test_effort_fields(self, tmp_path):
        """Effort records have correct fields."""
        effort_path = _write_effort_json(tmp_path)
        df = GFWAdapter().normalize_fishing_effort(
            [_make_artifact(effort_path)]
        )
        assert df is not None
        assert len(df) == 2
        assert "date" in df.columns
        assert "lat" in df.columns
        assert "lon" in df.columns
        assert "flag" in df.columns
        assert "geartype" in df.columns
        assert "vessel_hours" in df.columns

    def test_date_type(self, tmp_path):
        """Date column is pl.Date."""
        effort_path = _write_effort_json(tmp_path)
        df = GFWAdapter().normalize_fishing_effort(
            [_make_artifact(effort_path)]
        )
        assert df["date"].dtype == pl.Date

    def test_vessel_hours_from_hours(self, tmp_path):
        """4Wings 'hours' field maps to vessel_hours."""
        effort_path = _write_effort_json(tmp_path)
        df = GFWAdapter().normalize_fishing_effort(
            [_make_artifact(effort_path)]
        )
        assert df["vessel_hours"][0] == 26.6

    def test_source_field(self, tmp_path):
        effort_path = _write_effort_json(tmp_path)
        df = GFWAdapter().normalize_fishing_effort(
            [_make_artifact(effort_path)]
        )
        assert (df["source"] == "gfw").all()

    def test_empty_effort_returns_empty_df(self, tmp_path):
        """Empty effort list returns empty DataFrame with correct schema."""
        effort_path = _write_effort_json(tmp_path, [])
        df = GFWAdapter().normalize_fishing_effort(
            [_make_artifact(effort_path)]
        )
        assert df is not None
        assert len(df) == 0

    def test_no_effort_artifact_returns_none(self, tmp_path):
        """If no effort artifact is in the list, returns None."""
        events_path = _write_events_json(tmp_path)
        df = GFWAdapter().normalize_fishing_effort(
            [_make_artifact(events_path)]
        )
        assert df is None


# ---------------------------------------------------------------------------
# Async bridge
# ---------------------------------------------------------------------------


class TestGFWAsyncBridge:
    def test_run_async_from_sync(self):
        """_run_async works from a sync context."""
        import asyncio

        async def sample():
            return 42

        result = GFWAdapter._run_async(sample())
        assert result == 42

    def test_run_async_from_async_raises(self):
        """_run_async raises when called inside a running event loop."""
        import asyncio

        async def inner():
            async def sample():
                return 42

            GFWAdapter._run_async(sample())

        with pytest.raises(RuntimeError, match="async event loop"):
            asyncio.run(inner())


# ---------------------------------------------------------------------------
# QC rules
# ---------------------------------------------------------------------------


class TestGFWQC:
    def test_qc_rules_empty(self):
        """GFW handles quirks in normalization, not via QC rules."""
        assert GFWAdapter().qc_rules() == []


# ---------------------------------------------------------------------------
# Event type map
# ---------------------------------------------------------------------------


class TestEventTypeMap:
    def test_all_mapped_types_are_valid_event_types(self):
        """All mapped Neptune event types are in the events vocabulary."""
        from neptune_ais.datasets.events import EVENT_TYPES

        for neptune_type in GFW_EVENT_TYPE_MAP.values():
            assert neptune_type in EVENT_TYPES, (
                f"Mapped type {neptune_type!r} not in EVENT_TYPES"
            )
