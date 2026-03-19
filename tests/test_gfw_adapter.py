"""Tests for the GFW (Global Fishing Watch) adapter.

Covers: registration, capabilities, normalization, vessel extraction,
sentinel handling, column mapping, and SourceAdapter protocol conformance.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from neptune_ais.adapters.gfw import (
    ADAPTER_VERSION,
    COLUMN_MAP,
    GFWAdapter,
    SOURCE_ID,
    _build_filename,
    _build_url,
)
from neptune_ais.adapters.base import SourceAdapter, SourceCapabilities


# ---------------------------------------------------------------------------
# Sample GFW data
# ---------------------------------------------------------------------------


def _sample_gfw_records() -> list[dict]:
    """Return sample GFW API response records."""
    return [
        {
            "mmsi": 123456789,
            "timestamp": "2024-06-15T00:00:00Z",
            "lat": 40.0,
            "lon": -74.0,
            "speed": 5.5,
            "course": 180.0,
            "heading": 179.0,
            "shipname": "FISHING VESSEL A",
            "imo": 9876543,
            "callsign": "ABCD1",
            "flag": "NOR",
            "vessel_type": "fishing",
            "length": 25.0,
        },
        {
            "mmsi": 987654321,
            "timestamp": "2024-06-15T00:01:00Z",
            "lat": 41.0,
            "lon": -73.0,
            "speed": 12.0,
            "course": 90.0,
            "heading": 88.0,
            "shipname": "CARGO SHIP B",
            "imo": 0,  # unavailable
            "callsign": "EFGH2",
            "flag": "PAN",
            "vessel_type": "cargo",
            "length": 150.0,
        },
        {
            "mmsi": 111222333,
            "timestamp": "2024-06-15T00:02:00Z",
            "lat": 42.0,
            "lon": -72.0,
            "speed": 0.0,
            "course": 0.0,
            "shipname": "UNKNOWN",
            "imo": 0,
            "flag": "USA",
            "vessel_type": "other",
            "length": 10.0,
        },
    ]


def _write_gfw_json(path: Path, records: list[dict] | None = None) -> str:
    """Write sample GFW data to a JSON file, return the path."""
    if records is None:
        records = _sample_gfw_records()
    filepath = path / "gfw_2024-06-15.json"
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
        assert ADAPTER_VERSION == "gfw/0.1.0"

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
        assert caps.supports_server_side_bbox is True
        assert caps.auth_scheme == "api_key"
        assert "positions" in caps.datasets_provided
        assert "vessels" in caps.datasets_provided
        assert caps.history_start == "2012-01-01"
        assert caps.coverage == "Global"

    def test_capabilities_summary(self):
        summary = GFWAdapter().capabilities.summary()
        assert summary["source"] == "gfw"
        assert summary["backfill"] == "yes"
        assert summary["streaming"] == "no"

    def test_available_dates(self):
        adapter = GFWAdapter()
        start, end = adapter.available_dates()
        assert start == date(2012, 1, 1)
        assert end <= date.today()

    def test_known_quirks(self):
        caps = GFWAdapter().capabilities
        assert len(caps.known_quirks) >= 3
        assert any("IMO" in q for q in caps.known_quirks)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


class TestGFWURLHelpers:
    def test_build_url(self):
        url = _build_url(date(2024, 6, 15))
        assert "2024-06-15" in url
        assert "globalfishingwatch" in url

    def test_build_filename(self):
        assert _build_filename(date(2024, 6, 15)) == "gfw_2024-06-15.json"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestGFWNormalization:
    def test_normalize_basic(self, tmp_path):
        """Basic normalization produces correct columns and types."""
        path = _write_gfw_json(tmp_path)
        adapter = GFWAdapter()
        df = adapter.normalize_positions([_make_artifact(path)])

        assert len(df) == 3
        assert "mmsi" in df.columns
        assert "timestamp" in df.columns
        assert "lat" in df.columns
        assert "lon" in df.columns
        assert "sog" in df.columns  # renamed from speed
        assert "cog" in df.columns  # renamed from course
        assert "source" in df.columns
        assert df["source"][0] == "gfw"

    def test_timestamp_parsed(self, tmp_path):
        """Timestamps are parsed as non-null UTC datetimes."""
        path = _write_gfw_json(tmp_path)
        df = GFWAdapter().normalize_positions([_make_artifact(path)])
        assert df["timestamp"].null_count() == 0
        assert df["timestamp"].dtype == pl.Datetime("us", "UTC")
        assert df["timestamp"][0] == datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)

    def test_mmsi_type(self, tmp_path):
        path = _write_gfw_json(tmp_path)
        df = GFWAdapter().normalize_positions([_make_artifact(path)])
        assert df["mmsi"].dtype == pl.Int64

    def test_coordinate_types(self, tmp_path):
        path = _write_gfw_json(tmp_path)
        df = GFWAdapter().normalize_positions([_make_artifact(path)])
        assert df["lat"].dtype == pl.Float64
        assert df["lon"].dtype == pl.Float64

    def test_speed_course_mapping(self, tmp_path):
        """GFW 'speed' → 'sog', 'course' → 'cog'."""
        path = _write_gfw_json(tmp_path)
        df = GFWAdapter().normalize_positions([_make_artifact(path)])
        assert df["sog"][0] == 5.5
        assert df["cog"][0] == 180.0

    def test_imo_sentinel_normalized(self, tmp_path):
        """IMO=0 is normalized to null."""
        path = _write_gfw_json(tmp_path)
        df = GFWAdapter().normalize_positions([_make_artifact(path)])
        # First record has valid IMO.
        assert df["imo"][0] is not None
        # Second record has IMO=0 → null.
        assert df["imo"][1] is None

    def test_flag_preserved(self, tmp_path):
        path = _write_gfw_json(tmp_path)
        df = GFWAdapter().normalize_positions([_make_artifact(path)])
        assert df["flag"][0] == "NOR"
        assert df["flag"][1] == "PAN"

    def test_vessel_name_preserved(self, tmp_path):
        path = _write_gfw_json(tmp_path)
        df = GFWAdapter().normalize_positions([_make_artifact(path)])
        assert df["vessel_name"][0] == "FISHING VESSEL A"

    def test_wrapper_object_format(self, tmp_path):
        """Handles GFW response wrapped in {"entries": [...]}."""
        records = _sample_gfw_records()
        filepath = tmp_path / "gfw_wrapped.json"
        filepath.write_text(json.dumps({"entries": records}))
        df = GFWAdapter().normalize_positions([_make_artifact(str(filepath))])
        assert len(df) == 3

    def test_data_key_format(self, tmp_path):
        """Handles GFW response wrapped in {"data": [...]}."""
        records = _sample_gfw_records()
        filepath = tmp_path / "gfw_data.json"
        filepath.write_text(json.dumps({"data": records}))
        df = GFWAdapter().normalize_positions([_make_artifact(str(filepath))])
        assert len(df) == 3

    def test_empty_records_raises(self, tmp_path):
        filepath = tmp_path / "gfw_empty.json"
        filepath.write_text(json.dumps([]))
        with pytest.raises(ValueError, match="No data frames"):
            GFWAdapter().normalize_positions([_make_artifact(str(filepath))])

    def test_missing_optional_columns(self, tmp_path):
        """Records with missing optional fields are handled gracefully."""
        records = [{
            "mmsi": 111222333,
            "timestamp": "2024-06-15T00:00:00Z",
            "lat": 40.0,
            "lon": -74.0,
        }]
        filepath = tmp_path / "gfw_minimal.json"
        filepath.write_text(json.dumps(records))
        df = GFWAdapter().normalize_positions([_make_artifact(str(filepath))])
        assert len(df) == 1
        assert "mmsi" in df.columns
        assert "source" in df.columns


# ---------------------------------------------------------------------------
# Vessel extraction
# ---------------------------------------------------------------------------


class TestGFWVessels:
    def test_normalize_vessels(self, tmp_path):
        path = _write_gfw_json(tmp_path)
        adapter = GFWAdapter()
        vessels = adapter.normalize_vessels([_make_artifact(path)])
        assert vessels is not None
        assert len(vessels) == 3  # 3 unique MMSIs
        assert "mmsi" in vessels.columns
        assert "source" in vessels.columns

    def test_vessel_identity_fields(self, tmp_path):
        path = _write_gfw_json(tmp_path)
        vessels = GFWAdapter().normalize_vessels([_make_artifact(path)])
        assert "vessel_name" in vessels.columns


# ---------------------------------------------------------------------------
# QC rules
# ---------------------------------------------------------------------------


class TestGFWQC:
    def test_qc_rules_empty(self):
        """GFW handles quirks in normalization, not via QC rules."""
        assert GFWAdapter().qc_rules() == []


# ---------------------------------------------------------------------------
# Column map completeness
# ---------------------------------------------------------------------------


class TestColumnMap:
    def test_all_canonical_targets_are_valid(self):
        """All mapped target names are recognized canonical columns."""
        from neptune_ais.datasets.positions import Col
        known_cols = {v for k, v in vars(Col).items() if not k.startswith("_")}
        # flag is an extra column from GFW not in standard positions schema.
        known_cols.add("flag")
        for target in COLUMN_MAP.values():
            assert target in known_cols, f"Unknown canonical column: {target}"
