"""Tests for the AISHub community-sourced adapter.

Covers: registration, capabilities, normalization, vessel extraction,
scaling (SOG/COG/draught ÷ 10), sentinel handling (heading=511,
IMO=0/empty/Unknown), dimension derivation (A+B=length, C+D=beam),
variable-quality metadata, and SourceAdapter protocol conformance.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from neptune_ais.adapters.aishub import (
    ADAPTER_VERSION,
    AISHUB_SCALE_FACTOR,
    AISHubAdapter,
    COLUMN_MAP,
    IMO_UNAVAILABLE_VALUES,
    SOURCE_ID,
    _build_filename,
    _build_url,
)
from neptune_ais.adapters.base import RawArtifact, SourceAdapter, SourceCapabilities


# ---------------------------------------------------------------------------
# Sample AISHub data
# ---------------------------------------------------------------------------


def _sample_records() -> list[dict]:
    """Return sample AISHub API response records."""
    return [
        {
            "MMSI": 211000001,
            "TIME": "2024-06-15T00:00:00Z",
            "LATITUDE": 53.5,
            "LONGITUDE": 9.9,
            "SPEED": 55,       # 5.5 knots × 10
            "COURSE": 1800,    # 180.0° × 10
            "HEADING": 179,
            "NAME": "CONTAINER SHIP A",
            "IMO": "9876543",
            "CALLSIGN": "DABC1",
            "TYPE": 70,
            "A": 100,
            "B": 50,
            "C": 10,
            "D": 15,
            "DRAUGHT": 85,     # 8.5m × 10
            "DEST": "HAMBURG",
            "FLAG": "DE",
            "NAVSTAT": 0,
        },
        {
            "MMSI": 211000002,
            "TIME": "2024-06-15T00:01:00Z",
            "LATITUDE": 54.0,
            "LONGITUDE": 10.0,
            "SPEED": 0,
            "COURSE": 0,
            "HEADING": 511,     # unavailable
            "NAME": "FISHING BOAT B",
            "IMO": "0",         # unavailable
            "CALLSIGN": "DXYZ2",
            "TYPE": 30,
            "A": 8,
            "B": 4,
            "C": 2,
            "D": 3,
            "DRAUGHT": 30,
            "DEST": "",
            "FLAG": "DE",
            "NAVSTAT": 1,
        },
        {
            "MMSI": 211000003,
            "TIME": "2024-06-15T00:02:00Z",
            "LATITUDE": 53.0,
            "LONGITUDE": 9.5,
            "SPEED": 120,
            "COURSE": 900,
            "HEADING": 88,
            "NAME": "TANKER C",
            "IMO": "Unknown",   # unavailable
            "CALLSIGN": "DQRS3",
            "TYPE": 80,
            "A": 150,
            "B": 100,
            "C": 20,
            "D": 25,
            "DRAUGHT": 120,
            "DEST": "ROTTERDAM",
            "FLAG": "NL",
            "NAVSTAT": 5,
        },
    ]


def _write_json(path: Path, records: list[dict] | None = None) -> str:
    if records is None:
        records = _sample_records()
    filepath = path / "aishub_2024-06-15.json"
    filepath.write_text(json.dumps(records))
    return str(filepath)


def _make_artifact(path: str) -> RawArtifact:
    return RawArtifact(
        source_url="https://data.aishub.net/test",
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


class TestAISHubRegistration:
    def test_adapter_registered(self):
        from neptune_ais.adapters.registry import _ADAPTERS
        assert "aishub" in _ADAPTERS

    def test_source_id(self):
        assert AISHubAdapter().source_id == "aishub"

    def test_adapter_version(self):
        assert ADAPTER_VERSION == "aishub/0.1.0"

    def test_protocol_conformance(self):
        assert isinstance(AISHubAdapter(), SourceAdapter)


# ---------------------------------------------------------------------------
# Capabilities — variable quality explicitly documented
# ---------------------------------------------------------------------------


class TestAISHubCapabilities:
    def test_capabilities_type(self):
        assert isinstance(AISHubAdapter().capabilities, SourceCapabilities)

    def test_capabilities_values(self):
        caps = AISHubAdapter().capabilities
        assert caps.source_id == "aishub"
        assert caps.provider == "AISHub"
        assert caps.supports_backfill is False  # snapshot-oriented
        assert caps.supports_streaming is False
        assert caps.supports_server_side_bbox is True
        assert caps.auth_scheme == "api_key"
        assert caps.history_start is None

    def test_variable_quality_documented(self):
        """Quality variance is explicitly in known_quirks, not hidden."""
        caps = AISHubAdapter().capabilities
        quirk_text = " ".join(caps.known_quirks).lower()
        assert "variable" in quirk_text or "community" in quirk_text
        assert "clock drift" in quirk_text
        assert "gps" in quirk_text.lower() or "spoofing" in quirk_text

    def test_available_dates_is_none(self):
        """Snapshot API has no date enumeration."""
        assert AISHubAdapter().available_dates() is None

    def test_capabilities_summary(self):
        summary = AISHubAdapter().capabilities.summary()
        assert summary["source"] == "aishub"
        assert summary["backfill"] == "no"

    def test_api_key_required_for_fetch(self):
        adapter = AISHubAdapter(download_dir=Path("/tmp"))
        with pytest.raises(ValueError, match="api_key"):
            adapter.fetch_raw(FetchSpec(date=date(2024, 6, 15)))


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

from neptune_ais.adapters.base import FetchSpec


class TestAISHubURLHelpers:
    def test_build_url(self):
        url = _build_url("test_key")
        assert "aishub.net" in url
        assert "test_key" in url
        assert "json" in url

    def test_build_filename(self):
        assert _build_filename(date(2024, 6, 15)) == "aishub_2024-06-15.json"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestAISHubNormalization:
    def test_normalize_basic(self, tmp_path):
        path = _write_json(tmp_path)
        df = AISHubAdapter().normalize_positions([_make_artifact(path)])

        assert len(df) == 3
        assert "mmsi" in df.columns
        assert "timestamp" in df.columns
        assert "lat" in df.columns
        assert "lon" in df.columns
        assert "sog" in df.columns
        assert "source" in df.columns
        assert df["source"][0] == "aishub"

    def test_timestamp_parsed(self, tmp_path):
        path = _write_json(tmp_path)
        df = AISHubAdapter().normalize_positions([_make_artifact(path)])
        assert df["timestamp"].null_count() == 0
        assert df["timestamp"].dtype == pl.Datetime("us", "UTC")
        assert df["timestamp"][0] == datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)

    def test_mmsi_type(self, tmp_path):
        path = _write_json(tmp_path)
        df = AISHubAdapter().normalize_positions([_make_artifact(path)])
        assert df["mmsi"].dtype == pl.Int64

    def test_sog_scaling(self, tmp_path):
        """SOG ÷ 10: 55 → 5.5 knots."""
        path = _write_json(tmp_path)
        df = AISHubAdapter().normalize_positions([_make_artifact(path)])
        assert df["sog"][0] == pytest.approx(5.5)
        assert df["sog"][2] == pytest.approx(12.0)

    def test_cog_scaling(self, tmp_path):
        """COG ÷ 10: 1800 → 180.0°."""
        path = _write_json(tmp_path)
        df = AISHubAdapter().normalize_positions([_make_artifact(path)])
        assert df["cog"][0] == pytest.approx(180.0)

    def test_draught_scaling(self, tmp_path):
        """Draught ÷ 10: 85 → 8.5m."""
        path = _write_json(tmp_path)
        df = AISHubAdapter().normalize_positions([_make_artifact(path)])
        assert df["draught"][0] == pytest.approx(8.5)

    def test_heading_sentinel(self, tmp_path):
        """Heading=511 → null."""
        path = _write_json(tmp_path)
        df = AISHubAdapter().normalize_positions([_make_artifact(path)])
        assert df["heading"][0] == 179.0
        assert df["heading"][1] is None

    def test_imo_sentinel_zero(self, tmp_path):
        """IMO='0' → null."""
        path = _write_json(tmp_path)
        df = AISHubAdapter().normalize_positions([_make_artifact(path)])
        assert df["imo"][0] is not None       # valid
        assert df["imo"][1] is None           # "0" → null

    def test_imo_sentinel_unknown(self, tmp_path):
        """IMO='Unknown' → null."""
        path = _write_json(tmp_path)
        df = AISHubAdapter().normalize_positions([_make_artifact(path)])
        assert df["imo"][2] is None           # "Unknown" → null

    def test_dimension_derivation_length(self, tmp_path):
        """Length = A + B: 100 + 50 = 150m."""
        path = _write_json(tmp_path)
        df = AISHubAdapter().normalize_positions([_make_artifact(path)])
        assert "length" in df.columns
        assert df["length"][0] == pytest.approx(150.0)

    def test_dimension_derivation_beam(self, tmp_path):
        """Beam = C + D: 10 + 15 = 25m."""
        path = _write_json(tmp_path)
        df = AISHubAdapter().normalize_positions([_make_artifact(path)])
        assert "beam" in df.columns
        assert df["beam"][0] == pytest.approx(25.0)

    def test_no_dimension_fields_graceful(self, tmp_path):
        """Records without A/B/C/D fields don't break normalization."""
        records = [{
            "MMSI": 211000001,
            "TIME": "2024-06-15T00:00:00Z",
            "LATITUDE": 53.5,
            "LONGITUDE": 9.9,
        }]
        filepath = tmp_path / "aishub_minimal.json"
        filepath.write_text(json.dumps(records))
        df = AISHubAdapter().normalize_positions([_make_artifact(str(filepath))])
        assert len(df) == 1
        assert "length" not in df.columns
        assert "beam" not in df.columns

    def test_flag_preserved(self, tmp_path):
        path = _write_json(tmp_path)
        df = AISHubAdapter().normalize_positions([_make_artifact(path)])
        assert df["flag"][0] == "DE"

    def test_nav_status_as_string(self, tmp_path):
        path = _write_json(tmp_path)
        df = AISHubAdapter().normalize_positions([_make_artifact(path)])
        assert df["nav_status"].dtype == pl.String
        assert df["nav_status"][0] == "0"

    def test_metadata_wrapper_format(self, tmp_path):
        """Handles AISHub [metadata, [records]] format."""
        records = _sample_records()
        metadata = {"type": "FeatureCollection"}
        filepath = tmp_path / "aishub_wrapped.json"
        filepath.write_text(json.dumps([metadata, records]))
        df = AISHubAdapter().normalize_positions([_make_artifact(str(filepath))])
        assert len(df) == 3

    def test_empty_records_raises(self, tmp_path):
        filepath = tmp_path / "aishub_empty.json"
        filepath.write_text(json.dumps([]))
        with pytest.raises(ValueError, match="No data frames"):
            AISHubAdapter().normalize_positions([_make_artifact(str(filepath))])


# ---------------------------------------------------------------------------
# Vessel extraction
# ---------------------------------------------------------------------------


class TestAISHubVessels:
    def test_normalize_vessels(self, tmp_path):
        path = _write_json(tmp_path)
        vessels = AISHubAdapter().normalize_vessels([_make_artifact(path)])
        assert vessels is not None
        assert len(vessels) == 3
        assert "mmsi" in vessels.columns

    def test_vessel_identity_fields(self, tmp_path):
        path = _write_json(tmp_path)
        vessels = AISHubAdapter().normalize_vessels([_make_artifact(path)])
        assert "vessel_name" in vessels.columns
        assert "callsign" in vessels.columns


# ---------------------------------------------------------------------------
# QC rules
# ---------------------------------------------------------------------------


class TestAISHubQC:
    def test_qc_rules_empty(self):
        assert AISHubAdapter().qc_rules() == []
