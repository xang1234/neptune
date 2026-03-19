"""Tests for the Finland (Digitraffic) adapter.

Covers: registration, capabilities, normalization, vessel extraction,
scaling (SOG/COG/draught ÷ 10), sentinel handling (heading=511, IMO=0),
epoch timestamp parsing, and SourceAdapter protocol conformance.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from neptune_ais.adapters.finland import (
    ADAPTER_VERSION,
    COLUMN_MAP,
    DIGITRAFFIC_SCALE_FACTOR,
    FinlandAdapter,
    SOURCE_ID,
    _build_filename,
    _build_url,
)
from neptune_ais.adapters.base import RawArtifact, SourceAdapter, SourceCapabilities


# ---------------------------------------------------------------------------
# Sample Digitraffic data
# ---------------------------------------------------------------------------


def _sample_records() -> list[dict]:
    """Return sample Digitraffic API response records."""
    # timestampExternal is Unix epoch milliseconds.
    return [
        {
            "mmsi": 230000001,
            "timestampExternal": 1718409600000,  # 2024-06-15T00:00:00Z
            "y": 60.1,    # lat
            "x": 24.9,    # lon
            "sog": 55,    # 5.5 knots × 10
            "cog": 1800,  # 180.0 degrees × 10
            "heading": 179,
            "name": "FERRY ALPHA",
            "imo": 9876543,
            "callSign": "OIABC",
            "destination": "HELSINKI",
            "shipType": 60,
            "draught": 55,  # 5.5 meters × 10
        },
        {
            "mmsi": 230000002,
            "timestampExternal": 1718409660000,  # 2024-06-15T00:01:00Z
            "y": 60.2,
            "x": 25.0,
            "sog": 0,
            "cog": 0,
            "heading": 511,  # unavailable
            "name": "CARGO BETA",
            "imo": 0,  # unavailable
            "callSign": "OIXYZ",
            "destination": "TURKU",
            "shipType": 70,
            "draught": 80,  # 8.0 meters × 10
        },
        {
            "mmsi": 230000003,
            "timestampExternal": 1718409720000,  # 2024-06-15T00:02:00Z
            "y": 59.9,
            "x": 24.8,
            "sog": 120,  # 12.0 knots × 10
            "cog": 900,  # 90.0 degrees × 10
            "heading": 88,
            "name": "TANKER GAMMA",
            "imo": 1234567,
            "callSign": "OIQRS",
            "shipType": 80,
            "draught": 100,  # 10.0 meters × 10
        },
    ]


def _write_json(path: Path, records: list[dict] | None = None) -> str:
    if records is None:
        records = _sample_records()
    filepath = path / "finland_2024-06-15.json"
    filepath.write_text(json.dumps(records))
    return str(filepath)


def _make_artifact(path: str) -> RawArtifact:
    return RawArtifact(
        source_url="https://meri.digitraffic.fi/test",
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


class TestFinlandRegistration:
    def test_adapter_registered(self):
        from neptune_ais.adapters.registry import _ADAPTERS
        assert "finland" in _ADAPTERS

    def test_source_id(self):
        assert FinlandAdapter().source_id == "finland"

    def test_adapter_version(self):
        assert ADAPTER_VERSION == "finland/0.1.0"

    def test_protocol_conformance(self):
        assert isinstance(FinlandAdapter(), SourceAdapter)


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestFinlandCapabilities:
    def test_capabilities_type(self):
        assert isinstance(FinlandAdapter().capabilities, SourceCapabilities)

    def test_capabilities_values(self):
        caps = FinlandAdapter().capabilities
        assert caps.source_id == "finland"
        assert caps.provider == "Fintraffic Digitraffic"
        assert caps.supports_backfill is True
        assert caps.supports_streaming is True  # mixed delivery
        assert caps.auth_scheme is None  # open data
        assert "positions" in caps.datasets_provided
        assert caps.history_start == "2019-01-01"

    def test_capabilities_summary(self):
        summary = FinlandAdapter().capabilities.summary()
        assert summary["source"] == "finland"
        assert summary["backfill"] == "yes"
        assert summary["streaming"] == "yes"
        assert summary["auth"] == "none"

    def test_available_dates(self):
        start, end = FinlandAdapter().available_dates()
        assert start == date(2019, 1, 1)
        assert end <= date.today()

    def test_known_quirks(self):
        caps = FinlandAdapter().capabilities
        assert len(caps.known_quirks) >= 5
        assert any("SOG" in q for q in caps.known_quirks)
        assert any("heading" in q.lower() for q in caps.known_quirks)


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


class TestFinlandURLHelpers:
    def test_build_url(self):
        url = _build_url(date(2024, 6, 15))
        assert "2024-06-15" in url
        assert "digitraffic" in url

    def test_build_filename(self):
        assert _build_filename(date(2024, 6, 15)) == "finland_2024-06-15.json"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestFinlandNormalization:
    def test_normalize_basic(self, tmp_path):
        path = _write_json(tmp_path)
        df = FinlandAdapter().normalize_positions([_make_artifact(path)])

        assert len(df) == 3
        assert "mmsi" in df.columns
        assert "timestamp" in df.columns
        assert "lat" in df.columns
        assert "lon" in df.columns
        assert "sog" in df.columns
        assert "cog" in df.columns
        assert "source" in df.columns
        assert df["source"][0] == "finland"

    def test_timestamp_parsed(self, tmp_path):
        """Epoch milliseconds are parsed as non-null UTC datetimes."""
        path = _write_json(tmp_path)
        df = FinlandAdapter().normalize_positions([_make_artifact(path)])
        assert df["timestamp"].null_count() == 0
        assert df["timestamp"].dtype == pl.Datetime("us", "UTC")
        assert df["timestamp"][0] == datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)

    def test_mmsi_type(self, tmp_path):
        path = _write_json(tmp_path)
        df = FinlandAdapter().normalize_positions([_make_artifact(path)])
        assert df["mmsi"].dtype == pl.Int64

    def test_coordinate_mapping(self, tmp_path):
        """x → lon, y → lat (Digitraffic uses reversed naming)."""
        path = _write_json(tmp_path)
        df = FinlandAdapter().normalize_positions([_make_artifact(path)])
        assert df["lat"][0] == 60.1  # y
        assert df["lon"][0] == 24.9  # x

    def test_sog_scaling(self, tmp_path):
        """SOG is divided by 10 (Digitraffic sends knots × 10)."""
        path = _write_json(tmp_path)
        df = FinlandAdapter().normalize_positions([_make_artifact(path)])
        assert df["sog"][0] == pytest.approx(5.5)
        assert df["sog"][2] == pytest.approx(12.0)

    def test_cog_scaling(self, tmp_path):
        """COG is divided by 10 (Digitraffic sends degrees × 10)."""
        path = _write_json(tmp_path)
        df = FinlandAdapter().normalize_positions([_make_artifact(path)])
        assert df["cog"][0] == pytest.approx(180.0)
        assert df["cog"][2] == pytest.approx(90.0)

    def test_draught_scaling(self, tmp_path):
        """Draught is divided by 10 (Digitraffic sends meters × 10)."""
        path = _write_json(tmp_path)
        df = FinlandAdapter().normalize_positions([_make_artifact(path)])
        assert df["draught"][0] == pytest.approx(5.5)
        assert df["draught"][1] == pytest.approx(8.0)

    def test_heading_sentinel(self, tmp_path):
        """Heading=511 is normalized to null."""
        path = _write_json(tmp_path)
        df = FinlandAdapter().normalize_positions([_make_artifact(path)])
        assert df["heading"][0] == 179.0  # valid
        assert df["heading"][1] is None   # 511 → null

    def test_imo_sentinel(self, tmp_path):
        """IMO=0 is normalized to null."""
        path = _write_json(tmp_path)
        df = FinlandAdapter().normalize_positions([_make_artifact(path)])
        assert df["imo"][0] is not None    # valid
        assert df["imo"][1] is None        # 0 → null

    def test_vessel_name_preserved(self, tmp_path):
        path = _write_json(tmp_path)
        df = FinlandAdapter().normalize_positions([_make_artifact(path)])
        assert df["vessel_name"][0] == "FERRY ALPHA"

    def test_destination_preserved(self, tmp_path):
        path = _write_json(tmp_path)
        df = FinlandAdapter().normalize_positions([_make_artifact(path)])
        assert df["destination"][0] == "HELSINKI"

    def test_ship_type_as_string(self, tmp_path):
        """Numeric ship type is cast to string."""
        path = _write_json(tmp_path)
        df = FinlandAdapter().normalize_positions([_make_artifact(path)])
        assert df["ship_type"].dtype == pl.String
        assert df["ship_type"][0] == "60"

    def test_wrapper_object_format(self, tmp_path):
        """Handles response wrapped in {"features": [...]}."""
        records = _sample_records()
        filepath = tmp_path / "finland_wrapped.json"
        filepath.write_text(json.dumps({"features": records}))
        df = FinlandAdapter().normalize_positions([_make_artifact(str(filepath))])
        assert len(df) == 3

    def test_empty_records_raises(self, tmp_path):
        filepath = tmp_path / "finland_empty.json"
        filepath.write_text(json.dumps([]))
        with pytest.raises(ValueError, match="No data frames"):
            FinlandAdapter().normalize_positions([_make_artifact(str(filepath))])

    def test_missing_optional_columns(self, tmp_path):
        """Records with only required fields are handled."""
        records = [{
            "mmsi": 230000001,
            "timestampExternal": 1718409600000,
            "y": 60.1,
            "x": 24.9,
        }]
        filepath = tmp_path / "finland_minimal.json"
        filepath.write_text(json.dumps(records))
        df = FinlandAdapter().normalize_positions([_make_artifact(str(filepath))])
        assert len(df) == 1
        assert "source" in df.columns


# ---------------------------------------------------------------------------
# Vessel extraction
# ---------------------------------------------------------------------------


class TestFinlandVessels:
    def test_normalize_vessels(self, tmp_path):
        path = _write_json(tmp_path)
        vessels = FinlandAdapter().normalize_vessels([_make_artifact(path)])
        assert vessels is not None
        assert len(vessels) == 3
        assert "mmsi" in vessels.columns
        assert "source" in vessels.columns

    def test_vessel_identity_fields(self, tmp_path):
        path = _write_json(tmp_path)
        vessels = FinlandAdapter().normalize_vessels([_make_artifact(path)])
        assert "vessel_name" in vessels.columns
        assert "callsign" in vessels.columns


# ---------------------------------------------------------------------------
# QC rules
# ---------------------------------------------------------------------------


class TestFinlandQC:
    def test_qc_rules_empty(self):
        assert FinlandAdapter().qc_rules() == []


# ---------------------------------------------------------------------------
# Column map
# ---------------------------------------------------------------------------


class TestColumnMap:
    def test_coordinate_swap(self):
        """x maps to lon, y maps to lat (Digitraffic convention)."""
        assert COLUMN_MAP["x"] == "lon"
        assert COLUMN_MAP["y"] == "lat"

    def test_all_canonical_targets_valid(self):
        from neptune_ais.datasets.positions import Col
        known = {v for k, v in vars(Col).items() if not k.startswith("_")}
        for target in COLUMN_MAP.values():
            assert target in known, f"Unknown canonical column: {target}"
