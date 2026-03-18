"""Tests for AISStream adapter: subscription, normalization, registration.

Covers: subscription building, message normalization, sentinel handling,
capabilities registration, and integration with NeptuneStream.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from neptune_ais.adapters.aisstream import (
    CAPABILITIES,
    SOURCE_ID,
    build_subscription,
    normalize_message,
)


# ---------------------------------------------------------------------------
# Subscription
# ---------------------------------------------------------------------------


class TestBuildSubscription:
    def test_default_subscription(self):
        sub = json.loads(build_subscription("test-key"))
        assert sub["APIKey"] == "test-key"
        assert sub["FilterMessageTypes"] == ["PositionReport"]
        assert len(sub["BoundingBoxes"]) == 1
        # Full globe default.
        assert sub["BoundingBoxes"][0] == [[-90, -180], [90, 180]]

    def test_bbox_subscription(self):
        sub = json.loads(build_subscription(
            "test-key",
            bbox=(-74.0, 40.0, -73.0, 41.0),
        ))
        bb = sub["BoundingBoxes"][0]
        # AISStream format: [[south, west], [north, east]]
        assert bb == [[40.0, -74.0], [41.0, -73.0]]


# ---------------------------------------------------------------------------
# Message normalization
# ---------------------------------------------------------------------------


def _raw_position_report(
    mmsi: int = 123456789,
    lat: float = 40.0,
    lon: float = -74.0,
    sog: int = 105,
    cog: int = 1800,
    heading: int = 178,
    nav_status: int = 0,
    ship_name: str = "TEST VESSEL",
    time_utc: str = "2024-06-15 00:00:00.000000+00:00",
) -> dict[str, Any]:
    """Build a synthetic AISStream position report message."""
    return {
        "MessageType": "PositionReport",
        "MetaData": {
            "MMSI": mmsi,
            "time_utc": time_utc,
            "ShipName": ship_name,
            "latitude": lat,
            "longitude": lon,
        },
        "Message": {
            "PositionReport": {
                "Sog": sog,
                "Cog": cog,
                "TrueHeading": heading,
                "NavigationalStatus": nav_status,
            }
        },
    }


class TestNormalizeMessage:
    def test_basic_normalization(self):
        msg = normalize_message(_raw_position_report())
        assert msg is not None
        assert msg["mmsi"] == 123456789
        assert msg["lat"] == 40.0
        assert msg["lon"] == -74.0
        assert msg["source"] == "aisstream"

    def test_sog_scaled(self):
        """SOG is in 1/10 knot units — should be divided by 10."""
        msg = normalize_message(_raw_position_report(sog=105))
        assert msg["sog"] == 10.5

    def test_cog_scaled(self):
        """COG is in 1/10 degree units — should be divided by 10."""
        msg = normalize_message(_raw_position_report(cog=1800))
        assert msg["cog"] == 180.0

    def test_heading_preserved(self):
        msg = normalize_message(_raw_position_report(heading=178))
        assert msg["heading"] == 178.0

    def test_heading_511_is_null(self):
        """Heading=511 means unavailable → should be None."""
        msg = normalize_message(_raw_position_report(heading=511))
        assert msg["heading"] is None

    def test_nav_status_decoded(self):
        msg = normalize_message(_raw_position_report(nav_status=0))
        assert msg["nav_status"] == "Under way using engine"

    def test_nav_status_at_anchor(self):
        msg = normalize_message(_raw_position_report(nav_status=1))
        assert msg["nav_status"] == "At anchor"

    def test_vessel_name_stripped(self):
        msg = normalize_message(_raw_position_report(ship_name="  TEST  "))
        assert msg["vessel_name"] == "TEST"

    def test_empty_vessel_name_is_none(self):
        msg = normalize_message(_raw_position_report(ship_name=""))
        assert msg["vessel_name"] is None

    def test_timestamp_parsed(self):
        msg = normalize_message(_raw_position_report(
            time_utc="2024-06-15 12:30:00.000000+00:00"
        ))
        assert "2024-06-15" in msg["timestamp"]
        assert "12:30" in msg["timestamp"]

    def test_non_position_report_returns_none(self):
        raw = {"MessageType": "StaticDataReport", "MetaData": {}, "Message": {}}
        assert normalize_message(raw) is None

    def test_missing_mmsi_returns_none(self):
        raw = _raw_position_report()
        del raw["MetaData"]["MMSI"]
        assert normalize_message(raw) is None

    def test_missing_lat_returns_none(self):
        raw = _raw_position_report()
        del raw["MetaData"]["latitude"]
        assert normalize_message(raw) is None


# ---------------------------------------------------------------------------
# Capabilities registration
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_source_id(self):
        assert CAPABILITIES.source_id == "aisstream"

    def test_streaming_supported(self):
        assert CAPABILITIES.supports_streaming is True

    def test_backfill_not_supported(self):
        assert CAPABILITIES.supports_backfill is False

    def test_registered_in_catalog(self):
        from neptune_ais.adapters import registry
        sources = registry.registered_sources()
        assert "aisstream" in sources

    def test_info_returns_capabilities(self):
        from neptune_ais.adapters import registry
        caps = registry.info("aisstream")
        assert caps.source_id == "aisstream"
        assert caps.supports_streaming is True


# ---------------------------------------------------------------------------
# NeptuneStream integration
# ---------------------------------------------------------------------------


class TestStreamIntegration:
    """Verify normalized messages feed into NeptuneStream correctly."""

    def test_ingest_normalized_message(self):
        from neptune_ais.stream import NeptuneStream

        async def _test():
            async with NeptuneStream(source="aisstream") as stream:
                raw = _raw_position_report(mmsi=111)
                normalized = normalize_message(raw)
                assert normalized is not None

                accepted = await stream.ingest(normalized)
                assert accepted is True
                assert stream.stats.messages_delivered == 1

        asyncio.run(_test())

    def test_duplicate_raw_messages_deduped(self):
        from neptune_ais.stream import NeptuneStream

        async def _test():
            async with NeptuneStream(source="aisstream") as stream:
                raw = _raw_position_report(mmsi=111)
                msg = normalize_message(raw)

                await stream.ingest(msg)
                accepted = await stream.ingest(msg)  # same message
                assert accepted is False
                assert stream.stats.messages_deduplicated == 1

        asyncio.run(_test())
