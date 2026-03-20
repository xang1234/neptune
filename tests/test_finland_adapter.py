"""Tests for the Finland (Digitraffic) streaming adapter.

Covers: streaming registration, capabilities, normalization of location
and metadata MQTT messages, sentinel handling (heading=511, IMO=0),
timestamp parsing (epoch seconds for location, epoch ms for metadata),
MMSI extraction from MQTT topic, and connect_and_stream with mocked aiomqtt.

All test fixtures are derived from live MQTT capture (2026-03-20).
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neptune_ais.adapters.finland import (
    CAPABILITIES,
    DRAUGHT_SCALE_FACTOR,
    HEADING_UNAVAILABLE,
    IMO_UNAVAILABLE,
    LOCATION_TOPIC,
    METADATA_TOPIC,
    MQTT_BROKER,
    SOURCE_ID,
    connect_and_stream,
    normalize_message,
)


# ---------------------------------------------------------------------------
# Sample MQTT payloads (from live capture 2026-03-20)
# ---------------------------------------------------------------------------


def _sample_location() -> dict:
    """Return a sample location payload from the MQTT broker."""
    return {
        "time": 1773972038,
        "sog": 12.3,
        "cog": 72.2,
        "navStat": 0,
        "rot": 0,
        "posAcc": False,
        "raim": False,
        "heading": 66,
        "lon": 22.96629,
        "lat": 59.47799,
    }


def _sample_metadata() -> dict:
    """Return a sample metadata payload from the MQTT broker."""
    return {
        "timestamp": 1773972046685,
        "destination": "TURKU",
        "name": "SATURN",
        "draught": 40,
        "eta": 100737,
        "posType": 1,
        "refA": 8,
        "refB": 11,
        "refC": 5,
        "refD": 5,
        "callSign": "OJUV",
        "imo": 9315410,
        "type": 52,
    }


LOCATION_TOPIC_SAMPLE = "vessels-v2/249788000/location"
METADATA_TOPIC_SAMPLE = "vessels-v2/230710000/metadata"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestFinlandStreamingRegistration:
    def test_streaming_registered(self):
        from neptune_ais.adapters.registry import _STREAMING_CAPABILITIES
        assert "finland" in _STREAMING_CAPABILITIES

    def test_not_in_archival(self):
        from neptune_ais.adapters.registry import _ADAPTERS
        assert "finland" not in _ADAPTERS


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class TestFinlandCapabilities:
    def test_streaming_true(self):
        assert CAPABILITIES.supports_streaming is True

    def test_backfill_false(self):
        assert CAPABILITIES.supports_backfill is False

    def test_auth_none(self):
        assert CAPABILITIES.auth_scheme is None

    def test_datasets_include_vessels(self):
        assert "positions" in CAPABILITIES.datasets_provided
        assert "vessels" in CAPABILITIES.datasets_provided

    def test_source_id(self):
        assert CAPABILITIES.source_id == "finland"

    def test_provider(self):
        assert CAPABILITIES.provider == "Fintraffic Digitraffic"

    def test_delivery_format(self):
        assert "MQTT" in CAPABILITIES.delivery_format


# ---------------------------------------------------------------------------
# Normalization — location messages
# ---------------------------------------------------------------------------


class TestFinlandNormalizationLocation:
    def test_normalize_location_message(self):
        result = normalize_message(_sample_location(), LOCATION_TOPIC_SAMPLE)
        assert result is not None
        assert result["mmsi"] == 249788000
        assert result["lat"] == 59.47799
        assert result["lon"] == 22.96629
        assert result["sog"] == 12.3
        assert result["cog"] == 72.2
        assert result["heading"] == 66.0
        assert result["source"] == "finland"

    def test_sog_cog_are_prescaled(self):
        """SOG/COG from MQTT are pre-scaled floats, NOT ×10."""
        loc = _sample_location()
        loc["sog"] = 10.5
        loc["cog"] = 180.0
        result = normalize_message(loc, LOCATION_TOPIC_SAMPLE)
        # Values should be passed through as-is, not divided by 10.
        assert result["sog"] == 10.5
        assert result["cog"] == 180.0

    def test_timestamp_epoch_seconds(self):
        """Location timestamps are epoch seconds (not milliseconds)."""
        result = normalize_message(_sample_location(), LOCATION_TOPIC_SAMPLE)
        ts = datetime.fromisoformat(result["timestamp"])
        assert ts.tzinfo is not None
        # 1773972038 seconds → 2026-03-17T11:40:38+00:00
        expected = datetime.fromtimestamp(1773972038, tz=timezone.utc)
        assert ts == expected

    def test_missing_coords_returns_none(self):
        loc = _sample_location()
        del loc["lat"]
        assert normalize_message(loc, LOCATION_TOPIC_SAMPLE) is None

    def test_missing_lon_returns_none(self):
        loc = _sample_location()
        del loc["lon"]
        assert normalize_message(loc, LOCATION_TOPIC_SAMPLE) is None

    def test_heading_sentinel(self):
        """Heading=511 is normalized to None."""
        loc = _sample_location()
        loc["heading"] = 511
        result = normalize_message(loc, LOCATION_TOPIC_SAMPLE)
        assert result["heading"] is None

    def test_heading_valid(self):
        loc = _sample_location()
        loc["heading"] = 180
        result = normalize_message(loc, LOCATION_TOPIC_SAMPLE)
        assert result["heading"] == 180.0

    def test_nav_status_decoded(self):
        result = normalize_message(_sample_location(), LOCATION_TOPIC_SAMPLE)
        assert result["nav_status"] == "Under way using engine"

    def test_nav_status_moored(self):
        loc = _sample_location()
        loc["navStat"] = 5
        result = normalize_message(loc, LOCATION_TOPIC_SAMPLE)
        assert result["nav_status"] == "Moored"

    def test_source_field(self):
        result = normalize_message(_sample_location(), LOCATION_TOPIC_SAMPLE)
        assert result["source"] == "finland"

    def test_mmsi_from_topic(self):
        """MMSI is extracted from the topic, not the payload."""
        result = normalize_message(
            _sample_location(), "vessels-v2/123456789/location"
        )
        assert result["mmsi"] == 123456789


# ---------------------------------------------------------------------------
# Normalization — metadata messages
# ---------------------------------------------------------------------------


class TestFinlandNormalizationMetadata:
    def test_normalize_metadata_message(self):
        result = normalize_message(_sample_metadata(), METADATA_TOPIC_SAMPLE)
        assert result is not None
        assert result["mmsi"] == 230710000
        assert result["vessel_name"] == "SATURN"
        assert result["callsign"] == "OJUV"
        assert result["destination"] == "TURKU"
        assert result["source"] == "finland"

    def test_timestamp_epoch_ms(self):
        """Metadata timestamps are epoch milliseconds."""
        result = normalize_message(_sample_metadata(), METADATA_TOPIC_SAMPLE)
        ts = datetime.fromisoformat(result["timestamp"])
        expected = datetime.fromtimestamp(1773972046685 / 1000, tz=timezone.utc)
        assert ts == expected

    def test_imo_valid(self):
        result = normalize_message(_sample_metadata(), METADATA_TOPIC_SAMPLE)
        assert result["imo"] == "9315410"

    def test_imo_sentinel(self):
        """IMO=0 is normalized to None."""
        meta = _sample_metadata()
        meta["imo"] = 0
        result = normalize_message(meta, METADATA_TOPIC_SAMPLE)
        assert result["imo"] is None

    def test_imo_missing(self):
        meta = _sample_metadata()
        del meta["imo"]
        result = normalize_message(meta, METADATA_TOPIC_SAMPLE)
        assert result["imo"] is None

    def test_draught_scaled(self):
        """Draught in metadata is ×10 (40 → 4.0 meters)."""
        result = normalize_message(_sample_metadata(), METADATA_TOPIC_SAMPLE)
        assert result["draught"] == pytest.approx(4.0)

    def test_ship_type_as_string(self):
        result = normalize_message(_sample_metadata(), METADATA_TOPIC_SAMPLE)
        assert result["ship_type"] == "52"

    def test_destination_stripped(self):
        meta = _sample_metadata()
        meta["destination"] = "RUPRIMORSK      "
        result = normalize_message(meta, METADATA_TOPIC_SAMPLE)
        assert result["destination"] == "RUPRIMORSK"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestFinlandNormalizationEdgeCases:
    def test_missing_mmsi_returns_none(self):
        """Invalid topic without MMSI returns None."""
        assert normalize_message(_sample_location(), "vessels-v2") is None

    def test_invalid_mmsi_returns_none(self):
        assert normalize_message(_sample_location(), "vessels-v2/abc/location") is None

    def test_unknown_topic_suffix(self):
        assert normalize_message({}, "vessels-v2/123/unknown") is None

    def test_sog_none(self):
        loc = _sample_location()
        del loc["sog"]
        result = normalize_message(loc, LOCATION_TOPIC_SAMPLE)
        assert result["sog"] is None

    def test_empty_name_normalized(self):
        meta = _sample_metadata()
        meta["name"] = ""
        result = normalize_message(meta, METADATA_TOPIC_SAMPLE)
        assert result["vessel_name"] is None

    def test_missing_time_uses_now(self):
        loc = _sample_location()
        del loc["time"]
        result = normalize_message(loc, LOCATION_TOPIC_SAMPLE)
        # Should not crash; timestamp should be close to now.
        ts = datetime.fromisoformat(result["timestamp"])
        assert ts.tzinfo is not None


# ---------------------------------------------------------------------------
# connect_and_stream (mocked aiomqtt)
# ---------------------------------------------------------------------------


def _make_mock_client(messages_iter):
    """Build a mock aiomqtt.Client async context manager."""
    mock_client = AsyncMock()
    mock_client.subscribe = AsyncMock()
    mock_client.messages = messages_iter
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


@contextmanager
def _patch_aiomqtt(mock_client):
    """Temporarily inject a mock aiomqtt module into sys.modules."""
    import neptune_ais.adapters.finland as _finland_mod

    mock_mod = ModuleType("aiomqtt")
    mock_mod.Client = MagicMock(return_value=mock_client)
    saved_mod = sys.modules.get("aiomqtt")
    saved_tls = _finland_mod._tls_context
    sys.modules["aiomqtt"] = mock_mod
    _finland_mod._tls_context = None
    try:
        yield mock_mod
    finally:
        if saved_mod is not None:
            sys.modules["aiomqtt"] = saved_mod
        else:
            sys.modules.pop("aiomqtt", None)
        _finland_mod._tls_context = saved_tls


class TestFinlandConnectAndStream:
    @pytest.mark.asyncio
    async def test_requires_aiomqtt(self):
        """Gives a helpful ImportError if aiomqtt is not installed."""
        stream = MagicMock()
        # _patch_aiomqtt expects a mock_client; for ImportError test we
        # directly manipulate sys.modules with None.
        saved = sys.modules.get("aiomqtt")
        sys.modules["aiomqtt"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="aiomqtt"):
                await connect_and_stream(stream)
        finally:
            if saved is not None:
                sys.modules["aiomqtt"] = saved
            else:
                sys.modules.pop("aiomqtt", None)

    @pytest.mark.asyncio
    async def test_subscribe_and_ingest(self):
        """Mocks aiomqtt.Client to verify subscription and ingestion."""
        stream = MagicMock()
        stream.ingest = AsyncMock(return_value=True)

        # Build mock messages.
        loc_msg = MagicMock()
        loc_msg.topic = MagicMock()
        loc_msg.topic.__str__ = lambda self: "vessels-v2/249788000/location"
        loc_msg.payload = json.dumps(_sample_location()).encode()

        meta_msg = MagicMock()
        meta_msg.topic = MagicMock()
        meta_msg.topic.__str__ = lambda self: "vessels-v2/230710000/metadata"
        meta_msg.payload = json.dumps(_sample_metadata()).encode()

        # After two messages, stop the stream.
        call_count = 0
        is_running = True

        def side_effect(*args, **kwargs):
            nonlocal call_count, is_running
            call_count += 1
            if call_count >= 2:
                is_running = False
            return True

        stream.ingest.side_effect = side_effect
        type(stream).is_running = property(lambda self: is_running)

        async def mock_messages():
            yield loc_msg
            yield meta_msg

        mock_client = _make_mock_client(mock_messages())

        with _patch_aiomqtt(mock_client):
            await connect_and_stream(stream)

        # Verify subscriptions.
        subscribe_calls = mock_client.subscribe.call_args_list
        topics_subscribed = [call.args[0] for call in subscribe_calls]
        assert LOCATION_TOPIC in topics_subscribed
        assert METADATA_TOPIC in topics_subscribed

        # Verify ingest was called with normalized data.
        assert stream.ingest.call_count == 2
        loc_call = stream.ingest.call_args_list[0].args[0]
        assert loc_call["mmsi"] == 249788000
        assert loc_call["lat"] == 59.47799
        meta_call = stream.ingest.call_args_list[1].args[0]
        assert meta_call["mmsi"] == 230710000
        assert meta_call["vessel_name"] == "SATURN"

    @pytest.mark.asyncio
    async def test_invalid_json_skipped(self):
        """Invalid JSON payloads are skipped without crashing."""
        stream = MagicMock()
        stream.ingest = AsyncMock(return_value=True)

        bad_msg = MagicMock()
        bad_msg.topic = MagicMock()
        bad_msg.topic.__str__ = lambda self: "vessels-v2/123/location"
        bad_msg.payload = b"not json"

        good_msg = MagicMock()
        good_msg.topic = MagicMock()
        good_msg.topic.__str__ = lambda self: "vessels-v2/249788000/location"
        good_msg.payload = json.dumps(_sample_location()).encode()

        # Stop after the good message.
        def stop_stream(*args, **kwargs):
            type(stream).is_running = property(lambda self: False)
            return True
        stream.ingest.side_effect = stop_stream
        type(stream).is_running = property(lambda self: True)

        async def mock_messages():
            yield bad_msg
            yield good_msg

        mock_client = _make_mock_client(mock_messages())

        with _patch_aiomqtt(mock_client):
            await connect_and_stream(stream)

        # Only the valid message should be ingested.
        assert stream.ingest.call_count == 1

    @pytest.mark.asyncio
    async def test_bbox_filtering(self):
        """Messages outside the bbox are silently discarded."""
        stream = MagicMock()
        stream.ingest = AsyncMock(return_value=True)

        # Location at lat=59.47, lon=22.96 — outside bbox (0,0,1,1).
        msg = MagicMock()
        msg.topic = MagicMock()
        msg.topic.__str__ = lambda self: "vessels-v2/249788000/location"
        msg.payload = json.dumps(_sample_location()).encode()

        type(stream).is_running = property(lambda self: True)

        async def mock_messages():
            yield msg

        mock_client = _make_mock_client(mock_messages())

        with _patch_aiomqtt(mock_client):
            # bbox=(0,0,1,1) — tiny box near null island.
            await connect_and_stream(stream, bbox=(0.0, 0.0, 1.0, 1.0))

        # Message should be filtered out by bbox.
        assert stream.ingest.call_count == 0
