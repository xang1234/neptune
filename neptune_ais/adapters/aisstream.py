"""AISStream — aisstream.io WebSocket streaming adapter.

Real-time global AIS via WebSocket JSON. Streaming-only, no backfill.

AISStream.io delivers AIS messages as JSON over a WebSocket connection.
Each message contains vessel position data that this adapter normalizes
into Neptune's canonical position schema.

Message format (simplified)::

    {
        "MessageType": "PositionReport",
        "MetaData": {
            "MMSI": 123456789,
            "time_utc": "2024-06-15 00:00:00.000000+00:00",
            "ShipName": "VESSEL NAME",
            "latitude": 40.0,
            "longitude": -74.0
        },
        "Message": {
            "PositionReport": {
                "Sog": 10.5,
                "Cog": 180.0,
                "TrueHeading": 178,
                "NavigationalStatus": 0
            }
        }
    }

Requires: ``pip install neptune-ais[stream]`` (websockets).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from neptune_ais.adapters.base import SourceCapabilities
from neptune_ais.adapters.registry import register_streaming

logger = logging.getLogger(__name__)

SOURCE_ID = "aisstream"
ADAPTER_VERSION = "aisstream/0.1.0"

WEBSOCKET_URL = "wss://stream.aisstream.io/v0/stream"

# AIS navigational status codes → human-readable strings.
_NAV_STATUS = {
    0: "Under way using engine",
    1: "At anchor",
    2: "Not under command",
    3: "Restricted maneuverability",
    4: "Constrained by draught",
    5: "Moored",
    6: "Aground",
    7: "Engaged in fishing",
    8: "Under way sailing",
    9: "Reserved for HSC",
    10: "Reserved for WIG",
    14: "AIS-SART",
    15: "Not defined",
}

# Heading sentinel value (511 = not available).
_HEADING_UNAVAILABLE = 511

CAPABILITIES = SourceCapabilities(
    source_id=SOURCE_ID,
    provider="aisstream.io",
    description="Real-time global AIS data via WebSocket",
    supports_backfill=False,
    supports_streaming=True,
    supports_server_side_bbox=True,
    auth_scheme="api_key",
    expected_latency="real-time",
    coverage="Global (terrestrial AIS receivers)",
    delivery_format="WebSocket JSON",
    typical_daily_rows="10M+",
    datasets_provided=["positions"],
    known_quirks=[
        "Heading=511 means unavailable",
        "SOG/COG delivered as pre-scaled floats (not raw NMEA 1/10 units)",
        "time_utc may include trailing ' UTC' suffix",
    ],
)

# Register as a streaming-only source (no SourceAdapter protocol).
register_streaming(CAPABILITIES)


def build_subscription(
    api_key: str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
) -> str:
    """Build the JSON subscription message for aisstream.io.

    Args:
        api_key: AISStream API key.
        bbox: Optional bounding box filter ``(west, south, east, north)``.
            If provided, only positions within this bbox are delivered.

    Returns:
        A JSON string to send after WebSocket connection.
    """
    sub: dict[str, Any] = {
        "APIKey": api_key,
        "BoundingBoxes": [
            [[-90, -180], [90, 180]]  # full globe default
        ],
        "FilterMessageTypes": ["PositionReport"],
    }

    if bbox is not None:
        west, south, east, north = bbox
        sub["BoundingBoxes"] = [
            [[south, west], [north, east]]
        ]

    return json.dumps(sub)


def normalize_message(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a raw AISStream JSON message to a position dict.

    Returns None if the message is not a position report or cannot
    be parsed.

    The returned dict has the same field names as the canonical
    positions schema, ready for ``NeptuneStream.ingest()``.

    Args:
        raw: A parsed JSON dict from the AISStream WebSocket.

    Returns:
        A normalized position dict, or None if not a position report.
    """
    msg_type = raw.get("MessageType")
    if msg_type != "PositionReport":
        return None

    meta = raw.get("MetaData", {})
    message = raw.get("Message", {})
    pos_report = message.get("PositionReport", {})

    mmsi = meta.get("MMSI")
    if mmsi is None:
        return None

    # Parse timestamp. AISStream uses "2022-12-29 18:22:32.318353 +0000 UTC".
    time_str = meta.get("time_utc", "")
    try:
        cleaned = time_str.strip()
        if cleaned.endswith(" UTC"):
            cleaned = cleaned[:-4].strip()
        # Normalize bare offset "+0000" → "+00:00" for fromisoformat.
        cleaned = re.sub(r"([+-])(\d{2})(\d{2})$", r"\1\2:\3", cleaned)
        cleaned = cleaned.replace(" ", "T", 1)  # date-time boundary only
        ts = datetime.fromisoformat(cleaned)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        ts = datetime.now(timezone.utc)

    lat = meta.get("latitude")
    lon = meta.get("longitude")
    if lat is None or lon is None:
        return None

    # Normalize heading sentinel.
    heading = pos_report.get("TrueHeading")
    if heading == _HEADING_UNAVAILABLE:
        heading = None

    # Navigational status.
    nav_code = pos_report.get("NavigationalStatus")
    nav_status = _NAV_STATUS.get(nav_code) if nav_code is not None else None

    return {
        "mmsi": int(mmsi),
        "timestamp": ts.isoformat(),
        "lat": float(lat),
        "lon": float(lon),
        # AISStream delivers pre-scaled floats (not raw NMEA 1/10 units).
        "sog": float(pos_report.get("Sog")) if pos_report.get("Sog") is not None else None,
        "cog": float(pos_report.get("Cog")) if pos_report.get("Cog") is not None else None,
        "heading": float(heading) if heading is not None else None,
        "nav_status": nav_status,
        "vessel_name": (meta.get("ShipName") or "").strip() or None,
        "source": SOURCE_ID,
    }


async def connect_and_stream(
    stream,  # NeptuneStream instance
    *,
    api_key: str,
    bbox: tuple[float, float, float, float] | None = None,
) -> None:
    """Connect to AISStream and feed messages into a NeptuneStream.

    This is the adapter's main entry point for live streaming. It
    connects to the WebSocket, sends the subscription, and feeds
    normalized messages into ``stream.ingest()`` until the stream
    is closed or the connection drops.

    Args:
        stream: A running NeptuneStream instance.
        api_key: AISStream API key.
        bbox: Optional server-side bbox filter.

    Raises:
        ImportError: If websockets is not installed.
    """
    try:
        import websockets
    except ImportError:
        raise ImportError(
            "websockets is required for AISStream streaming. "
            "Install with: pip install neptune-ais[stream]"
        ) from None

    sub_msg = build_subscription(api_key, bbox=bbox)

    async with websockets.connect(WEBSOCKET_URL) as ws:
        await ws.send(sub_msg)
        logger.info("Connected to AISStream, subscription sent")

        async for raw_text in ws:
            if not stream.is_running:
                break

            try:
                raw = json.loads(raw_text)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON from AISStream, skipping")
                continue

            normalized = normalize_message(raw)
            if normalized is not None:
                await stream.ingest(normalized)
