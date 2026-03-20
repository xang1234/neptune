"""Finland — Fintraffic Digitraffic marine AIS streaming adapter.

Real-time vessel positions and metadata via MQTT over WebSocket from the
Digitraffic marine API. Streaming-only, no backfill.

Digitraffic provides:
- MQTT: ``wss://meri.digitraffic.fi:443/mqtt`` — real-time updates
  - ``vessels-v2/{mmsi}/location`` — position reports
  - ``vessels-v2/{mmsi}/metadata`` — vessel identity/static data

Coverage: Finnish waters and Baltic Sea.
Delivery: MQTT over WSS (open data, no authentication).
Latency: Real-time (< 1 second).

MQTT location message wire format (captured 2026-03-20)::

    topic: vessels-v2/{mmsi}/location
    {
        "time": 1773972038,        # Unix epoch seconds
        "sog": 12.3,               # pre-scaled knots (NOT ×10)
        "cog": 72.2,               # pre-scaled degrees (NOT ×10)
        "navStat": 0,              # AIS navigation status code
        "rot": 0,                  # rate of turn
        "posAcc": false,
        "raim": false,
        "heading": 66,             # 511 = unavailable
        "lon": 22.96629,
        "lat": 59.47799
    }

MQTT metadata message wire format::

    topic: vessels-v2/{mmsi}/metadata
    {
        "timestamp": 1773972046685,  # epoch milliseconds
        "destination": "TURKU",
        "name": "SATURN",
        "draught": 40,               # meters × 10 (divide by 10)
        "eta": 100737,
        "posType": 1,
        "refA": 8, "refB": 11, "refC": 5, "refD": 5,
        "callSign": "OJUV",
        "imo": 9315410,
        "type": 52                   # numeric ship type
    }

Requires: ``pip install neptune-ais[stream]`` (aiomqtt).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from neptune_ais.adapters.base import AIS_NAV_STATUS, SourceCapabilities
from neptune_ais.adapters.registry import register_streaming

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_ID = "finland"

MQTT_BROKER = "meri.digitraffic.fi"
MQTT_PORT = 443
MQTT_WEBSOCKET_PATH = "/mqtt"

LOCATION_TOPIC = "vessels-v2/+/location"
METADATA_TOPIC = "vessels-v2/+/metadata"

# Sentinel values.
HEADING_UNAVAILABLE = 511
IMO_UNAVAILABLE = 0

# Draught is still ×10 in metadata messages.
DRAUGHT_SCALE_FACTOR = 10.0

# Cached TLS context for MQTT connections (avoids re-loading the system
# trust store on every reconnection).
_tls_context = None


# ---------------------------------------------------------------------------
# Capabilities & registration
# ---------------------------------------------------------------------------


CAPABILITIES = SourceCapabilities(
    source_id=SOURCE_ID,
    provider="Fintraffic Digitraffic",
    description=(
        "Finnish waters and Baltic Sea AIS from Digitraffic. "
        "Open data, real-time MQTT streaming."
    ),
    supports_backfill=False,
    supports_streaming=True,
    supports_server_side_bbox=False,
    supports_incremental=False,
    auth_scheme=None,
    rate_limit=None,
    expected_latency="real-time",
    license_requirements=(
        "CC BY 4.0 — Fintraffic / Digitraffic. "
        "See https://www.digitraffic.fi/en/terms/"
    ),
    coverage="Finnish waters and Baltic Sea",
    delivery_format="MQTT over WSS",
    datasets_provided=["positions", "vessels"],
    known_quirks=[
        "MMSI is extracted from the MQTT topic, not the payload",
        "Location timestamps are Unix epoch seconds (not milliseconds)",
        "Metadata timestamps are Unix epoch milliseconds",
        "SOG/COG are pre-scaled floats (NOT raw NMEA ×10 units)",
        "Draught in metadata IS ×10 (divide by 10)",
        "Heading=511 means unavailable (normalized to null)",
        "IMO=0 means unavailable (normalized to null)",
        "Ship type is numeric (cast to string)",
    ],
)

register_streaming(CAPABILITIES)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_message(raw: dict[str, Any], topic: str) -> dict[str, Any] | None:
    """Normalize a raw Digitraffic MQTT message to a canonical dict.

    Parses the topic once to extract both MMSI and message type, then
    dispatches: ``location`` → position dict, ``metadata`` → vessel dict.

    Returns None if the message is invalid (missing MMSI, missing
    coordinates for location messages, unknown topic structure).

    Args:
        raw: Parsed JSON payload from the MQTT message.
        topic: The full MQTT topic string (e.g. ``vessels-v2/230000001/location``).

    Returns:
        A normalized dict ready for ``NeptuneStream.ingest()``, or None.
    """
    parts = topic.split("/")
    if len(parts) < 3:
        return None
    try:
        mmsi = int(parts[1])
    except ValueError:
        return None

    kind = parts[2]
    if kind == "location":
        return _normalize_location(raw, mmsi)
    elif kind == "metadata":
        return _normalize_metadata(raw, mmsi)
    return None


def _normalize_location(raw: dict[str, Any], mmsi: int) -> dict[str, Any] | None:
    """Normalize a location message."""
    lat = raw.get("lat")
    lon = raw.get("lon")
    if lat is None or lon is None:
        return None

    # Timestamp: epoch seconds → ISO-8601.
    time_val = raw.get("time")
    if time_val is not None:
        ts = datetime.fromtimestamp(int(time_val), tz=timezone.utc).isoformat()
    else:
        ts = datetime.now(timezone.utc).isoformat()

    # Heading sentinel.
    heading = raw.get("heading")
    if heading == HEADING_UNAVAILABLE:
        heading = None
    elif heading is not None:
        heading = float(heading)

    # Nav status.
    nav_code = raw.get("navStat")
    nav_status = AIS_NAV_STATUS.get(nav_code) if nav_code is not None else None

    return {
        "mmsi": mmsi,
        "timestamp": ts,
        "lat": float(lat),
        "lon": float(lon),
        "sog": float(raw["sog"]) if raw.get("sog") is not None else None,
        "cog": float(raw["cog"]) if raw.get("cog") is not None else None,
        "heading": heading,
        "nav_status": nav_status,
        "source": SOURCE_ID,
    }


def _normalize_metadata(raw: dict[str, Any], mmsi: int) -> dict[str, Any] | None:
    """Normalize a metadata message."""
    # Timestamp: epoch milliseconds → ISO-8601.
    ts_ms = raw.get("timestamp")
    if ts_ms is not None:
        ts = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).isoformat()
    else:
        ts = datetime.now(timezone.utc).isoformat()

    # IMO sentinel.
    imo = raw.get("imo")
    if imo == IMO_UNAVAILABLE or imo is None:
        imo = None
    else:
        imo = str(imo)

    # Draught: ×10 in metadata messages.
    draught = raw.get("draught")
    if draught is not None:
        draught = float(draught) / DRAUGHT_SCALE_FACTOR

    # Ship type: numeric → string.
    ship_type = raw.get("type")
    if ship_type is not None:
        ship_type = str(ship_type)

    return {
        "mmsi": mmsi,
        "timestamp": ts,
        "vessel_name": (raw.get("name") or "").strip() or None,
        "imo": imo,
        "callsign": (raw.get("callSign") or "").strip() or None,
        "destination": (raw.get("destination") or "").strip() or None,
        "ship_type": ship_type,
        "draught": draught,
        "source": SOURCE_ID,
    }


# ---------------------------------------------------------------------------
# Streaming connector
# ---------------------------------------------------------------------------


async def connect_and_stream(
    stream,  # NeptuneStream instance
    *,
    bbox: tuple[float, float, float, float] | None = None,
) -> None:
    """Connect to Digitraffic MQTT and feed messages into a NeptuneStream.

    Subscribes to both location and metadata topics. Messages are
    normalized and ingested until the stream is closed or the connection
    drops.

    Args:
        stream: A running NeptuneStream instance.
        bbox: Optional client-side bbox filter ``(west, south, east, north)``.
            Digitraffic does not support server-side filtering; positions
            outside the bbox are silently discarded.

    Raises:
        ImportError: If aiomqtt is not installed.
    """
    try:
        import aiomqtt
    except ImportError:
        raise ImportError(
            "aiomqtt is required for Digitraffic Finland streaming. "
            "Install with: pip install neptune-ais[stream]"
        ) from None

    global _tls_context
    if _tls_context is None:
        import ssl
        _tls_context = ssl.create_default_context()

    async with aiomqtt.Client(
        hostname=MQTT_BROKER,
        port=MQTT_PORT,
        transport="websockets",
        tls_context=_tls_context,
        websocket_path=MQTT_WEBSOCKET_PATH,
    ) as client:
        await client.subscribe(LOCATION_TOPIC)
        await client.subscribe(METADATA_TOPIC)
        logger.info("Connected to Digitraffic MQTT, subscribed to location + metadata")

        async for msg in client.messages:
            if not stream.is_running:
                break

            try:
                raw = json.loads(msg.payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("Invalid payload from Digitraffic, skipping")
                continue

            topic = str(msg.topic)
            normalized = normalize_message(raw, topic)
            if normalized is None:
                continue

            # Client-side bbox filtering for location messages.
            if bbox is not None and "lat" in normalized and "lon" in normalized:
                west, south, east, north = bbox
                lat, lon = normalized["lat"], normalized["lon"]
                if not (south <= lat <= north and west <= lon <= east):
                    continue

            await stream.ingest(normalized)
