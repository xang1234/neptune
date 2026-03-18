"""Events — inferred or source-native maritime events.

Covers port calls, EEZ crossings, encounters, loitering, fishing effort
markers, and other typed event records with optional geometry.

Schema version: ``events/v1``
"""

from __future__ import annotations

import hashlib
from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Schema identity
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "events/v1"
"""Catalog-level identifier for compatibility checks."""

DATASET_NAME = "events"

# ---------------------------------------------------------------------------
# Event type vocabulary
# ---------------------------------------------------------------------------

EVENT_TYPE_PORT_CALL = "port_call"
EVENT_TYPE_EEZ_CROSSING = "eez_crossing"
EVENT_TYPE_ENCOUNTER = "encounter"
EVENT_TYPE_LOITERING = "loitering"

EVENT_TYPES: frozenset[str] = frozenset(
    {
        EVENT_TYPE_PORT_CALL,
        EVENT_TYPE_EEZ_CROSSING,
        EVENT_TYPE_ENCOUNTER,
        EVENT_TYPE_LOITERING,
    }
)
"""Known event type values. Detectors may add new types, but these are
the initial families supported by the derivation pipeline."""

# ---------------------------------------------------------------------------
# Column names — single source of truth for string literals.
# ---------------------------------------------------------------------------


class Col:
    """Canonical column names for the events dataset."""

    # --- required: identity ---
    EVENT_ID = "event_id"
    EVENT_TYPE = "event_type"
    MMSI = "mmsi"

    # --- optional: secondary vessel ---
    OTHER_MMSI = "other_mmsi"

    # --- required: temporal bounds ---
    START_TIME = "start_time"
    END_TIME = "end_time"

    # --- required: representative location ---
    LAT = "lat"
    LON = "lon"

    # --- optional: geometry ---
    GEOMETRY_WKB = "geometry_wkb"

    # --- required: confidence and provenance ---
    CONFIDENCE_SCORE = "confidence_score"
    SOURCE = "source"
    RECORD_PROVENANCE = "record_provenance"


# ---------------------------------------------------------------------------
# Polars schema
# ---------------------------------------------------------------------------

SCHEMA: dict[str, pl.DataType] = {
    # identity (required)
    Col.EVENT_ID: pl.String,
    Col.EVENT_TYPE: pl.String,
    Col.MMSI: pl.Int64,
    # secondary vessel (optional — only for encounters)
    Col.OTHER_MMSI: pl.Int64,
    # temporal bounds (required)
    Col.START_TIME: pl.Datetime("us", "UTC"),
    Col.END_TIME: pl.Datetime("us", "UTC"),
    # representative location (required)
    Col.LAT: pl.Float64,
    Col.LON: pl.Float64,
    # geometry (optional)
    Col.GEOMETRY_WKB: pl.Binary,
    # confidence and provenance (required)
    Col.CONFIDENCE_SCORE: pl.Float64,
    Col.SOURCE: pl.String,
    Col.RECORD_PROVENANCE: pl.String,
}
"""Full Polars schema for the events dataset."""

# ---------------------------------------------------------------------------
# Required / optional column sets
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        Col.EVENT_ID,
        Col.EVENT_TYPE,
        Col.MMSI,
        Col.START_TIME,
        Col.END_TIME,
        Col.LAT,
        Col.LON,
        Col.CONFIDENCE_SCORE,
        Col.SOURCE,
        Col.RECORD_PROVENANCE,
    }
)
"""Columns that must be present and non-null in every canonical row."""

OPTIONAL_COLUMNS: frozenset[str] = frozenset(SCHEMA.keys()) - REQUIRED_COLUMNS
"""Columns that may be absent or contain nulls."""

# ---------------------------------------------------------------------------
# Sort order
# ---------------------------------------------------------------------------

SORT_ORDER: list[str] = [Col.EVENT_TYPE, Col.MMSI, Col.START_TIME]
"""Default sort order within Parquet partitions."""

# ---------------------------------------------------------------------------
# Partition keys
# ---------------------------------------------------------------------------

PARTITION_KEYS: list[str] = [Col.EVENT_TYPE]
"""Hive-style partition columns. Events are partitioned by event_type
(e.g. ``event_type=port_call/date=2024-06-15/``)."""

# ---------------------------------------------------------------------------
# event_id construction — deterministic and stable
# ---------------------------------------------------------------------------


def make_event_id(
    event_type: str,
    mmsi: int,
    start_time_us: int,
    source: str,
    config_hash: str = "",
) -> str:
    """Generate a deterministic event_id.

    The event_id is a SHA-1 hex prefix (16 chars) of the concatenated
    inputs. This ensures:
    - Same event type + vessel + start time + source + config → same ID.
    - Re-running detection with the same parameters reproduces IDs.
    - Different detector configs produce different IDs.

    Args:
        event_type: Event type (e.g. ``"port_call"``).
        mmsi: Vessel MMSI.
        start_time_us: Event start time as epoch microseconds.
        source: Source identifier.
        config_hash: Hash of the detector configuration. Empty string
            if not applicable.

    Returns:
        A 16-character hex string, e.g. ``"a1b2c3d4e5f6a7b8"``.
    """
    key = f"{event_type}:{mmsi}:{start_time_us}:{source}:{config_hash}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Confidence semantics — what confidence_score means for events
# ---------------------------------------------------------------------------

CONFIDENCE_HIGH = 0.7
"""Threshold above which an event is considered high-confidence.
High-confidence events have strong heuristic support: multiple
confirming signals, long duration, clear spatial pattern."""

CONFIDENCE_LOW = 0.3
"""Threshold below which an event is considered low-confidence.
Low-confidence events are speculative: sparse data, ambiguous
pattern, or single weak signal."""

CONFIDENCE_BANDS: dict[str, tuple[float, float]] = {
    "high": (CONFIDENCE_HIGH, 1.0),
    "medium": (CONFIDENCE_LOW, CONFIDENCE_HIGH),
    "low": (0.0, CONFIDENCE_LOW),
}
"""Named confidence bands for filtering and reporting.

Each band is a ``[lower, upper)`` half-open interval — the lower bound
is inclusive and the upper bound is exclusive (except ``"high"`` which
includes 1.0). This matches ``classify_confidence()`` semantics.

Usage::

    # Filter to high-confidence events only
    events.filter(pl.col("confidence_score") >= CONFIDENCE_HIGH)

    # Or use the band thresholds (half-open: [lo, hi))
    lo, hi = CONFIDENCE_BANDS["medium"]
    events.filter(
        (pl.col("confidence_score") >= lo)
        & (pl.col("confidence_score") < hi)
    )
"""


def classify_confidence(score: float) -> str:
    """Classify a confidence score into a named band.

    Args:
        score: Confidence value in [0.0, 1.0].

    Returns:
        ``"high"``, ``"medium"``, or ``"low"``.
    """
    if score >= CONFIDENCE_HIGH:
        return "high"
    if score >= CONFIDENCE_LOW:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Valid value ranges
# ---------------------------------------------------------------------------

VALID_RANGES: dict[str, tuple[Any, Any]] = {
    Col.LAT: (-90.0, 90.0),
    Col.LON: (-180.0, 180.0),
    Col.CONFIDENCE_SCORE: (0.0, 1.0),
}
"""Inclusive min/max ranges for numeric fields."""

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_schema(df: pl.DataFrame | pl.LazyFrame) -> list[str]:
    """Check that *df* conforms to the events schema contract.

    Returns a list of human-readable error strings. An empty list means
    the DataFrame is schema-conformant.
    """
    from neptune_ais.datasets import validate_schema as _validate

    return _validate(df, SCHEMA, REQUIRED_COLUMNS)
