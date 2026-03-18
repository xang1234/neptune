"""Tracks — derived trip/trajectory segments.

Trip segments grouped by vessel and split by time gaps. Includes
per-track statistics, optional WKB geometry, and per-vertex timestamp
offsets for animated visualization.

Schema version: ``tracks/v1``
"""

from __future__ import annotations

import hashlib
from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Schema identity
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "tracks/v1"
"""Catalog-level identifier for compatibility checks."""

DATASET_NAME = "tracks"

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------


class Col:
    """Canonical column names for the tracks dataset."""

    # --- required: identity ---
    TRACK_ID = "track_id"
    MMSI = "mmsi"

    # --- required: temporal bounds ---
    START_TIME = "start_time"
    END_TIME = "end_time"

    # --- required: statistics ---
    POINT_COUNT = "point_count"
    DISTANCE_M = "distance_m"
    DURATION_S = "duration_s"
    MEAN_SPEED = "mean_speed"
    MAX_SPEED = "max_speed"

    # --- required: spatial bounds ---
    BBOX_WEST = "bbox_west"
    BBOX_SOUTH = "bbox_south"
    BBOX_EAST = "bbox_east"
    BBOX_NORTH = "bbox_north"

    # --- optional: geometry ---
    GEOMETRY_WKB = "geometry_wkb"
    TIMESTAMP_OFFSETS_MS = "timestamp_offsets_ms"

    # --- required: provenance ---
    SOURCE = "source"
    RECORD_PROVENANCE = "record_provenance"


# ---------------------------------------------------------------------------
# Polars schema
# ---------------------------------------------------------------------------

SCHEMA: dict[str, pl.DataType] = {
    # identity (required)
    Col.TRACK_ID: pl.String,
    Col.MMSI: pl.Int64,
    # temporal bounds (required)
    Col.START_TIME: pl.Datetime("us", "UTC"),
    Col.END_TIME: pl.Datetime("us", "UTC"),
    # statistics (required)
    Col.POINT_COUNT: pl.Int64,
    Col.DISTANCE_M: pl.Float64,
    Col.DURATION_S: pl.Float64,
    Col.MEAN_SPEED: pl.Float64,
    Col.MAX_SPEED: pl.Float64,
    # spatial bounds (required)
    Col.BBOX_WEST: pl.Float64,
    Col.BBOX_SOUTH: pl.Float64,
    Col.BBOX_EAST: pl.Float64,
    Col.BBOX_NORTH: pl.Float64,
    # geometry (optional)
    Col.GEOMETRY_WKB: pl.Binary,
    Col.TIMESTAMP_OFFSETS_MS: pl.List(pl.Int64),
    # provenance (required)
    Col.SOURCE: pl.String,
    Col.RECORD_PROVENANCE: pl.String,
}
"""Full Polars schema for the tracks dataset."""

# ---------------------------------------------------------------------------
# Required / optional column sets
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        Col.TRACK_ID,
        Col.MMSI,
        Col.START_TIME,
        Col.END_TIME,
        Col.POINT_COUNT,
        Col.DISTANCE_M,
        Col.DURATION_S,
        Col.MEAN_SPEED,
        Col.MAX_SPEED,
        Col.BBOX_WEST,
        Col.BBOX_SOUTH,
        Col.BBOX_EAST,
        Col.BBOX_NORTH,
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

SORT_ORDER: list[str] = [Col.MMSI, Col.START_TIME]
"""Default sort order within Parquet partitions."""

# ---------------------------------------------------------------------------
# Partition keys
# ---------------------------------------------------------------------------

PARTITION_KEYS: list[str] = [Col.SOURCE]
"""Hive-style partition columns."""

# ---------------------------------------------------------------------------
# track_id construction — deterministic and stable
# ---------------------------------------------------------------------------


def make_track_id(
    mmsi: int,
    start_time_us: int,
    source: str,
    config_hash: str = "",
) -> str:
    """Generate a deterministic track_id.

    The track_id is a SHA-1 hex prefix (16 chars) of the concatenated
    inputs. This ensures:
    - Same vessel + same start time + same source + same config → same ID.
    - Re-running segmentation with the same parameters reproduces IDs.
    - Different configs (gap threshold, etc.) produce different IDs.

    Args:
        mmsi: Vessel MMSI.
        start_time_us: Segment start time as epoch microseconds.
        source: Source identifier.
        config_hash: Hash of the segmentation configuration. Empty string
            if not applicable.

    Returns:
        A 16-character hex string, e.g. ``"a1b2c3d4e5f6a7b8"``.
    """
    key = f"{mmsi}:{start_time_us}:{source}:{config_hash}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Valid value ranges
# ---------------------------------------------------------------------------

VALID_RANGES: dict[str, tuple[Any, Any]] = {
    Col.POINT_COUNT: (1, 1_000_000),
    Col.DISTANCE_M: (0.0, 100_000_000.0),  # ~circumference of Earth
    Col.DURATION_S: (0.0, 86400.0 * 365),  # max 1 year
    Col.MEAN_SPEED: (0.0, 102.3),  # AIS max SOG
    Col.MAX_SPEED: (0.0, 102.3),
    Col.BBOX_WEST: (-180.0, 180.0),
    Col.BBOX_SOUTH: (-90.0, 90.0),
    Col.BBOX_EAST: (-180.0, 180.0),
    Col.BBOX_NORTH: (-90.0, 90.0),
}
"""Inclusive min/max ranges for numeric fields."""

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_schema(df: pl.DataFrame | pl.LazyFrame) -> list[str]:
    """Check that *df* conforms to the tracks schema contract.

    Returns a list of human-readable error strings. An empty list means
    the DataFrame is schema-conformant.
    """
    from neptune_ais.datasets import validate_schema as _validate

    return _validate(df, SCHEMA, REQUIRED_COLUMNS)
