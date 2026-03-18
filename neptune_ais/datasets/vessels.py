"""Vessels — vessel identity and slowly changing reference attributes.

Stores MMSI-keyed identity records with standardized names, types,
dimensions, flag state, and temporal first/last-seen bounds.

This is a *reference* dataset, not a point-observation dataset. Each row
represents a vessel's resolved identity over a time window, not a single
AIS message. Identity fields that also appear in ``positions`` (imo,
callsign, ship_type, etc.) use identical column names so that joins
between the two datasets are natural.

Schema version: ``vessels/v1``
"""

from __future__ import annotations

from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Schema identity
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "vessels/v1"
"""Catalog-level identifier for compatibility checks."""

DATASET_NAME = "vessels"

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------


class Col:
    """Canonical column names for the vessels dataset.

    Where column names overlap with ``positions.Col`` (mmsi, imo, callsign,
    ship_type, vessel_name, length, beam, flag, source, record_provenance),
    the string values are intentionally identical to enable seamless joins.
    """

    # --- required: identity key ---
    MMSI = "mmsi"

    # --- optional: identity attributes ---
    IMO = "imo"
    CALLSIGN = "callsign"
    VESSEL_NAME = "vessel_name"
    SHIP_TYPE = "ship_type"
    LENGTH = "length"
    BEAM = "beam"
    FLAG = "flag"

    # --- required: temporal bounds ---
    FIRST_SEEN = "first_seen"
    LAST_SEEN = "last_seen"

    # --- required: provenance ---
    SOURCE = "source"
    RECORD_PROVENANCE = "record_provenance"


# ---------------------------------------------------------------------------
# Polars schema
# ---------------------------------------------------------------------------

SCHEMA: dict[str, pl.DataType] = {
    # identity key (required)
    Col.MMSI: pl.Int64,
    # identity attributes (optional — not every source provides all)
    Col.IMO: pl.String,
    Col.CALLSIGN: pl.String,
    Col.VESSEL_NAME: pl.String,
    Col.SHIP_TYPE: pl.String,
    Col.LENGTH: pl.Float64,
    Col.BEAM: pl.Float64,
    Col.FLAG: pl.String,
    # temporal bounds (required)
    Col.FIRST_SEEN: pl.Datetime("us", "UTC"),
    Col.LAST_SEEN: pl.Datetime("us", "UTC"),
    # provenance (required)
    Col.SOURCE: pl.String,
    Col.RECORD_PROVENANCE: pl.String,
}
"""Full Polars schema for the vessels dataset."""

# ---------------------------------------------------------------------------
# Required / optional column sets
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        Col.MMSI,
        Col.FIRST_SEEN,
        Col.LAST_SEEN,
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

SORT_ORDER: list[str] = [Col.MMSI, Col.LAST_SEEN]
"""Default sort order within Parquet partitions.

Sorted by MMSI then last_seen so that the most recent identity record
for a given vessel is at the end of its group."""

# ---------------------------------------------------------------------------
# Partition keys
# ---------------------------------------------------------------------------

PARTITION_KEYS: list[str] = [Col.SOURCE]
"""Hive-style partition columns."""

# ---------------------------------------------------------------------------
# Dedup key
# ---------------------------------------------------------------------------

DEDUP_KEY: list[str] = [Col.MMSI, Col.SOURCE]
"""Columns that together identify a unique vessel record per source.

Within a single source, there should be at most one active identity
record per MMSI. Cross-source fusion may produce merged records with
combined provenance."""

# ---------------------------------------------------------------------------
# Valid value ranges
# ---------------------------------------------------------------------------

VALID_RANGES: dict[str, tuple[Any, Any]] = {
    Col.LENGTH: (0.0, 500.0),  # largest ships ~400m
    Col.BEAM: (0.0, 80.0),  # largest beams ~60m
}
"""Inclusive min/max ranges for numeric fields."""

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_schema(df: pl.DataFrame | pl.LazyFrame) -> list[str]:
    """Check that *df* conforms to the vessels schema contract.

    Returns a list of human-readable error strings. An empty list means
    the DataFrame is schema-conformant.
    """
    from neptune_ais.datasets import validate_schema as _validate

    return _validate(df, SCHEMA, REQUIRED_COLUMNS)
