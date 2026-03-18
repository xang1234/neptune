"""Positions — timestamped AIS point observations.

The most frequently queried canonical dataset. Each row is a single
vessel observation with location, kinematics, identity, QC flags,
and provenance.

Schema version: ``positions/v1``
"""

from __future__ import annotations

from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Schema identity
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "positions/v1"
"""Catalog-level identifier for compatibility checks."""

DATASET_NAME = "positions"

# ---------------------------------------------------------------------------
# Column names — single source of truth for string literals.
# Adapters, QC, fusion, and storage should reference these rather than
# hardcoding column name strings.
# ---------------------------------------------------------------------------

class Col:
    """Canonical column names for the positions dataset."""

    # --- required: observation core ---
    MMSI = "mmsi"
    TIMESTAMP = "timestamp"
    LAT = "lat"
    LON = "lon"

    # --- optional: kinematics ---
    SOG = "sog"
    COG = "cog"
    HEADING = "heading"
    NAV_STATUS = "nav_status"

    # --- optional: vessel identity (denormalized from message) ---
    IMO = "imo"
    CALLSIGN = "callsign"
    SHIP_TYPE = "ship_type"
    VESSEL_NAME = "vessel_name"
    LENGTH = "length"
    BEAM = "beam"
    DRAUGHT = "draught"
    DESTINATION = "destination"
    ETA = "eta"
    FLAG = "flag"

    # --- required: provenance ---
    SOURCE = "source"
    SOURCE_FILE = "source_file"
    INGEST_ID = "ingest_id"
    RECORD_PROVENANCE = "record_provenance"

    # --- optional: provenance ---
    SOURCE_RECORD_ID = "source_record_id"

    # --- optional: spatial ---
    SPATIAL_KEY = "spatial_key"

    # --- required: quality ---
    QC_SEVERITY = "qc_severity"

    # --- optional: quality ---
    QC_FLAGS = "qc_flags"
    CONFIDENCE_SCORE = "confidence_score"


# ---------------------------------------------------------------------------
# Polars schema — maps every column to its Polars dtype.
# ---------------------------------------------------------------------------

SCHEMA: dict[str, pl.DataType] = {
    # observation core (required)
    Col.MMSI: pl.Int64,
    Col.TIMESTAMP: pl.Datetime("us", "UTC"),
    Col.LAT: pl.Float64,
    Col.LON: pl.Float64,
    # kinematics (optional)
    Col.SOG: pl.Float64,
    Col.COG: pl.Float64,
    Col.HEADING: pl.Float64,
    Col.NAV_STATUS: pl.String,
    # vessel identity (optional, denormalized from message)
    Col.IMO: pl.String,
    Col.CALLSIGN: pl.String,
    Col.SHIP_TYPE: pl.String,
    Col.VESSEL_NAME: pl.String,
    Col.LENGTH: pl.Float64,
    Col.BEAM: pl.Float64,
    Col.DRAUGHT: pl.Float64,
    Col.DESTINATION: pl.String,
    Col.ETA: pl.Datetime("us", "UTC"),
    Col.FLAG: pl.String,
    # provenance (required except source_record_id)
    Col.SOURCE: pl.String,
    Col.SOURCE_RECORD_ID: pl.String,
    Col.SOURCE_FILE: pl.String,
    Col.INGEST_ID: pl.String,
    Col.RECORD_PROVENANCE: pl.String,
    # spatial (optional)
    Col.SPATIAL_KEY: pl.String,
    # quality
    Col.QC_FLAGS: pl.List(pl.String),
    Col.QC_SEVERITY: pl.String,
    Col.CONFIDENCE_SCORE: pl.Float64,
}
"""Full Polars schema for the positions dataset.

Every column that may appear in a canonical positions Parquet file is listed
here. Required columns must be non-null; optional columns may be entirely
absent or contain nulls.
"""

# ---------------------------------------------------------------------------
# Required / optional column sets
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        Col.MMSI,
        Col.TIMESTAMP,
        Col.LAT,
        Col.LON,
        Col.SOURCE,
        Col.SOURCE_FILE,
        Col.INGEST_ID,
        Col.QC_SEVERITY,
        Col.RECORD_PROVENANCE,
    }
)
"""Columns that must be present and non-null in every canonical row."""

OPTIONAL_COLUMNS: frozenset[str] = frozenset(SCHEMA.keys()) - REQUIRED_COLUMNS
"""Columns that may be absent or contain nulls."""

# ---------------------------------------------------------------------------
# Sort order — storage uses this to maintain Parquet row-group layout.
# ---------------------------------------------------------------------------

SORT_ORDER: list[str] = [Col.MMSI, Col.TIMESTAMP]
"""Default sort order within Parquet partitions."""

# ---------------------------------------------------------------------------
# Partition keys — storage uses these for directory layout.
# ---------------------------------------------------------------------------

PARTITION_KEYS: list[str] = [Col.SOURCE]
"""Hive-style partition columns (source and date are path-level; date is
derived from the timestamp range and managed by storage, not stored as a
column)."""

# ---------------------------------------------------------------------------
# Dedup key — fusion uses this for near-duplicate detection.
# ---------------------------------------------------------------------------

DEDUP_KEY: list[str] = [Col.MMSI, Col.TIMESTAMP, Col.SOURCE]
"""Columns that together identify a unique observation for dedup purposes.
Fusion may additionally apply timestamp tolerance and coordinate proximity."""

# ---------------------------------------------------------------------------
# Valid value ranges — QC uses these for built-in checks.
# ---------------------------------------------------------------------------

VALID_RANGES: dict[str, tuple[Any, Any]] = {
    Col.LAT: (-90.0, 90.0),
    Col.LON: (-180.0, 180.0),
    Col.SOG: (0.0, 102.3),  # AIS max SOG value
    Col.COG: (0.0, 360.0),
    Col.HEADING: (0.0, 360.0),
    Col.CONFIDENCE_SCORE: (0.0, 1.0),
}
"""Inclusive min/max ranges for numeric fields. Values outside these
ranges are hard-invalid (QC severity = error)."""

# ---------------------------------------------------------------------------
# QC severity vocabulary
# ---------------------------------------------------------------------------

QC_SEVERITY_OK = "ok"
QC_SEVERITY_WARNING = "warning"
QC_SEVERITY_ERROR = "error"

QC_SEVERITY_VALUES: frozenset[str] = frozenset(
    {QC_SEVERITY_OK, QC_SEVERITY_WARNING, QC_SEVERITY_ERROR}
)
"""Allowed values for the qc_severity column."""

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_schema(df: pl.DataFrame | pl.LazyFrame) -> list[str]:
    """Check that *df* conforms to the positions schema contract.

    Returns a list of human-readable error strings. An empty list means
    the DataFrame is schema-conformant.
    """
    from neptune_ais.datasets import validate_schema as _validate

    return _validate(df, SCHEMA, REQUIRED_COLUMNS)
