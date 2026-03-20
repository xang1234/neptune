"""Fishing effort — aggregated spatial fishing activity grids.

Represents gridded fishing effort data from the GFW 4Wings API, where
each row is a spatial-temporal cell with aggregated vessel activity
hours, not an individual vessel position.

Schema version: ``fishing_effort/v1``
"""

from __future__ import annotations

from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Schema identity
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "fishing_effort/v1"
"""Catalog-level identifier for compatibility checks."""

DATASET_NAME = "fishing_effort"

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------


class Col:
    """Canonical column names for the fishing effort dataset."""

    DATE = "date"
    LAT = "lat"
    LON = "lon"
    FLAG = "flag"
    GEARTYPE = "geartype"
    VESSEL_HOURS = "vessel_hours"
    SOURCE = "source"
    RECORD_PROVENANCE = "record_provenance"


# ---------------------------------------------------------------------------
# Polars schema
# ---------------------------------------------------------------------------

SCHEMA: dict[str, pl.DataType] = {
    Col.DATE: pl.Date,
    Col.LAT: pl.Float64,
    Col.LON: pl.Float64,
    Col.FLAG: pl.String,
    Col.GEARTYPE: pl.String,
    Col.VESSEL_HOURS: pl.Float64,
    Col.SOURCE: pl.String,
    Col.RECORD_PROVENANCE: pl.String,
}
"""Full Polars schema for the fishing effort dataset."""

# ---------------------------------------------------------------------------
# Required / optional column sets
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        Col.DATE,
        Col.VESSEL_HOURS,
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

SORT_ORDER: list[str] = [Col.DATE, Col.FLAG, Col.GEARTYPE]
"""Default sort order within Parquet partitions."""

# ---------------------------------------------------------------------------
# Valid value ranges
# ---------------------------------------------------------------------------

VALID_RANGES: dict[str, tuple[Any, Any]] = {
    Col.LAT: (-90.0, 90.0),
    Col.LON: (-180.0, 180.0),
    Col.VESSEL_HOURS: (0.0, 1e6),
}
"""Inclusive min/max ranges for numeric fields."""

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_schema(df: pl.DataFrame | pl.LazyFrame) -> list[str]:
    """Check that *df* conforms to the fishing effort schema contract.

    Returns a list of human-readable error strings. An empty list means
    the DataFrame is schema-conformant.
    """
    from neptune_ais.datasets import validate_schema as _validate

    return _validate(df, SCHEMA, REQUIRED_COLUMNS)
