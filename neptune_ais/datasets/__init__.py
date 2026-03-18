"""Datasets — canonical dataset definitions and schema contracts.

Submodules define the schema, validation, and access logic for each canonical
dataset: positions, vessels, tracks, and events.

Subsystem boundary
------------------
**Owns:**
- Polars schema definitions (column names, types, required/optional) for each
  canonical dataset.
- Schema version identifiers and migration helpers.
- Row-level validation logic that enforces schema contracts.
- Dataset-specific constants (sentinel values, enum vocabularies).

**Delegates to:**
- ``neptune_ais.storage`` — partition layout, file I/O, atomic writes.
- ``neptune_ais.catalog`` — manifest creation, version tracking, staleness.
- ``neptune_ais.qc`` — quality checks and confidence scoring.
- ``neptune_ais.fusion`` — multi-source merge and dedup.

**Rule:** Dataset modules must never import from ``adapters``, ``derive``,
``geometry``, or ``cli``. They define *what* canonical data looks like, not
how it is produced or consumed.
"""

from __future__ import annotations

import polars as pl

__all__ = [
    "positions",
    "vessels",
    "tracks",
    "events",
    "validate_schema",
]


def validate_schema(
    df: pl.DataFrame | pl.LazyFrame,
    schema: dict[str, pl.DataType],
    required_columns: frozenset[str],
) -> list[str]:
    """Check that *df* conforms to a dataset schema contract.

    Returns a list of human-readable error strings. An empty list means
    the DataFrame is schema-conformant.

    This checks:
    - All required columns are present.
    - Column dtypes match the schema (where columns exist).
    - No unexpected columns are present.
    """
    if isinstance(df, pl.LazyFrame):
        df_schema = df.collect_schema()
    else:
        df_schema = df.schema

    errors: list[str] = []

    for col in required_columns:
        if col not in df_schema:
            errors.append(f"missing required column: {col!r}")

    for col_name, expected_dtype in schema.items():
        if col_name in df_schema:
            actual = df_schema[col_name]
            if actual != expected_dtype:
                errors.append(
                    f"column {col_name!r}: expected {expected_dtype}, "
                    f"got {actual}"
                )

    known = set(schema.keys())
    for col_name in df_schema:
        if col_name not in known:
            errors.append(f"unexpected column: {col_name!r}")

    return errors
