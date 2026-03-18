"""Fusion — multi-source merge and dedup rules.

Implements explicit fusion policies (best / union / prefer:source) with
configurable dedup keys, timestamp tolerance, field precedence, and
provenance tagging.

Module role — cross-cutting infrastructure
------------------------------------------
Fusion is invoked by ``api`` when a user requests multiple sources. It
operates on already-normalized canonical DataFrames — it never touches
raw data or adapter logic.

**Owns:**
- Fusion policy definitions (best, union, prefer:<source>).
- Dedup key selection and near-duplicate matching.
- Field-level precedence resolution for conflicting values.
- Provenance token generation for fused rows.

**Does not own:**
- Fetching or normalizing source data — that is ``adapters``.
- Schema definitions — those are in ``datasets``.
- Writing fused output — that is ``storage`` via ``api``.

**Import rule:** Fusion may import from ``datasets`` (dedup key columns,
provenance field names). It must not import from ``adapters``, ``derive``,
``geometry``, ``cli``, or ``api``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import polars as pl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Merge modes
# ---------------------------------------------------------------------------


class MergeMode(str, Enum):
    """How Neptune combines data from multiple sources.

    Each mode determines dedup behavior and provenance tagging:

    - ``UNION`` — concatenate all sources, tag provenance, no dedup.
      Every row from every source is retained. Provenance is tagged so
      users can filter or group by source. This is the simplest mode
      and preserves maximum information.

    - ``BEST`` — deduplicate using configured precedence, keep the
      winning row per dedup key. Source precedence order determines
      which source "wins" for near-duplicate observations. Losing rows
      are discarded but provenance is tagged on the winner.

    - ``PREFER`` — deterministic source preference. Like ``BEST`` but
      with an explicit source priority string (e.g. ``"prefer:noaa"``).
      The preferred source wins ties; other sources fill gaps.
    """

    UNION = "union"
    """Concatenate all sources with provenance tags, no dedup."""

    BEST = "best"
    """Deduplicate using configured precedence and tolerance."""

    PREFER = "prefer"
    """Deterministic source preference (requires prefer_source)."""


# ---------------------------------------------------------------------------
# Fusion configuration
# ---------------------------------------------------------------------------


@dataclass
class FusionConfig:
    """Configuration for multi-source fusion.

    Controls how Neptune merges data when multiple sources are requested.
    Validated at construction time to reject ambiguous combinations.

    Usage::

        # Simple: just union everything
        config = FusionConfig(mode=MergeMode.UNION)

        # Prefer NOAA, fill gaps with DMA
        config = FusionConfig(
            mode=MergeMode.PREFER,
            prefer_source="noaa",
        )

        # Best-quality dedup with custom tolerance
        config = FusionConfig(
            mode=MergeMode.BEST,
            source_precedence=["noaa", "dma"],
            timestamp_tolerance_seconds=60,
        )
    """

    mode: MergeMode = MergeMode.BEST
    """Merge mode: union, best, or prefer."""

    # --- source precedence ---

    source_precedence: list[str] = field(default_factory=list)
    """Ordered list of source IDs from highest to lowest priority.
    Used by ``BEST`` mode to break ties. If empty, alphabetical order
    is used as a deterministic fallback."""

    prefer_source: str | None = None
    """The preferred source for ``PREFER`` mode. Required when
    mode is PREFER, ignored otherwise."""

    # --- near-duplicate tolerance ---

    timestamp_tolerance_seconds: int = 30
    """Maximum time difference (seconds) between two observations to
    consider them near-duplicates of the same event. Used by BEST and
    PREFER modes during dedup matching."""

    coordinate_tolerance_degrees: float = 0.01
    """Maximum lat/lon difference (degrees) for near-duplicate matching.
    ~0.01° ≈ ~1.1 km at the equator. Set to 0 to require exact match."""

    # --- field precedence ---

    field_precedence: dict[str, list[str]] = field(default_factory=dict)
    """Per-field source precedence for conflict resolution.

    When two sources report different values for the same field on a
    near-duplicate observation, this dict determines which source wins
    per field. Keys are column names, values are ordered source lists.

    Example::

        {"vessel_name": ["noaa", "dma"], "ship_type": ["dma", "noaa"]}

    If a field is not listed, ``source_precedence`` is used.
    If ``source_precedence`` is also empty, alphabetical order applies.
    """

    # --- confidence weights ---

    source_confidence_weights: dict[str, float] = field(default_factory=dict)
    """Per-source confidence weights (0.0–1.0) for fused rows.

    When mode is BEST, the winning source's weight is used as a base
    for the fused row's confidence_score. Sources not listed default
    to 1.0.

    Example::

        {"noaa": 1.0, "dma": 0.9, "aishub": 0.5}
    """

    # --- provenance ---

    tag_provenance: bool = True
    """Whether to update the record_provenance column on fused rows.
    Always True for BEST/PREFER; can be disabled for UNION if the
    caller wants raw concatenation only."""

    def __post_init__(self) -> None:
        """Validate the configuration."""
        errors = self.validate()
        if errors:
            raise ValueError(
                "Invalid fusion configuration:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def validate(self) -> list[str]:
        """Check this configuration for errors.

        Returns a list of human-readable error strings. An empty list
        means the configuration is valid.
        """
        errors: list[str] = []

        # Mode-specific validation.
        if self.mode == MergeMode.PREFER and not self.prefer_source:
            errors.append(
                "PREFER mode requires prefer_source to be set"
            )

        if self.mode != MergeMode.PREFER and self.prefer_source:
            errors.append(
                f"prefer_source={self.prefer_source!r} is only valid "
                f"with PREFER mode (current mode: {self.mode.value})"
            )

        # Tolerance ranges.
        if self.timestamp_tolerance_seconds < 0:
            errors.append(
                f"timestamp_tolerance_seconds must be >= 0, "
                f"got {self.timestamp_tolerance_seconds}"
            )

        if self.coordinate_tolerance_degrees < 0:
            errors.append(
                f"coordinate_tolerance_degrees must be >= 0, "
                f"got {self.coordinate_tolerance_degrees}"
            )

        # Confidence weights must be in [0, 1].
        for source, weight in self.source_confidence_weights.items():
            if not 0.0 <= weight <= 1.0:
                errors.append(
                    f"confidence weight for {source!r} must be in "
                    f"[0.0, 1.0], got {weight}"
                )

        # Field precedence must not include core observation columns
        # that the fusion engine relies on (source, mmsi, timestamp, etc.).
        protected = {"mmsi", "timestamp", "lat", "lon", "source",
                      "record_provenance", "qc_severity", "ingest_id"}
        for field_name in self.field_precedence:
            if field_name in protected:
                errors.append(
                    f"field_precedence must not include protected column "
                    f"{field_name!r}"
                )

        return errors


# ---------------------------------------------------------------------------
# Near-duplicate matching — what constitutes the "same observation"
# ---------------------------------------------------------------------------
#
# Two observations from different sources are considered near-duplicates
# (i.e., the "same" real-world event) if ALL of these match:
#
# 1. MMSI — exact match (same vessel).
# 2. Timestamp — within timestamp_tolerance_seconds of each other.
# 3. Coordinates — within coordinate_tolerance_degrees of each other
#    (set to 0.0 to skip coordinate matching).
#
# The dedup key does NOT use source_record_id because different sources
# assign independent record IDs. Cross-source dedup must rely on the
# physical observation properties (vessel, time, position).


def compute_dedup_buckets(
    df: pl.DataFrame,
    config: FusionConfig,
) -> pl.DataFrame:
    """Add dedup bucket columns to a DataFrame for near-duplicate grouping.

    Adds internal columns:
    - ``_dedup_ts_bucket``: timestamp rounded to the tolerance window
    - ``_dedup_lat_bucket``: lat rounded to tolerance grid (if enabled)
    - ``_dedup_lon_bucket``: lon rounded to tolerance grid (if enabled)

    Uses ``config.timestamp_tolerance_seconds`` and
    ``config.coordinate_tolerance_degrees`` directly — no intermediate
    objects needed.
    """
    tolerance_us = config.timestamp_tolerance_seconds * 1_000_000
    exprs: list[pl.Expr] = []

    if tolerance_us > 0:
        exprs.append(
            (pl.col("timestamp").dt.epoch("us") // tolerance_us * tolerance_us)
            .alias("_dedup_ts_bucket")
        )
    else:
        exprs.append(
            pl.col("timestamp").dt.epoch("us").alias("_dedup_ts_bucket")
        )

    if config.coordinate_tolerance_degrees > 0.0:
        tol = config.coordinate_tolerance_degrees
        if "lat" in df.columns and "lon" in df.columns:
            exprs.append((pl.col("lat") / tol).round(0).alias("_dedup_lat_bucket"))
            exprs.append((pl.col("lon") / tol).round(0).alias("_dedup_lon_bucket"))

    return df.with_columns(exprs)


def dedup_subset_columns(config: FusionConfig, df: pl.DataFrame) -> list[str]:
    """Return the column names that form the dedup group key.

    Always includes ``mmsi`` and ``_dedup_ts_bucket``. Adds coordinate
    buckets if coordinate tolerance is enabled and columns exist.
    """
    subset = ["mmsi", "_dedup_ts_bucket"]
    if (
        config.coordinate_tolerance_degrees > 0.0
        and "_dedup_lat_bucket" in df.columns
        and "_dedup_lon_bucket" in df.columns
    ):
        subset.extend(["_dedup_lat_bucket", "_dedup_lon_bucket"])
    return subset


DEDUP_INTERNAL_COLUMNS: frozenset[str] = frozenset({
    "_dedup_ts_bucket",
    "_dedup_lat_bucket",
    "_dedup_lon_bucket",
    "_fusion_rank",
})
"""Internal columns added during dedup that must be dropped before return."""


# ---------------------------------------------------------------------------
# Parsing helper — convert user-facing merge string to FusionConfig
# ---------------------------------------------------------------------------


def parse_merge_arg(merge: str, sources: list[str] | None = None) -> FusionConfig:
    """Parse the user-facing ``merge`` argument into a ``FusionConfig``.

    Accepts:
    - ``"union"`` → MergeMode.UNION
    - ``"best"`` → MergeMode.BEST with sources as precedence
    - ``"prefer:noaa"`` → MergeMode.PREFER with prefer_source="noaa"

    This is the bridge between the Neptune constructor's simple string
    API and the full FusionConfig.
    """
    merge_lower = merge.lower().strip()

    if merge_lower == "union":
        return FusionConfig(mode=MergeMode.UNION)

    if merge_lower == "best":
        return FusionConfig(
            mode=MergeMode.BEST,
            source_precedence=sources or [],
        )

    if merge_lower.startswith("prefer:"):
        preferred = merge_lower.split(":", 1)[1].strip()
        if not preferred:
            raise ValueError(
                "prefer: merge mode requires a source ID, "
                "e.g. merge='prefer:noaa'"
            )
        return FusionConfig(
            mode=MergeMode.PREFER,
            prefer_source=preferred,
            source_precedence=sources or [],
        )

    raise ValueError(
        f"Unknown merge mode: {merge!r}. "
        f"Valid modes: 'union', 'best', 'prefer:<source>'"
    )


# ---------------------------------------------------------------------------
# Merge engine — apply fusion to multi-source DataFrames
# ---------------------------------------------------------------------------


def merge(
    frames: dict[str, pl.DataFrame],
    config: FusionConfig,
) -> pl.DataFrame:
    """Merge DataFrames from multiple sources according to the fusion config.

    Args:
        frames: Mapping of source_id → normalized canonical DataFrame.
            Each DataFrame must have at least ``mmsi``, ``timestamp``,
            and ``source`` columns.
        config: Fusion configuration controlling merge behavior.

    Returns:
        A single merged DataFrame with provenance tagging.
    """
    if not frames:
        raise ValueError("No frames to merge")

    if len(frames) == 1:
        # Single source — no fusion needed, just tag provenance.
        source_id, df = next(iter(frames.items()))
        if config.tag_provenance:
            df = _tag_provenance(df, f"{source_id}:only")
        return df

    if config.mode == MergeMode.UNION:
        return _merge_union(frames, config)
    elif config.mode == MergeMode.BEST:
        return _merge_best(frames, config)
    elif config.mode == MergeMode.PREFER:
        return _merge_best(frames, config)
    else:
        raise ValueError(f"Unsupported merge mode: {config.mode}")


# ---------------------------------------------------------------------------
# Union merge
# ---------------------------------------------------------------------------


def _merge_union(
    frames: dict[str, pl.DataFrame],
    config: FusionConfig,
) -> pl.DataFrame:
    """Concatenate all sources with provenance tags, no dedup.

    Every row from every source is retained. The ``record_provenance``
    column is updated to indicate union membership.
    """
    tagged: list[pl.DataFrame] = []

    for source_id, df in frames.items():
        if config.tag_provenance:
            df = _tag_provenance(df, f"{source_id}:union")
        tagged.append(df)

    # Align columns across sources before concatenation.
    all_cols = _collect_all_columns(tagged)
    aligned = [_align_columns(df, all_cols) for df in tagged]

    result = pl.concat(aligned, how="vertical_relaxed")

    logger.info(
        "Union merge: %d sources, %d total rows",
        len(frames),
        len(result),
    )
    return result


# ---------------------------------------------------------------------------
# Best merge — dedup with source precedence
# ---------------------------------------------------------------------------


def _merge_best(
    frames: dict[str, pl.DataFrame],
    config: FusionConfig,
) -> pl.DataFrame:
    """Deduplicate near-duplicate observations, keeping the best source.

    Algorithm:
    1. Concatenate all sources.
    2. Assign a precedence rank to each row based on its source.
    3. Round timestamps to the tolerance window.
    4. Group by (mmsi, rounded_timestamp).
    5. Within each group, keep the row with the best (lowest) rank.
    6. Tag provenance on the winner.
    """
    # Build source precedence.
    precedence = _build_precedence(config, list(frames.keys()))

    # Concatenate and add rank column.
    tagged: list[pl.DataFrame] = []
    for source_id, df in frames.items():
        rank = precedence.get(source_id, len(precedence))
        df = df.with_columns(
            pl.lit(rank).alias("_fusion_rank").cast(pl.Int32),
        )
        tagged.append(df)

    all_cols = _collect_all_columns(tagged)
    aligned = [_align_columns(df, all_cols) for df in tagged]
    combined = pl.concat(aligned, how="vertical_relaxed")

    # Compute dedup buckets (timestamp + optional coordinates).
    combined = compute_dedup_buckets(combined, config)
    subset = dedup_subset_columns(config, combined)

    # Within each dedup group, keep the row with the best (lowest) rank.
    sort_cols = subset + ["_fusion_rank", "timestamp"]
    combined = combined.sort(sort_cols)

    if config.field_precedence:
        # Field-level conflict resolution: for specified fields, pick
        # the best non-null value per-field within each dedup group.
        deduped = _resolve_field_conflicts(combined, subset, config)
    else:
        # Simple row-level dedup: keep entire row from best source.
        deduped = combined.unique(subset=subset, keep="first")

    # Count how many rows were deduped.
    n_before = len(combined)
    n_after = len(deduped)
    n_deduped = n_before - n_after

    # Build provenance: collect contributing sources per dedup group.
    if config.tag_provenance:
        # Compute which sources contributed to each dedup group.
        contributors = (
            combined
            .group_by(subset)
            .agg(pl.col("source").unique().sort().alias("_contributors"))
        )
        deduped = deduped.join(contributors, on=subset, how="left")

        # Build provenance token: "winner:mode[+contributor1+contributor2]"
        if config.mode == MergeMode.PREFER:
            mode_tag = f"prefer:{config.prefer_source}"
        else:
            mode_tag = "best"

        deduped = deduped.with_columns(
            _build_provenance_token(mode_tag).alias("record_provenance"),
        )

        # Drop internal contributor column.
        deduped = deduped.drop("_contributors")

    # Apply confidence weights if configured.
    if config.source_confidence_weights:
        deduped = _apply_confidence_weights(deduped, config)

    # Drop internal columns.
    drop_cols = [c for c in DEDUP_INTERNAL_COLUMNS if c in deduped.columns]
    deduped = deduped.drop(drop_cols)

    logger.info(
        "Best merge: %d sources, %d → %d rows (%d near-duplicates removed)",
        len(frames),
        n_before,
        n_after,
        n_deduped,
    )

    return deduped


# ---------------------------------------------------------------------------
# Field-level conflict resolution
# ---------------------------------------------------------------------------


def _resolve_field_conflicts(
    combined: pl.DataFrame,
    subset: list[str],
    config: FusionConfig,
) -> pl.DataFrame:
    """Resolve per-field conflicts within dedup groups.

    For each dedup group (identified by ``subset`` columns):
    - Fields listed in ``config.field_precedence`` pick the best non-null
      value according to that field's source priority.
    - All other fields take the first value (from the best-ranked row,
      since the input is pre-sorted by rank).

    Protected columns (mmsi, timestamp, source, etc.) are rejected by
    ``FusionConfig.validate()`` and cannot appear in ``field_precedence``.

    Returns one row per dedup group.
    """
    # Build per-field rank columns for fields with custom precedence.
    field_rank_exprs: list[pl.Expr] = []
    field_rank_cols: dict[str, str] = {}  # field_name → rank_col_name

    for field_name, source_order in config.field_precedence.items():
        if field_name not in combined.columns:
            continue
        rank_col = f"_fprec_{field_name}"
        # Build a when/then chain: source_order[0] → 0, source_order[1] → 1, etc.
        expr = pl.lit(len(source_order))  # default rank for unlisted sources
        for i, src in enumerate(source_order):
            expr = (
                pl.when(pl.col("source") == src)
                .then(pl.lit(i))
                .otherwise(expr)
            )
        field_rank_exprs.append(expr.alias(rank_col).cast(pl.Int32))
        field_rank_cols[field_name] = rank_col

    if field_rank_exprs:
        combined = combined.with_columns(field_rank_exprs)

    # Build aggregation expressions per column.
    agg_exprs: list[pl.Expr] = []

    for col_name in combined.columns:
        if col_name in subset:
            # Group-by key — not aggregated.
            continue

        if col_name in field_rank_cols:
            # Custom field precedence: pick best non-null value by
            # field-specific rank.
            rank_col = field_rank_cols[col_name]
            # Sort by field rank, drop nulls, take first.
            agg_exprs.append(
                pl.col(col_name)
                .sort_by(rank_col)
                .drop_nulls()
                .first()
                .alias(col_name)
            )
        elif col_name.startswith("_fprec_"):
            # Internal field-precedence rank column — skip.
            continue
        else:
            # Default: take first value (from best-ranked row due to sort).
            agg_exprs.append(pl.col(col_name).first())

    result = combined.group_by(subset).agg(agg_exprs)

    # Drop field-precedence rank columns.
    fprec_cols = [c for c in result.columns if c.startswith("_fprec_")]
    if fprec_cols:
        result = result.drop(fprec_cols)

    return result


def _apply_confidence_weights(
    df: pl.DataFrame,
    config: FusionConfig,
) -> pl.DataFrame:
    """Apply per-source confidence weights to a DataFrame.

    Sets the ``confidence_score`` column based on the source of each row.
    Sources not listed in ``config.source_confidence_weights`` default to 1.0.
    """
    weight_expr = pl.lit(1.0)
    for source_id, weight in config.source_confidence_weights.items():
        weight_expr = (
            pl.when(pl.col("source") == source_id)
            .then(pl.lit(weight))
            .otherwise(weight_expr)
        )
    return df.with_columns(weight_expr.alias("confidence_score"))


# ---------------------------------------------------------------------------
# Provenance token construction
# ---------------------------------------------------------------------------


def _build_provenance_token(mode_tag: str) -> pl.Expr:
    """Build a provenance token expression for fused rows.

    Token format: ``"winner:mode"`` or ``"winner:mode[+src1+src2]"``

    Examples:
    - ``"noaa:best"`` — single source, no dedup needed
    - ``"noaa:best[+dma]"`` — noaa won, dma also contributed
    - ``"noaa:prefer:noaa[+dma+aishub]"`` — noaa preferred, others contributed

    Requires ``_contributors`` column (List[String]) with sorted unique
    source IDs that contributed to each dedup group.
    """
    winner = pl.col("source")
    n_contributors = pl.col("_contributors").list.len()
    all_sources = pl.col("_contributors").list.join("+")

    base = winner + pl.lit(f":{mode_tag}")
    return (
        pl.when(n_contributors > 1)
        .then(base + pl.lit("[+") + all_sources + pl.lit("]"))
        .otherwise(base)
    )


@dataclass
class FusionSummary:
    """Summary of a fusion operation for inspection and debugging.

    Returned by ``merge()`` when summary is requested, providing
    row-level statistics and contributor breakdowns.
    """

    mode: str
    """Merge mode used: 'union', 'best', 'prefer:<source>'."""

    sources: list[str]
    """Source IDs that were merged."""

    rows_before: int
    """Total rows across all sources before dedup."""

    rows_after: int
    """Rows after dedup."""

    rows_deduped: int
    """Number of near-duplicate rows removed."""

    dedup_groups_with_conflicts: int = 0
    """Number of dedup groups where sources disagreed on at least one field."""

    field_precedence_applied: list[str] = field(default_factory=list)
    """Fields that had custom precedence applied."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_precedence(
    config: FusionConfig, source_ids: list[str]
) -> dict[str, int]:
    """Build a source → rank mapping from config.

    Lower rank = higher priority.
    """
    if config.mode == MergeMode.PREFER and config.prefer_source:
        # Preferred source gets rank 0, others get rank based on
        # source_precedence or alphabetical order.
        remaining = [s for s in source_ids if s != config.prefer_source]
        if config.source_precedence:
            # Use precedence order for remaining, then alphabetical for unlisted.
            ordered = [
                s for s in config.source_precedence
                if s in remaining and s != config.prefer_source
            ]
            unlisted = sorted(s for s in remaining if s not in ordered)
            remaining = ordered + unlisted
        else:
            remaining = sorted(remaining)

        result = {config.prefer_source: 0}
        for i, sid in enumerate(remaining, start=1):
            result[sid] = i
        return result

    if config.source_precedence:
        # Use explicit precedence order, then alphabetical for unlisted.
        result: dict[str, int] = {}
        rank = 0
        for sid in config.source_precedence:
            if sid in source_ids:
                result[sid] = rank
                rank += 1
        for sid in sorted(source_ids):
            if sid not in result:
                result[sid] = rank
                rank += 1
        return result

    # Default: alphabetical order.
    return {sid: i for i, sid in enumerate(sorted(source_ids))}


def _tag_provenance(df: pl.DataFrame, tag: str) -> pl.DataFrame:
    """Set the record_provenance column to the given tag."""
    return df.with_columns(pl.lit(tag).alias("record_provenance"))


def _collect_all_columns(frames: list[pl.DataFrame]) -> dict[str, pl.DataType]:
    """Collect the union of all columns and their types across frames."""
    all_cols: dict[str, pl.DataType] = {}
    for df in frames:
        for col_name, dtype in df.schema.items():
            if col_name not in all_cols:
                all_cols[col_name] = dtype
    return all_cols


def _align_columns(
    df: pl.DataFrame, target_cols: dict[str, pl.DataType]
) -> pl.DataFrame:
    """Add missing columns as null and reorder to match target schema."""
    missing = [
        pl.lit(None).cast(dtype).alias(col_name)
        for col_name, dtype in target_cols.items()
        if col_name not in df.columns
    ]
    if missing:
        df = df.with_columns(missing)
    return df.select([c for c in target_cols if c in df.columns])
