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

        return errors


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
    # Build source → rank mapping.
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

    # Round timestamps to tolerance window for grouping.
    tolerance_us = config.timestamp_tolerance_seconds * 1_000_000
    if tolerance_us > 0:
        combined = combined.with_columns(
            (pl.col("timestamp").dt.epoch("us") // tolerance_us * tolerance_us)
            .alias("_fusion_ts_bucket"),
        )
    else:
        combined = combined.with_columns(
            pl.col("timestamp").dt.epoch("us").alias("_fusion_ts_bucket"),
        )

    # Within each (mmsi, ts_bucket), keep the row with the lowest rank.
    # Use sort + unique to get deterministic dedup.
    combined = combined.sort(["mmsi", "_fusion_ts_bucket", "_fusion_rank", "timestamp"])
    deduped = combined.unique(
        subset=["mmsi", "_fusion_ts_bucket"],
        keep="first",
    )

    # Count how many rows were deduped.
    n_before = len(combined)
    n_after = len(deduped)
    n_deduped = n_before - n_after

    # Tag provenance — distinguish mode in the tag.
    if config.tag_provenance:
        if config.mode == MergeMode.PREFER:
            tag_suffix = f":prefer:{config.prefer_source}"
        else:
            tag_suffix = ":best"
        deduped = deduped.with_columns(
            (pl.col("source") + pl.lit(tag_suffix)).alias("record_provenance"),
        )

    # Apply confidence weights if configured.
    if config.source_confidence_weights:
        weight_expr = pl.lit(1.0)
        for source_id, weight in config.source_confidence_weights.items():
            weight_expr = pl.when(pl.col("source") == source_id).then(
                pl.lit(weight)
            ).otherwise(weight_expr)
        deduped = deduped.with_columns(
            weight_expr.alias("confidence_score"),
        )

    # Drop internal columns.
    deduped = deduped.drop(["_fusion_rank", "_fusion_ts_bucket"])

    logger.info(
        "Best merge: %d sources, %d → %d rows (%d near-duplicates removed)",
        len(frames),
        n_before,
        n_after,
        n_deduped,
    )

    return deduped


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
    for col_name, dtype in target_cols.items():
        if col_name not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col_name))
    # Select in target order, keeping only columns that exist in target.
    return df.select([c for c in target_cols if c in df.columns])
