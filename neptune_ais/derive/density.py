"""Density derivation — spatial density and heatmap products.

Computes vessel density grids at configurable resolution for
visualization and analysis. Produces a reusable derived dataset
that map layers (``viz.prepare_density``) can consume.

Unlike ``viz.prepare_density`` which is a presentation-time helper,
this module produces a **stored derived product** with provenance
and cache identity — suitable for cross-session reuse.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class DensityConfig:
    """Configuration for density grid derivation.

    Args:
        resolution: Grid resolution. For the lat/lon fallback, this
            maps to decimal rounding places. For H3 (when available),
            this is the H3 resolution (0–15). Default 4.
        time_bucket_hours: Aggregate positions into time windows of
            this width. Each (cell, time_bucket) pair counts as one
            observation. Default 24 (daily density).
    """

    resolution: int = 4
    time_bucket_hours: int = 24

    def config_hash(self) -> str:
        key = f"res={self.resolution}:bucket={self.time_bucket_hours}"
        return hashlib.sha1(key.encode()).hexdigest()[:12]


def compute_density(
    positions: pl.DataFrame,
    *,
    config: DensityConfig | None = None,
) -> pl.DataFrame:
    """Compute vessel density grid from positions.

    Bins positions into spatial cells and counts distinct vessels
    and total observations per cell.

    Args:
        positions: Positions DataFrame with ``mmsi``, ``timestamp``,
            ``lat``, and ``lon`` columns.
        config: Density configuration. Uses defaults if None.

    Returns:
        A DataFrame with columns:

        - ``cell_id`` (String): Grid cell identifier.
        - ``center_lat`` (Float64): Cell center latitude.
        - ``center_lon`` (Float64): Cell center longitude.
        - ``vessel_count`` (Int64): Distinct vessels in the cell.
        - ``observation_count`` (Int64): Total position observations.
    """
    if config is None:
        config = DensityConfig()

    if len(positions) == 0:
        return pl.DataFrame(
            schema={
                "cell_id": pl.String,
                "center_lat": pl.Float64,
                "center_lon": pl.Float64,
                "vessel_count": pl.Int64,
                "observation_count": pl.Int64,
            }
        )

    # Map resolution to decimal places for lat/lon rounding.
    decimals = max(0, (config.resolution - 1) // 3 + 1)

    grid = positions.with_columns(
        pl.col("lat").round(decimals).alias("center_lat"),
        pl.col("lon").round(decimals).alias("center_lon"),
    )

    counts = (
        grid.group_by(["center_lat", "center_lon"])
        .agg(
            pl.col("mmsi").n_unique().cast(pl.Int64).alias("vessel_count"),
            pl.len().cast(pl.Int64).alias("observation_count"),
        )
    )

    counts = counts.with_columns(
        pl.concat_str(
            [pl.col("center_lat").cast(pl.String),
             pl.col("center_lon").cast(pl.String)],
            separator=",",
        ).alias("cell_id"),
    )

    return counts.select(
        "cell_id", "center_lat", "center_lon",
        "vessel_count", "observation_count",
    ).sort("observation_count", descending=True)
