"""Viz — map layer helpers and visualization support.

Builds viewport-aware, Arrow/GeoArrow-friendly map layers for positions,
tracks, trips, density, and events using lonboard.

Module role — presentation layer (optional dependency)
------------------------------------------------------
**Owns:**
- Map layer construction: positions, tracks, trips, density, events.
- Viewport clipping and sampling for dense point clouds.
- Color-by logic and layer styling.
- HTML export for standalone map files.

**Does not own:**
- Data access or derivation — receives DataFrames/LazyFrames from ``api``.
- Geometry conversions — delegates to ``geometry.bridges`` if needed.

**Import rule:** Viz may import from ``datasets`` (column names for color-by)
and ``geometry.bridges`` (for GeoArrow conversion). lonboard is an optional
dependency — viz must handle its absence gracefully. Viz must not import
from ``adapters``, ``derive``, ``storage``, ``catalog``, or ``cli``.

**Install extra:** ``pip install neptune-ais[geo]`` (lonboard is part of the
geo extra since it is used alongside spatial data).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from neptune_ais.datasets.positions import Col as PosCol
from neptune_ais.datasets.tracks import Col as TrackCol

# Viz-only derived column name (not part of the tracks schema).
_TRIP_PROGRESS = "trip_progress"


# ---------------------------------------------------------------------------
# Viewport — shared bounding box type for clipping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Viewport:
    """WGS-84 bounding box for viewport clipping.

    Args:
        west: Minimum longitude (-180 to 180).
        south: Minimum latitude (-90 to 90).
        east: Maximum longitude (-180 to 180).
        north: Maximum latitude (-90 to 90).
    """

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if not (-90 <= self.south <= self.north <= 90):
            raise ValueError(
                f"Invalid latitude range: south={self.south}, north={self.north}"
            )
        if not (-180 <= self.west <= 180 and -180 <= self.east <= 180):
            raise ValueError(
                f"Invalid longitude range: west={self.west}, east={self.east}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Collect a LazyFrame or pass through a DataFrame."""
    return df.collect() if isinstance(df, pl.LazyFrame) else df


def _clip_positions(df: pl.DataFrame, viewport: Viewport) -> pl.DataFrame:
    """Filter positions to those within a viewport."""
    return df.filter(
        (pl.col(PosCol.LAT) >= viewport.south)
        & (pl.col(PosCol.LAT) <= viewport.north)
        & (pl.col(PosCol.LON) >= viewport.west)
        & (pl.col(PosCol.LON) <= viewport.east)
    )


def _clip_tracks(df: pl.DataFrame, viewport: Viewport) -> pl.DataFrame:
    """Filter tracks whose bbox intersects a viewport.

    Two bounding boxes intersect when neither is entirely left/right/above/below
    the other.
    """
    return df.filter(
        (pl.col(TrackCol.BBOX_EAST) >= viewport.west)
        & (pl.col(TrackCol.BBOX_WEST) <= viewport.east)
        & (pl.col(TrackCol.BBOX_NORTH) >= viewport.south)
        & (pl.col(TrackCol.BBOX_SOUTH) <= viewport.north)
    )


def _sample(
    df: pl.DataFrame, max_rows: int | None, *, seed: int | None = None
) -> pl.DataFrame:
    """Downsample to at most *max_rows* rows if the frame is larger."""
    if max_rows is not None and len(df) > max_rows:
        return df.sample(n=max_rows, seed=seed)
    return df


# ---------------------------------------------------------------------------
# Positions layer
# ---------------------------------------------------------------------------


def prepare_positions(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    viewport: Viewport | None = None,
    max_points: int | None = None,
) -> pl.DataFrame:
    """Prepare a positions DataFrame for map rendering.

    Applies viewport clipping (if provided) then optional downsampling.
    Returns a materialized DataFrame ready for GeoDataFrame conversion
    or direct Arrow consumption.

    Args:
        df: Positions LazyFrame or DataFrame.
        viewport: Optional bounding box to clip to.
        max_points: If set, downsample to at most this many points.

    Returns:
        A Polars DataFrame with position rows, clipped and sampled.
    """
    result = _collect(df)

    if viewport is not None:
        result = _clip_positions(result, viewport)

    result = _sample(result, max_points)
    return result


# ---------------------------------------------------------------------------
# Tracks layer
# ---------------------------------------------------------------------------


def prepare_tracks(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    viewport: Viewport | None = None,
    max_tracks: int | None = None,
) -> pl.DataFrame:
    """Prepare a tracks DataFrame for map rendering.

    Viewport clipping uses bbox intersection — a track is included if its
    bounding box overlaps the viewport. This avoids decoding WKB geometry
    for the filter step.

    Args:
        df: Tracks LazyFrame or DataFrame.
        viewport: Optional bounding box to clip to.
        max_tracks: If set, downsample to at most this many tracks.

    Returns:
        A Polars DataFrame with track rows, clipped and sampled.
    """
    result = _collect(df)

    if viewport is not None:
        result = _clip_tracks(result, viewport)

    result = _sample(result, max_tracks)
    return result


# ---------------------------------------------------------------------------
# Trip layer — animated track playback prerequisites
# ---------------------------------------------------------------------------


def prepare_trips(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    viewport: Viewport | None = None,
    max_tracks: int | None = None,
) -> pl.DataFrame:
    """Prepare tracks for animated trip playback.

    Trip-style rendering (e.g. deck.gl TripsLayer) requires per-vertex
    timestamps. This function filters tracks to those that have
    ``timestamp_offsets_ms`` and ``geometry_wkb``, then adds a normalized
    ``trip_progress`` column (0.0–1.0) for animation scaling.

    Tracks without geometry or timestamp offsets are dropped — call
    ``Neptune.tracks(include_geometry=True)`` to populate them.

    Args:
        df: Tracks LazyFrame or DataFrame (must include geometry columns).
        viewport: Optional bounding box to clip to.
        max_tracks: If set, downsample to at most this many tracks.

    Returns:
        A Polars DataFrame with trip-ready track rows. Includes
        ``trip_progress`` (Float64) column: duration_s normalized to [0, 1]
        across all returned tracks for uniform animation speed.
    """
    # Check schema before collecting — avoids materializing a large
    # LazyFrame only to discover the required columns are absent.
    if isinstance(df, pl.LazyFrame):
        cols = df.collect_schema().names()
    else:
        cols = df.columns
    required = {TrackCol.GEOMETRY_WKB, TrackCol.TIMESTAMP_OFFSETS_MS}
    if not required.issubset(cols):
        if isinstance(df, pl.LazyFrame):
            schema = dict(df.collect_schema())
        else:
            schema = dict(df.schema)
        schema[_TRIP_PROGRESS] = pl.Float64
        return pl.DataFrame(schema=schema)

    result = _collect(df)

    # Only keep tracks with geometry and timestamp offsets.
    result = result.filter(
        pl.col(TrackCol.GEOMETRY_WKB).is_not_null()
        & pl.col(TrackCol.TIMESTAMP_OFFSETS_MS).is_not_null()
    )

    if viewport is not None:
        result = _clip_tracks(result, viewport)

    result = _sample(result, max_tracks)

    # Normalize duration to [0, 1] for animation.
    max_dur = result[TrackCol.DURATION_S].max()
    if max_dur is not None and max_dur > 0:
        result = result.with_columns(
            (pl.col(TrackCol.DURATION_S) / max_dur).alias(_TRIP_PROGRESS),
        )
    else:
        result = result.with_columns(
            pl.lit(0.0).alias(_TRIP_PROGRESS),
        )

    return result


# ---------------------------------------------------------------------------
# Density layer — H3-binned position counts
# ---------------------------------------------------------------------------


def prepare_density(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    viewport: Viewport | None = None,
    resolution: int = 4,
    max_points: int | None = None,
) -> pl.DataFrame:
    """Prepare a density grid from positions for heatmap rendering.

    Bins positions into H3 hexagonal cells at the given resolution and
    counts observations per cell. Returns a DataFrame with one row per
    occupied H3 cell, suitable for ``lonboard.H3HexagonLayer`` or
    similar heatmap renderers.

    If the ``h3`` package is not installed, falls back to a simple
    lat/lon grid (rounded to resolution-dependent decimal places) with
    counts.

    Args:
        df: Positions LazyFrame or DataFrame.
        viewport: Optional bounding box to clip to before binning.
        resolution: H3 resolution (0–15). Higher = smaller hexagons.
            Default 4 (~1,770 km² per hex) is good for overview maps.
        max_points: If set, sample positions before binning (useful for
            very large datasets where exact counts aren't needed).

    Returns:
        A Polars DataFrame with columns:

        - ``h3_index`` (String): H3 cell index (or ``grid_key`` if
          falling back to lat/lon grid).
        - ``count`` (Int64): Number of positions in the cell.
        - ``center_lat`` (Float64): Cell center latitude.
        - ``center_lon`` (Float64): Cell center longitude.
    """
    result = _collect(df)

    if viewport is not None:
        result = _clip_positions(result, viewport)

    result = _sample(result, max_points)

    if len(result) == 0:
        return pl.DataFrame(
            schema={
                "h3_index": pl.String,
                "count": pl.Int64,
                "center_lat": pl.Float64,
                "center_lon": pl.Float64,
            }
        )

    try:
        return _density_h3(result, resolution)
    except (ImportError, AttributeError):
        # ImportError: h3 not installed.
        # AttributeError: h3 v3 installed (different API names).
        return _density_grid_fallback(result, resolution)


def _density_h3(df: pl.DataFrame, resolution: int) -> pl.DataFrame:
    """Bin positions into H3 cells and count per cell."""
    import h3  # caller guards with try/except ImportError

    # Use struct map_elements to avoid materializing two Python lists.
    h3_series = (
        df.select(pl.struct([PosCol.LAT, PosCol.LON])
            .map_elements(
                lambda r: h3.latlng_to_cell(r[PosCol.LAT], r[PosCol.LON], resolution),
                return_dtype=pl.String,
            )
            .alias("h3_index"))
        .to_series()
    )

    h3_df = pl.DataFrame({"h3_index": h3_series})
    counts = h3_df.group_by("h3_index").agg(pl.len().cast(pl.Int64).alias("count"))

    # Compute cell centers — return struct to avoid intermediate List column.
    _center_dtype = pl.Struct({"center_lat": pl.Float64, "center_lon": pl.Float64})
    centers = counts["h3_index"].map_elements(
        lambda idx: dict(zip(("center_lat", "center_lon"), h3.cell_to_latlng(idx))),
        return_dtype=_center_dtype,
    )
    counts = counts.with_columns(centers.struct.unnest())

    return counts.sort("count", descending=True)


def _density_grid_fallback(df: pl.DataFrame, resolution: int) -> pl.DataFrame:
    """Fallback: bin positions into a rounded lat/lon grid.

    Uses resolution-dependent rounding as an approximation of H3 cell
    size. Not as accurate as H3, but works without the h3 dependency.
    """
    # Map H3 resolution to approximate decimal places for rounding.
    # H3 res 0 → 0 decimals, res 2 → 1, res 5 → 2, res 8 → 3, etc.
    decimals = max(0, (resolution - 1) // 3 + 1)

    grid = df.select(
        pl.col(PosCol.LAT).round(decimals).alias("center_lat"),
        pl.col(PosCol.LON).round(decimals).alias("center_lon"),
    )

    counts = (
        grid.group_by(["center_lat", "center_lon"])
        .agg(pl.len().cast(pl.Int64).alias("count"))
    )

    # Create a grid_key for the h3_index column.
    counts = counts.with_columns(
        pl.concat_str(
            [pl.col("center_lat").cast(pl.String),
             pl.col("center_lon").cast(pl.String)],
            separator=",",
        ).alias("h3_index"),
    )

    return counts.select(
        "h3_index", "count", "center_lat", "center_lon"
    ).sort("count", descending=True)
