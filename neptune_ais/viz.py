"""Viz — map layer helpers and visualization support.

Builds viewport-aware, Arrow/GeoArrow-friendly map layers for positions,
tracks, trips, density, and events using lonboard.

Module role — presentation layer (optional dependency)
------------------------------------------------------
**Owns:**
- Map layer construction: positions, tracks, trips, density, events.
- Viewport clipping and sampling for dense point clouds.
- Color-by logic and layer styling.
- HTML export for standalone map files and dashboards.

**Does not own:**
- Data access or derivation — receives DataFrames/LazyFrames from ``api``.
- Geometry conversions — delegates to ``geometry.bridges`` if needed.

**Import rule:** Viz may import from ``datasets`` (column names for color-by),
``geometry.bridges`` (for GeoArrow conversion), and ``derive.crossings``
(for dashboard gate-crossing analytics). lonboard is an optional
dependency — viz must handle its absence gracefully. Viz must not import
from ``adapters``, ``storage``, ``catalog``, or ``cli``.

**Install extra:** ``pip install neptune-ais[geo]`` (lonboard is part of the
geo extra since it is used alongside spatial data).
"""

from __future__ import annotations

import json
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from neptune_ais.datasets.events import Col as EventCol
from neptune_ais.datasets.positions import Col as PosCol
from neptune_ais.datasets.tracks import Col as TrackCol
from neptune_ais.datasets.vessels import Col as VesselCol
from neptune_ais.derive.crossings import GateLine  # noqa: F401 — re-exported

if TYPE_CHECKING:
    from neptune_ais.ports._index import PortIndex

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


@dataclass(frozen=True)
class InfrastructurePoint:
    """Named point feature for map display (port, refinery, etc.).

    Args:
        name: Human-readable label.
        lat: WGS-84 latitude.
        lon: WGS-84 longitude.
        kind: Feature category (``"port"``, ``"refinery"``,
            ``"desalination"``, ``"anchorage"``, etc.).
    """

    name: str
    lat: float
    lon: float
    kind: str = "port"


@dataclass(frozen=True)
class TimelapsConfig:
    """Configuration for a timelapse corridor visualization.

    Parameterizes the cinematic vessel-corridor timelapse effect —
    positions accumulate over time on a dark basemap to reveal shipping
    lanes through density, similar to long-exposure photography.

    Args:
        title: Display title (e.g., ``"VESSEL MARKET - AIS TIMELAPSE"``).
        subtitle: Subtitle line below the title.
        date_from: Start of analysis period (ISO-8601 string).
        date_to: End of analysis period (ISO-8601 string).
        center_lat: Map center latitude. Auto-computed from data if None.
        center_lon: Map center longitude. Auto-computed if None.
        zoom: Initial map zoom level. Auto-computed if None.
        dot_radius: Radius of each position dot in pixels. Default 2.
        dot_alpha: Base alpha per dot (0.0–1.0). Default 0.3.
        bin_interval_minutes: Time bin size for animation steps. Default 60.
        speed: Animation speed — bins per second of playback. Default 4.
        color_by_type: Color-code dots by vessel type. Default True.
        fade_factor: Per-frame multiplicative fade for the accumulation
            canvas (1.0 = no fade, keep all trails). Default 0.998.
        bloom: Apply gaussian-blur bloom post-processing. Default True.
        layout: Multi-panel layout direction. ``"vertical"`` (rows,
            default), ``"horizontal"`` (columns), or ``"grid"``.
    """

    title: str = "AIS TIMELAPSE"
    subtitle: str = ""
    date_from: str = ""
    date_to: str = ""
    center_lat: float | None = None
    center_lon: float | None = None
    zoom: int | None = None
    dot_radius: float = 1.0
    dot_alpha: float = 0.10
    bin_interval_minutes: int = 30
    speed: float = 2.0
    color_by_type: bool = True
    fade_factor: float = 0.998
    bloom: bool = True
    layout: str = "vertical"


@dataclass(frozen=True)
class DashboardConfig:
    """Configuration for a maritime intelligence dashboard.

    Parameterizes the analysis scenario so the same function can
    generate dashboards for any chokepoint or region.

    Args:
        title: Dashboard title (e.g., ``"STRAIT OF HORMUZ"``).
        description: Analysis description paragraph.
        gate: Optional chokepoint line. Enables crossing analytics
            (inbound/outbound counts, transit detection, reversals).
        event_date: ISO-8601 date string for the regime-change event
            (e.g., ``"2026-03-01"``). Anchors before/after comparison.
        date_from: Start of analysis period (ISO-8601 string).
        date_to: End of analysis period (ISO-8601 string).
        center_lat: Map center latitude. Auto-computed from tracks if None.
        center_lon: Map center longitude. Auto-computed if None.
        zoom: Initial map zoom level. Auto-computed if None.
        pitch: Map pitch in degrees.
        bearing: Map bearing in degrees.
        trail_length: TripsLayer trail length in seconds.
        speed: Playback speed — seconds of vessel time per second of
            animation.  ``21600`` = 6 hours/second, ``86400`` = 1 day/second.
        infrastructure: Optional named point features for the map.
    """

    title: str
    description: str = ""
    gate: GateLine | None = None
    event_date: str | None = None
    date_from: str = ""
    date_to: str = ""
    center_lat: float | None = None
    center_lon: float | None = None
    zoom: int | None = None
    pitch: float = 35.0
    bearing: float = 0.0
    trail_length: int = 180
    speed: float = 21600.0
    infrastructure: list[InfrastructurePoint] = field(default_factory=list)


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
    counts = counts.with_columns(centers.alias("_center")).unnest("_center")

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


# ---------------------------------------------------------------------------
# Events layer
# ---------------------------------------------------------------------------


def prepare_events(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    viewport: Viewport | None = None,
    event_type: str | None = None,
    min_confidence: float | None = None,
    max_events: int | None = None,
) -> pl.DataFrame:
    """Prepare an events DataFrame for map rendering.

    Applies viewport clipping on the event's representative lat/lon,
    optional event type and confidence filters, and downsampling.

    This follows the same pattern as ``prepare_positions`` and
    ``prepare_tracks`` — the output is a materialized DataFrame
    ready for point-marker or icon rendering on a map.

    Args:
        df: Events LazyFrame or DataFrame.
        viewport: Optional bounding box to clip to.
        event_type: Filter to a single event type (e.g. ``"port_call"``).
        min_confidence: Minimum confidence score to include.
        max_events: If set, downsample to at most this many events.

    Returns:
        A Polars DataFrame with event rows, clipped, filtered, and sampled.
    """
    result = _collect(df)

    if event_type is not None:
        result = result.filter(pl.col(EventCol.EVENT_TYPE) == event_type)

    if min_confidence is not None:
        result = result.filter(
            pl.col(EventCol.CONFIDENCE_SCORE) >= min_confidence
        )

    if viewport is not None:
        # Events have lat/lon like positions — reuse the same clipper.
        result = _clip_positions(result, viewport)

    result = _sample(result, max_events)
    return result


# ---------------------------------------------------------------------------
# Shared trip-building helper
# ---------------------------------------------------------------------------

# Vessel color palette — 10 distinct colors for up to 10 vessels.
PALETTE = [
    [0, 200, 255],    # cyan
    [255, 100, 50],   # orange
    [50, 255, 130],   # green
    [255, 50, 200],   # pink
    [255, 230, 50],   # yellow
    [130, 80, 255],   # purple
    [255, 160, 130],  # salmon
    [80, 255, 255],   # light cyan
    [255, 100, 100],  # red
    [100, 200, 100],  # forest
]


# Vessel type palette — muted tones tuned for additive blending.
# Overlapping types produce natural white highlights instead of neon clashing.
VESSEL_TYPE_PALETTE: dict[str, list[int]] = {
    "cargo": [30, 120, 255],       # vivid blue
    "tanker": [255, 40, 130],      # vivid magenta
    "passenger": [40, 230, 150],   # bright teal
    "fishing": [255, 200, 40],     # vivid amber
    "tug": [160, 50, 255],         # vivid purple
    "other": [140, 150, 180],      # cool gray
}

# Ordered list for index-based lookups in the JS template.
_VESSEL_TYPE_ORDER = list(VESSEL_TYPE_PALETTE.keys())


def _categorize_vessel_type(ship_type: str | None) -> str:
    """Map AIS ship type code to a visualization category.

    Handles both numeric codes (NOAA) and text descriptions (DMA/Finland).
    """
    if not ship_type:
        return "other"
    try:
        code = int(ship_type)
    except (ValueError, TypeError):
        lower = ship_type.lower()
        if "cargo" in lower or "container" in lower or "bulk" in lower:
            return "cargo"
        if "tanker" in lower or "oil" in lower or "chemical" in lower:
            return "tanker"
        if "passenger" in lower or "cruise" in lower or "ferry" in lower:
            return "passenger"
        if "fish" in lower:
            return "fishing"
        if "tug" in lower or "pilot" in lower or "tow" in lower:
            return "tug"
        return "other"
    if 70 <= code <= 79:
        return "cargo"
    if 80 <= code <= 89:
        return "tanker"
    if 60 <= code <= 69:
        return "passenger"
    if 30 <= code <= 39:
        return "fishing"
    if 31 <= code <= 32 or 50 <= code <= 59:
        return "tug"
    return "other"


def _decode_wkb_linestring(wkb: bytes) -> list[list[float]]:
    """Decode a WKB LineString to [[lon, lat], ...].

    Handles the little-endian WKB format produced by _encode_wkb_linestring
    in derive/tracks.py. Uses bulk struct.unpack for efficiency.
    """

    if wkb is None or len(wkb) < 13:
        return []

    byte_order = wkb[0]
    fmt_prefix = "<" if byte_order == 1 else ">"
    _geom_type, n_points = struct.unpack_from(f"{fmt_prefix}II", wkb, 1)
    if n_points == 0:
        return []
    flat = struct.unpack_from(f"{fmt_prefix}{n_points * 2}d", wkb, 9)
    return [
        [round(flat[i], 6), round(flat[i + 1], 6)]
        for i in range(0, len(flat), 2)
    ]


def _build_trips(
    tracks: pl.DataFrame,
) -> tuple[list[dict], dict[int, list[int]], float, int | None]:
    """Decode tracks into trip dicts for deck.gl.

    Shared by :func:`generate_replay` and :func:`generate_dashboard`.

    Args:
        tracks: Tracks DataFrame already filtered to rows with
            ``geometry_wkb`` and ``timestamp_offsets_ms`` present.

    Returns:
        A 4-tuple of:

        - *trips*: list of ``{path, timestamps, color, mmsi}`` dicts.
        - *mmsi_to_color*: mapping of MMSI → ``[r, g, b]``.
        - *max_time*: latest timestamp across all trips (seconds).
        - *global_start_ms*: epoch millis of the earliest track start,
          or ``None`` if unavailable.
    """
    min_start = tracks[TrackCol.START_TIME].min()
    global_start_ms: int | None = (
        int(min_start.timestamp() * 1000) if min_start is not None else None
    )

    trips: list[dict] = []
    mmsi_to_color: dict[int, list[int]] = {}
    color_idx = 0

    for row in tracks.iter_rows(named=True):
        wkb = row[TrackCol.GEOMETRY_WKB]
        offsets_ms = row[TrackCol.TIMESTAMP_OFFSETS_MS]
        mmsi = row[TrackCol.MMSI]

        coords = _decode_wkb_linestring(wkb)
        if len(coords) < 2 or not offsets_ms or len(offsets_ms) != len(coords):
            continue

        if mmsi not in mmsi_to_color:
            mmsi_to_color[mmsi] = PALETTE[color_idx % len(PALETTE)]
            color_idx += 1

        start_time = row[TrackCol.START_TIME]
        if start_time is not None and global_start_ms is not None:
            base_s = (int(start_time.timestamp() * 1000) - global_start_ms) / 1000.0
            timestamps_s = [base_s + t / 1000.0 for t in offsets_ms]
        else:
            timestamps_s = [t / 1000.0 for t in offsets_ms]

        trips.append({
            "path": coords,
            "timestamps": timestamps_s,
            "color": mmsi_to_color[mmsi],
            "mmsi": mmsi,
        })

    max_time = max(t["timestamps"][-1] for t in trips) if trips else 0.0
    return trips, mmsi_to_color, max_time, global_start_ms


def _auto_view(
    tracks: pl.DataFrame,
) -> tuple[float, float, int]:
    """Compute map center and zoom from track bounding boxes."""
    center_lon = float(
        (tracks[TrackCol.BBOX_WEST].mean() + tracks[TrackCol.BBOX_EAST].mean()) / 2
    )
    center_lat = float(
        (tracks[TrackCol.BBOX_SOUTH].mean() + tracks[TrackCol.BBOX_NORTH].mean()) / 2
    )
    lon_spread = float(
        tracks[TrackCol.BBOX_EAST].max() - tracks[TrackCol.BBOX_WEST].min()
    )
    lat_spread = float(
        tracks[TrackCol.BBOX_NORTH].max() - tracks[TrackCol.BBOX_SOUTH].min()
    )
    spread = max(lon_spread, lat_spread, 0.01)
    if spread > 50:
        zoom = 3
    elif spread > 10:
        zoom = 5
    elif spread > 2:
        zoom = 7
    elif spread > 0.5:
        zoom = 9
    else:
        zoom = 11
    return center_lat, center_lon, zoom


def _auto_view_positions(
    df: pl.DataFrame,
) -> tuple[float, float, int]:
    """Compute map center and zoom from positions lat/lon."""
    center_lat = float(df[PosCol.LAT].mean())
    center_lon = float(df[PosCol.LON].mean())
    lat_spread = float(df[PosCol.LAT].max() - df[PosCol.LAT].min())
    lon_spread = float(df[PosCol.LON].max() - df[PosCol.LON].min())
    spread = max(lon_spread, lat_spread, 0.01)
    if spread > 50:
        zoom = 3
    elif spread > 10:
        zoom = 5
    elif spread > 2:
        zoom = 7
    elif spread > 0.5:
        zoom = 9
    else:
        zoom = 11
    return center_lat, center_lon, zoom


def _validate_track_geometry(
    tracks: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame:
    """Collect, validate, and filter tracks to those with geometry."""
    df = _collect(tracks)
    required = {TrackCol.GEOMETRY_WKB, TrackCol.TIMESTAMP_OFFSETS_MS}
    if not required.issubset(df.columns):
        raise ValueError(
            "Tracks must have geometry_wkb and timestamp_offsets_ms columns. "
            "Use Neptune.tracks(include_geometry=True) to generate them."
        )
    df = df.filter(
        pl.col(TrackCol.GEOMETRY_WKB).is_not_null()
        & pl.col(TrackCol.TIMESTAMP_OFFSETS_MS).is_not_null()
    )
    if len(df) == 0:
        raise ValueError("No tracks with geometry found.")
    return df


def _safe_json_embed(obj: Any) -> str:
    """Serialize *obj* to JSON safe for embedding in ``<script>`` tags.

    Escapes sequences that could break out of a ``<script>`` block
    (``</script>``, ``<!--``) to prevent XSS from untrusted string
    fields like vessel names or destinations.
    """
    s = json.dumps(obj)
    return s.replace("<", r"\u003c")


# ---------------------------------------------------------------------------
# Animated vessel replay — standalone HTML with deck.gl TripsLayer
# ---------------------------------------------------------------------------


def generate_replay(
    tracks: pl.DataFrame | pl.LazyFrame,
    output: str = "replay.html",
    *,
    trail_length: int = 180,
    speed: float = 60.0,
) -> str:
    """Generate a standalone HTML vessel replay animation.

    Takes tracks with ``geometry_wkb`` and ``timestamp_offsets_ms``
    (from ``Neptune.tracks(include_geometry=True)``) and produces a
    self-contained HTML file with an animated deck.gl TripsLayer
    showing vessels moving along their routes.

    The output file loads deck.gl and MapLibre GL from CDNs — no
    Python server or Jupyter required. Open it in any browser.

    Args:
        tracks: Tracks DataFrame with geometry columns. Must have
            ``geometry_wkb`` and ``timestamp_offsets_ms`` columns.
            Call ``Neptune.tracks(include_geometry=True)`` to get these.
        output: Output file path. Default ``"replay.html"``.
        trail_length: Seconds of glowing trail behind each vessel.
            Default 180 (3 minutes).
        speed: Playback speed multiplier. At ``speed=60``, one real
            second of animation = 60 seconds of vessel time. Default 60.

    Returns:
        The absolute path to the generated HTML file.

    Raises:
        ValueError: If the tracks DataFrame lacks geometry columns.
    """
    tracks_df = _validate_track_geometry(tracks)
    trips, mmsi_to_color, max_time, _global_start = _build_trips(tracks_df)
    if not trips:
        raise ValueError("No valid trip geometries found.")

    center_lat, center_lon, zoom = _auto_view(tracks_df)

    # Build vessel legend entries.
    legend_entries = [
        {"mmsi": mmsi, "color": color}
        for mmsi, color in mmsi_to_color.items()
    ]

    # Serialize trip data (drop mmsi field — not needed by deck.gl).
    trips_json = _safe_json_embed(
        [{"path": t["path"], "timestamps": t["timestamps"], "color": t["color"]}
         for t in trips],
    )

    html = _REPLAY_HTML_TEMPLATE.format(
        trips_json=trips_json,
        center_lon=center_lon,
        center_lat=center_lat,
        zoom=zoom,
        max_time=max_time,
        trail_length=trail_length,
        speed=speed,
        legend_json=_safe_json_embed(legend_entries),
        n_vessels=len(mmsi_to_color),
        n_tracks=len(trips),
    )

    out_path = Path(output).resolve()
    out_path.write_text(html)
    return str(out_path)


_REPLAY_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Neptune AIS — Vessel Replay</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://unpkg.com/deck.gl@9.1.4/dist.min.js"></script>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0a0a1a; color: #e0e0e0; overflow: hidden; }}
  #map {{ position: absolute; top: 0; left: 0; right: 0; bottom: 0; }}
  #controls {{
    position: absolute; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: rgba(10, 10, 30, 0.92); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px; padding: 14px 20px; display: flex; align-items: center;
    gap: 14px; backdrop-filter: blur(12px); z-index: 10;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
  }}
  #controls button {{
    background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
    color: #fff; padding: 6px 14px; border-radius: 6px; cursor: pointer;
    font-size: 14px; transition: background 0.15s;
  }}
  #controls button:hover {{ background: rgba(255,255,255,0.2); }}
  #controls button.active {{ background: rgba(0,200,255,0.3); border-color: rgba(0,200,255,0.5); }}
  #slider {{ width: 280px; accent-color: #00c8ff; }}
  #clock {{ font-variant-numeric: tabular-nums; font-size: 15px; min-width: 60px;
            color: #00c8ff; font-weight: 600; }}
  #speed-btn {{ font-variant-numeric: tabular-nums; min-width: 44px; text-align: center; }}
  #legend {{
    position: absolute; top: 16px; right: 16px;
    background: rgba(10, 10, 30, 0.88); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px; padding: 12px 16px; z-index: 10;
    backdrop-filter: blur(12px); font-size: 13px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
  }}
  #legend h3 {{ font-size: 11px; text-transform: uppercase; letter-spacing: 1px;
                color: rgba(255,255,255,0.5); margin-bottom: 8px; }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  #header {{
    position: absolute; top: 16px; left: 16px; z-index: 10;
    background: rgba(10, 10, 30, 0.88); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px; padding: 12px 16px; backdrop-filter: blur(12px);
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
  }}
  #header h2 {{ font-size: 15px; font-weight: 600; margin-bottom: 2px; }}
  #header p {{ font-size: 12px; color: rgba(255,255,255,0.5); }}
</style>
</head>
<body>
<div id="map"></div>

<div id="header">
  <h2>Neptune AIS Replay</h2>
  <p>{n_vessels} vessels &middot; {n_tracks} tracks</p>
</div>

<div id="legend">
  <h3>Vessels</h3>
  <div id="legend-items"></div>
</div>

<div id="controls">
  <button id="play-btn" class="active" title="Play / Pause">&#9654;</button>
  <input type="range" id="slider" min="0" max="1000" value="0">
  <span id="clock">00:00:00</span>
  <button id="speed-btn" title="Playback speed">{speed:.0f}x</button>
</div>

<script>
const TRIPS = {trips_json};
const LEGEND = {legend_json};
const MAX_TIME = {max_time};
const TRAIL_LENGTH = {trail_length};
let speed = {speed};
let playing = true;
let currentTime = 0;
const speedSteps = [10, 30, 60, 120, 300, 600];
let speedIdx = speedSteps.indexOf(speed);
if (speedIdx < 0) speedIdx = 2;

// Legend
const legendEl = document.getElementById('legend-items');
LEGEND.forEach(v => {{
  const item = document.createElement('div');
  item.className = 'legend-item';
  item.innerHTML = '<div class="legend-dot" style="background:rgb(' +
    v.color.join(',') + ')"></div><span>' + v.mmsi + '</span>';
  legendEl.appendChild(item);
}});

// Format seconds as HH:MM:SS
function fmt(s) {{
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  return String(h).padStart(2,'0') + ':' + String(m).padStart(2,'0') + ':' +
         String(sec).padStart(2,'0');
}}

// deck.gl overlay
const deckgl = new deck.DeckGL({{
  container: 'map',
  mapStyle: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
  initialViewState: {{
    longitude: {center_lon},
    latitude: {center_lat},
    zoom: {zoom},
    pitch: 35,
    bearing: 0,
  }},
  controller: true,
  getTooltip: ({{object}}) => object && ('MMSI: ' + (object.mmsi || '')),
  layers: [],
}});

function updateLayers() {{
  deckgl.setProps({{
    layers: [
      new deck.TripsLayer({{
        id: 'trips',
        data: TRIPS,
        getPath: d => d.path,
        getTimestamps: d => d.timestamps,
        getColor: d => d.color,
        currentTime: currentTime,
        trailLength: TRAIL_LENGTH,
        widthMinPixels: 3,
        widthMaxPixels: 8,
        capRounded: true,
        jointRounded: true,
        opacity: 0.9,
      }}),
      new deck.ScatterplotLayer({{
        id: 'heads',
        data: TRIPS.filter(d => {{
          const ts = d.timestamps;
          return ts[0] <= currentTime && currentTime <= ts[ts.length - 1];
        }}).map(d => {{
          const ts = d.timestamps;
          let idx = 0;
          for (let i = 0; i < ts.length - 1; i++) {{
            if (ts[i + 1] >= currentTime) {{ idx = i; break; }}
          }}
          const frac = ts[idx + 1] !== ts[idx]
            ? (currentTime - ts[idx]) / (ts[idx + 1] - ts[idx]) : 0;
          const p0 = d.path[idx], p1 = d.path[Math.min(idx + 1, d.path.length - 1)];
          return {{
            position: [p0[0] + (p1[0] - p0[0]) * frac, p0[1] + (p1[1] - p0[1]) * frac],
            color: d.color,
          }};
        }}),
        getPosition: d => d.position,
        getFillColor: d => [...d.color, 255],
        getLineColor: [255, 255, 255, 180],
        radiusMinPixels: 5,
        radiusMaxPixels: 12,
        lineWidthMinPixels: 2,
        stroked: true,
      }}),
    ],
  }});
}}

// Animation loop
let lastFrame = performance.now();
function animate(now) {{
  if (playing) {{
    const dt = (now - lastFrame) / 1000;
    currentTime += dt * speed;
    if (currentTime > MAX_TIME) currentTime = 0;
    document.getElementById('slider').value = (currentTime / MAX_TIME * 1000) | 0;
    document.getElementById('clock').textContent = fmt(currentTime);
    updateLayers();
  }}
  lastFrame = now;
  requestAnimationFrame(animate);
}}
requestAnimationFrame(animate);

// Controls
document.getElementById('play-btn').onclick = () => {{
  playing = !playing;
  const btn = document.getElementById('play-btn');
  btn.textContent = playing ? '\u25B6' : '\u23F8';
  btn.classList.toggle('active', playing);
}};
document.getElementById('slider').oninput = (e) => {{
  currentTime = (e.target.value / 1000) * MAX_TIME;
  document.getElementById('clock').textContent = fmt(currentTime);
  updateLayers();
}};
document.getElementById('speed-btn').onclick = () => {{
  speedIdx = (speedIdx + 1) % speedSteps.length;
  speed = speedSteps[speedIdx];
  document.getElementById('speed-btn').textContent = speed + 'x';
}};
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Maritime intelligence dashboard
# ---------------------------------------------------------------------------


def _compute_dashboard_analytics(
    trips: list[dict],
    crossings: list[dict],
    reversals: list[dict],
    vessels_df: pl.DataFrame | None,
    positions_df: pl.DataFrame | None,
    config: DashboardConfig,
    global_start_ms: int | None,
    total_track_count: int,
) -> dict:
    """Pre-compute all analytics for the dashboard.

    Returns a dict of JSON-serializable analytics data.
    """
    from neptune_ais.datasets.positions import Col as PosCol_

    has_gate = config.gate is not None and len(crossings) > 0

    # -- vessel index from vessels DataFrame ---------------------------------
    vessel_index: dict[int, dict] = {}
    if vessels_df is not None:
        for row in vessels_df.iter_rows(named=True):
            mmsi = row.get(VesselCol.MMSI)
            if mmsi is None:
                continue
            vessel_index[int(mmsi)] = {
                "name": row.get(VesselCol.VESSEL_NAME, ""),
                "type": row.get(VesselCol.SHIP_TYPE, ""),
                "flag": row.get(VesselCol.FLAG, ""),
                "imo": row.get(VesselCol.IMO, ""),
                "callsign": row.get(VesselCol.CALLSIGN, ""),
                "length": row.get(VesselCol.LENGTH),
                "beam": row.get(VesselCol.BEAM),
            }

    # -- enrich vessel index with last-known destination/draught from positions
    if positions_df is not None:
        pos_cols = positions_df.columns
        if PosCol_.DESTINATION in pos_cols or "draught" in pos_cols:
            agg_exprs = [pl.col(PosCol_.MMSI)]
            if PosCol_.DESTINATION in pos_cols:
                agg_exprs.append(pl.col(PosCol_.DESTINATION).drop_nulls().last().alias("_dest"))
            if "draught" in pos_cols:
                agg_exprs.append(pl.col("draught").drop_nulls().last().alias("_draught"))
            last_info = (
                positions_df.sort(PosCol_.TIMESTAMP)
                .group_by(PosCol_.MMSI)
                .agg(*agg_exprs[1:])
            )
            for row in last_info.iter_rows(named=True):
                mmsi = int(row[PosCol_.MMSI])
                if mmsi not in vessel_index:
                    vessel_index[mmsi] = {}
                if "_dest" in row and row["_dest"]:
                    vessel_index[mmsi]["destination"] = str(row["_dest"])
                if "_draught" in row and row["_draught"] is not None:
                    vessel_index[mmsi]["draught"] = row["_draught"]

    # -- per-trip metadata for filtering -------------------------------------
    transit_mmsis: set[int] = set()
    if has_gate:
        transit_mmsis = {c["mmsi"] for c in crossings}

    # -- flag and type counts ------------------------------------------------
    flag_counts: dict[str, int] = defaultdict(int)
    type_counts: dict[str, int] = defaultdict(int)
    trip_mmsis: set[int] = set()
    for trip in trips:
        mmsi = trip["mmsi"]
        if mmsi in trip_mmsis:
            continue
        trip_mmsis.add(mmsi)
        info = vessel_index.get(mmsi, {})
        flag = info.get("flag", "")
        if flag:
            flag_counts[flag] += 1
        ship_type = info.get("type", "")
        if ship_type:
            type_counts[ship_type] += 1

    # -- daily crossings time-series -----------------------------------------
    daily_crossings: list[dict] = []
    if has_gate and global_start_ms is not None:
        epoch_s = global_start_ms / 1000.0
        by_day: dict[str, dict] = defaultdict(
            lambda: {"inbound": 0, "outbound": 0, "mmsis": set()}
        )
        for c in crossings:
            dt = datetime.fromtimestamp(epoch_s + c["timestamp_s"], tz=timezone.utc)
            day_str = dt.strftime("%Y-%m-%d")
            rec = by_day[day_str]
            rec[c["direction"]] += 1
            rec["mmsis"].add(c["mmsi"])

        for day_str in sorted(by_day):
            rec = by_day[day_str]
            daily_crossings.append({
                "date": day_str,
                "inbound": rec["inbound"],
                "outbound": rec["outbound"],
                "unique_vessels": len(rec["mmsis"]),
            })

    # -- rolling 7-day averages (sliding window) ------------------------------
    rolling_7d: list[dict] = []
    if len(daily_crossings) >= 7:
        in_sum = sum(d["inbound"] for d in daily_crossings[:7])
        out_sum = sum(d["outbound"] for d in daily_crossings[:7])
        rolling_7d.append({
            "date": daily_crossings[6]["date"],
            "inbound": round(in_sum / 7, 1),
            "outbound": round(out_sum / 7, 1),
        })
        for i in range(7, len(daily_crossings)):
            in_sum += daily_crossings[i]["inbound"] - daily_crossings[i - 7]["inbound"]
            out_sum += daily_crossings[i]["outbound"] - daily_crossings[i - 7]["outbound"]
            rolling_7d.append({
                "date": daily_crossings[i]["date"],
                "inbound": round(in_sum / 7, 1),
                "outbound": round(out_sum / 7, 1),
            })

    # -- summary stats -------------------------------------------------------
    total_crossings = len(crossings)
    n_days = max(len(daily_crossings), 1)
    avg_per_day = round(total_crossings / n_days, 1) if has_gate else 0

    # Before/after split using event_date.
    before_count = 0
    after_count = 0
    delta_pct: float | None = None
    if has_gate and config.event_date and global_start_ms is not None:
        epoch_s = global_start_ms / 1000.0
        event_dt = datetime.fromisoformat(config.event_date).replace(
            tzinfo=timezone.utc
        )
        event_s = (event_dt.timestamp() - epoch_s)
        for c in crossings:
            if c["timestamp_s"] < event_s:
                before_count += 1
            else:
                after_count += 1
        if before_count > 0:
            delta_pct = round(
                (after_count - before_count) / before_count * 100, 1
            )

    # -- sparkline (200-bucket histogram) ------------------------------------
    sparkline: list[int] = []
    sparkline_inbound: list[int] = []
    sparkline_outbound: list[int] = []
    n_buckets = 200
    if trips:
        t_min = min(t["timestamps"][0] for t in trips)
        t_max = max(t["timestamps"][-1] for t in trips)
        t_range = t_max - t_min
        if t_range > 0:
            bucket_size = t_range / n_buckets
            buckets = [0] * n_buckets
            for trip in trips:
                for ts in trip["timestamps"][::max(1, len(trip["timestamps"]) // 20)]:
                    idx = min(int((ts - t_min) / bucket_size), n_buckets - 1)
                    buckets[idx] += 1
            sparkline = buckets

    # Directional sparklines from crossings.
    if has_gate and sparkline:
        t_min = min(t["timestamps"][0] for t in trips)
        t_max = max(t["timestamps"][-1] for t in trips)
        t_range = t_max - t_min
        if t_range > 0:
            bucket_size = t_range / n_buckets
            in_buckets = [0] * n_buckets
            out_buckets = [0] * n_buckets
            for c in crossings:
                idx = min(int((c["timestamp_s"] - t_min) / bucket_size), n_buckets - 1)
                if idx >= 0:
                    if c["direction"] == "inbound":
                        in_buckets[idx] += 1
                    else:
                        out_buckets[idx] += 1
            sparkline_inbound = in_buckets
            sparkline_outbound = out_buckets

    # -- crossing times for gate pulse animation -----------------------------
    crossing_times: list[float] = [c["timestamp_s"] for c in crossings]

    # -- position counts -----------------------------------------------------
    total_positions = 0
    positions_before = 0
    positions_after = 0
    if positions_df is not None:
        total_positions = len(positions_df)
        if config.event_date and global_start_ms is not None:
            from neptune_ais.datasets.positions import Col as PosCol_
            epoch_s = global_start_ms / 1000.0
            event_dt = datetime.fromisoformat(config.event_date).replace(
                tzinfo=timezone.utc
            )
            positions_before = len(positions_df.filter(
                pl.col(PosCol_.TIMESTAMP) < event_dt
            ))
            positions_after = total_positions - positions_before

    summary = {
        "total_tracked": len(trip_mmsis),
        "total_transit": len(transit_mmsis),
        "total_crossings": total_crossings,
        "avg_per_day": avg_per_day,
        "delta_pct": delta_pct,
        "before_count": before_count,
        "after_count": after_count,
        "total_positions": total_positions,
        "positions_before": positions_before,
        "positions_after": positions_after,
    }

    return {
        "daily_crossings": daily_crossings,
        "rolling_7d": rolling_7d,
        "summary": summary,
        "vessel_index": {str(k): v for k, v in vessel_index.items()},
        "transit_mmsis": transit_mmsis,
        "flag_counts": sorted(
            [{"flag": k, "count": v} for k, v in flag_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        ),
        "type_counts": sorted(
            [{"type": k, "count": v} for k, v in type_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        ),
        "reversals": reversals,
        "sparkline": sparkline,
        "sparkline_inbound": sparkline_inbound,
        "sparkline_outbound": sparkline_outbound,
        "crossing_times": crossing_times,
    }


def generate_dashboard(
    tracks: pl.DataFrame | pl.LazyFrame,
    *,
    positions: pl.DataFrame | pl.LazyFrame | None = None,
    vessels: pl.DataFrame | pl.LazyFrame | None = None,
    events: pl.DataFrame | pl.LazyFrame | None = None,
    config: DashboardConfig,
    output: str = "dashboard.html",
    max_tracks: int | None = 1000,
) -> str:
    """Generate a self-contained maritime intelligence dashboard HTML file.

    Produces a standalone HTML file with animated vessel tracks,
    crossing analytics, filters, and interactive controls. No Python
    server or Jupyter required — open in any browser.

    Args:
        tracks: Tracks DataFrame with geometry columns
            (from ``Neptune.tracks(include_geometry=True)``).
        positions: Optional positions DataFrame. Enables the density
            heatmap layer.
        vessels: Optional vessels DataFrame. Enables vessel detail
            cards with name, type, flag, and dimensions.
        events: Optional events DataFrame. Shows event markers on
            the timeline.
        config: Dashboard configuration (title, gate, date range, etc.).
        output: Output file path. Default ``"dashboard.html"``.
        max_tracks: If set, downsample to at most this many tracks.
            Default 1000.

    Returns:
        The absolute path to the generated HTML file.
    """
    from neptune_ais._dashboard_template import render_dashboard
    from neptune_ais.derive.crossings import (
        detect_gate_crossings,
        detect_reversals,
    )

    tracks_df = _validate_track_geometry(tracks)
    total_track_count = len(tracks_df)

    # Capture the true epoch before downsampling so wall-clock timestamps
    # remain correct even if the earliest track is dropped by _sample().
    true_min_start = tracks_df[TrackCol.START_TIME].min()
    true_global_start_ms: int | None = (
        int(true_min_start.timestamp() * 1000)
        if true_min_start is not None
        else None
    )

    # Downsample if requested.
    tracks_df = _sample(tracks_df, max_tracks)

    trips, mmsi_to_color, max_time, _sample_start_ms = _build_trips(tracks_df)
    global_start_ms = true_global_start_ms
    if not trips:
        raise ValueError("No valid trip geometries found.")

    # Gate crossing detection.
    crossings: list[dict] = []
    reversals_list: list[dict] = []
    if config.gate is not None:
        crossings = detect_gate_crossings(trips, config.gate)
        reversals_list = detect_reversals(crossings)

    # Collect optional DataFrames.
    vessels_df = _collect(vessels) if vessels is not None else None
    positions_df = _collect(positions) if positions is not None else None
    events_df = _collect(events) if events is not None else None

    # Pre-compute analytics.
    analytics = _compute_dashboard_analytics(
        trips, crossings, reversals_list, vessels_df, positions_df, config,
        global_start_ms, total_track_count,
    )

    # Density data.
    density_data: list[dict] = []
    if positions_df is not None:
        density_df = prepare_density(positions_df, max_points=50_000)
        density_data = density_df.to_dicts()

    # Event data.
    event_data: list[dict] = []
    if events_df is not None:
        event_data = prepare_events(events_df).to_dicts()

    # Map view.
    center_lat = config.center_lat
    center_lon = config.center_lon
    zoom = config.zoom
    if center_lat is None or center_lon is None or zoom is None:
        auto_lat, auto_lon, auto_zoom = _auto_view(tracks_df)
        if center_lat is None:
            center_lat = auto_lat
        if center_lon is None:
            center_lon = auto_lon
        if zoom is None:
            zoom = auto_zoom

    # Gate line for map rendering.
    gate_coords: list[list[float]] | None = None
    if config.gate is not None:
        gate_coords = [
            [config.gate.point_a[1], config.gate.point_a[0]],
            [config.gate.point_b[1], config.gate.point_b[0]],
        ]

    # Infrastructure markers.
    infra_data = [
        {"name": p.name, "lat": p.lat, "lon": p.lon, "kind": p.kind}
        for p in config.infrastructure
    ]

    # Trip data with metadata for filtering.
    transit_mmsis = analytics["transit_mmsis"]
    vessel_index = analytics["vessel_index"]
    trip_data = []
    for t in trips:
        info = vessel_index.get(str(t["mmsi"]), {})
        trip_data.append({
            "path": t["path"],
            "timestamps": t["timestamps"],
            "color": t["color"],
            "mmsi": t["mmsi"],
            "isTransit": t["mmsi"] in transit_mmsis,
            "flag": info.get("flag", ""),
            "shipType": info.get("type", ""),
            "name": info.get("name", ""),
        })

    # Serialize analytics without vessel_index (embedded separately to
    # avoid doubling the payload — JS uses the top-level VESSEL_INDEX).
    analytics_slim = {
        k: v for k, v in analytics.items()
        if k not in ("vessel_index", "transit_mmsis")
    }

    # Assemble template data.
    data = {
        "title": config.title,
        "description": config.description,
        "has_gate": config.gate is not None and len(crossings) > 0,
        "gate_name": config.gate.name if config.gate else "",
        "gate_coords": _safe_json_embed(gate_coords),
        "event_date": config.event_date or "",
        "date_from": config.date_from,
        "date_to": config.date_to,
        "trips_json": _safe_json_embed(trip_data),
        "analytics_json": _safe_json_embed(analytics_slim),
        "density_json": _safe_json_embed(density_data),
        "events_json": _safe_json_embed(event_data),
        "infra_json": _safe_json_embed(infra_data),
        "vessel_index_json": _safe_json_embed(vessel_index),
        "crossing_times_json": _safe_json_embed(analytics.get("crossing_times", [])),
        "center_lat": center_lat,
        "center_lon": center_lon,
        "zoom": zoom,
        "pitch": config.pitch,
        "bearing": config.bearing,
        "max_time": max_time,
        "trail_length": config.trail_length,
        "default_speed": config.speed,
        "n_vessels": len(mmsi_to_color),
        "n_tracks": len(trips),
        "showing_subset": len(tracks_df) < total_track_count,
        "total_track_count": total_track_count,
        "global_start_ms": global_start_ms or 0,
    }

    html = render_dashboard(data)

    out_path = Path(output).resolve()
    out_path.write_text(html)
    return str(out_path)


# ---------------------------------------------------------------------------
# Timelapse corridor visualization
# ---------------------------------------------------------------------------


def prepare_timelapse(
    df: pl.DataFrame | pl.LazyFrame,
    *,
    vessels: pl.DataFrame | pl.LazyFrame | None = None,
    viewport: Viewport | None = None,
    max_points: int | None = 200_000,
    bin_interval_minutes: int = 60,
    color_by_type: bool = True,
) -> dict:
    """Prepare positions for timelapse corridor rendering.

    Groups positions into time bins and extracts minimal fields for
    compact JSON embedding. Optionally enriches ``ship_type`` from a
    vessels DataFrame.

    Args:
        df: Positions DataFrame or LazyFrame.
        vessels: Optional vessels DataFrame for ``ship_type`` enrichment
            via MMSI join.
        viewport: Optional bounding box to clip to.
        max_points: Max total points. Default 200K.
        bin_interval_minutes: Size of each time bin in minutes.
        color_by_type: Whether to assign type-based color indices.

    Returns:
        A dict with keys:

        - ``bins``: list of lists — each inner list contains
          ``[lat, lon, type_idx]`` triples for one time bin.
        - ``type_counts``: dict mapping type name → count.
        - ``cumul_vessels``: list of cumulative unique vessel counts
          (one per bin).
        - ``bin_timestamps_ms``: list of epoch-millis for each bin start.
        - ``center_lat``, ``center_lon``, ``zoom``: auto-computed view.
        - ``color_by_type``: whether type coloring is active (may be
          auto-disabled if too few typed positions).
        - ``palette``: list of ``[r, g, b]`` colors in type-index order.
        - ``type_names``: list of type names in index order.
    """
    result = _collect(df)

    if viewport is not None:
        result = _clip_positions(result, viewport)

    result = _sample(result, max_points)

    if len(result) == 0:
        return {
            "bins": [],
            "type_counts": {},
            "cumul_vessels": [],
            "bin_timestamps_ms": [],
            "center_lat": 0.0,
            "center_lon": 0.0,
            "zoom": 3,
            "color_by_type": False,
            "palette": [c for c in VESSEL_TYPE_PALETTE.values()],
            "type_names": _VESSEL_TYPE_ORDER[:],
        }

    # Enrich ship_type from vessels table if available.
    if vessels is not None:
        if PosCol.SHIP_TYPE not in result.columns:
            result = result.with_columns(
                pl.lit(None, dtype=pl.String).alias(PosCol.SHIP_TYPE),
            )
        vessels_df = _collect(vessels)
        if VesselCol.SHIP_TYPE in vessels_df.columns:
            type_lookup = vessels_df.select(
                pl.col(VesselCol.MMSI), pl.col(VesselCol.SHIP_TYPE).alias("_vtype"),
            ).unique(VesselCol.MMSI)
            result = result.join(type_lookup, on=PosCol.MMSI, how="left")
            result = result.with_columns(
                pl.coalesce(PosCol.SHIP_TYPE, "_vtype").alias(PosCol.SHIP_TYPE),
            ).drop("_vtype")

    # Categorize vessel types.
    ship_type_col = PosCol.SHIP_TYPE if PosCol.SHIP_TYPE in result.columns else None
    if ship_type_col is not None and color_by_type:
        type_series = result[ship_type_col].fill_null("").map_elements(
            _categorize_vessel_type, return_dtype=pl.String,
        )
    else:
        type_series = pl.Series("_vcat", ["other"] * len(result))

    result = result.with_columns(type_series.alias("_vcat"))

    # Auto-fallback: if >60% are "other", disable type coloring.
    other_frac = (result["_vcat"] == "other").sum() / max(len(result), 1)
    effective_color_by_type = color_by_type and other_frac <= 0.6

    if not effective_color_by_type:
        result = result.with_columns(pl.lit("other").alias("_vcat"))

    # Map type names to indices.
    type_to_idx = {name: i for i, name in enumerate(_VESSEL_TYPE_ORDER)}
    other_idx = type_to_idx["other"]

    result = result.with_columns(
        result["_vcat"].fill_null("other").map_elements(
            lambda v: type_to_idx.get(v, other_idx),
            return_dtype=pl.Int32,
        ).alias("_tidx"),
    )

    # Type counts.
    type_counts = dict(
        result.group_by("_vcat")
        .agg(pl.len().alias("n"))
        .iter_rows()
    )

    # Round coordinates.
    result = result.with_columns(
        pl.col(PosCol.LAT).round(4).alias(PosCol.LAT),
        pl.col(PosCol.LON).round(4).alias(PosCol.LON),
    )

    # Sort by timestamp and bin.
    result = result.sort(PosCol.TIMESTAMP)
    bin_dur = f"{bin_interval_minutes}m"
    result = result.with_columns(
        pl.col(PosCol.TIMESTAMP).dt.truncate(bin_dur).alias("_bin"),
    )

    # Group into bins.
    grouped = result.group_by("_bin", maintain_order=True).agg(
        pl.col(PosCol.LAT).alias("_lats"),
        pl.col(PosCol.LON).alias("_lons"),
        pl.col("_tidx").alias("_types"),
        pl.col(PosCol.MMSI).alias("_mmsis"),
    ).sort("_bin")

    # Build MMSI → index mapping for compact JS-side vessel tracking.
    unique_mmsis = result[PosCol.MMSI].unique().sort().to_list()
    mmsi_to_idx = {m: i for i, m in enumerate(unique_mmsis)}

    bins: list[list[list[float | int]]] = []
    bin_timestamps_ms: list[int] = []
    cumul_vessels: list[int] = []
    seen_mmsis: set[int] = set()

    for row in grouped.iter_rows(named=True):
        bin_ts = row["_bin"]
        lats = row["_lats"]
        lons = row["_lons"]
        types = row["_types"]
        mmsis = row["_mmsis"]

        bin_data: list[list[float | int]] = []
        for lat, lon, tidx, mmsi in zip(lats, lons, types, mmsis):
            bin_data.append([lat, lon, tidx, mmsi_to_idx.get(mmsi, 0)])

        bins.append(bin_data)
        bin_timestamps_ms.append(int(bin_ts.timestamp() * 1000))
        seen_mmsis.update(mmsis)
        cumul_vessels.append(len(seen_mmsis))

    # Auto view.
    center_lat, center_lon, zoom = _auto_view_positions(result)

    return {
        "bins": bins,
        "type_counts": type_counts,
        "cumul_vessels": cumul_vessels,
        "bin_timestamps_ms": bin_timestamps_ms,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "zoom": zoom,
        "color_by_type": effective_color_by_type,
        "palette": [VESSEL_TYPE_PALETTE[name] for name in _VESSEL_TYPE_ORDER],
        "type_names": _VESSEL_TYPE_ORDER[:],
    }


def generate_timelapse(
    positions: pl.DataFrame | pl.LazyFrame,
    *,
    vessels: pl.DataFrame | pl.LazyFrame | None = None,
    config: TimelapsConfig | None = None,
    output: str = "timelapse.html",
    max_points: int | None = 200_000,
    panels: list[dict] | None = None,
) -> str:
    """Generate a standalone HTML timelapse corridor visualization.

    Produces a self-contained HTML file showing AIS positions
    accumulating over time to reveal shipping corridors — similar to
    Kpler's AIS timelapse videos. Uses Canvas 2D with additive blending
    over a MapLibre GL dark basemap.

    For multi-panel mode, pass ``panels`` — a list of dicts each with
    ``"positions"`` (DataFrame), ``"config"`` (TimelapsConfig), and
    ``"label"`` (str) keys. The top-level ``positions``/``config`` are
    ignored when ``panels`` is set.

    Args:
        positions: Positions DataFrame or LazyFrame.
        vessels: Optional vessels DataFrame for ship-type enrichment.
        config: Timelapse configuration. Default settings if None.
        output: Output file path. Default ``"timelapse.html"``.
        max_points: Max positions per panel. Default 200K.
        panels: Optional list of panel specifications for multi-panel.

    Returns:
        The absolute path to the generated HTML file.
    """
    from neptune_ais._timelapse_template import render_timelapse

    if config is None:
        config = TimelapsConfig()

    if panels is not None:
        # Multi-panel mode.
        panels_data = []
        for p in panels:
            p_config = p.get("config") or TimelapsConfig()
            prep = prepare_timelapse(
                p["positions"],
                vessels=p.get("vessels", vessels),
                max_points=max_points,
                bin_interval_minutes=p_config.bin_interval_minutes,
                color_by_type=p_config.color_by_type,
            )
            center_lat = prep["center_lat"] if p_config.center_lat is None else p_config.center_lat
            center_lon = prep["center_lon"] if p_config.center_lon is None else p_config.center_lon
            zoom = prep["zoom"] if p_config.zoom is None else p_config.zoom
            panels_data.append({
                "label": p.get("label", ""),
                "bins_json": _safe_json_embed(prep["bins"]),
                "cumul_vessels_json": _safe_json_embed(prep["cumul_vessels"]),
                "bin_timestamps_ms_json": _safe_json_embed(
                    prep["bin_timestamps_ms"],
                ),
                "center_lat": center_lat,
                "center_lon": center_lon,
                "zoom": zoom,
                "n_bins": len(prep["bins"]),
                "config": {
                    "dot_radius": p_config.dot_radius,
                    "dot_alpha": p_config.dot_alpha,
                    "fade_factor": p_config.fade_factor,
                    "bloom": p_config.bloom,
                },
            })

        # Use global palette + type info from first panel.
        first_prep = prepare_timelapse(
            panels[0]["positions"],
            vessels=panels[0].get("vessels", vessels),
            max_points=1,
            color_by_type=config.color_by_type,
        )

        data = {
            "multi": True,
            "panels_json": _safe_json_embed(panels_data),
            "n_panels": len(panels_data),
            "title": config.title,
            "subtitle": config.subtitle,
            "palette_json": _safe_json_embed(first_prep["palette"]),
            "type_names_json": _safe_json_embed(first_prep["type_names"]),
            "color_by_type": first_prep["color_by_type"],
            "speed": config.speed,
            "layout": config.layout,
            "dot_radius": config.dot_radius,
            "dot_alpha": config.dot_alpha,
            "fade_factor": config.fade_factor,
            "bloom": config.bloom,
        }
    else:
        # Single-panel mode.
        prep = prepare_timelapse(
            positions,
            vessels=vessels,
            max_points=max_points,
            bin_interval_minutes=config.bin_interval_minutes,
            color_by_type=config.color_by_type,
        )

        if not prep["bins"]:
            raise ValueError("No positions found for timelapse.")

        center_lat = prep["center_lat"] if config.center_lat is None else config.center_lat
        center_lon = prep["center_lon"] if config.center_lon is None else config.center_lon
        zoom = prep["zoom"] if config.zoom is None else config.zoom

        data = {
            "multi": False,
            "title": config.title,
            "subtitle": config.subtitle,
            "bins_json": _safe_json_embed(prep["bins"]),
            "cumul_vessels_json": _safe_json_embed(prep["cumul_vessels"]),
            "bin_timestamps_ms_json": _safe_json_embed(
                prep["bin_timestamps_ms"],
            ),
            "type_counts_json": _safe_json_embed(prep["type_counts"]),
            "palette_json": _safe_json_embed(prep["palette"]),
            "type_names_json": _safe_json_embed(prep["type_names"]),
            "color_by_type": prep["color_by_type"],
            "center_lat": center_lat,
            "center_lon": center_lon,
            "zoom": zoom,
            "speed": config.speed,
            "dot_radius": config.dot_radius,
            "dot_alpha": config.dot_alpha,
            "fade_factor": config.fade_factor,
            "bloom": config.bloom,
        }

    html = render_timelapse(data)

    out_path = Path(output).resolve()
    out_path.write_text(html)
    return str(out_path)


# ---------------------------------------------------------------------------
# Port boundaries layer
# ---------------------------------------------------------------------------


def prepare_ports(
    port_index: PortIndex | None = None,
    *,
    derived_polygons: pl.DataFrame | None = None,
    viewport: Viewport | None = None,
    show_centers: bool = True,
    show_polygons: bool = True,
    min_confidence: float = 0.0,
) -> dict[str, pl.DataFrame]:
    """Prepare port boundaries and centers for map rendering.

    Returns a dict of DataFrames ready for map rendering:

    - ``"centers"``: Port center points with labels (for markers).
    - ``"polygons"``: Port boundary polygons with tier metadata
      (for polygon/outline rendering). Includes a ``polygon_source``
      column: ``"tier2"`` for derived polygons, ``"tier1_bbox"`` for
      bbox-only ports.

    Tier 2 derived polygons (from ``derive_port_polygons()``) are
    preferred when available. Remaining ports fall back to Tier 1
    bbox rectangles (requires ``shapely``).

    Color coding guidance (for rendering):

    - ``tier2`` + confidence >= 0.7 → solid fill (high confidence)
    - ``tier2`` + confidence < 0.7 → dashed outline (low confidence)
    - ``tier1_bbox`` → gray outline (no derived data)

    Args:
        port_index: Optional PortIndex. If None, loads the default
            singleton.
        derived_polygons: Optional DataFrame from
            ``derive_port_polygons()`` with ``port_name``,
            ``geometry_wkb``, ``confidence``, and bbox columns.
        viewport: Optional bounding box to clip to.
        show_centers: Include center points in the result.
        show_polygons: Include polygon boundaries in the result.
        min_confidence: Minimum confidence for derived polygons.
            Polygons below this threshold are excluded from Tier 2
            (the port falls back to Tier 1 bbox). Default 0.0.

    Returns:
        A dict with keys ``"centers"`` and ``"polygons"``, each a
        Polars DataFrame. Either key may be absent if the
        corresponding ``show_*`` flag is False.
    """
    if port_index is None:
        from neptune_ais.ports import index
        port_index = index()

    result: dict[str, pl.DataFrame] = {}

    ports = port_index.ports

    if viewport is not None:
        ports = _clip_positions(ports, viewport)

    if show_centers:
        result["centers"] = ports.select(
            "wpi_number", "name", "lat", "lon",
            "harbor_size", "unlocode", "country_code",
        )

    if show_polygons:
        result["polygons"] = _build_port_polygons(
            ports, derived_polygons,
            viewport=viewport,
            min_confidence=min_confidence,
        )

    return result


def _build_port_polygons(
    ports: pl.DataFrame,
    derived_polygons: pl.DataFrame | None,
    *,
    viewport: Viewport | None,
    min_confidence: float,
) -> pl.DataFrame:
    """Build a unified polygon DataFrame from Tier 2 + Tier 1 sources."""
    all_rows: list[pl.DataFrame] = []

    # Tier 2: derived polygons (preferred)
    tier2_names: set[str] = set()
    if derived_polygons is not None and len(derived_polygons) > 0:
        t2 = derived_polygons
        if min_confidence > 0:
            t2 = t2.filter(pl.col("confidence") >= min_confidence)

        if viewport is not None:
            t2 = _clip_tracks(t2, viewport)

        if len(t2) > 0:
            # Use the best zone per port (highest confidence)
            best = (
                t2.sort("confidence", descending=True)
                .unique(subset=["port_name"], keep="first")
            )
            tier2_names = set(best["port_name"].to_list())

            all_rows.append(
                best.select(
                    pl.col("port_name").alias("name"),
                    "geometry_wkb",
                    "confidence",
                    "bbox_west", "bbox_south", "bbox_east", "bbox_north",
                    pl.lit("tier2").alias("polygon_source"),
                )
            )

    # Tier 1: bbox rectangles for ports without derived polygons
    tier1_ports = ports.filter(~pl.col("name").is_in(tier2_names))

    if len(tier1_ports) > 0:
        tier1_wkb = _bbox_to_wkb_series(tier1_ports)
        if tier1_wkb is not None:
            all_rows.append(
                tier1_ports.with_columns(
                    tier1_wkb.alias("geometry_wkb"),
                    pl.lit(None).cast(pl.Float64).alias("confidence"),
                    pl.lit("tier1_bbox").alias("polygon_source"),
                ).select(
                    "name", "geometry_wkb", "confidence",
                    "bbox_west", "bbox_south", "bbox_east", "bbox_north",
                    "polygon_source",
                )
            )

    if not all_rows:
        return pl.DataFrame(
            schema={
                "name": pl.String,
                "geometry_wkb": pl.Binary,
                "confidence": pl.Float64,
                "bbox_west": pl.Float64,
                "bbox_south": pl.Float64,
                "bbox_east": pl.Float64,
                "bbox_north": pl.Float64,
                "polygon_source": pl.String,
            }
        )

    return pl.concat(all_rows, how="vertical_relaxed")


def _bbox_to_wkb_series(ports: pl.DataFrame) -> pl.Series | None:
    """Convert port bbox columns to WKB polygon Series (vectorized).

    Uses ``shapely.box()`` (Shapely 2.0+ vectorized ufunc) for
    fast batch conversion. Returns None if shapely is not available.
    """
    try:
        import shapely
    except ImportError:
        return None

    west = ports["bbox_west"].to_numpy()
    south = ports["bbox_south"].to_numpy()
    east = ports["bbox_east"].to_numpy()
    north = ports["bbox_north"].to_numpy()
    boxes = shapely.box(west, south, east, north)
    wkb_arr = shapely.to_wkb(boxes)
    return pl.Series("geometry_wkb", wkb_arr.tolist(), dtype=pl.Binary)
