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

from neptune_ais.datasets.events import Col as EventCol
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
# Animated vessel replay — standalone HTML with deck.gl TripsLayer
# ---------------------------------------------------------------------------

# Vessel color palette — 10 distinct colors for up to 10 vessels.
_PALETTE = [
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


def _decode_wkb_linestring(wkb: bytes) -> list[list[float]]:
    """Decode a WKB LineString to [[lon, lat], ...].

    Handles the little-endian WKB format produced by _encode_wkb_linestring
    in derive/tracks.py. Uses bulk struct.unpack for efficiency.
    """
    import struct

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
    import json
    from pathlib import Path

    tracks = _collect(tracks)

    # Validate required columns.
    required = {TrackCol.GEOMETRY_WKB, TrackCol.TIMESTAMP_OFFSETS_MS}
    if not required.issubset(tracks.columns):
        raise ValueError(
            "Tracks must have geometry_wkb and timestamp_offsets_ms columns. "
            "Use Neptune.tracks(include_geometry=True) to generate them."
        )

    # Filter to tracks with geometry.
    tracks = tracks.filter(
        pl.col(TrackCol.GEOMETRY_WKB).is_not_null()
        & pl.col(TrackCol.TIMESTAMP_OFFSETS_MS).is_not_null()
    )

    if len(tracks) == 0:
        raise ValueError("No tracks with geometry found.")

    # Precompute global start time so all tracks share the same time origin.
    min_start = tracks[TrackCol.START_TIME].min()
    global_start_ms = (
        int(min_start.timestamp() * 1000) if min_start is not None else None
    )

    # Build trip data for deck.gl in a single pass.
    trips = []
    mmsi_to_color: dict[int, list[int]] = {}
    color_idx = 0

    for row in tracks.iter_rows(named=True):
        wkb = row[TrackCol.GEOMETRY_WKB]
        offsets_ms = row[TrackCol.TIMESTAMP_OFFSETS_MS]
        mmsi = row[TrackCol.MMSI]

        coords = _decode_wkb_linestring(wkb)
        if len(coords) < 2 or not offsets_ms or len(offsets_ms) != len(coords):
            continue

        # Assign color per vessel.
        if mmsi not in mmsi_to_color:
            mmsi_to_color[mmsi] = _PALETTE[color_idx % len(_PALETTE)]
            color_idx += 1

        # Compute timestamps as seconds from global start.
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

    if not trips:
        raise ValueError("No valid trip geometries found.")

    max_time = max(t["timestamps"][-1] for t in trips)

    # Compute map center and zoom from track bounding boxes.
    center_lon = float(
        (tracks[TrackCol.BBOX_WEST].mean() + tracks[TrackCol.BBOX_EAST].mean()) / 2
    )
    center_lat = float(
        (tracks[TrackCol.BBOX_SOUTH].mean() + tracks[TrackCol.BBOX_NORTH].mean()) / 2
    )
    lon_spread = float(tracks[TrackCol.BBOX_EAST].max() - tracks[TrackCol.BBOX_WEST].min())
    lat_spread = float(tracks[TrackCol.BBOX_NORTH].max() - tracks[TrackCol.BBOX_SOUTH].min())
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

    # Build vessel legend entries.
    legend_entries = [
        {"mmsi": mmsi, "color": color}
        for mmsi, color in mmsi_to_color.items()
    ]

    # Serialize trip data (drop mmsi field — not needed by deck.gl).
    trips_json = json.dumps(
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
        legend_json=json.dumps(legend_entries),
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
