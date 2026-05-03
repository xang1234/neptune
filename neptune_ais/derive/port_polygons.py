"""Port polygon derivation — auto-derive port boundaries from AIS data.

Infers port polygon boundaries from vessel position data using a
position-first algorithm: assign low-speed positions to the nearest
known port, group by port, compute concave hulls, and split into
spatial zones.

This is the Tier 2 polygon source in the World Port Index system.
It produces empirical port footprints that improve as more AIS data
is ingested — the same approach used by commercial providers
(MarineTraffic, Spire, Kpler) but open-source.

Algorithm
---------
1. Filter to low-speed positions (SOG <= ``max_speed_knots``).
2. Assign each position to the nearest WPI port within
   ``search_radius_km`` (via ``vectorized_port_lookup``).
3. Group by assigned port.
4. For each port group with >= ``min_positions`` points, compute a
   concave hull (``shapely.concave_hull``).
5. Split the point cloud into spatial clusters (zones) using
   leader-based clustering with ``cluster_separation_m``.
6. Compute per-zone concave hull and statistics.
7. Score confidence using a multi-factor model (temporal span ×
   vessel diversity × coverage density).

Heuristic assumptions
---------------------
- SOG <= 3.0 knots is "stopped" (matches ``PortCallConfig``).
- A 20 km search radius captures positions in the port approach
  and anchorage, not just the berths.
- ``concave_ratio=0.3`` balances following major indentations
  without fragmenting the polygon.
- Zones are numbered (``zone_0``, ``zone_1``), not auto-labeled.
  Use ``suggest_zone_type()`` for heuristic suggestions.

Known limits
------------
- Requires ``shapely >= 2.0`` (the ``[geo]`` extra).
- Antimeridian-crossing ports may produce incorrect polygons.
- Concave hulls on unprojected WGS-84 coordinates have slight
  distortion at high latitudes (negligible at port scale <50 km).
- Zone identification uses simple leader clustering, not DBSCAN.

Requires: ``pip install neptune-ais[geo]``
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from neptune_ais.ports._index import PortIndex

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PortPolygonConfig:
    """Configuration for AIS-derived port polygon generation.

    All parameters participate in the config hash, which determines
    cache validity. Changing any parameter triggers recomputation.

    Args:
        search_radius_km: How far from a WPI port center to look
            for low-speed positions. Default 20.0 km.
        max_speed_knots: SOG threshold — positions above this speed
            are excluded. Default 3.0 (matches ``PortCallConfig``).
        min_positions: Minimum low-speed positions required to
            derive a polygon for a port. Default 50.
        concave_ratio: Ratio for ``shapely.concave_hull()``.
            0.0 = convex hull, 1.0 = maximally concave.
            Default 0.3 — good balance for port shapes.
        cluster_separation_m: If two groups of points are more than
            this distance apart, they are separate zones. Default 2000 m.
        min_cluster_points: Minimum points per zone cluster.
            Default 20.
    """

    search_radius_km: float = 20.0
    max_speed_knots: float = 3.0
    min_positions: int = 50
    concave_ratio: float = 0.3
    cluster_separation_m: float = 2000.0
    min_cluster_points: int = 20

    def config_hash(self) -> str:
        """Deterministic hash of all config parameters (12-char hex)."""
        key = (
            f"radius={self.search_radius_km},"
            f"speed={self.max_speed_knots},"
            f"min_pos={self.min_positions},"
            f"ratio={self.concave_ratio},"
            f"sep={self.cluster_separation_m},"
            f"min_cluster={self.min_cluster_points}"
        )
        return hashlib.sha1(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Stage 1: Position-first nearest-port assignment
# ---------------------------------------------------------------------------


def assign_positions_to_ports(
    positions: pl.DataFrame,
    port_index: PortIndex,
    *,
    config: PortPolygonConfig | None = None,
) -> pl.DataFrame:
    """Assign each low-speed position to its nearest WPI port.

    This is the first stage of the derivation pipeline:

    1. Filter to low-speed positions (SOG <= ``max_speed_knots``).
    2. Use ``vectorized_port_lookup()`` to assign each position to the
       port whose bbox contains it (smallest bbox wins for overlaps).
    3. Drop positions with no port assignment (open sea).

    Args:
        positions: Positions DataFrame with ``lat``, ``lon``, ``sog``,
            ``mmsi``, and ``timestamp`` columns.
        port_index: PortIndex with built-in port data.
        config: Derivation configuration. Uses defaults if None.

    Returns:
        A DataFrame with all original columns plus
        ``assigned_port`` (String) — the WPI port name.
        Only rows assigned to a port are returned.
    """
    from neptune_ais.ports._registry_bridge import vectorized_port_lookup

    if config is None:
        config = PortPolygonConfig()

    n_input = len(positions)
    if n_input == 0:
        return positions.with_columns(
            pl.lit(None).cast(pl.String).alias("assigned_port"),
        )

    # Step 1: Filter to low-speed positions
    slow = positions.filter(pl.col("sog") <= config.max_speed_knots)
    n_slow = len(slow)
    logger.info(
        "Low-speed filter: %d → %d positions (SOG <= %.1f kn)",
        n_input, n_slow, config.max_speed_knots,
    )

    if n_slow == 0:
        return positions.head(0).with_columns(
            pl.lit(None).cast(pl.String).alias("assigned_port"),
        )

    # Step 2: Build port boundaries for lookup (only ports within search radius)
    # Use the full port bounds with lat/lon so H3 path is available
    port_bounds = port_index.ports.select(
        "wpi_number", "name", "lat", "lon",
        "bbox_west", "bbox_south", "bbox_east", "bbox_north",
    )

    # Assign ports
    port_names = vectorized_port_lookup(slow, port_bounds)

    # Step 3: Add assignment and filter to assigned positions
    result = slow.with_columns(port_names.alias("assigned_port"))
    result = result.filter(pl.col("assigned_port").is_not_null())

    n_assigned = len(result)
    n_ports = result["assigned_port"].n_unique()
    logger.info(
        "Port assignment: %d of %d low-speed positions assigned to %d ports",
        n_assigned, n_slow, n_ports,
    )

    return result


# ---------------------------------------------------------------------------
# Stage 2: Concave hull computation per port
# ---------------------------------------------------------------------------


def compute_port_hulls(
    assigned: pl.DataFrame,
    port_index: PortIndex | None = None,
    *,
    config: PortPolygonConfig | None = None,
) -> pl.DataFrame:
    """Compute a concave hull polygon for each port's assigned positions.

    Groups ``assigned`` by ``assigned_port``, skips ports with fewer
    than ``min_positions`` points, and computes
    ``shapely.concave_hull(MultiPoint, ratio=concave_ratio)``
    for each group.

    Requires ``shapely >= 2.0`` (the ``[geo]`` extra).

    Args:
        assigned: DataFrame from ``assign_positions_to_ports()`` with
            ``lat``, ``lon``, ``assigned_port``, ``mmsi``, ``timestamp``.
        port_index: Optional PortIndex for resolving ``wpi_number``
            from port names. If provided, output includes ``wpi_number``.
        config: Derivation configuration. Uses defaults if None.

    Returns:
        A DataFrame with one row per port:
            ``port_name``, ``wpi_number`` (if port_index provided),
            ``geometry_wkb`` (Binary),
            ``bbox_west/south/east/north`` (Float64),
            ``position_count`` (Int64), ``vessel_count`` (Int64),
            ``first_seen`` (Datetime), ``last_seen`` (Datetime).
        Ports with degenerate hulls (< 3 points → not a Polygon)
        are excluded.
    """
    try:
        import numpy as np
        import shapely
        from shapely.geometry import MultiPoint
    except ImportError:
        raise ImportError(
            "Port polygon derivation requires shapely >= 2.0 and numpy. "
            "Install with: pip install neptune-ais[geo]"
        ) from None

    if config is None:
        config = PortPolygonConfig()

    if len(assigned) == 0:
        return _empty_hull_df()

    # Aggregate statistics per port
    port_stats = assigned.group_by("assigned_port").agg(
        pl.col("lat").alias("_lats"),
        pl.col("lon").alias("_lons"),
        pl.len().cast(pl.Int64).alias("position_count"),
        pl.col("mmsi").n_unique().cast(pl.Int64).alias("vessel_count"),
        pl.col("timestamp").min().alias("first_seen"),
        pl.col("timestamp").max().alias("last_seen"),
    )

    # Filter to ports with enough positions
    port_stats = port_stats.filter(
        pl.col("position_count") >= config.min_positions
    )

    if len(port_stats) == 0:
        return _empty_hull_df()

    # Build name → wpi_number lookup if port_index provided
    name_to_wpi: dict[str, int] = {}
    if port_index is not None:
        for r in port_index.ports.select("name", "wpi_number").iter_rows():
            name_to_wpi[r[0]] = r[1]

    rows: list[dict] = []
    for row in port_stats.iter_rows(named=True):
        lats = row["_lats"]
        lons = row["_lons"]
        coords = np.column_stack((lons, lats))  # shapely uses (x, y) = (lon, lat)
        points = MultiPoint(coords)

        hull = shapely.concave_hull(points, ratio=config.concave_ratio)

        # Skip degenerate results (Point, LineString, etc.)
        if hull.geom_type not in ("Polygon", "MultiPolygon"):
            logger.debug(
                "Skipping %s: hull is %s (need Polygon)",
                row["assigned_port"], hull.geom_type,
            )
            continue

        bounds = hull.bounds  # (minx, miny, maxx, maxy) = (west, south, east, north)
        port_name = row["assigned_port"]
        entry: dict = {
            "port_name": port_name,
        }
        if name_to_wpi:
            entry["wpi_number"] = name_to_wpi.get(port_name)
        entry.update({
            "geometry_wkb": shapely.to_wkb(hull),
            "bbox_west": bounds[0],
            "bbox_south": bounds[1],
            "bbox_east": bounds[2],
            "bbox_north": bounds[3],
            "position_count": row["position_count"],
            "vessel_count": row["vessel_count"],
            "first_seen": row["first_seen"],
            "last_seen": row["last_seen"],
        })
        rows.append(entry)

    if not rows:
        return _empty_hull_df()

    logger.info(
        "Computed concave hulls for %d ports (ratio=%.2f)",
        len(rows), config.concave_ratio,
    )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 3: Zone splitting within ports
# ---------------------------------------------------------------------------


def split_port_zones(
    assigned: pl.DataFrame,
    port_index: PortIndex | None = None,
    *,
    config: PortPolygonConfig | None = None,
) -> pl.DataFrame:
    """Split each port's positions into spatial zones and compute per-zone hulls.

    Uses leader-based clustering: for each position, assign to the
    nearest existing cluster centroid; start a new cluster if the
    distance to all centroids exceeds ``cluster_separation_m``.
    Zones are numbered ``zone_0``, ``zone_1``, etc. — NOT auto-labeled.

    Args:
        assigned: DataFrame from ``assign_positions_to_ports()``.
        port_index: Optional PortIndex for resolving ``wpi_number``.
        config: Derivation configuration. Uses defaults if None.

    Returns:
        A DataFrame with one row per zone:
            ``port_name``, ``wpi_number`` (if port_index provided),
            ``zone_id`` (String), ``geometry_wkb`` (Binary),
            ``bbox_west/south/east/north`` (Float64),
            ``center_lat``, ``center_lon`` (Float64),
            ``position_count`` (Int64), ``vessel_count`` (Int64),
            ``mean_sog`` (Float64),
            ``temporal_span_days`` (Int64),
            ``first_seen``, ``last_seen`` (Datetime).
    """
    try:
        import numpy as np
        import shapely
        from shapely.geometry import MultiPoint
    except ImportError:
        raise ImportError(
            "Port polygon derivation requires shapely >= 2.0 and numpy. "
            "Install with: pip install neptune-ais[geo]"
        ) from None

    if config is None:
        config = PortPolygonConfig()

    if len(assigned) == 0:
        return _empty_zone_df()

    # Build name → wpi_number lookup
    name_to_wpi: dict[str, int] = {}
    if port_index is not None:
        for r in port_index.ports.select("name", "wpi_number").iter_rows():
            name_to_wpi[r[0]] = r[1]

    has_destinations = "resolved_port_name" in assigned.columns

    # Group positions by port
    agg_exprs = [
        pl.col("lat").alias("_lats"),
        pl.col("lon").alias("_lons"),
        pl.col("sog").alias("_sogs"),
        pl.col("mmsi").alias("_mmsis"),
        pl.col("timestamp").alias("_timestamps"),
    ]
    if has_destinations:
        agg_exprs.append(pl.col("resolved_port_name").alias("_resolved"))

    port_groups = assigned.group_by("assigned_port").agg(agg_exprs)

    all_zones: list[dict] = []

    for port_row in port_groups.iter_rows(named=True):
        port_name = port_row["assigned_port"]
        lats = port_row["_lats"]
        lons = port_row["_lons"]
        sogs = port_row["_sogs"]
        mmsis = port_row["_mmsis"]
        timestamps = port_row["_timestamps"]
        resolved = port_row["_resolved"] if has_destinations else None

        if len(lats) < config.min_positions:
            continue

        # Leader-based clustering
        clusters = _leader_cluster(
            lats, lons, config.cluster_separation_m,
        )

        # Build per-zone stats and hulls
        zone_idx = 0
        for cluster_indices in clusters:
            if len(cluster_indices) < config.min_cluster_points:
                continue

            c_lats = [lats[i] for i in cluster_indices]
            c_lons = [lons[i] for i in cluster_indices]
            c_sogs = [sogs[i] for i in cluster_indices]
            c_mmsis = [mmsis[i] for i in cluster_indices]
            c_ts = [timestamps[i] for i in cluster_indices]

            # Concave hull
            coords = np.column_stack((c_lons, c_lats))
            points = MultiPoint(coords)
            hull = shapely.concave_hull(points, ratio=config.concave_ratio)

            if hull.geom_type not in ("Polygon", "MultiPolygon"):
                continue

            bounds = hull.bounds
            center_lat = sum(c_lats) / len(c_lats)
            center_lon = sum(c_lons) / len(c_lons)
            unique_vessels = len(set(c_mmsis))
            ts_sorted = sorted(t for t in c_ts if t is not None)
            span_days = 0.0
            if len(ts_sorted) >= 2:
                delta = ts_sorted[-1] - ts_sorted[0]
                span_days = delta.total_seconds() / 86400.0

            dest_match_rate: float | None = None
            if resolved is not None:
                matches = sum(1 for i in cluster_indices if resolved[i] == port_name)
                dest_match_rate = matches / len(cluster_indices)

            entry: dict = {"port_name": port_name}
            if name_to_wpi:
                entry["wpi_number"] = name_to_wpi.get(port_name)

            # Compute area in km² (approximate, from bbox)
            area_km2 = _bbox_area_km2(bounds[0], bounds[1], bounds[2], bounds[3])

            confidence = compute_confidence(
                position_count=len(cluster_indices),
                vessel_count=unique_vessels,
                temporal_span_days=span_days,
                area_km2=area_km2,
                destination_match_rate=dest_match_rate or 0.0,
            )

            entry.update({
                "zone_id": f"zone_{zone_idx}",
                "geometry_wkb": shapely.to_wkb(hull),
                "bbox_west": bounds[0],
                "bbox_south": bounds[1],
                "bbox_east": bounds[2],
                "bbox_north": bounds[3],
                "center_lat": center_lat,
                "center_lon": center_lon,
                "position_count": len(cluster_indices),
                "vessel_count": unique_vessels,
                "mean_sog": sum(c_sogs) / len(c_sogs),
                "temporal_span_days": span_days,
                "destination_match_rate": dest_match_rate,
                "confidence": confidence,
                "first_seen": ts_sorted[0] if ts_sorted else None,
                "last_seen": ts_sorted[-1] if ts_sorted else None,
            })
            all_zones.append(entry)
            zone_idx += 1

    if not all_zones:
        return _empty_zone_df()

    logger.info("Split %d zones across ports", len(all_zones))
    return pl.DataFrame(all_zones)


def _leader_cluster(
    lats: list[float],
    lons: list[float],
    separation_m: float,
) -> list[list[int]]:
    """Leader-based spatial clustering. O(n × k).

    For each point, find the nearest existing cluster centroid.
    If distance > ``separation_m``, start a new cluster.
    Returns a list of clusters, each a list of point indices.
    """
    from neptune_ais.ports._spatial import haversine_distance

    clusters: list[list[int]] = []
    centroids: list[tuple[float, float]] = []  # (lat, lon) per cluster

    for i, (lat, lon) in enumerate(zip(lats, lons)):
        if lat is None or lon is None:
            continue

        # Find nearest centroid
        best_idx = -1
        best_dist = float("inf")
        for ci, (clat, clon) in enumerate(centroids):
            d = haversine_distance(lat, lon, clat, clon)
            if d < best_dist:
                best_dist = d
                best_idx = ci

        if best_idx >= 0 and best_dist <= separation_m:
            clusters[best_idx].append(i)
            # Update centroid (running mean)
            n = len(clusters[best_idx])
            old_lat, old_lon = centroids[best_idx]
            centroids[best_idx] = (
                old_lat + (lat - old_lat) / n,
                old_lon + (lon - old_lon) / n,
            )
        else:
            clusters.append([i])
            centroids.append((lat, lon))

    # Sort clusters by size descending (zone_0 = largest)
    clusters.sort(key=len, reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# Multi-factor confidence model
# ---------------------------------------------------------------------------


def compute_confidence(
    *,
    position_count: int,
    vessel_count: int,
    temporal_span_days: float,
    area_km2: float,
    destination_match_rate: float = 0.0,
) -> float:
    """Multi-factor confidence score for a derived polygon or zone.

    All three factors must be high for confidence to be high:

    - **temporal_factor**: ``min(temporal_span_days / 90, 1.0)`` —
      ~3 months of data needed for full confidence.
    - **vessel_factor**: ``min(unique_vessels / 30, 1.0)`` —
      30+ different vessels confirms a shared operational area.
    - **density_factor**: ``min(positions / (area_km2 * 100), 1.0)`` —
      100 positions per km² is good spatial resolution.

    The ``min()`` is deliberate: a polygon with 100 vessels but only
    1 day of data still gets low confidence.

    An optional **destination boost** adds up to +0.1 when AIS
    destination fields confirm the spatial assignment: vessels
    declaring they are going to this port validates the polygon.

    Returns:
        A float in [0.0, 1.0].
    """
    temporal = min(temporal_span_days / 90.0, 1.0) if temporal_span_days > 0 else 0.0
    vessel = min(vessel_count / 30.0, 1.0)
    density = min(position_count / max(area_km2 * 100.0, 1.0), 1.0)
    return min(min(temporal, vessel, density) + destination_match_rate * 0.1, 1.0)


def _bbox_area_km2(
    west: float, south: float, east: float, north: float,
) -> float:
    """Approximate area of a bbox in km² (equirectangular)."""
    import math

    mid_lat = (south + north) / 2.0
    cos_lat = math.cos(math.radians(mid_lat))
    width_km = (east - west) * 111.32 * cos_lat
    height_km = (north - south) * 111.32
    return abs(width_km * height_km)


# ---------------------------------------------------------------------------
# Zone type suggestion (heuristic, NOT baked into zone data)
# ---------------------------------------------------------------------------


def suggest_zone_type(
    zone: dict,
) -> tuple[str, float]:
    """Suggest a zone type label with a confidence score.

    This is a heuristic suggestion — NOT an authoritative label.
    Zones are always numbered (``zone_0``, ``zone_1``), never
    auto-labeled. Users should validate suggestions visually.

    Heuristic signals (weighted):
    - Low mean_sog + high vessel_count + near center → terminal
    - Low mean_sog + far from center + fewer vessels → anchorage
    - Higher mean_sog → approach/fairway

    Args:
        zone: A dict with keys: ``mean_sog``, ``vessel_count``,
            ``position_count``, ``center_lat``, ``center_lon``.

    Returns:
        ``(suggested_type, suggestion_confidence)`` where type is
        one of ``"terminal"``, ``"anchorage"``, ``"approach"``,
        ``"unknown"`` and confidence is 0.0–1.0.
    """
    mean_sog = zone.get("mean_sog", 0.0) or 0.0
    vessel_count = zone.get("vessel_count", 0) or 0
    position_count = zone.get("position_count", 0) or 0

    # Simple heuristic scoring
    if mean_sog > 2.0:
        return ("approach", 0.4)

    if vessel_count >= 10 and position_count >= 100:
        return ("terminal", 0.5)

    if vessel_count < 10 and mean_sog < 1.0:
        return ("anchorage", 0.4)

    return ("unknown", 0.2)


def _empty_zone_df() -> pl.DataFrame:
    """Return an empty DataFrame with the zone output schema."""
    return pl.DataFrame(
        schema={
            "port_name": pl.String,
            "zone_id": pl.String,
            "geometry_wkb": pl.Binary,
            "bbox_west": pl.Float64,
            "bbox_south": pl.Float64,
            "bbox_east": pl.Float64,
            "bbox_north": pl.Float64,
            "center_lat": pl.Float64,
            "center_lon": pl.Float64,
            "position_count": pl.Int64,
            "vessel_count": pl.Int64,
            "mean_sog": pl.Float64,
            "temporal_span_days": pl.Float64,
            "destination_match_rate": pl.Float64,
            "confidence": pl.Float64,
            "first_seen": pl.Datetime("us", "UTC"),
            "last_seen": pl.Datetime("us", "UTC"),
        }
    )


def _empty_hull_df() -> pl.DataFrame:
    """Return an empty DataFrame with the hull output schema."""
    return pl.DataFrame(
        schema={
            "port_name": pl.String,
            "geometry_wkb": pl.Binary,
            "bbox_west": pl.Float64,
            "bbox_south": pl.Float64,
            "bbox_east": pl.Float64,
            "bbox_north": pl.Float64,
            "position_count": pl.Int64,
            "vessel_count": pl.Int64,
            "first_seen": pl.Datetime("us", "UTC"),
            "last_seen": pl.Datetime("us", "UTC"),
        }
    )


# ---------------------------------------------------------------------------
# Top-level pipeline: derive + persist
# ---------------------------------------------------------------------------


def derive_port_polygons(
    positions: pl.DataFrame,
    port_index: PortIndex,
    *,
    config: PortPolygonConfig | None = None,
    output_dir: str | Path | None = None,
    resolved_destinations: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Run the full port polygon derivation pipeline.

    Orchestrates all stages:
    1. Assign low-speed positions to nearest ports.
    2. Split into spatial zones with concave hulls.
    3. Score confidence (boosted by destination match if available).
    4. Persist to Parquet (if ``output_dir`` provided).

    Args:
        positions: Positions DataFrame with ``lat``, ``lon``, ``sog``,
            ``mmsi``, ``timestamp`` columns.
        port_index: PortIndex with built-in port data.
        config: Derivation configuration. Uses defaults if None.
        output_dir: Directory to write the output Parquet file.
            If None, returns the DataFrame without persisting.
        resolved_destinations: Optional DataFrame from
            ``resolve_destination_column()`` with a
            ``resolved_port_name`` column (same row order as
            ``positions``). When provided, zones where vessels'
            AIS destinations match the assigned port receive a
            soft confidence boost.

    Returns:
        A DataFrame with per-zone rows including ``port_name``,
        ``wpi_number``, ``zone_id``, ``geometry_wkb``, bbox,
        statistics, ``destination_match_rate``, and ``confidence``.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    if config is None:
        config = PortPolygonConfig()

    if resolved_destinations is not None:
        if "resolved_port_name" not in resolved_destinations.columns:
            raise ValueError(
                "resolved_destinations must contain a 'resolved_port_name' column"
            )
        if len(resolved_destinations) != len(positions):
            raise ValueError(
                f"resolved_destinations has {len(resolved_destinations)} rows "
                f"but positions has {len(positions)} — must be equal"
            )
        positions = positions.with_columns(
            resolved_destinations["resolved_port_name"],
        )

    # Stage 1: assign positions to ports
    assigned = assign_positions_to_ports(positions, port_index, config=config)

    def _finalize(zones: pl.DataFrame) -> pl.DataFrame:
        """Add metadata columns for consistent output schema."""
        if "wpi_number" not in zones.columns:
            zones = zones.with_columns(
                pl.lit(None).cast(pl.Int64).alias("wpi_number"),
            )
        return zones.with_columns(
            pl.lit(config.config_hash()).alias("config_hash"),
            pl.lit(datetime.now(timezone.utc)).alias("derived_at"),
        )

    if len(assigned) == 0:
        logger.info("No positions assigned to ports — nothing to derive")
        return _finalize(_empty_zone_df())

    # Stage 2+3: split zones with hulls + confidence
    zones = split_port_zones(assigned, port_index, config=config)

    if len(zones) == 0:
        logger.info("No zones produced — insufficient data")
        return _finalize(_empty_zone_df())

    zones = _finalize(zones)

    logger.info(
        "Derived %d zones across %d ports (config %s)",
        len(zones),
        zones["port_name"].n_unique(),
        config.config_hash(),
    )

    # Persist if output_dir provided
    if output_dir is not None:
        out_path = Path(output_dir) / "port_polygons" / f"{config.config_hash()}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        zones.write_parquet(out_path, compression="zstd", compression_level=9)
        logger.info("Persisted to %s", out_path)

    return zones
