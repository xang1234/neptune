"""Spatial math utilities for the port dictionary.

Pure-Python and Polars-expression helpers that work without shapely.
Provides haversine distance, bbox computation, and bbox containment
for the port query API and vectorized spatial join.

The existing codebase uses equirectangular distance (``derive/events.py``,
``derive/tracks.py``) which is fine for <100 km scales. This module uses
full haversine because ``ports.near()`` queries can span hundreds of km
where equirectangular error exceeds 1%.
"""

from __future__ import annotations

import math

import polars as pl

# Earth's mean radius in meters (WGS-84 volumetric mean).
EARTH_RADIUS_M = 6_371_008.8

# Degrees per km at the equator (used for bbox computation).
_DEG_PER_KM = 1.0 / 111.32


# ---------------------------------------------------------------------------
# Scalar haversine
# ---------------------------------------------------------------------------


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float,
) -> float:
    """Great-circle distance between two points in meters.

    Uses the haversine formula, which is numerically stable for
    all distances (unlike the spherical law of cosines).

    Args:
        lat1: Latitude of point 1 (WGS-84 degrees).
        lon1: Longitude of point 1 (WGS-84 degrees).
        lat2: Latitude of point 2 (WGS-84 degrees).
        lon2: Longitude of point 2 (WGS-84 degrees).

    Returns:
        Distance in meters.
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Vectorized haversine (Polars expressions)
# ---------------------------------------------------------------------------


def haversine_distance_polars(
    lat1: pl.Expr,
    lon1: pl.Expr,
    lat2: pl.Expr,
    lon2: pl.Expr,
) -> pl.Expr:
    """Great-circle distance as a Polars expression (meters).

    All four arguments are Polars expressions producing Float64
    columns in degrees. The result is a Float64 expression in meters.

    This runs entirely inside Polars' Rust engine — no Python
    per-row overhead. Use for vectorized nearest-port assignment
    and ``ports.near()`` on large DataFrames.

    Example::

        df.with_columns(
            haversine_distance_polars(
                pl.col("lat"), pl.col("lon"),
                pl.lit(51.9), pl.lit(4.5),
            ).alias("distance_m")
        )
    """
    lat1_r = lat1.radians()
    lat2_r = lat2.radians()
    dlat = (lat2 - lat1).radians()
    dlon = (lon2 - lon1).radians()

    a = (
        (dlat / 2).sin().pow(2)
        + lat1_r.cos() * lat2_r.cos() * (dlon / 2).sin().pow(2)
    )
    # Equivalent to 2 * atan2(sqrt(a), sqrt(1-a)) but using arcsin
    # which Polars supports as a method. Clamp to [0, 1] for safety.
    return pl.lit(EARTH_RADIUS_M) * 2 * a.clip(0.0, 1.0).sqrt().arcsin()


# ---------------------------------------------------------------------------
# Bbox computation
# ---------------------------------------------------------------------------


def bbox_from_center(
    lat: float, lon: float, radius_km: float,
) -> tuple[float, float, float, float]:
    """Compute ``(west, south, east, north)`` bbox from a center and radius.

    Adjusts longitude span for latitude (cos correction). Clamps to
    valid WGS-84 bounds. Does not handle antimeridian wrapping.

    Args:
        lat: Center latitude (degrees).
        lon: Center longitude (degrees).
        radius_km: Radius in kilometers.

    Returns:
        ``(west, south, east, north)`` bounding box.
    """
    lat_delta = radius_km * _DEG_PER_KM
    cos_lat = math.cos(math.radians(lat)) if abs(lat) < 89.9 else 0.01
    lon_delta = radius_km * _DEG_PER_KM / max(cos_lat, 0.01)

    return (
        max(lon - lon_delta, -180.0),
        max(lat - lat_delta, -90.0),
        min(lon + lon_delta, 180.0),
        min(lat + lat_delta, 90.0),
    )


# ---------------------------------------------------------------------------
# Vectorized bbox containment (Polars expression)
# ---------------------------------------------------------------------------


def bbox_contains_polars(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pl.Expr:
    """Polars expression that tests if ``(lat, lon)`` is inside a bbox.

    Returns a Boolean expression. Does not handle antimeridian wrapping.

    Args:
        west: Western bound (longitude).
        south: Southern bound (latitude).
        east: Eastern bound (longitude).
        north: Northern bound (latitude).
        lat_col: Name of the latitude column.
        lon_col: Name of the longitude column.

    Returns:
        A ``pl.Expr`` evaluating to ``True`` for rows inside the bbox.
    """
    return (
        (pl.col(lat_col) >= south)
        & (pl.col(lat_col) <= north)
        & (pl.col(lon_col) >= west)
        & (pl.col(lon_col) <= east)
    )
