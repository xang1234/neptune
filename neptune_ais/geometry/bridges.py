"""Bridges — optional GeoPandas and MovingPandas conversions.

Converts Neptune's native Polars/Arrow representations to GeoDataFrame
or MovingPandas TrajectoryCollection when the user opts in.

Requires: ``pip install neptune-ais[geo]`` (geopandas, movingpandas).
"""

from __future__ import annotations

from typing import Any

import polars as pl

from neptune_ais.geometry import _missing_geo_extra


def positions_to_geodataframe(
    df: pl.DataFrame | pl.LazyFrame,
) -> Any:
    """Convert a positions DataFrame to a GeoDataFrame with Point geometry.

    Creates a GeoDataFrame with a ``geometry`` column of Point(lon, lat)
    from the ``lat`` and ``lon`` columns.

    Args:
        df: Positions DataFrame or LazyFrame (will be collected).

    Returns:
        A ``geopandas.GeoDataFrame`` with WGS-84 CRS.

    Raises:
        ImportError: If geopandas or shapely is not installed.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        raise _missing_geo_extra("geopandas") from None

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    pdf = df.to_pandas()
    geometry = [Point(lon, lat) for lon, lat in zip(pdf["lon"], pdf["lat"])]
    return gpd.GeoDataFrame(pdf, geometry=geometry, crs="EPSG:4326")


def tracks_to_geodataframe(
    df: pl.DataFrame | pl.LazyFrame,
) -> Any:
    """Convert a tracks DataFrame to a GeoDataFrame.

    If the tracks have ``geometry_wkb`` (from ``include_geometry=True``),
    uses the WKB geometry. Otherwise, creates LineString geometry from
    bbox corners (a rough bounding rectangle, not the actual track).

    Args:
        df: Tracks DataFrame or LazyFrame (will be collected).

    Returns:
        A ``geopandas.GeoDataFrame`` with WGS-84 CRS.
    """
    try:
        import geopandas as gpd
        from shapely import wkb as shapely_wkb
        from shapely.geometry import box
    except ImportError:
        raise _missing_geo_extra("geopandas") from None

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    pdf = df.to_pandas()

    if "geometry_wkb" in pdf.columns and pdf["geometry_wkb"].notna().any():
        geometry = [
            shapely_wkb.loads(g) if g is not None else None
            for g in pdf["geometry_wkb"]
        ]
    else:
        # Fallback: bbox rectangle.
        geometry = [
            box(row["bbox_west"], row["bbox_south"], row["bbox_east"], row["bbox_north"])
            for _, row in pdf.iterrows()
        ]

    return gpd.GeoDataFrame(pdf, geometry=geometry, crs="EPSG:4326")


def tracks_to_movingpandas(
    df: pl.DataFrame | pl.LazyFrame,
    positions: pl.DataFrame | pl.LazyFrame | None = None,
) -> Any:
    """Convert tracks to a MovingPandas TrajectoryCollection.

    MovingPandas works with point-level data, not pre-aggregated tracks.
    This function requires the underlying positions DataFrame to build
    trajectories from the raw points, using ``track_id`` or ``_segment_id``
    as the trajectory identifier.

    If ``positions`` is not provided, this function cannot build
    trajectories and raises ValueError.

    Args:
        df: Tracks DataFrame (used for track metadata, not geometry).
        positions: The underlying positions data with a ``_segment_id``
            column or joinable via mmsi+timestamp.

    Returns:
        A ``movingpandas.TrajectoryCollection``.
    """
    if positions is None:
        raise ValueError(
            "positions DataFrame required for MovingPandas conversion. "
            "Pass the positions data used to derive the tracks."
        )

    try:
        import geopandas as gpd
        import movingpandas as mpd
        from shapely.geometry import Point
    except ImportError:
        raise _missing_geo_extra("movingpandas") from None

    if isinstance(positions, pl.LazyFrame):
        positions = positions.collect()
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    pdf = positions.to_pandas()

    # Build GeoDataFrame from positions.
    geometry = [Point(lon, lat) for lon, lat in zip(pdf["lon"], pdf["lat"])]
    gdf = gpd.GeoDataFrame(pdf, geometry=geometry, crs="EPSG:4326")

    # Set timestamp as index for MovingPandas.
    gdf = gdf.set_index("timestamp")

    # Use mmsi as trajectory ID (one trajectory per vessel).
    return mpd.TrajectoryCollection(gdf, traj_id_col="mmsi")
