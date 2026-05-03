"""Registry bridge — vectorized spatial join and BoundaryDataset builder.

Provides a fast alternative to ``BoundaryRegistry.lookup_column()``
for assigning positions to ports. The existing ``lookup_column()`` is
O(rows × regions) in Python; this module uses Polars vectorized
expressions to achieve the same result orders of magnitude faster.

Also provides ``build_boundary_dataset()`` which assembles a
``BoundaryDataset`` from the best available polygon per port
(Tier 2 derived > Tier 1 bbox).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from neptune_ais.geometry.boundaries import BoundaryDataset
    from neptune_ais.ports._index import PortIndex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vectorized port lookup
# ---------------------------------------------------------------------------


def vectorized_port_lookup(
    positions: pl.DataFrame,
    port_boundaries: pl.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pl.Series:
    """Assign each position to the nearest containing port.

    For each position row, finds the first port whose bbox contains
    the position's (lat, lon). If multiple ports match, the one with
    the smallest bbox area wins (most specific match).

    Uses H3 spatial indexing when available (via ``[geo]`` extra)
    for O(N) performance. Falls back to a broadcast loop otherwise.

    Args:
        positions: DataFrame with lat/lon columns.
        port_boundaries: DataFrame with columns: ``name``,
            ``bbox_west``, ``bbox_south``, ``bbox_east``, ``bbox_north``.
            Must also have ``lat``, ``lon`` columns for H3 path.
        lat_col: Name of the latitude column in ``positions``.
        lon_col: Name of the longitude column in ``positions``.

    Returns:
        A String Series (length = ``len(positions)``) with the port
        name for each position, or None if no port contains the point.
    """
    n_positions = len(positions)
    if n_positions == 0 or len(port_boundaries) == 0:
        return pl.Series("port_name", [None] * n_positions, dtype=pl.String)

    # Choose strategy based on data characteristics.
    # H3 is faster for global/wide-area data (many candidate ports after
    # extent filter). Broadcast is faster for regional data (few candidates).
    has_h3 = False
    try:
        import h3  # noqa: F401
        has_h3 = "lat" in port_boundaries.columns and "lon" in port_boundaries.columns
    except ImportError:
        pass

    # Pre-filter ports by positions' bounding box (shared by both paths)
    pos_south = positions[lat_col].min()
    pos_north = positions[lat_col].max()
    pos_west = positions[lon_col].min()
    pos_east = positions[lon_col].max()

    candidates = port_boundaries.filter(
        (pl.col("bbox_north") >= pos_south)
        & (pl.col("bbox_south") <= pos_north)
        & (pl.col("bbox_east") >= pos_west)
        & (pl.col("bbox_west") <= pos_east)
    )

    if len(candidates) == 0:
        return pl.Series("port_name", [None] * n_positions, dtype=pl.String)

    # H3 wins when many ports remain after extent filtering
    if has_h3 and len(candidates) > 200:
        return _h3_port_lookup(
            positions, candidates, lat_col=lat_col, lon_col=lon_col,
        )

    return _broadcast_port_lookup(
        positions, candidates, lat_col=lat_col, lon_col=lon_col,
    )


# ---------------------------------------------------------------------------
# H3 spatial index path
# ---------------------------------------------------------------------------

# Resolution 4: ~22 km edge-to-edge. Largest port bbox (harbor_size L)
# is 10 km radius = 20 km diameter, fits within one cell + ring(1).
_H3_RESOLUTION = 4


def _h3_port_lookup(
    positions: pl.DataFrame,
    port_boundaries: pl.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pl.Series:
    """H3-accelerated port lookup. Requires h3 package."""
    import h3

    n_positions = len(positions)

    # Step 1: Build port → H3 cell index (port center + disk(1) for coverage)
    # Store only the fields needed for bbox refinement (not all 27 columns)
    cell_to_ports: dict[str, list[tuple[str, float, float, float, float]]] = {}
    for row in port_boundaries.iter_rows(named=True):
        center_cell = h3.latlng_to_cell(row["lat"], row["lon"], _H3_RESOLUTION)
        entry = (row["name"], row["bbox_south"], row["bbox_north"], row["bbox_west"], row["bbox_east"])
        for cell in h3.grid_disk(center_cell, 1):
            cell_to_ports.setdefault(cell, []).append(entry)

    logger.debug(
        "H3 index: %d ports → %d cell entries (res %d)",
        len(port_boundaries), sum(len(v) for v in cell_to_ports.values()), _H3_RESOLUTION,
    )

    # Step 2: For each position, find its H3 cell → candidate ports → bbox refine
    pos_lats = positions[lat_col].to_list()
    pos_lons = positions[lon_col].to_list()

    result: list[str | None] = [None] * n_positions
    matched_count = 0

    for i, (lat, lon) in enumerate(zip(pos_lats, pos_lons)):
        if lat is None or lon is None:
            continue

        cell = h3.latlng_to_cell(lat, lon, _H3_RESOLUTION)
        candidates = cell_to_ports.get(cell)
        if not candidates:
            continue

        # Refine: bbox containment, prefer smallest bbox (most specific)
        best_name = None
        best_area = float("inf")
        for name, south, north, west, east in candidates:
            if south <= lat <= north and west <= lon <= east:
                area = (east - west) * (north - south)
                if area < best_area:
                    best_area = area
                    best_name = name

        if best_name is not None:
            result[i] = best_name
            matched_count += 1

    logger.debug(
        "H3 port lookup: %d positions, %d matched (res %d)",
        n_positions, matched_count, _H3_RESOLUTION,
    )
    return pl.Series("port_name", result, dtype=pl.String)


# ---------------------------------------------------------------------------
# Broadcast fallback (no H3)
# ---------------------------------------------------------------------------


def _broadcast_port_lookup(
    positions: pl.DataFrame,
    port_boundaries: pl.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pl.Series:
    """Broadcast-loop port lookup. No external deps beyond Polars.

    Expects ``port_boundaries`` to be pre-filtered by the caller
    (extent filter already applied in ``vectorized_port_lookup``).
    """
    n_positions = len(positions)

    boundaries = port_boundaries.with_columns(
        (
            (pl.col("bbox_east") - pl.col("bbox_west"))
            * (pl.col("bbox_north") - pl.col("bbox_south"))
        ).alias("_bbox_area")
    ).sort("_bbox_area")

    lats = positions[lat_col]
    lons = positions[lon_col]

    result: list[str | None] = [None] * n_positions
    unmatched = pl.Series("_idx", range(n_positions), dtype=pl.UInt32)

    for row in boundaries.iter_rows(named=True):
        if len(unmatched) == 0:
            break

        name = row["name"]
        west = row["bbox_west"]
        south = row["bbox_south"]
        east = row["bbox_east"]
        north = row["bbox_north"]

        sub_lats = lats.gather(unmatched)
        sub_lons = lons.gather(unmatched)

        inside = (
            (sub_lats >= south)
            & (sub_lats <= north)
            & (sub_lons >= west)
            & (sub_lons <= east)
        )

        if inside.any():
            matched_orig = unmatched.filter(inside)
            for idx in matched_orig.to_list():
                result[idx] = name
            unmatched = unmatched.filter(~inside)

    logger.debug(
        "Broadcast port lookup: %d positions, %d ports, %d matched",
        n_positions, len(boundaries), n_positions - len(unmatched),
    )
    return pl.Series("port_name", result, dtype=pl.String)


# ---------------------------------------------------------------------------
# BoundaryDataset builder with tiered selection
# ---------------------------------------------------------------------------


def build_boundary_dataset(
    port_index: PortIndex,
    *,
    derived_polygons: pl.DataFrame | None = None,
    name: str = "builtin_ports",
    version: str = "wpi_2024",
) -> BoundaryDataset:
    """Build a ``BoundaryDataset`` using the best available polygon per port.

    Selection order per port:

    1. **Tier 2** derived polygon (if available with ``confidence >= 0.7``).
    2. **Tier 1** bbox from harbor size (always available).

    The resulting ``BoundaryDataset`` can be registered with
    ``BoundaryRegistry`` and consumed by ``detect_port_calls()``
    without any changes to the detector.

    Args:
        port_index: The port index to build from.
        derived_polygons: Optional DataFrame of Tier 2 derived polygons
            with columns: ``wpi_number``, ``port_name``, ``geometry_wkb``,
            ``confidence``. Only polygons with ``confidence >= 0.7``
            are used.
        name: Dataset name for provenance.
        version: Dataset version for provenance.

    Returns:
        A ``BoundaryDataset`` instance.
    """
    from neptune_ais.geometry.boundaries import BoundaryDataset, BoundaryRegion

    # Build a set of WPI numbers that have high-confidence derived polygons
    derived_ports: dict[int, bytes] = {}
    if derived_polygons is not None and len(derived_polygons) > 0:
        dp = derived_polygons
        # Filter by confidence if the column exists; treat all as high-confidence otherwise
        if "confidence" in dp.columns:
            dp = dp.filter(pl.col("confidence") >= 0.7)
        # Key by wpi_number if available, else by port_name → wpi_number lookup
        if "wpi_number" in dp.columns:
            for row in dp.iter_rows(named=True):
                wpi = row["wpi_number"]
                if wpi is not None and wpi not in derived_ports:
                    derived_ports[wpi] = row["geometry_wkb"]
        elif "port_name" in dp.columns:
            # Resolve port names to WPI numbers via port_index
            name_to_wpi = {
                r[0]: r[1]
                for r in port_index.ports.select("name", "wpi_number").iter_rows()
            }
            for row in dp.iter_rows(named=True):
                wpi = name_to_wpi.get(row["port_name"])
                if wpi is not None and wpi not in derived_ports:
                    derived_ports[wpi] = row["geometry_wkb"]
        if derived_ports:
            logger.info(
                "Using %d Tier 2 derived polygons",
                len(derived_ports),
            )

    # Build BoundaryRegions: prefer derived polygon, fall back to bbox
    regions: list[BoundaryRegion] = []
    for row in port_index.ports.iter_rows(named=True):
        wpi = row["wpi_number"]
        bbox = (row["bbox_west"], row["bbox_south"],
                row["bbox_east"], row["bbox_north"])

        geometry = None
        if wpi in derived_ports:
            try:
                import shapely
                import shapely.errors
                geometry = shapely.from_wkb(derived_ports[wpi])
            except ImportError:
                pass  # No shapely — fall back to bbox-only
            except shapely.errors.ShapelyError:
                logger.warning("Invalid WKB for port %d, using bbox", wpi)

        regions.append(BoundaryRegion(
            name=row["name"],
            bbox=bbox,
            geometry=geometry,
        ))

    prov_suffix = ""
    if derived_ports:
        prov_suffix = f"+derived_{len(derived_ports)}"

    return BoundaryDataset(
        name=name,
        version=f"{version}{prov_suffix}",
        source_url="https://msi.nga.mil/Publications/WPI",
        description=(
            f"NGA World Port Index — {len(regions)} ports"
            f" ({len(derived_ports)} with derived polygons)"
        ),
        regions=tuple(regions),
    )
