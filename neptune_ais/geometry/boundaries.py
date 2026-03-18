"""Boundaries — EEZ, port, and region spatial reference data.

Provides lookups for Exclusive Economic Zones, port polygons, and
named maritime regions used by event derivation and spatial queries.

Boundary datasets are external reference data (e.g. from Natural Earth
or MarineRegions) — they are versioned separately from AIS data. The
``BoundaryDataset`` dataclass captures provenance so that derived events
can record which boundary version contributed to their inference.

This module defines a registry-based architecture:

1. ``BoundaryDataset`` — metadata + geometry for one boundary source.
2. ``BoundaryRegistry`` — loads, caches, and queries boundary datasets.
3. ``lookup_point`` — point-in-polygon query over a registry.

Requires: ``pip install neptune-ais[geo]`` (shapely) for polygon
containment. Without shapely, only bounding-box containment is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from neptune_ais.geometry import _missing_geo_extra


# ---------------------------------------------------------------------------
# Boundary dataset — metadata + geometry for one boundary source
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryRegion:
    """A single named region within a boundary dataset.

    Each region has a name, a bounding box for fast pre-filtering, and
    an optional shapely geometry for precise containment checks.

    Args:
        name: Region identifier (e.g. ``"USA_EEZ"``, ``"Port_Rotterdam"``).
        bbox: Bounding box as ``(west, south, east, north)``.
        geometry: Optional shapely geometry for precise containment.
            If None, only bounding-box containment is available.
    """

    name: str
    bbox: tuple[float, float, float, float]  # (west, south, east, north)
    geometry: Any = None  # shapely.Geometry when available


@dataclass(frozen=True)
class BoundaryDataset:
    """Metadata and regions for a boundary source.

    Captures enough provenance to explain which boundary data contributed
    to an event inference. The ``version`` and ``source_url`` fields
    flow into ``EventProvenance.upstream_datasets`` as ``"boundaries"``.

    Args:
        name: Dataset identifier (e.g. ``"natural_earth_eez"``).
        version: Version string (e.g. ``"5.1.0"``).
        source_url: Where the data was obtained.
        description: Human-readable description.
        regions: List of named regions in this dataset.

    Usage::

        ds = BoundaryDataset(
            name="world_ports",
            version="2024.1",
            source_url="https://example.com/ports.geojson",
            regions=[
                BoundaryRegion("Rotterdam", bbox=(3.8, 51.8, 4.2, 52.0)),
                BoundaryRegion("Singapore", bbox=(103.6, 1.1, 104.1, 1.5)),
            ],
        )
    """

    name: str
    version: str
    source_url: str = ""
    description: str = ""
    regions: tuple[BoundaryRegion, ...] = ()

    def provenance_tag(self) -> str:
        """Return a compact provenance string for this boundary dataset.

        Format: ``"<name>/<version>"``

        Example: ``"natural_earth_eez/5.1.0"``
        """
        return f"{self.name}/{self.version}"


# ---------------------------------------------------------------------------
# Boundary registry — loads, caches, and queries boundary datasets
# ---------------------------------------------------------------------------


class BoundaryRegistry:
    """Registry of boundary datasets for spatial lookups.

    Manages one or more ``BoundaryDataset`` instances and provides
    point-in-region queries across all registered datasets.

    Usage::

        registry = BoundaryRegistry()
        registry.register(eez_dataset)
        registry.register(ports_dataset)

        # Point lookup
        matches = registry.lookup(lat=51.9, lon=4.0)
        # → [("world_ports", "Rotterdam"), ...]

        # Provenance
        tags = registry.provenance_tags()
        # → ["natural_earth_eez/5.1.0", "world_ports/2024.1"]
    """

    def __init__(self) -> None:
        self._datasets: dict[str, BoundaryDataset] = {}

    def register(self, dataset: BoundaryDataset) -> None:
        """Register a boundary dataset.

        Args:
            dataset: The boundary dataset to register. Replaces any
                existing dataset with the same name.
        """
        self._datasets[dataset.name] = dataset

    def get(self, name: str) -> BoundaryDataset | None:
        """Return a registered dataset by name, or None."""
        return self._datasets.get(name)

    @property
    def datasets(self) -> list[BoundaryDataset]:
        """Return all registered datasets."""
        return list(self._datasets.values())

    def provenance_tags(self) -> list[str]:
        """Return provenance tags for all registered datasets.

        Returns:
            A list of ``"name/version"`` strings, sorted by name.
        """
        return sorted(ds.provenance_tag() for ds in self._datasets.values())

    def lookup(self, lat: float, lon: float) -> list[tuple[str, str]]:
        """Find all regions containing a point.

        First checks bounding-box containment (fast, no dependencies).
        If a region has a shapely geometry, uses precise point-in-polygon
        containment. Otherwise, bounding-box containment is the final
        answer.

        Args:
            lat: Latitude (WGS-84).
            lon: Longitude (WGS-84).

        Returns:
            A list of ``(dataset_name, region_name)`` tuples for all
            regions that contain the point.
        """
        matches: list[tuple[str, str]] = []
        for ds in self._datasets.values():
            for region in ds.regions:
                if _bbox_contains(region.bbox, lat, lon):
                    if region.geometry is not None:
                        if _point_in_geometry(region.geometry, lat, lon):
                            matches.append((ds.name, region.name))
                    else:
                        # No precise geometry — bbox is the best we have.
                        matches.append((ds.name, region.name))
        return matches

    def lookup_column(
        self,
        df: pl.DataFrame,
        dataset_name: str,
        *,
        lat_col: str = "lat",
        lon_col: str = "lon",
    ) -> pl.Series:
        """Add a region-name column by looking up each row's (lat, lon).

        Returns a String Series with the **first matching** region name
        for each row, or None if no region in the named dataset contains
        the point. Unlike ``lookup()`` which returns all matches, this
        method returns at most one region per row (first match wins).

        This is a row-level operation suitable for small-to-medium
        DataFrames. For large datasets, consider spatial indexing.

        Args:
            df: DataFrame with lat/lon columns.
            dataset_name: Which registered dataset to look up against.
            lat_col: Name of the latitude column.
            lon_col: Name of the longitude column.

        Returns:
            A ``pl.Series`` of String type with one entry per row.
        """
        ds = self._datasets.get(dataset_name)
        if ds is None:
            return pl.Series("region", [None] * len(df), dtype=pl.String)

        lats = df[lat_col].to_list()
        lons = df[lon_col].to_list()

        results: list[str | None] = []
        for lat, lon in zip(lats, lons):
            if lat is None or lon is None:
                results.append(None)
                continue
            found = None
            for region in ds.regions:
                if _bbox_contains(region.bbox, lat, lon):
                    if region.geometry is not None:
                        if _point_in_geometry(region.geometry, lat, lon):
                            found = region.name
                            break
                    else:
                        found = region.name
                        break
            results.append(found)

        return pl.Series("region", results, dtype=pl.String)


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------


def _bbox_contains(
    bbox: tuple[float, float, float, float], lat: float, lon: float
) -> bool:
    """Check if a point is within a bounding box.

    Args:
        bbox: ``(west, south, east, north)`` in WGS-84.
        lat: Point latitude.
        lon: Point longitude.

    Returns:
        True if the point is inside the bbox (inclusive).
    """
    west, south, east, north = bbox
    return south <= lat <= north and west <= lon <= east


_shapely_Point: Any = None


def _point_in_geometry(geometry: Any, lat: float, lon: float) -> bool:
    """Check if a point is within a shapely geometry.

    Args:
        geometry: A shapely geometry object.
        lat: Point latitude.
        lon: Point longitude.

    Returns:
        True if the point is inside the geometry.

    Raises:
        ImportError: If shapely is not installed.
    """
    global _shapely_Point
    if _shapely_Point is None:
        try:
            from shapely.geometry import Point
        except ImportError:
            raise _missing_geo_extra("shapely") from None
        _shapely_Point = Point

    return geometry.contains(_shapely_Point(lon, lat))
