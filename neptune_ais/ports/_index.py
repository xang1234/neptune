"""PortIndex — queryable index over the built-in World Port Index.

Lazy-loaded: no disk I/O until the first query method is called.
Thread-safe for reads after initialization.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from neptune_ais.geometry.boundaries import BoundaryDataset

from neptune_ais.ports._models import EEZRegion, Port
from neptune_ais.ports._spatial import (
    bbox_contains_polars,
    bbox_from_center,
    haversine_distance,
    haversine_distance_polars,
)


class PortIndex:
    """Queryable index over the built-in World Port Index.

    All query methods return Polars DataFrames for efficiency.
    Single-item lookups return ``Port`` or ``EEZRegion`` dataclass
    instances for convenience.

    Usage::

        idx = PortIndex()
        idx.search("Rotterdam")          # → DataFrame
        idx.near(51.9, 4.5)              # → DataFrame with distance_m
        idx.by_unlocode("NLRTM")         # → Port
    """

    def __init__(self, *, user_overlays: list[Path] | None = None) -> None:
        self._user_overlays = user_overlays or []
        self._ports_df: pl.DataFrame | None = None
        self._unlocode_df: pl.DataFrame | None = None
        self._eez_df: pl.DataFrame | None = None

    # --- Lazy loading ---

    @property
    def ports(self) -> pl.DataFrame:
        """The full ports DataFrame (lazy-loaded on first access)."""
        if self._ports_df is None:
            from neptune_ais.ports._loader import load_ports

            self._ports_df = load_ports(user_overlays=self._user_overlays)
        return self._ports_df

    @property
    def unlocodes(self) -> pl.DataFrame:
        """The full UNLOCODE DataFrame (lazy-loaded on first access)."""
        if self._unlocode_df is None:
            from neptune_ais.ports._loader import load_unlocodes

            self._unlocode_df = load_unlocodes()
        return self._unlocode_df

    @property
    def eez(self) -> pl.DataFrame:
        """The full EEZ metadata DataFrame (lazy-loaded on first access)."""
        if self._eez_df is None:
            from neptune_ais.ports._loader import load_eez_meta

            self._eez_df = load_eez_meta()
        return self._eez_df

    # --- Core queries ---

    def search(self, query: str, *, limit: int = 20) -> pl.DataFrame:
        """Full-text search over port names, alternate names, and UNLOCODEs.

        Matches are scored: exact UNLOCODE > starts-with > contains.
        Results are sorted by relevance then harbor size.

        Args:
            query: Search string (case-insensitive).
            limit: Maximum results to return.

        Returns:
            A DataFrame of matching ports, sorted by relevance.
        """
        q = query.strip().upper()
        if not q:
            return self.ports.head(0)  # empty query → empty result

        df = self.ports

        # Score: 3 = exact UNLOCODE, 2 = name starts with, 1 = name/alt contains
        score = (
            pl.when(pl.col("unlocode") == q)
            .then(3)
            .when(pl.col("name").str.to_uppercase().str.starts_with(q))
            .then(2)
            .when(
                pl.col("name").str.to_uppercase().str.contains(q, literal=True)
                | pl.col("alternate_name")
                .str.to_uppercase()
                .str.contains(q, literal=True)
                | pl.col("unlocode").str.contains(q, literal=True)
            )
            .then(1)
            .otherwise(0)
        )

        # Harbor size order for tiebreaking (L > M > S > V > null)
        size_rank = (
            pl.when(pl.col("harbor_size") == "L").then(0)
            .when(pl.col("harbor_size") == "M").then(1)
            .when(pl.col("harbor_size") == "S").then(2)
            .when(pl.col("harbor_size") == "V").then(3)
            .otherwise(4)
        )

        return (
            df.with_columns(
                score.alias("_score"),
                size_rank.alias("_size_rank"),
            )
            .filter(pl.col("_score") > 0)
            .sort(["_score", "_size_rank"], descending=[True, False])
            .drop(["_score", "_size_rank"])
            .head(limit)
        )

    def near(
        self,
        lat: float,
        lon: float,
        *,
        radius_km: float = 50.0,
        limit: int = 10,
    ) -> pl.DataFrame:
        """Find ports within ``radius_km`` of a point, sorted by distance.

        Uses bbox pre-filter then haversine refinement.

        Args:
            lat: Latitude (WGS-84 degrees).
            lon: Longitude (WGS-84 degrees).
            radius_km: Search radius in kilometers.
            limit: Maximum results to return.

        Returns:
            A DataFrame with an extra ``distance_km`` column, sorted
            by distance ascending.
        """
        # Bbox pre-filter
        west, south, east, north = bbox_from_center(lat, lon, radius_km)
        df = self.ports.filter(
            bbox_contains_polars(west, south, east, north)
        )

        # Haversine refinement
        radius_m = radius_km * 1000.0
        df = df.with_columns(
            haversine_distance_polars(
                pl.col("lat"), pl.col("lon"),
                pl.lit(lat), pl.lit(lon),
            ).alias("distance_m")
        )
        return (
            df.filter(pl.col("distance_m") <= radius_m)
            .with_columns(
                (pl.col("distance_m") / 1000.0).alias("distance_km"),
            )
            .drop("distance_m")
            .sort("distance_km")
            .head(limit)
        )

    def by_unlocode(self, code: str) -> Port | None:
        """Lookup a single port by UN/LOCODE (e.g. ``'NLRTM'``).

        Returns None if not found.
        """
        code = code.strip().upper().replace(" ", "")
        row = self.ports.filter(pl.col("unlocode") == code)
        if len(row) == 0:
            return None
        return Port(**row.row(0, named=True))

    def by_country(self, country_code: str) -> pl.DataFrame:
        """All ports in a country (ISO alpha-2 code).

        Args:
            country_code: Two-letter ISO code (e.g. ``'NL'``).

        Returns:
            A DataFrame of ports in the country, sorted by harbor size.
        """
        cc = country_code.strip().upper()
        return self.ports.filter(pl.col("country_code") == cc).sort(
            "harbor_size", nulls_last=True,
        )

    def by_wpi(self, wpi_number: int) -> Port | None:
        """Lookup a single port by WPI index number.

        Returns None if not found.
        """
        row = self.ports.filter(pl.col("wpi_number") == wpi_number)
        if len(row) == 0:
            return None
        return Port(**row.row(0, named=True))

    def get(self, identifier: str | int) -> Port | None:
        """Smart lookup: tries UNLOCODE, then WPI number, then name search.

        Args:
            identifier: UNLOCODE string, WPI number (int or numeric
                string), or port name fragment.

        Returns:
            The best matching ``Port``, or None.
        """
        # Try as int (WPI number)
        if isinstance(identifier, int):
            return self.by_wpi(identifier)

        s = str(identifier).strip()

        # Try as WPI number string
        if s.isdigit():
            return self.by_wpi(int(s))

        # Try as UNLOCODE (5 chars, alphanumeric)
        if len(s) <= 6:
            result = self.by_unlocode(s)
            if result is not None:
                return result

        # Fall back to name search
        matches = self.search(s, limit=1)
        if len(matches) > 0:
            return Port(**matches.row(0, named=True))

        return None

    # --- Filtered views ---

    def large_ports(self) -> pl.DataFrame:
        """Ports with ``harbor_size = 'L'`` (large)."""
        return self.ports.filter(pl.col("harbor_size") == "L")

    def with_facilities(self, **kwargs: bool) -> pl.DataFrame:
        """Filter ports by facility flags.

        Example::

            idx.with_facilities(has_drydock=True, has_fuel=True)

        Args:
            **kwargs: Facility flag names and required values.

        Returns:
            A DataFrame of matching ports.
        """
        df = self.ports
        for col, val in kwargs.items():
            if col not in df.columns:
                raise ValueError(f"Unknown facility column: {col!r}")
            df = df.filter(pl.col(col) == val)
        return df

    def in_bbox(
        self,
        west: float,
        south: float,
        east: float,
        north: float,
    ) -> pl.DataFrame:
        """All ports within a bounding box.

        Args:
            west: Western bound (longitude).
            south: Southern bound (latitude).
            east: Eastern bound (longitude).
            north: Northern bound (latitude).

        Returns:
            A DataFrame of ports inside the bbox.
        """
        return self.ports.filter(
            bbox_contains_polars(west, south, east, north)
        )

    # --- EEZ queries ---

    def eez_for_point(self, lat: float, lon: float) -> EEZRegion | None:
        """Which EEZ contains this point (bbox match).

        For precise polygon containment, use the BoundaryRegistry
        with EEZ polygons loaded.

        Returns:
            The first matching ``EEZRegion``, or None.
        """
        matches = self.eez.filter(
            (pl.col("bbox_south") <= lat)
            & (pl.col("bbox_north") >= lat)
            & (pl.col("bbox_west") <= lon)
            & (pl.col("bbox_east") >= lon)
        )
        if len(matches) == 0:
            return None
        return EEZRegion(**matches.row(0, named=True))

    def eez_by_country(self, iso_3: str) -> list[EEZRegion]:
        """All EEZ regions for a country (ISO alpha-3 code).

        Args:
            iso_3: Three-letter ISO code (e.g. ``'NLD'``).

        Returns:
            A list of ``EEZRegion`` instances.
        """
        iso = iso_3.strip().upper()
        matches = self.eez.filter(pl.col("iso_3") == iso)
        return [EEZRegion(**row) for row in matches.iter_rows(named=True)]

    # --- BoundaryRegistry integration ---

    @staticmethod
    def _df_to_boundary_dataset(
        df: pl.DataFrame,
        *,
        name: str,
        version: str,
        source_url: str,
        description: str,
    ) -> BoundaryDataset:
        """Build a ``BoundaryDataset`` from a DataFrame with bbox columns."""
        from neptune_ais.geometry.boundaries import BoundaryDataset, BoundaryRegion

        regions = tuple(
            BoundaryRegion(
                name=row["name"],
                bbox=(row["bbox_west"], row["bbox_south"],
                      row["bbox_east"], row["bbox_north"]),
            )
            for row in df.iter_rows(named=True)
        )
        return BoundaryDataset(
            name=name, version=version,
            source_url=source_url, description=description,
            regions=regions,
        )

    def to_boundary_dataset(
        self,
        *,
        name: str = "builtin_ports",
        version: str = "wpi_2024",
    ) -> BoundaryDataset:
        """Convert ports to a ``BoundaryDataset`` for ``BoundaryRegistry``.

        Each port becomes a ``BoundaryRegion`` with its bbox.
        This enables ``detect_port_calls()`` to use built-in port data
        without manual boundary loading.
        """
        return self._df_to_boundary_dataset(
            self.ports, name=name, version=version,
            source_url="https://msi.nga.mil/Publications/WPI",
            description="NGA World Port Index — built-in port boundaries",
        )

    def eez_to_boundary_dataset(
        self,
        *,
        name: str = "builtin_eez",
        version: str = "eez_v12",
    ) -> BoundaryDataset:
        """Convert EEZ metadata to a ``BoundaryDataset``.

        Uses bbox-only matching. For polygon containment, load EEZ
        polygon data and attach shapely geometries to the regions.
        """
        return self._df_to_boundary_dataset(
            self.eez, name=name, version=version,
            source_url="https://www.marineregions.org/eez.php",
            description="MarineRegions.org EEZ v12 — bbox-only boundaries",
        )
