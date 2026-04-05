"""Port data models — frozen dataclasses for the port dictionary.

Follows the existing Neptune pattern of frozen dataclasses for
lightweight reference objects (matching ``BoundaryRegion`` and
``BoundaryDataset`` in ``geometry/boundaries.py``).

These types are returned by single-item lookups (e.g.
``PortIndex.by_unlocode()``) and serve as the schema contract
between the Parquet data files and the query/integration layers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Port:
    """A port from the NGA World Port Index.

    Each instance represents one of ~3,800 ports worldwide with
    identity, location, classification, depths, and service flags.

    The ``wpi_number`` is the primary key (unique per port).
    Polygons are stored separately and joined by ``wpi_number``.

    Args:
        wpi_number: NGA WPI index number (unique identifier).
        name: Port name (e.g. ``"Rotterdam"``).
        alternate_name: Local or alternate name, or None.
        unlocode: UN/LOCODE (e.g. ``"NLRTM"``), or None.
        country_code: ISO 3166-1 alpha-2 (e.g. ``"NL"``).
        country_name: Full country name.
        lat: WGS-84 latitude of port center.
        lon: WGS-84 longitude of port center.
        bbox_west: Bounding box west (computed from harbor size).
        bbox_south: Bounding box south.
        bbox_east: Bounding box east.
        bbox_north: Bounding box north.
        harbor_size: ``"L"`` / ``"M"`` / ``"S"`` / ``"V"`` / None.
        harbor_type: Two-letter code (``"CN"`` coastal natural, etc.) or None.
        shelter_quality: ``"E"`` / ``"G"`` / ``"F"`` / ``"P"`` / ``"N"`` / None.
        channel_depth_m: Channel depth in meters, or None if unknown.
        anchorage_depth_m: Anchorage depth in meters, or None.
        cargo_pier_depth_m: Cargo pier depth in meters, or None.
        max_vessel_length_m: Maximum vessel length in meters, or None.
        tide_range_m: Tidal range in meters, or None.
        has_pilotage: Compulsory pilotage available.
        has_tugs: Tug assistance available.
        has_fuel: Fuel oil supply available.
        has_drydock: Dry dock facility present.
        has_cranes: Any crane type (fixed/mobile/floating) present.
        has_medical: Medical facilities available.
        has_derived_polygon: True if a Tier 2 AIS-derived polygon exists.
    """

    # Identity
    wpi_number: int
    name: str
    alternate_name: str | None
    unlocode: str | None
    country_code: str
    country_name: str

    # Location (WGS-84)
    lat: float
    lon: float

    # Bounding box (computed from harbor_size radius)
    bbox_west: float
    bbox_south: float
    bbox_east: float
    bbox_north: float

    # Classification
    harbor_size: str | None
    harbor_type: str | None
    shelter_quality: str | None

    # Depths (meters, None = unknown)
    channel_depth_m: float | None
    anchorage_depth_m: float | None
    cargo_pier_depth_m: float | None

    # Vessel limits
    max_vessel_length_m: float | None
    tide_range_m: float | None

    # Service flags
    has_pilotage: bool
    has_tugs: bool
    has_fuel: bool
    has_drydock: bool
    has_cranes: bool
    has_medical: bool

    # Tier 2 polygon status (set at load time)
    has_derived_polygon: bool

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box as ``(west, south, east, north)``."""
        return (self.bbox_west, self.bbox_south, self.bbox_east, self.bbox_north)


@dataclass(frozen=True)
class EEZRegion:
    """An Exclusive Economic Zone from MarineRegions.org.

    Args:
        mrgid: MarineRegions GeoObject ID (unique identifier).
        name: EEZ name (e.g. ``"Dutch Exclusive Economic Zone"``).
        sovereign: Sovereign state name.
        iso_3: ISO 3166-1 alpha-3 country code (e.g. ``"NLD"``).
        bbox_west: Bounding box west.
        bbox_south: Bounding box south.
        bbox_east: Bounding box east.
        bbox_north: Bounding box north.
        area_km2: EEZ area in square kilometers.
    """

    mrgid: int
    name: str
    sovereign: str
    iso_3: str
    bbox_west: float
    bbox_south: float
    bbox_east: float
    bbox_north: float
    area_km2: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box as ``(west, south, east, north)``."""
        return (self.bbox_west, self.bbox_south, self.bbox_east, self.bbox_north)
