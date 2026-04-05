"""Ports — built-in World Port Index and EEZ reference data.

Quick start::

    from neptune_ais import ports

    ports.search("Rotterdam")           # full-text search
    ports.near(51.9, 4.5)               # nearby ports with distances
    ports.by_unlocode("NLRTM")          # → Port dataclass
    ports.by_country("NL")              # all Dutch ports

Subsystem boundary
------------------
**Owns:**
- Built-in port, UNLOCODE, and EEZ reference data (Parquet files).
- Query API: search, near, by_country, by_unlocode, etc.
- User overlay merging logic.
- BoundaryRegistry bridge (auto-registration).

**Delegates to:**
- ``neptune_ais.geometry.boundaries`` — BoundaryDataset/BoundaryRegistry types.
- ``neptune_ais.derive.events`` — event detectors consume boundaries.

**Rule:** The ports module must not import from ``adapters``, ``storage``,
``catalog``, ``cli``, or ``api``. It is a self-contained reference data
provider. Core code (no [geo]) can use basic lookups (lat/lon, bbox).
Polygon containment requires [geo].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neptune_ais.ports._index import PortIndex
from neptune_ais.ports._models import EEZRegion, Port

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    "EEZRegion",
    "Port",
    "PortIndex",
    "by_country",
    "by_unlocode",
    "index",
    "near",
    "search",
]

# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_default_index: PortIndex | None = None


def index() -> PortIndex:
    """Return the default (singleton) ``PortIndex`` with built-in data.

    The index is created on first call and reused thereafter.
    No disk I/O occurs until the first query method is called on it.
    """
    global _default_index
    if _default_index is None:
        _default_index = PortIndex()
    return _default_index


# ---------------------------------------------------------------------------
# Module-level convenience functions (delegate to singleton)
# ---------------------------------------------------------------------------


def search(query: str, *, limit: int = 20) -> pl.DataFrame:
    """Full-text search over port names, alternate names, and UNLOCODEs.

    See :meth:`PortIndex.search` for details.
    """
    return index().search(query, limit=limit)


def near(
    lat: float,
    lon: float,
    *,
    radius_km: float = 50.0,
    limit: int = 10,
) -> pl.DataFrame:
    """Find ports within ``radius_km`` of a point, sorted by distance.

    See :meth:`PortIndex.near` for details.
    """
    return index().near(lat, lon, radius_km=radius_km, limit=limit)


def by_unlocode(code: str) -> Port | None:
    """Lookup a single port by UN/LOCODE (e.g. ``'NLRTM'``).

    See :meth:`PortIndex.by_unlocode` for details.
    """
    return index().by_unlocode(code)


def by_country(country_code: str) -> pl.DataFrame:
    """All ports in a country (ISO alpha-2 code).

    See :meth:`PortIndex.by_country` for details.
    """
    return index().by_country(country_code)
