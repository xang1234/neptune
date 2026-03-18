"""Geometry — optional spatial support and conversion bridges.

EEZ/port/region boundary lookups and optional GeoPandas/MovingPandas
conversions. These dependencies are optional — the core package does
not require them.

Subsystem boundary
------------------
**Owns:**
- Spatial reference data management: loading, caching, and querying EEZ
  polygons, port boundaries, and named maritime regions (``boundaries``).
- Format conversion bridges: Polars/Arrow → GeoDataFrame, Polars/Arrow →
  MovingPandas TrajectoryCollection (``bridges``).
- Any shapely, geopandas, or movingpandas imports are confined to this
  subsystem.

**Delegates to:**
- ``neptune_ais.datasets`` — schema definitions for the data being converted.
  Geometry does not redefine column names or types.
- ``neptune_ais.storage`` — if boundary reference files need local caching,
  storage manages the file layout.

**Rule:** All optional geo dependencies (shapely, geopandas, movingpandas, h3)
must be imported lazily and only within this subsystem. Core modules
(``datasets``, ``adapters``, ``derive``, ``api``) must access spatial
operations through ``geometry`` — never by importing geo libraries directly.
This keeps the optional install boundary clean.

**Install extra:** ``pip install neptune-ais[geo]``
"""

from __future__ import annotations

__all__ = [
    "boundaries",
    "bridges",
]


def _missing_geo_extra(package: str) -> ImportError:
    """Return an ImportError with install instructions for a missing geo dep.

    Intended usage::

        try:
            import shapely
        except ImportError:
            raise _missing_geo_extra("shapely") from None
    """
    return ImportError(
        f"{package!r} is required for this operation. "
        f"Install it with: pip install neptune-ais[geo]"
    )
