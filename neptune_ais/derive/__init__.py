"""Derive — derived dataset computation pipelines.

Native Polars/DuckDB pipelines for computing tracks, events, and
spatial density products from canonical datasets.

Subsystem boundary
------------------
**Owns:**
- Computation logic that transforms canonical datasets into derived products:
  track segmentation, event inference, density aggregation.
- Derivation configuration (gap thresholds, min-points, resolution, etc.).
- Derived dataset schemas (tracks, events, density grids) — these extend the
  canonical family defined in ``datasets``.

**Delegates to:**
- ``neptune_ais.datasets`` — input schemas. Derivation reads canonical data
  via schema-aware scans; it does not redefine what ``positions`` looks like.
- ``neptune_ais.storage`` — caching derived output to the derived store layer.
- ``neptune_ais.catalog`` — manifest registration for derived partitions.
- ``neptune_ais.geometry.boundaries`` — spatial reference data (EEZ polygons,
  port boundaries) needed by event derivation. Derive *uses* boundary lookups
  but does not own the reference datasets.

**Rule:** Derive modules operate on Polars LazyFrames and/or DuckDB relations.
They must not import from ``adapters`` (derive doesn't know about sources) or
``cli`` (derive is not presentation). Geometry usage must go through the
``geometry`` subsystem, not inline shapely calls.
"""

from __future__ import annotations

__all__ = [
    "tracks",
    "events",
    "density",
]
