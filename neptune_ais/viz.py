"""Viz — map layer helpers and visualization support.

Builds viewport-aware, Arrow/GeoArrow-friendly map layers for positions,
tracks, trips, density, and events using lonboard.

Module role — presentation layer (optional dependency)
------------------------------------------------------
**Owns:**
- Map layer construction: positions, tracks, trips, density, events.
- Viewport clipping and sampling for dense point clouds.
- Color-by logic and layer styling.
- HTML export for standalone map files.

**Does not own:**
- Data access or derivation — receives DataFrames/LazyFrames from ``api``.
- Geometry conversions — delegates to ``geometry.bridges`` if needed.

**Import rule:** Viz may import from ``datasets`` (column names for color-by)
and ``geometry.bridges`` (for GeoArrow conversion). lonboard is an optional
dependency — viz must handle its absence gracefully. Viz must not import
from ``adapters``, ``derive``, ``storage``, ``catalog``, or ``cli``.

**Install extra:** ``pip install neptune-ais[geo]`` (lonboard is part of the
geo extra since it is used alongside spatial data).
"""

from __future__ import annotations
