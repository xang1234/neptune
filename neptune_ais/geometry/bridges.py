"""Bridges — optional GeoPandas and MovingPandas conversions.

Converts Neptune's native Polars/Arrow representations to GeoDataFrame
or MovingPandas TrajectoryCollection when the user opts in.

Requires: ``pip install neptune-ais[geo]`` (geopandas, movingpandas).
"""

from __future__ import annotations
