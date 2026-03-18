"""Tracks derivation — native track segmentation pipeline.

Segments positions into trips using gap detection, jump filtering,
and configurable thresholds. Operates on Polars LazyFrames to avoid
the GeoDataFrame scaling bottleneck.
"""

from __future__ import annotations
