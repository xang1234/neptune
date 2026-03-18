"""Neptune AIS — Open AIS data platform for Python.

Neptune downloads, normalizes, catalogs, fuses, and analyzes open-source AIS
vessel tracking data. One interface, many archives, clean output.

Public entrypoints
------------------
Neptune        Main dataset object for archival AIS access.
NeptuneStream  Async streaming interface (separate lifecycle).
sources        Source adapter catalog and capability queries.

Quick start::

    from neptune_ais import Neptune

    n = Neptune("2024-06-15", sources=["noaa"])
    positions = n.positions()  # polars.LazyFrame
"""

from __future__ import annotations

# --- version ---
__version__ = "0.1.0dev0"

# --- lazy public API ---
# Imports are deferred so that heavy deps (polars, duckdb) are not paid at
# import time when users only need metadata or source inspection.

__all__ = [
    "Neptune",
    "NeptuneStream",
    "sources",
    "__version__",
]


def __getattr__(name: str):
    if name == "Neptune":
        from neptune_ais.api import Neptune

        globals()["Neptune"] = Neptune
        return Neptune
    if name == "NeptuneStream":
        from neptune_ais.stream import NeptuneStream

        globals()["NeptuneStream"] = NeptuneStream
        return NeptuneStream
    if name == "sources":
        from neptune_ais.adapters import registry as sources

        globals()["sources"] = sources
        return sources
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
