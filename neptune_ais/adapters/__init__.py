"""Adapters — source adapter framework and plugin registration.

Each adapter implements the SourceAdapter protocol to fetch, normalize,
and quality-check data from a specific AIS provider.

Subsystem boundary
------------------
**Owns:**
- The ``SourceAdapter`` protocol (defined in ``base``).
- Per-source fetch logic: HTTP downloads, WebSocket connections, API calls.
- Per-source normalization: raw provider formats → canonical Polars DataFrames
  conforming to ``datasets`` schemas.
- Source-specific QC rules returned via the adapter protocol.
- Adapter discovery and capability metadata (``registry``).

**Delegates to:**
- ``neptune_ais.datasets`` — canonical schema definitions that adapters
  normalize *into*. Adapters import dataset schemas; datasets never import
  adapters.
- ``neptune_ais.storage`` — writing normalized output to the local store.
  Adapters produce DataFrames; storage decides where they land on disk.
- ``neptune_ais.catalog`` — manifest creation after a successful write.
- ``neptune_ais.qc`` — executing QC checks on normalized output. Adapters
  may *supply* QC rules but do not run the QC pipeline themselves.

**Rule:** Adapters must not call storage or catalog directly. They return
normalized DataFrames (and optional raw artifacts) to the orchestration
layer in ``api``, which coordinates the write-and-catalog sequence.
"""

from __future__ import annotations

__all__ = [
    "base",
    "registry",
]
