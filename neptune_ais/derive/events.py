"""Events derivation — port calls, crossings, encounters, loitering.

Infers maritime events from positions and tracks using configurable
heuristics and spatial reference data.

This module defines the cache identity contract for derived events.
Individual event detectors will be added in subsequent tasks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from neptune_ais.datasets.events import SCHEMA_VERSION as _EVENTS_SCHEMA_VERSION
from neptune_ais.datasets.tracks import SCHEMA_VERSION as _TRACKS_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Event cache key — determines when cached events are valid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventCacheKey:
    """Cache identity for a derived event partition.

    A cached event result is valid if and only if the cache key matches.
    Any change to the upstream data, schema, or detector config produces
    a different key, triggering recomputation.

    Follows the same pattern as ``TrackCacheKey``:

    Invalidation triggers:
    1. **Upstream data changed** — different ``upstream_manifest_hash``.
       For events that depend on tracks, this is the tracks manifest hash.
       For events derived directly from positions, this is the positions
       manifest hash.
    2. **Schema version changed** — different ``upstream_schema_version``
       or ``events_schema_version``.
    3. **Detector config changed** — different ``config_hash`` from
       the event detector's configuration.

    Usage::

        key = EventCacheKey.from_manifest(
            event_type="port_call",
            source="noaa",
            date="2024-06-15",
            config_hash=detector.config_hash(),
            upstream_manifest_hash=tracks_hash,
        )
        if cached_key == key.cache_key():
            return cached_result
    """

    event_type: str
    """Event family (e.g. ``port_call``, ``encounter``)."""

    source: str
    """Source identifier for the upstream data."""

    date: str
    """Partition date (YYYY-MM-DD)."""

    config_hash: str
    """Hash of the detector configuration that produced this result."""

    upstream_schema_version: str
    """Schema version of the upstream dataset (positions or tracks)."""

    events_schema_version: str
    """Schema version of the events output."""

    upstream_manifest_hash: str
    """Hash of the upstream manifest (ties to specific data).
    Computed from the manifest's content checksums and record count."""

    def cache_key(self) -> str:
        """Compute the full cache key as a hex string.

        Two derived partitions with the same cache_key are guaranteed
        to produce identical results (assuming deterministic computation).

        Returns a 16-character hex string.
        """
        key = (
            f"{self.event_type}:{self.source}:{self.date}"
            f":{self.config_hash}"
            f":{self.upstream_schema_version}"
            f":{self.events_schema_version}"
            f":{self.upstream_manifest_hash}"
        )
        return hashlib.sha1(key.encode()).hexdigest()[:16]

    @classmethod
    def from_manifest(
        cls,
        event_type: str,
        source: str,
        date: str,
        config_hash: str,
        upstream_manifest_hash: str,
        upstream_schema_version: str = _TRACKS_SCHEMA_VERSION,
        events_schema_version: str = _EVENTS_SCHEMA_VERSION,
    ) -> EventCacheKey:
        """Create a cache key from an upstream manifest and detector config.

        Args:
            event_type: Event family (e.g. ``"port_call"``).
            source: Source identifier.
            date: Partition date string.
            config_hash: Detector configuration hash.
            upstream_manifest_hash: Hash of the upstream manifest.
            upstream_schema_version: Schema version of the upstream
                dataset. Defaults to ``"tracks/v1"`` since most events
                depend on tracks.
            events_schema_version: Schema version of the events output.
        """
        return cls(
            event_type=event_type,
            source=source,
            date=date,
            config_hash=config_hash,
            upstream_schema_version=upstream_schema_version,
            events_schema_version=events_schema_version,
            upstream_manifest_hash=upstream_manifest_hash,
        )
