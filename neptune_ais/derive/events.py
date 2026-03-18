"""Events derivation — port calls, crossings, encounters, loitering.

Infers maritime events from positions and tracks using configurable
heuristics and spatial reference data.

This module defines the cache identity contract for derived events.
Individual event detectors will be added in subsequent tasks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

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


# ---------------------------------------------------------------------------
# Event provenance — structured lineage for inferred events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventProvenance:
    """Structured provenance for an inferred event.

    Captures enough context to explain *how* an event was detected and
    *from what data*, without requiring access to the original detector
    state. Serializes to a compact ``record_provenance`` string.

    Token format::

        <source>:<detector>/<version>[<upstream>]

    Examples::

        "noaa:port_call_detector/0.1.0[tracks]"
        "dma:encounter_detector/0.2.0[positions]"
        "noaa:loitering_detector/0.1.0[tracks+boundaries]"

    The token is designed to be:
    - Human-readable in a DataFrame column.
    - Parseable back into components via ``parse_provenance()``.
    - Compatible with the fusion provenance token format (both use
      ``source:tag`` as the leading structure).

    Usage::

        prov = EventProvenance(
            source="noaa",
            detector="port_call_detector",
            detector_version="0.1.0",
            upstream_datasets=["tracks"],
        )
        token = prov.to_token()  # "noaa:port_call_detector/0.1.0[tracks]"
    """

    source: str
    """Source identifier for the upstream data."""

    detector: str
    """Detector name (e.g. ``"port_call_detector"``)."""

    detector_version: str
    """Detector version string (e.g. ``"0.1.0"``)."""

    upstream_datasets: tuple[str, ...] = field(default_factory=lambda: ("tracks",))
    """Upstream datasets this event was derived from (always sorted).
    Common values: ``("tracks",)``, ``("positions",)``,
    ``("boundaries", "tracks")``."""

    def __post_init__(self) -> None:
        # Ensure upstream_datasets is always a non-empty sorted tuple for
        # deterministic tokens and equality comparisons.
        sorted_ds = tuple(sorted(self.upstream_datasets))
        if not sorted_ds:
            raise ValueError("upstream_datasets must not be empty")
        object.__setattr__(self, "upstream_datasets", sorted_ds)

    def to_token(self) -> str:
        """Serialize to a compact provenance token string.

        Returns:
            A string like ``"noaa:port_call_detector/0.1.0[tracks]"``.
        """
        upstream = "+".join(self.upstream_datasets)
        return f"{self.source}:{self.detector}/{self.detector_version}[{upstream}]"


def parse_provenance(token: str) -> EventProvenance:
    """Parse a provenance token back into an ``EventProvenance``.

    Args:
        token: A provenance string produced by ``EventProvenance.to_token()``.

    Returns:
        An ``EventProvenance`` instance.

    Raises:
        ValueError: If the token cannot be parsed.

    Example::

        prov = parse_provenance("noaa:port_call_detector/0.1.0[tracks]")
        assert prov.source == "noaa"
        assert prov.detector == "port_call_detector"
        assert prov.detector_version == "0.1.0"
        assert prov.upstream_datasets == ("tracks",)
    """
    try:
        source, rest = token.split(":", 1)
        # rest = "port_call_detector/0.1.0[tracks]"
        detector_and_version, bracket_part = rest.split("[", 1)
        if not bracket_part.endswith("]"):
            raise ValueError("missing closing bracket")
        upstream_str = bracket_part[:-1]
        detector, version = detector_and_version.split("/", 1)
        upstream = tuple(sorted(upstream_str.split("+")))
        return EventProvenance(
            source=source,
            detector=detector,
            detector_version=version,
            upstream_datasets=upstream,
        )
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Cannot parse event provenance token: {token!r}"
        ) from e
