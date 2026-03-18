"""Events derivation — port calls, crossings, encounters, loitering.

Infers maritime events from positions and tracks using configurable
heuristics and spatial reference data.

This module defines the cache identity contract for derived events.
Individual event detectors will be added in subsequent tasks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import polars as pl

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


# ---------------------------------------------------------------------------
# Port-call detector
# ---------------------------------------------------------------------------

_DETECTOR_NAME = "port_call_detector"
_DETECTOR_VERSION = "0.1.0"


@dataclass(frozen=True)
class PortCallConfig:
    """Configuration for port-call detection.

    Controls how positions are classified as port calls based on
    speed, duration, and boundary containment.

    Args:
        max_speed_knots: Maximum SOG (knots) for a position to be
            considered "in port". Default 3.0.
        min_duration_s: Minimum duration (seconds) of low-speed
            positions in a port to qualify as a port call. Default
            3600 (1 hour).
        min_points: Minimum number of low-speed positions in the
            port call. Default 3.
        gap_seconds: Maximum gap (seconds) between consecutive
            low-speed positions before splitting into separate
            port calls. Default 7200 (2 hours).
    """

    max_speed_knots: float = 3.0
    min_duration_s: int = 3600
    min_points: int = 3
    gap_seconds: int = 7200

    def config_hash(self) -> str:
        """Deterministic hash of detection parameters."""
        key = (
            f"speed={self.max_speed_knots}"
            f":dur={self.min_duration_s}"
            f":pts={self.min_points}"
            f":gap={self.gap_seconds}"
        )
        return hashlib.sha1(key.encode()).hexdigest()[:12]


def detect_port_calls(
    positions: pl.DataFrame,
    port_regions: pl.Series,
    *,
    config: PortCallConfig | None = None,
    source: str = "",
) -> pl.DataFrame:
    """Detect port-call events from positions and port region labels.

    Algorithm:
    1. Filter to low-speed positions (SOG <= ``max_speed_knots``).
    2. Filter to positions that fall within a port region.
    3. Group consecutive low-speed, in-port positions by vessel and
       port, splitting on time gaps > ``gap_seconds``.
    4. Aggregate each group into a candidate port-call event.
    5. Filter candidates by ``min_duration_s`` and ``min_points``.
    6. Compute confidence scores and emit events.

    Args:
        positions: Positions DataFrame sorted by ``(mmsi, timestamp)``.
            Must include ``mmsi``, ``timestamp``, ``lat``, ``lon``,
            ``sog``, and ``source`` columns.
        port_regions: A String Series (same length as ``positions``)
            with the port region name for each position, or None if
            the position is not in any port. Produced by
            ``BoundaryRegistry.lookup_column()``.
        config: Detection configuration. Uses defaults if None.
        source: Source identifier for provenance.

    Returns:
        A DataFrame conforming to ``events/v1`` schema with
        ``event_type = "port_call"``.
    """
    from neptune_ais.datasets.events import (
        Col as EventCol,
        EVENT_TYPE_PORT_CALL,
        SCHEMA,
        make_event_id,
    )

    if config is None:
        config = PortCallConfig()

    cfg_hash = config.config_hash()

    # Step 1-2: filter to low-speed, in-port positions.
    df = positions.with_columns(port_regions.alias("_port_region"))
    df = df.filter(
        pl.col("_port_region").is_not_null()
        & (pl.col("sog") <= config.max_speed_knots)
    )

    if len(df) == 0:
        return pl.DataFrame(schema=SCHEMA)

    # Ensure sorted by (mmsi, timestamp) after filtering.
    df = df.sort(["mmsi", "timestamp"])

    # Step 3: detect boundaries between port-call groups.
    # Use .over("mmsi") for timestamp diff so vessel boundaries are
    # handled correctly even if vessels are interleaved.
    gap_us = config.gap_seconds * 1_000_000

    df = df.with_columns(
        pl.col("timestamp").diff().dt.total_microseconds()
        .over("mmsi")
        .alias("_dt_us"),
    )
    df = df.with_columns(
        (
            (pl.col("mmsi") != pl.col("mmsi").shift(1))
            | (pl.col("_port_region") != pl.col("_port_region").shift(1))
            | pl.col("_dt_us").is_null()
            | (pl.col("_dt_us") > gap_us)
        )
        .cast(pl.Int64)
        .cum_sum()
        .alias("_group_id")
    )
    df = df.drop("_dt_us")

    # Step 4: aggregate each group into a candidate event.
    candidates = df.group_by("_group_id").agg(
        pl.col("mmsi").first().alias(EventCol.MMSI),
        pl.col("_port_region").first().alias("_port_name"),
        pl.col("timestamp").min().alias(EventCol.START_TIME),
        pl.col("timestamp").max().alias(EventCol.END_TIME),
        pl.col("lat").mean().alias(EventCol.LAT),
        pl.col("lon").mean().alias(EventCol.LON),
        pl.len().cast(pl.Int64).alias("_point_count"),
        (
            (pl.col("timestamp").max() - pl.col("timestamp").min())
            .dt.total_seconds()
            .cast(pl.Float64)
        ).alias("_duration_s"),
        pl.col("sog").mean().alias("_mean_sog"),
        pl.col("source").first().alias(EventCol.SOURCE),
    )

    # Step 5: filter by thresholds.
    candidates = candidates.filter(
        (pl.col("_duration_s") >= config.min_duration_s)
        & (pl.col("_point_count") >= config.min_points)
    )

    if len(candidates) == 0:
        return pl.DataFrame(schema=SCHEMA)

    # Step 6: compute confidence and build output.
    # Confidence is based on stay duration:
    #   >= 4 hours → 0.9 (high), >= 2 hours → 0.7, else → 0.5 (medium).
    candidates = candidates.with_columns(
        (
            pl.when(pl.col("_duration_s") >= 14400)  # >= 4 hours
            .then(0.9)
            .when(pl.col("_duration_s") >= 7200)  # >= 2 hours
            .then(0.7)
            .otherwise(0.5)
        ).alias(EventCol.CONFIDENCE_SCORE),
    )

    # Generate deterministic event IDs.
    candidates = candidates.with_columns(
        pl.struct([EventCol.MMSI, EventCol.START_TIME, EventCol.SOURCE])
        .map_elements(
            lambda row: make_event_id(
                EVENT_TYPE_PORT_CALL,
                row[EventCol.MMSI],
                int(row[EventCol.START_TIME].timestamp() * 1e6),
                row[EventCol.SOURCE],
                cfg_hash,
            ),
            return_dtype=pl.String,
        )
        .alias(EventCol.EVENT_ID),
    )

    # Build provenance token.
    prov = EventProvenance(
        source=source or "derived",
        detector=_DETECTOR_NAME,
        detector_version=_DETECTOR_VERSION,
        upstream_datasets=["boundaries", "positions"],
    )

    # Assemble final output.
    result = candidates.with_columns(
        pl.lit(EVENT_TYPE_PORT_CALL).alias(EventCol.EVENT_TYPE),
        pl.lit(None).cast(pl.Int64).alias(EventCol.OTHER_MMSI),
        pl.lit(None).cast(pl.Binary).alias(EventCol.GEOMETRY_WKB),
        pl.lit(prov.to_token()).alias(EventCol.RECORD_PROVENANCE),
    )

    output_cols = [
        EventCol.EVENT_ID, EventCol.EVENT_TYPE, EventCol.MMSI,
        EventCol.OTHER_MMSI, EventCol.START_TIME, EventCol.END_TIME,
        EventCol.LAT, EventCol.LON, EventCol.GEOMETRY_WKB,
        EventCol.CONFIDENCE_SCORE, EventCol.SOURCE,
        EventCol.RECORD_PROVENANCE,
    ]

    return result.select(output_cols).sort(
        [EventCol.MMSI, EventCol.START_TIME]
    )


# ---------------------------------------------------------------------------
# EEZ-crossing detector
# ---------------------------------------------------------------------------

_EEZ_DETECTOR_NAME = "eez_crossing_detector"
_EEZ_DETECTOR_VERSION = "0.1.0"


@dataclass(frozen=True)
class EEZCrossingConfig:
    """Configuration for EEZ-crossing detection.

    Controls how transitions between EEZ regions are detected and
    filtered.

    Args:
        max_gap_s: Maximum time gap (seconds) between the two
            positions forming a crossing. Gaps larger than this are
            ignored (the vessel may have transited through unknown
            waters). Default 7200 (2 hours).
        max_distance_m: Maximum distance (meters) between the two
            positions. Crossings with larger distance gaps are
            lower-confidence. Default 100000 (100 km).
    """

    max_gap_s: int = 7200
    max_distance_m: float = 100_000.0

    def config_hash(self) -> str:
        """Deterministic hash of detection parameters."""
        key = f"gap={self.max_gap_s}:dist={self.max_distance_m}"
        return hashlib.sha1(key.encode()).hexdigest()[:12]


# Approximate meters per degree of latitude.
_METERS_PER_DEG = 111_320.0


def detect_eez_crossings(
    positions: pl.DataFrame,
    eez_regions: pl.Series,
    *,
    config: EEZCrossingConfig | None = None,
    source: str = "",
) -> pl.DataFrame:
    """Detect EEZ-crossing events from positions and EEZ region labels.

    Algorithm:
    1. Attach EEZ region labels to positions.
    2. Sort by ``(mmsi, timestamp)`` and compute per-vessel diffs.
    3. Detect transitions where the EEZ label changes between
       consecutive positions of the same vessel.
    4. Filter by maximum time gap.
    5. Compute crossing location (midpoint), confidence, and emit events.

    Confidence is based on the time gap and distance between the two
    positions forming the crossing:
    - Gap <= 30 min and distance <= 20 km → 0.9 (high)
    - Gap <= 2 hours and distance <= 100 km → 0.7 (medium)
    - Otherwise → 0.5 (low, but still within max_gap_s)

    Args:
        positions: Positions DataFrame sorted by ``(mmsi, timestamp)``.
            Must include ``mmsi``, ``timestamp``, ``lat``, ``lon``,
            and ``source`` columns.
        eez_regions: A String Series (same length as ``positions``)
            with the EEZ region name for each position, or None if
            the position is not in any EEZ. Produced by
            ``BoundaryRegistry.lookup_column()``.
        config: Detection configuration. Uses defaults if None.
        source: Source identifier for provenance.

    Returns:
        A DataFrame conforming to ``events/v1`` schema with
        ``event_type = "eez_crossing"``.
    """
    from neptune_ais.datasets.events import (
        Col as EventCol,
        EVENT_TYPE_EEZ_CROSSING,
        SCHEMA,
        make_event_id,
    )

    if config is None:
        config = EEZCrossingConfig()

    cfg_hash = config.config_hash()

    # Attach EEZ labels and sort.
    df = positions.with_columns(eez_regions.alias("_eez_region"))
    df = df.sort(["mmsi", "timestamp"])

    # Compute per-vessel previous values.
    df = df.with_columns(
        pl.col("_eez_region").shift(1).over("mmsi").alias("_prev_eez"),
        pl.col("timestamp").shift(1).over("mmsi").alias("_prev_ts"),
        pl.col("lat").shift(1).over("mmsi").alias("_prev_lat"),
        pl.col("lon").shift(1).over("mmsi").alias("_prev_lon"),
    )

    # Detect transitions: EEZ changed between consecutive positions.
    # Exclude transitions from/to null (entering/leaving coverage).
    crossings = df.filter(
        pl.col("_eez_region").is_not_null()
        & pl.col("_prev_eez").is_not_null()
        & (pl.col("_eez_region") != pl.col("_prev_eez"))
    )

    if len(crossings) == 0:
        return pl.DataFrame(schema=SCHEMA)

    # Compute gap and approximate distance.
    gap_us = config.max_gap_s * 1_000_000
    crossings = crossings.with_columns(
        (
            (pl.col("timestamp") - pl.col("_prev_ts"))
            .dt.total_microseconds()
        ).alias("_gap_us"),
        (
            (
                ((pl.col("lat") - pl.col("_prev_lat")) * _METERS_PER_DEG).pow(2)
                + (
                    (pl.col("lon") - pl.col("_prev_lon"))
                    * _METERS_PER_DEG
                    * pl.col("lat").radians().cos()
                ).pow(2)
            ).sqrt()
        ).alias("_dist_m"),
    )

    # Filter by max gap.
    crossings = crossings.filter(pl.col("_gap_us") <= gap_us)

    if len(crossings) == 0:
        return pl.DataFrame(schema=SCHEMA)

    # Compute crossing midpoint and confidence.
    crossings = crossings.with_columns(
        ((pl.col("lat") + pl.col("_prev_lat")) / 2).alias(EventCol.LAT),
        ((pl.col("lon") + pl.col("_prev_lon")) / 2).alias(EventCol.LON),
        pl.col("_prev_ts").alias(EventCol.START_TIME),
        pl.col("timestamp").alias(EventCol.END_TIME),
        pl.col("mmsi").alias(EventCol.MMSI),
        pl.col("source").alias(EventCol.SOURCE),
        # Confidence: tighter gap + shorter distance = higher confidence.
        (
            pl.when(
                (pl.col("_gap_us") <= 1_800_000_000)  # <= 30 min
                & (pl.col("_dist_m") <= 20_000)
            )
            .then(0.9)
            .when(
                (pl.col("_gap_us") <= 7_200_000_000)  # <= 2 hours
                & (pl.col("_dist_m") <= 100_000)
            )
            .then(0.7)
            .otherwise(0.5)
        ).alias(EventCol.CONFIDENCE_SCORE),
    )

    # Generate deterministic event IDs.
    crossings = crossings.with_columns(
        pl.struct([EventCol.MMSI, EventCol.START_TIME, EventCol.SOURCE])
        .map_elements(
            lambda row: make_event_id(
                EVENT_TYPE_EEZ_CROSSING,
                row[EventCol.MMSI],
                int(row[EventCol.START_TIME].timestamp() * 1e6),
                row[EventCol.SOURCE],
                cfg_hash,
            ),
            return_dtype=pl.String,
        )
        .alias(EventCol.EVENT_ID),
    )

    # Build provenance.
    prov = EventProvenance(
        source=source or "derived",
        detector=_EEZ_DETECTOR_NAME,
        detector_version=_EEZ_DETECTOR_VERSION,
        upstream_datasets=["boundaries", "positions"],
    )

    # Assemble output.
    result = crossings.with_columns(
        pl.lit(EVENT_TYPE_EEZ_CROSSING).alias(EventCol.EVENT_TYPE),
        pl.lit(None).cast(pl.Int64).alias(EventCol.OTHER_MMSI),
        pl.lit(None).cast(pl.Binary).alias(EventCol.GEOMETRY_WKB),
        pl.lit(prov.to_token()).alias(EventCol.RECORD_PROVENANCE),
    )

    output_cols = [
        EventCol.EVENT_ID, EventCol.EVENT_TYPE, EventCol.MMSI,
        EventCol.OTHER_MMSI, EventCol.START_TIME, EventCol.END_TIME,
        EventCol.LAT, EventCol.LON, EventCol.GEOMETRY_WKB,
        EventCol.CONFIDENCE_SCORE, EventCol.SOURCE,
        EventCol.RECORD_PROVENANCE,
    ]

    return result.select(output_cols).sort(
        [EventCol.MMSI, EventCol.START_TIME]
    )
