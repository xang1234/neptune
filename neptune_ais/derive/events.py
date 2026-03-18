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

    return _select_and_sort_events(result)


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

    # Filter by max gap, max distance, and exclude zero-duration
    # crossings (duplicate timestamps produce phantom events).
    crossings = crossings.filter(
        (pl.col("_gap_us") > 0)
        & (pl.col("_gap_us") <= gap_us)
        & (pl.col("_dist_m") <= config.max_distance_m)
    )

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

    return _select_and_sort_events(result)


# ---------------------------------------------------------------------------
# Encounter detector
# ---------------------------------------------------------------------------

_ENCOUNTER_DETECTOR_NAME = "encounter_detector"
_ENCOUNTER_DETECTOR_VERSION = "0.1.0"


@dataclass(frozen=True)
class EncounterConfig:
    """Configuration for encounter detection.

    Args:
        max_distance_m: Maximum distance (meters) between two vessels
            for a proximity observation. Default 500.
        min_duration_s: Minimum sustained proximity (seconds) to
            qualify as an encounter. Default 600 (10 min).
        min_observations: Minimum time-coincident observations.
            Default 2.
        time_bucket_s: Bucket width (seconds) for aligning
            observations from different vessels. Default 300 (5 min).
        gap_seconds: Maximum gap (seconds) between consecutive
            proximity observations before splitting. Default 3600.
    """

    max_distance_m: float = 500.0
    min_duration_s: int = 600
    min_observations: int = 2
    time_bucket_s: int = 300
    gap_seconds: int = 3600

    def config_hash(self) -> str:
        key = (
            f"dist={self.max_distance_m}"
            f":dur={self.min_duration_s}"
            f":obs={self.min_observations}"
            f":bucket={self.time_bucket_s}"
            f":gap={self.gap_seconds}"
        )
        return hashlib.sha1(key.encode()).hexdigest()[:12]


def detect_encounters(
    positions: pl.DataFrame,
    *,
    config: EncounterConfig | None = None,
    source: str = "",
) -> pl.DataFrame:
    """Detect encounter events between vessel pairs.

    Buckets positions by time, self-joins to find co-located pairs,
    groups consecutive proximity observations, and emits encounter
    events for sustained close-range interactions.

    Only pairs where ``mmsi_a < mmsi_b`` are kept (no duplicates/self).

    Args:
        positions: Positions DataFrame with ``mmsi``, ``timestamp``,
            ``lat``, ``lon``, and ``source`` columns.
        config: Detection configuration. Uses defaults if None.
        source: Source identifier for provenance.

    Returns:
        A DataFrame conforming to ``events/v1`` with
        ``event_type = "encounter"`` and ``other_mmsi`` populated.
    """
    from neptune_ais.datasets.events import (
        Col as EventCol,
        EVENT_TYPE_ENCOUNTER,
        SCHEMA,
        make_event_id,
    )

    if config is None:
        config = EncounterConfig()

    cfg_hash = config.config_hash()

    if len(positions) == 0 or positions["mmsi"].n_unique() < 2:
        return pl.DataFrame(schema=SCHEMA)

    # Step 1: bucket positions by time.
    # Emit each position into current and next bucket to catch pairs
    # that straddle a bucket boundary.
    bucket_us = config.time_bucket_s * 1_000_000
    epoch_col = pl.col("timestamp").dt.epoch(time_unit="us")
    df = positions.with_columns(
        (epoch_col // bucket_us).alias("_time_bucket"),
    )
    df_next = df.with_columns(
        (pl.col("_time_bucket") + 1).alias("_time_bucket"),
    )
    df = pl.concat([df, df_next])

    # Step 2: self-join within time buckets.
    df_a = df.select(
        pl.col("mmsi").alias("_mmsi_a"),
        pl.col("lat").alias("_lat_a"),
        pl.col("lon").alias("_lon_a"),
        pl.col("timestamp").alias("_ts_a"),
        pl.col("source").alias("_source_a"),
        pl.col("_time_bucket"),
    )
    df_b = df.select(
        pl.col("mmsi").alias("_mmsi_b"),
        pl.col("lat").alias("_lat_b"),
        pl.col("lon").alias("_lon_b"),
        pl.col("timestamp").alias("_ts_b"),
        pl.col("_time_bucket"),
    )

    pairs = df_a.join(df_b, on="_time_bucket", how="inner")
    pairs = pairs.filter(pl.col("_mmsi_a") < pl.col("_mmsi_b"))
    # Deduplicate pairs created by the double-bucketing.
    pairs = pairs.unique(["_mmsi_a", "_mmsi_b", "_ts_a", "_ts_b"])

    if len(pairs) == 0:
        return pl.DataFrame(schema=SCHEMA)

    # Step 3: distance and proximity filter.
    pairs = pairs.with_columns(
        (
            (
                ((pl.col("_lat_a") - pl.col("_lat_b")) * _METERS_PER_DEG).pow(2)
                + (
                    (pl.col("_lon_a") - pl.col("_lon_b"))
                    * _METERS_PER_DEG
                    * ((pl.col("_lat_a") + pl.col("_lat_b")) / 2).radians().cos()
                ).pow(2)
            ).sqrt()
        ).alias("_dist_m"),
    )

    proximate = pairs.filter(pl.col("_dist_m") <= config.max_distance_m)
    if len(proximate) == 0:
        return pl.DataFrame(schema=SCHEMA)

    # Use min timestamp as observation time for ordering, but keep
    # both timestamps for accurate start/end computation.
    proximate = proximate.with_columns(
        pl.min_horizontal("_ts_a", "_ts_b").alias("_obs_ts"),
    )

    # Step 4: group consecutive observations by pair.
    # Sort ensures contiguous pair groups; bare diff() is correct
    # since shift(1) pair-boundary check handles cross-pair resets.
    proximate = proximate.sort(["_mmsi_a", "_mmsi_b", "_obs_ts"])
    gap_us = config.gap_seconds * 1_000_000

    proximate = proximate.with_columns(
        pl.col("_obs_ts").diff().dt.total_microseconds()
        .alias("_dt_us"),
    )
    proximate = proximate.with_columns(
        (
            (pl.col("_mmsi_a") != pl.col("_mmsi_a").shift(1))
            | (pl.col("_mmsi_b") != pl.col("_mmsi_b").shift(1))
            | pl.col("_dt_us").is_null()
            | (pl.col("_dt_us") > gap_us)
        )
        .cast(pl.Int64)
        .cum_sum()
        .alias("_encounter_id")
    )

    # Step 5: aggregate and filter.
    encounters = proximate.group_by("_encounter_id").agg(
        pl.col("_mmsi_a").first().alias(EventCol.MMSI),
        pl.col("_mmsi_b").first().alias(EventCol.OTHER_MMSI),
        pl.min_horizontal(
            pl.col("_ts_a").min(), pl.col("_ts_b").min()
        ).alias(EventCol.START_TIME),
        pl.max_horizontal(
            pl.col("_ts_a").max(), pl.col("_ts_b").max()
        ).alias(EventCol.END_TIME),
        ((pl.col("_lat_a") + pl.col("_lat_b")) / 2).mean().alias(EventCol.LAT),
        ((pl.col("_lon_a") + pl.col("_lon_b")) / 2).mean().alias(EventCol.LON),
        pl.len().cast(pl.Int64).alias("_obs_count"),
        (
            (pl.col("_obs_ts").max() - pl.col("_obs_ts").min())
            .dt.total_seconds().cast(pl.Float64)
        ).alias("_duration_s"),
        pl.col("_dist_m").mean().alias("_mean_dist_m"),
        pl.col("_source_a").first().alias(EventCol.SOURCE),
    )

    encounters = encounters.filter(
        (pl.col("_duration_s") >= config.min_duration_s)
        & (pl.col("_obs_count") >= config.min_observations)
    )

    if len(encounters) == 0:
        return pl.DataFrame(schema=SCHEMA)

    # Confidence: closer + longer = higher.
    encounters = encounters.with_columns(
        (
            pl.when(
                (pl.col("_duration_s") >= 3600) & (pl.col("_mean_dist_m") <= 200)
            ).then(0.9)
            .when(
                (pl.col("_duration_s") >= 600) & (pl.col("_mean_dist_m") <= 500)
            ).then(0.7)
            .otherwise(0.5)
        ).alias(EventCol.CONFIDENCE_SCORE),
    )

    encounters = encounters.with_columns(
        pl.struct([EventCol.MMSI, EventCol.START_TIME, EventCol.SOURCE])
        .map_elements(
            lambda row: make_event_id(
                EVENT_TYPE_ENCOUNTER,
                row[EventCol.MMSI],
                int(row[EventCol.START_TIME].timestamp() * 1e6),
                row[EventCol.SOURCE],
                cfg_hash,
            ),
            return_dtype=pl.String,
        )
        .alias(EventCol.EVENT_ID),
    )

    prov = EventProvenance(
        source=source or "derived",
        detector=_ENCOUNTER_DETECTOR_NAME,
        detector_version=_ENCOUNTER_DETECTOR_VERSION,
        upstream_datasets=["positions"],
    )

    result = encounters.with_columns(
        pl.lit(EVENT_TYPE_ENCOUNTER).alias(EventCol.EVENT_TYPE),
        pl.lit(None).cast(pl.Binary).alias(EventCol.GEOMETRY_WKB),
        pl.lit(prov.to_token()).alias(EventCol.RECORD_PROVENANCE),
    )

    return _select_and_sort_events(result)


# ---------------------------------------------------------------------------
# Shared output helper
# ---------------------------------------------------------------------------

from neptune_ais.datasets.events import (
    Col as _EventCol,
    SORT_ORDER as _EVENT_SORT_ORDER,
)

_EVENT_OUTPUT_COLS: list[str] = [
    _EventCol.EVENT_ID, _EventCol.EVENT_TYPE, _EventCol.MMSI,
    _EventCol.OTHER_MMSI, _EventCol.START_TIME, _EventCol.END_TIME,
    _EventCol.LAT, _EventCol.LON, _EventCol.GEOMETRY_WKB,
    _EventCol.CONFIDENCE_SCORE, _EventCol.SOURCE,
    _EventCol.RECORD_PROVENANCE,
]


def _select_and_sort_events(df: pl.DataFrame) -> pl.DataFrame:
    """Select canonical event columns and sort by the schema's SORT_ORDER."""
    return df.select(_EVENT_OUTPUT_COLS).sort(_EVENT_SORT_ORDER)


# ---------------------------------------------------------------------------
# Loitering detector
# ---------------------------------------------------------------------------

_LOITERING_DETECTOR_NAME = "loitering_detector"
_LOITERING_DETECTOR_VERSION = "0.1.0"


@dataclass(frozen=True)
class LoiteringConfig:
    """Configuration for loitering detection.

    Detects vessels that stay in a small area for a sustained period.
    Uses spatial radius and speed thresholds.

    Args:
        max_speed_knots: Maximum SOG for a position to count as
            slow/stationary. Default 2.0.
        max_radius_m: Maximum spatial radius (meters) of the cluster.
            Default 1000.
        min_duration_s: Minimum sustained duration (seconds).
            Default 1800 (30 min).
        min_points: Minimum slow-speed positions. Default 3.
        gap_seconds: Maximum gap before splitting. Default 3600.
    """

    max_speed_knots: float = 2.0
    max_radius_m: float = 1000.0
    min_duration_s: int = 1800
    min_points: int = 3
    gap_seconds: int = 3600

    def config_hash(self) -> str:
        key = (
            f"speed={self.max_speed_knots}"
            f":radius={self.max_radius_m}"
            f":dur={self.min_duration_s}"
            f":pts={self.min_points}"
            f":gap={self.gap_seconds}"
        )
        return hashlib.sha1(key.encode()).hexdigest()[:12]


def detect_loitering(
    positions: pl.DataFrame,
    *,
    config: LoiteringConfig | None = None,
    source: str = "",
) -> pl.DataFrame:
    """Detect loitering events from positions.

    A vessel is loitering when it has sustained low-speed positions
    clustered in a small area. Unlike port calls, loitering does not
    require boundary context — it is purely position-based.

    Algorithm:
    1. Filter to low-speed positions (SOG <= ``max_speed_knots``).
    2. Sort by ``(mmsi, timestamp)`` and group by time gaps.
    3. Compute spatial spread per group; filter by ``max_radius_m``.
    4. Filter by ``min_duration_s`` and ``min_points``.
    5. Emit loitering events with centroid and confidence.

    Args:
        positions: Positions DataFrame with ``mmsi``, ``timestamp``,
            ``lat``, ``lon``, ``sog``, and ``source`` columns.
        config: Detection configuration. Uses defaults if None.
        source: Source identifier for provenance.

    Returns:
        A DataFrame conforming to ``events/v1`` with
        ``event_type = "loitering"``.
    """
    from neptune_ais.datasets.events import (
        Col as EventCol,
        EVENT_TYPE_LOITERING,
        SCHEMA,
        make_event_id,
    )

    if config is None:
        config = LoiteringConfig()

    cfg_hash = config.config_hash()

    # Step 1: filter to low-speed positions.
    df = positions.filter(pl.col("sog") <= config.max_speed_knots)

    if len(df) == 0:
        return pl.DataFrame(schema=SCHEMA)

    # Step 2: sort and group by time gaps per vessel.
    df = df.sort(["mmsi", "timestamp"])
    gap_us = config.gap_seconds * 1_000_000

    df = df.with_columns(
        pl.col("timestamp").diff().dt.total_microseconds()
        .over("mmsi")
        .alias("_dt_us"),
    )
    df = df.with_columns(
        (
            (pl.col("mmsi") != pl.col("mmsi").shift(1))
            | pl.col("_dt_us").is_null()
            | (pl.col("_dt_us") > gap_us)
        )
        .cast(pl.Int64)
        .cum_sum()
        .alias("_group_id")
    )
    df = df.drop("_dt_us")

    # Step 3: aggregate and check spatial spread.
    candidates = df.group_by("_group_id").agg(
        pl.col("mmsi").first().alias(EventCol.MMSI),
        pl.col("timestamp").min().alias(EventCol.START_TIME),
        pl.col("timestamp").max().alias(EventCol.END_TIME),
        pl.col("lat").mean().alias(EventCol.LAT),
        pl.col("lon").mean().alias(EventCol.LON),
        pl.len().cast(pl.Int64).alias("_point_count"),
        (
            (pl.col("timestamp").max() - pl.col("timestamp").min())
            .dt.total_seconds().cast(pl.Float64)
        ).alias("_duration_s"),
        # Max deviation from centroid (approximate equirectangular).
        (
            (
                ((pl.col("lat") - pl.col("lat").mean()) * _METERS_PER_DEG).pow(2)
                + (
                    (pl.col("lon") - pl.col("lon").mean())
                    * _METERS_PER_DEG
                    * pl.col("lat").mean().radians().cos()
                ).pow(2)
            ).sqrt().max()
        ).alias("_max_radius_m"),
        pl.col("source").first().alias(EventCol.SOURCE),
    )

    # Step 4: filter by thresholds.
    candidates = candidates.filter(
        (pl.col("_duration_s") >= config.min_duration_s)
        & (pl.col("_point_count") >= config.min_points)
        & (pl.col("_max_radius_m") <= config.max_radius_m)
    )

    if len(candidates) == 0:
        return pl.DataFrame(schema=SCHEMA)

    # Step 5: confidence and event IDs.
    # Longer + tighter radius = higher confidence.
    candidates = candidates.with_columns(
        (
            pl.when(
                (pl.col("_duration_s") >= 7200) & (pl.col("_max_radius_m") <= 500)
            ).then(0.9)
            .when(
                (pl.col("_duration_s") >= 1800) & (pl.col("_max_radius_m") <= 1000)
            ).then(0.7)
            .otherwise(0.5)
        ).alias(EventCol.CONFIDENCE_SCORE),
    )

    candidates = candidates.with_columns(
        pl.struct([EventCol.MMSI, EventCol.START_TIME, EventCol.SOURCE])
        .map_elements(
            lambda row: make_event_id(
                EVENT_TYPE_LOITERING,
                row[EventCol.MMSI],
                int(row[EventCol.START_TIME].timestamp() * 1e6),
                row[EventCol.SOURCE],
                cfg_hash,
            ),
            return_dtype=pl.String,
        )
        .alias(EventCol.EVENT_ID),
    )

    prov = EventProvenance(
        source=source or "derived",
        detector=_LOITERING_DETECTOR_NAME,
        detector_version=_LOITERING_DETECTOR_VERSION,
        upstream_datasets=["positions"],
    )

    result = candidates.with_columns(
        pl.lit(EVENT_TYPE_LOITERING).alias(EventCol.EVENT_TYPE),
        pl.lit(None).cast(pl.Int64).alias(EventCol.OTHER_MMSI),
        pl.lit(None).cast(pl.Binary).alias(EventCol.GEOMETRY_WKB),
        pl.lit(prov.to_token()).alias(EventCol.RECORD_PROVENANCE),
    )

    return _select_and_sort_events(result)
