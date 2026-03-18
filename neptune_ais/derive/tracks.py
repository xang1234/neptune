"""Tracks derivation — native track segmentation pipeline.

Segments positions into trips using gap detection, jump filtering,
and configurable thresholds. Operates on Polars LazyFrames to avoid
the GeoDataFrame scaling bottleneck.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class TrackConfig:
    """Configuration for track segmentation.

    Controls how positions are split into trip segments. All parameters
    that affect the output participate in the config hash, which in turn
    determines track_id stability and cache validity.

    Usage::

        # Default — practical for most AIS analyses
        config = TrackConfig()

        # Custom — tighter gap, more points required
        config = TrackConfig(
            gap_seconds=900,   # 15 minutes
            min_points=5,
            min_distance_m=500,
        )

        # With geometry for visualization
        config = TrackConfig(include_geometry=True)
    """

    # --- segment boundary detection ---

    gap_seconds: int = 1800
    """Maximum time gap (seconds) between consecutive positions before
    starting a new segment. Default 1800s = 30 minutes.

    Common values:
    - 900 (15m): tight, captures short port maneuvers
    - 1800 (30m): standard for coastal vessel tracking
    - 3600 (1h): loose, better for sparse data or long voyages
    """

    max_speed_knots: float = 50.0
    """Maximum plausible speed (knots) between consecutive positions.
    Position pairs implying speed above this threshold trigger a
    segment break (likely a GPS jump or data error)."""

    # --- segment filtering ---

    min_points: int = 3
    """Minimum number of positions in a segment. Segments with fewer
    points are discarded. Must be >= 1."""

    min_duration_seconds: int = 300
    """Minimum segment duration (seconds). Segments shorter than this
    are discarded. Default 300s = 5 minutes."""

    min_distance_m: float = 100.0
    """Minimum total distance (meters) for a segment. Segments covering
    less distance are discarded (likely stationary noise)."""

    # --- geometry ---

    include_geometry: bool = False
    """Whether to compute WKB LineString geometry and per-vertex
    timestamp offsets. When False, the geometry_wkb and
    timestamp_offsets_ms columns are omitted (faster, smaller)."""

    generalize_tolerance_m: float = 0.0
    """Douglas-Peucker simplification tolerance in meters. Set to 0
    to keep all points. Positive values reduce vertex count for
    visualization. Only applied when include_geometry=True."""

    # --- caching ---

    refresh: bool = False
    """If True, recompute tracks even if cached results exist for
    this configuration. Does not affect the config hash."""

    def config_hash(self) -> str:
        """Compute a deterministic hash of all parameters that affect output.

        The hash excludes ``refresh`` (which only controls caching behavior,
        not the computation result). Two configs with the same hash will
        produce identical tracks from the same input data.

        Returns a 12-character hex string.
        """
        key = (
            f"gap={self.gap_seconds}"
            f":speed={self.max_speed_knots}"
            f":pts={self.min_points}"
            f":dur={self.min_duration_seconds}"
            f":dist={self.min_distance_m}"
            f":geo={self.include_geometry}"
            f":gen={self.generalize_tolerance_m}"
        )
        return hashlib.sha1(key.encode()).hexdigest()[:12]

    def __post_init__(self) -> None:
        """Validate configuration values."""
        errors = self.validate()
        if errors:
            raise ValueError(
                "Invalid track configuration:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def validate(self) -> list[str]:
        """Check this configuration for errors.

        Returns a list of human-readable error strings.
        """
        errors: list[str] = []

        if self.gap_seconds <= 0:
            errors.append(f"gap_seconds must be > 0, got {self.gap_seconds}")
        if self.max_speed_knots <= 0:
            errors.append(f"max_speed_knots must be > 0, got {self.max_speed_knots}")
        if self.min_points < 1:
            errors.append(f"min_points must be >= 1, got {self.min_points}")
        if self.min_duration_seconds < 0:
            errors.append(f"min_duration_seconds must be >= 0, got {self.min_duration_seconds}")
        if self.min_distance_m < 0:
            errors.append(f"min_distance_m must be >= 0, got {self.min_distance_m}")
        if self.generalize_tolerance_m < 0:
            errors.append(f"generalize_tolerance_m must be >= 0, got {self.generalize_tolerance_m}")

        return errors


# ---------------------------------------------------------------------------
# Track cache key — determines when cached tracks are valid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackCacheKey:
    """Cache identity for a derived track partition.

    A cached track result is valid if and only if the cache key matches.
    Any change to the upstream data, schema, or derivation config
    produces a different key, triggering recomputation.

    Invalidation triggers:
    1. **Upstream positions data changed** — different ``upstream_manifest_hash``
       (computed from the positions manifest's content_hash + record_count).
    2. **Schema version changed** — different ``positions_schema_version``
       or ``tracks_schema_version``.
    3. **Derivation config changed** — different ``config_hash`` (from
       ``TrackConfig.config_hash()``).

    The cache key is reusable for any derived dataset (events, density)
    by following the same pattern: upstream hash + schema versions + config hash.
    """

    source: str
    """Source identifier for the upstream positions data."""

    date: str
    """Partition date (YYYY-MM-DD)."""

    config_hash: str
    """Hash of the TrackConfig that produced this result."""

    positions_schema_version: str
    """Schema version of the upstream positions data."""

    tracks_schema_version: str
    """Schema version of the tracks output."""

    upstream_manifest_hash: str
    """Hash of the upstream positions manifest (ties to specific data).
    Computed from the manifest's content checksums and record count."""

    def cache_key(self) -> str:
        """Compute the full cache key as a hex string.

        Two derived partitions with the same cache_key are guaranteed
        to produce identical results (assuming deterministic computation).

        Returns a 16-character hex string.
        """
        key = (
            f"{self.source}:{self.date}"
            f":{self.config_hash}"
            f":{self.positions_schema_version}"
            f":{self.tracks_schema_version}"
            f":{self.upstream_manifest_hash}"
        )
        return hashlib.sha1(key.encode()).hexdigest()[:16]

    @classmethod
    def from_manifest(
        cls,
        source: str,
        date: str,
        config: TrackConfig,
        positions_manifest_hash: str,
        positions_schema_version: str = "positions/v1",
        tracks_schema_version: str = "tracks/v1",
    ) -> TrackCacheKey:
        """Create a cache key from a positions manifest and track config."""
        return cls(
            source=source,
            date=date,
            config_hash=config.config_hash(),
            positions_schema_version=positions_schema_version,
            tracks_schema_version=tracks_schema_version,
            upstream_manifest_hash=positions_manifest_hash,
        )


def compute_upstream_hash(
    manifest_checksums: list[str],
    record_count: int,
) -> str:
    """Compute a hash of upstream positions manifest metadata.

    Combines the raw artifact checksums and record count into a single
    hash. If any upstream data changes (re-ingest, new raw file), this
    hash changes and invalidates the derived cache.

    Args:
        manifest_checksums: SHA-256 checksums from the positions manifest's
            raw_artifacts. Order matters (sorted by the caller).
        record_count: Number of rows in the upstream positions partition.

    Returns a 12-character hex string.
    """
    key = f"n={record_count}:" + "+".join(manifest_checksums)
    return hashlib.sha1(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Segmentation — boundary detection and segment ID assignment
# ---------------------------------------------------------------------------

# Knots to meters-per-second conversion factor.
_KNOTS_TO_MPS = 1852.0 / 3600.0

# Approximate meters per degree of latitude (constant, good enough for speed).
_METERS_PER_DEG_LAT = 111_320.0


def _equirect_distance_expr() -> pl.Expr:
    """Equirectangular distance expression between consecutive points.

    Computes approximate distance in meters using ``.diff()`` on ``lat``
    and ``lon`` columns. Suitable for use inside ``group_by().agg()``
    where ``.diff()`` operates within each group.

    Note: ``detect_boundaries`` uses its own variant with pre-computed
    ``.over("mmsi")`` diffs because it needs per-vessel (not per-group)
    differencing.
    """
    return (
        (
            (pl.col("lat").diff() * _METERS_PER_DEG_LAT).pow(2)
            + (
                pl.col("lon").diff()
                * _METERS_PER_DEG_LAT
                * pl.col("lat").radians().cos()
            ).pow(2)
        ).sqrt()
    )


def detect_boundaries(
    df: pl.DataFrame,
    config: TrackConfig,
) -> pl.DataFrame:
    """Detect segment boundaries in a sorted positions DataFrame.

    The input must be sorted by ``(mmsi, timestamp)``. Returns the same
    DataFrame with an added ``_segment_id`` column (Int64) that assigns
    a unique segment number to each contiguous group of positions.

    Boundary triggers (any one causes a new segment):
    1. **Different vessel** — MMSI changes between consecutive rows.
    2. **Time gap** — time difference exceeds ``config.gap_seconds``.
    3. **Non-monotonic timestamp** — timestamp decreases (data corruption).
    4. **Implausible jump** — implied speed between consecutive points
       exceeds ``config.max_speed_knots``.

    This is a pure Polars columnar operation — no Python row iteration.
    """
    gap_us = config.gap_seconds * 1_000_000  # microseconds
    max_speed_mps = config.max_speed_knots * _KNOTS_TO_MPS

    # Compute differences within each vessel group.
    df = df.with_columns(
        # Time diff in microseconds.
        pl.col("timestamp").diff().dt.total_microseconds()
        .over("mmsi")
        .alias("_dt_us"),

        # Lat/lon diffs for distance estimation.
        pl.col("lat").diff().over("mmsi").alias("_dlat"),
        pl.col("lon").diff().over("mmsi").alias("_dlon"),

        # MMSI change detection (first row of each vessel).
        (pl.col("mmsi") != pl.col("mmsi").shift(1)).alias("_new_vessel"),
    )

    # Approximate distance in meters (equirectangular projection).
    # Good enough for speed plausibility checks at AIS scales.
    df = df.with_columns(
        (
            (
                (pl.col("_dlat") * _METERS_PER_DEG_LAT).pow(2)
                + (
                    pl.col("_dlon")
                    * _METERS_PER_DEG_LAT
                    * pl.col("lat").radians().cos()
                ).pow(2)
            ).sqrt()
        ).alias("_dist_m"),
    )

    # Implied speed in m/s.
    df = df.with_columns(
        (
            pl.col("_dist_m")
            / (pl.col("_dt_us").cast(pl.Float64) / 1e6)
        ).alias("_speed_mps"),
    )

    # Detect boundaries.
    df = df.with_columns(
        (
            pl.col("_new_vessel")                        # new vessel
            | pl.col("_dt_us").is_null()                 # first row
            | (pl.col("_dt_us") > gap_us)                # time gap
            | (pl.col("_dt_us") <= 0)                    # non-monotonic
            | (pl.col("_speed_mps") > max_speed_mps)     # implausible jump
        ).alias("_is_boundary"),
    )

    # Assign segment IDs via cumulative sum of boundaries.
    df = df.with_columns(
        pl.col("_is_boundary").cast(pl.Int64).cum_sum().alias("_segment_id"),
    )

    # Drop intermediate columns.
    df = df.drop([
        "_dt_us", "_dlat", "_dlon", "_new_vessel",
        "_dist_m", "_speed_mps", "_is_boundary",
    ])

    return df


def filter_segments(
    df: pl.DataFrame,
    config: TrackConfig,
) -> pl.DataFrame:
    """Remove segments that don't meet minimum thresholds.

    Filters out segments with fewer than ``min_points``, shorter duration
    than ``min_duration_seconds``, or less distance than ``min_distance_m``.

    The input must have a ``_segment_id`` column (from ``detect_boundaries``).
    """
    # Compute per-segment stats for filtering.
    seg_stats = (
        df.group_by("_segment_id")
        .agg(
            pl.len().alias("_seg_points"),
            (
                (pl.col("timestamp").max() - pl.col("timestamp").min())
                .dt.total_seconds()
            ).alias("_seg_duration_s"),
            # Sum of pairwise distances (approximate total distance).
            _equirect_distance_expr().sum().alias("_seg_distance_m"),
        )
    )

    # Filter to segments meeting all thresholds.
    valid_ids = (
        seg_stats
        .filter(
            (pl.col("_seg_points") >= config.min_points)
            & (pl.col("_seg_duration_s") >= config.min_duration_seconds)
            & (pl.col("_seg_distance_m") >= config.min_distance_m)
        )
        .select("_segment_id")
    )

    return df.join(valid_ids, on="_segment_id", how="semi")


# ---------------------------------------------------------------------------
# Aggregation — compute per-track statistics and optional geometry
# ---------------------------------------------------------------------------


def aggregate_tracks(
    df: pl.DataFrame,
    config: TrackConfig,
    source: str = "",
) -> pl.DataFrame:
    """Aggregate segmented positions into canonical track records.

    Takes a positions DataFrame with ``_segment_id`` column (from
    ``detect_boundaries`` + ``filter_segments``) and produces one row
    per segment matching the ``tracks/v1`` schema.

    Args:
        df: Positions with ``_segment_id``, sorted by (mmsi, timestamp).
        config: Track derivation configuration.
        source: Source identifier for provenance.

    Returns:
        A DataFrame conforming to ``datasets.tracks.SCHEMA``.
    """
    from neptune_ais.datasets.tracks import Col, make_track_id

    cfg_hash = config.config_hash()

    # Core aggregation: one row per segment.
    agg_exprs = [
        # Identity.
        pl.col("mmsi").first().alias(Col.MMSI),

        # Temporal bounds.
        pl.col("timestamp").min().alias(Col.START_TIME),
        pl.col("timestamp").max().alias(Col.END_TIME),

        # Statistics.
        pl.len().cast(pl.Int64).alias(Col.POINT_COUNT),
        (
            (pl.col("timestamp").max() - pl.col("timestamp").min())
            .dt.total_seconds()
            .cast(pl.Float64)
        ).alias(Col.DURATION_S),

        # Distance: sum of pairwise distances.
        _equirect_distance_expr().sum().alias(Col.DISTANCE_M),

        # Speed: from SOG column if available.
        pl.col("sog").mean().alias(Col.MEAN_SPEED)
        if "sog" in df.columns
        else pl.lit(None).cast(pl.Float64).alias(Col.MEAN_SPEED),

        pl.col("sog").max().alias(Col.MAX_SPEED)
        if "sog" in df.columns
        else pl.lit(None).cast(pl.Float64).alias(Col.MAX_SPEED),

        # Spatial bounds.
        pl.col("lon").min().alias(Col.BBOX_WEST),
        pl.col("lat").min().alias(Col.BBOX_SOUTH),
        pl.col("lon").max().alias(Col.BBOX_EAST),
        pl.col("lat").max().alias(Col.BBOX_NORTH),

        # Provenance.
        pl.col("source").first().alias(Col.SOURCE),
    ]

    tracks = df.group_by("_segment_id").agg(agg_exprs)

    # Compute derived speed from distance/duration if SOG not available.
    if "sog" not in df.columns:
        tracks = tracks.with_columns(
            (
                pl.col(Col.DISTANCE_M)
                / pl.col(Col.DURATION_S).clip(lower_bound=1.0)
                / _KNOTS_TO_MPS
            ).alias(Col.MEAN_SPEED),
        )

    # Generate deterministic track_id.
    tracks = tracks.with_columns(
        pl.struct([Col.MMSI, Col.START_TIME, Col.SOURCE])
        .map_elements(
            lambda row: make_track_id(
                row[Col.MMSI],
                int(row[Col.START_TIME].timestamp() * 1e6),
                row[Col.SOURCE],
                cfg_hash,
            ),
            return_dtype=pl.String,
        )
        .alias(Col.TRACK_ID),
    )

    # Add provenance.
    tracks = tracks.with_columns(
        pl.lit(f"{source or 'derived'}:tracks").alias(Col.RECORD_PROVENANCE),
    )

    # Optional geometry.
    if config.include_geometry:
        tracks = _add_geometry(tracks, df)

    # Drop internal segment_id, select final columns.
    output_cols = [
        Col.TRACK_ID, Col.MMSI, Col.START_TIME, Col.END_TIME,
        Col.POINT_COUNT, Col.DISTANCE_M, Col.DURATION_S,
        Col.MEAN_SPEED, Col.MAX_SPEED,
        Col.BBOX_WEST, Col.BBOX_SOUTH, Col.BBOX_EAST, Col.BBOX_NORTH,
        Col.SOURCE, Col.RECORD_PROVENANCE,
    ]
    if config.include_geometry:
        output_cols.extend([Col.GEOMETRY_WKB, Col.TIMESTAMP_OFFSETS_MS])

    available = [c for c in output_cols if c in tracks.columns]
    return tracks.select(available).sort([Col.MMSI, Col.START_TIME])


def _add_geometry(
    tracks: pl.DataFrame,
    positions: pl.DataFrame,
) -> pl.DataFrame:
    """Add WKB geometry and timestamp offsets to track records.

    Collects lat/lon/timestamp per segment, encodes as WKB LineString,
    and computes per-vertex timestamp offsets relative to segment start.
    """
    from neptune_ais.datasets.tracks import Col

    # Collect per-segment coordinate and timestamp lists.
    geo_data = (
        positions.group_by("_segment_id")
        .agg(
            pl.col("lat").alias("_lats"),
            pl.col("lon").alias("_lons"),
            pl.col("timestamp").alias("_timestamps"),
        )
    )

    # Compute geometry and timestamp offsets in a single pass.
    geo_data = geo_data.with_columns(
        pl.struct(["_lats", "_lons", "_timestamps"])
        .map_elements(
            lambda row: {
                "wkb": _encode_wkb_linestring(row["_lats"], row["_lons"]),
                "offsets": _compute_timestamp_offsets(row["_timestamps"]),
            },
            return_dtype=pl.Struct({
                "wkb": pl.Binary,
                "offsets": pl.List(pl.Int64),
            }),
        )
        .alias("_geo_result"),
    )
    geo_data = geo_data.with_columns(
        pl.col("_geo_result").struct.field("wkb").alias(Col.GEOMETRY_WKB),
        pl.col("_geo_result").struct.field("offsets").alias(Col.TIMESTAMP_OFFSETS_MS),
    ).drop("_geo_result")

    # Join geometry columns back to tracks.
    return tracks.join(
        geo_data.select(["_segment_id", Col.GEOMETRY_WKB, Col.TIMESTAMP_OFFSETS_MS]),
        on="_segment_id",
        how="left",
    )


def _compute_timestamp_offsets(timestamps: list) -> list[int]:
    """Compute per-vertex ms offsets from the first timestamp."""
    if not timestamps:
        return []
    start = timestamps[0]
    return [int((t - start).total_seconds() * 1000) for t in timestamps]


def _encode_wkb_linestring(lats: list[float], lons: list[float]) -> bytes | None:
    """Encode a list of lat/lon pairs as a WKB LineString (little-endian).

    Returns None if fewer than 2 points (WKB LineString requires >= 2).

    WKB format:
    - 1 byte: byte order (01 = little-endian)
    - 4 bytes: geometry type (02000000 = LineString)
    - 4 bytes: number of points
    - For each point: 8 bytes lon (double) + 8 bytes lat (double)
    """
    import struct

    n = len(lats)
    if n < 2:
        return None

    parts = [
        struct.pack("<BII", 1, 2, n),  # byte order, type, num points
    ]
    for lat, lon in zip(lats, lons):
        parts.append(struct.pack("<dd", lon, lat))  # WKB: x=lon, y=lat

    return b"".join(parts)


def parse_track_args(
    gap: str = "30m",
    min_points: int = 3,
    min_duration: str = "5m",
    min_distance_m: float = 100.0,
    generalize: str = "0",
    include_geometry: bool = False,
    refresh: bool = False,
) -> TrackConfig:
    """Parse user-facing track arguments into a TrackConfig.

    Accepts human-readable duration strings (e.g. "30m", "1h", "15m")
    for gap and min_duration parameters.

    This is the bridge between ``Neptune.tracks(gap="30m", ...)`` and
    the internal ``TrackConfig``.
    """
    return TrackConfig(
        gap_seconds=_parse_duration(gap),
        min_points=min_points,
        min_duration_seconds=_parse_duration(min_duration),
        min_distance_m=min_distance_m,
        include_geometry=include_geometry,
        generalize_tolerance_m=_parse_distance(generalize),
        refresh=refresh,
    )


def _parse_duration(s: str) -> int:
    """Parse a duration string like '30m', '1h', '5m', '90s' to seconds."""
    s = s.strip().lower()
    if s.endswith("h"):
        return int(float(s[:-1]) * 3600)
    if s.endswith("m"):
        return int(float(s[:-1]) * 60)
    if s.endswith("s"):
        return int(float(s[:-1]))
    # Bare number → assume seconds.
    return int(float(s))


def _parse_distance(s: str) -> float:
    """Parse a distance string like '1m', '500m', '1km' to meters."""
    s = s.strip().lower()
    if s.endswith("km"):
        return float(s[:-2]) * 1000
    if s.endswith("m"):
        return float(s[:-1])
    # Bare number → assume meters.
    return float(s)
