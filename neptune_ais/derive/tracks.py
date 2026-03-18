"""Tracks derivation — native track segmentation pipeline.

Segments positions into trips using gap detection, jump filtering,
and configurable thresholds. Operates on Polars LazyFrames to avoid
the GeoDataFrame scaling bottleneck.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


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
