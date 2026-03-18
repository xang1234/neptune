"""Phase 3 acceptance: track derivation and map-preparation validation.

Validates the full pipeline from positions → track segmentation →
aggregation → viz layer preparation. Covers:

1. Segmentation edge cases (gaps, speed jumps, multi-vessel, non-monotonic).
2. Segment filtering thresholds (min_points, min_duration, min_distance).
3. Track aggregation schema conformance (tracks/v1).
4. Geometry encoding (WKB LineString, timestamp offsets).
5. Integration: derived tracks feed into viz.prepare_tracks/prepare_trips.
6. Config determinism (same inputs → same track_ids).
"""

from __future__ import annotations

import struct
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from neptune_ais.datasets.tracks import (
    Col as TrackCol,
    REQUIRED_COLUMNS as TRACK_REQUIRED,
    SCHEMA as TRACK_SCHEMA,
    make_track_id,
)
from neptune_ais.derive.tracks import (
    TrackConfig,
    aggregate_tracks,
    detect_boundaries,
    filter_segments,
    parse_track_args,
    _encode_wkb_linestring,
    _compute_timestamp_offsets,
)
from neptune_ais.viz import (
    Viewport,
    _TRIP_PROGRESS,
    prepare_density,
    prepare_positions,
    prepare_tracks,
    prepare_trips,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BASE = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)

# Permissive config that keeps all segments (for boundary detection tests).
_PERMISSIVE = TrackConfig(min_points=1, min_duration_seconds=0, min_distance_m=0)


def _ts(minutes: float) -> datetime:
    """Helper: timestamp at *minutes* after base time."""
    return _BASE + timedelta(minutes=minutes)


def _run_pipeline(
    df: pl.DataFrame,
    *,
    include_geometry: bool = False,
    config: TrackConfig | None = None,
) -> pl.DataFrame:
    """Run detect → filter → aggregate and return tracks."""
    if config is None:
        config = TrackConfig(
            min_points=1, min_duration_seconds=0, min_distance_m=0,
            include_geometry=include_geometry,
        )
    segmented = detect_boundaries(df, config)
    filtered = filter_segments(segmented, config)
    return aggregate_tracks(filtered, config, source="noaa")


# ---------------------------------------------------------------------------
# Position fixtures
# ---------------------------------------------------------------------------


def _two_vessel_positions() -> pl.DataFrame:
    """Two vessels, each with a clear single segment.

    Vessel 111: 5 points, 1-minute intervals, moving north.
    Vessel 222: 4 points, 2-minute intervals, moving east.
    Pre-sorted by (mmsi, timestamp).
    """
    return pl.DataFrame({
        "mmsi": pl.Series([111] * 5 + [222] * 4, dtype=pl.Int64),
        "timestamp": pl.Series([
            _ts(0), _ts(1), _ts(2), _ts(3), _ts(4),  # vessel 111
            _ts(0), _ts(2), _ts(4), _ts(6),           # vessel 222
        ], dtype=pl.Datetime("us", "UTC")),
        "lat": [40.0, 40.01, 40.02, 40.03, 40.04,
                55.0, 55.0, 55.0, 55.0],
        "lon": [-74.0, -74.0, -74.0, -74.0, -74.0,
                12.0, 12.01, 12.02, 12.03],
        "sog": [5.0, 5.0, 5.0, 5.0, 5.0,
                3.0, 3.0, 3.0, 3.0],
        "source": ["noaa"] * 9,
    })


def _gapped_positions() -> pl.DataFrame:
    """Single vessel with a 45-minute gap mid-track.

    Expected: 2 segments when gap_seconds=1800 (30m default).
    Points 0-3: minutes 0,1,2,3  (segment A)
    Points 4-6: minutes 48,49,50 (segment B, after 45m gap)
    """
    return pl.DataFrame({
        "mmsi": pl.Series([111] * 7, dtype=pl.Int64),
        "timestamp": pl.Series([
            _ts(0), _ts(1), _ts(2), _ts(3),
            _ts(48), _ts(49), _ts(50),
        ], dtype=pl.Datetime("us", "UTC")),
        "lat": [40.0, 40.01, 40.02, 40.03,
                40.10, 40.11, 40.12],
        "lon": [-74.0, -74.0, -74.0, -74.0,
                -74.0, -74.0, -74.0],
        "sog": [5.0] * 7,
        "source": ["noaa"] * 7,
    })


def _speed_jump_positions() -> pl.DataFrame:
    """Single vessel with a GPS jump (implausible speed).

    Points 0-2: normal motion
    Point 3: jumps ~500km instantly (1-minute gap → ~500 km/min → impossible)
    Points 3-5: normal motion at new location
    Expected: 2 segments (break at the jump).
    """
    return pl.DataFrame({
        "mmsi": pl.Series([111] * 6, dtype=pl.Int64),
        "timestamp": pl.Series([
            _ts(0), _ts(1), _ts(2),
            _ts(3), _ts(4), _ts(5),
        ], dtype=pl.Datetime("us", "UTC")),
        "lat": [40.0, 40.01, 40.02,
                45.0, 45.01, 45.02],  # 5-degree jump
        "lon": [-74.0, -74.0, -74.0,
                -74.0, -74.0, -74.0],
        "sog": [5.0] * 6,
        "source": ["noaa"] * 6,
    })


def _nonmonotonic_positions() -> pl.DataFrame:
    """Single vessel with a non-monotonic timestamp.

    Point 2 has a timestamp BEFORE point 1.
    Expected: segment break at the non-monotonic point.
    """
    return pl.DataFrame({
        "mmsi": pl.Series([111] * 5, dtype=pl.Int64),
        "timestamp": pl.Series([
            _ts(0), _ts(2), _ts(1),  # non-monotonic: 2 then 1
            _ts(3), _ts(4),
        ], dtype=pl.Datetime("us", "UTC")),
        "lat": [40.0, 40.01, 40.005, 40.02, 40.03],
        "lon": [-74.0] * 5,
        "sog": [5.0] * 5,
        "source": ["noaa"] * 5,
    })


# ---------------------------------------------------------------------------
# Segmentation — detect_boundaries
# ---------------------------------------------------------------------------


class TestDetectBoundaries:
    """Tests for segment boundary detection."""

    def test_single_vessel_no_gaps(self):
        """Continuous positions → single segment."""
        df = _two_vessel_positions().filter(pl.col("mmsi") == 111)
        result = detect_boundaries(df, _PERMISSIVE)
        assert "_segment_id" in result.columns
        assert result["_segment_id"].n_unique() == 1

    def test_two_vessels_separate_segments(self):
        """Different vessels get different segment IDs."""
        df = _two_vessel_positions()
        result = detect_boundaries(df, _PERMISSIVE)
        # Group by mmsi and check each vessel has unique segment IDs.
        seg_per_vessel = (
            result.group_by("mmsi")
            .agg(pl.col("_segment_id").n_unique().alias("n_segs"))
        )
        # Each vessel should have exactly 1 segment (no gaps within).
        assert (seg_per_vessel["n_segs"] == 1).all()
        # But the two vessels should have different segment IDs.
        assert result["_segment_id"].n_unique() == 2

    def test_time_gap_creates_boundary(self):
        """45-minute gap exceeds 30m default → 2 segments."""
        df = _gapped_positions()
        result = detect_boundaries(df, _PERMISSIVE)
        assert result["_segment_id"].n_unique() == 2

    def test_time_gap_with_larger_threshold(self):
        """45-minute gap within 60m threshold → 1 segment."""
        df = _gapped_positions()
        config = TrackConfig(
            gap_seconds=3600,  # 1 hour
            min_points=1, min_duration_seconds=0, min_distance_m=0,
        )
        result = detect_boundaries(df, config)
        assert result["_segment_id"].n_unique() == 1

    def test_speed_jump_creates_boundary(self):
        """GPS jump (implausible speed) → segment break."""
        df = _speed_jump_positions()
        result = detect_boundaries(df, _PERMISSIVE)
        assert result["_segment_id"].n_unique() == 2

    def test_nonmonotonic_timestamp_creates_boundary(self):
        """Decreasing timestamp → segment break."""
        df = _nonmonotonic_positions()
        result = detect_boundaries(df, _PERMISSIVE)
        # Non-monotonic at point 2 should create a boundary.
        assert result["_segment_id"].n_unique() >= 2

    def test_preserves_original_columns(self):
        """detect_boundaries adds _segment_id but doesn't drop input columns."""
        df = _two_vessel_positions()
        config = TrackConfig()
        result = detect_boundaries(df, config)
        for col in ["mmsi", "timestamp", "lat", "lon", "sog", "source"]:
            assert col in result.columns
        assert "_segment_id" in result.columns
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# Segment filtering
# ---------------------------------------------------------------------------


class TestFilterSegments:
    """Tests for segment threshold filtering."""

    def test_min_points_filters_short_segments(self):
        """Segments with < min_points are dropped."""
        df = _gapped_positions()
        config = TrackConfig(min_points=4, min_duration_seconds=0, min_distance_m=0)
        segmented = detect_boundaries(df, config)
        filtered = filter_segments(segmented, config)
        # Segment A has 4 points (kept), segment B has 3 points (dropped).
        assert filtered["_segment_id"].n_unique() == 1
        assert len(filtered) == 4

    def test_min_duration_filters_short_segments(self):
        """Segments shorter than min_duration are dropped."""
        df = _two_vessel_positions()
        config = TrackConfig(
            min_points=1,
            min_duration_seconds=300,  # 5m: above v111's 4m, below v222's 6m
            min_distance_m=0,
        )
        segmented = detect_boundaries(df, config)
        filtered = filter_segments(segmented, config)
        # Vessel 111: 4 min duration (dropped). Vessel 222: 6 min duration (kept).
        assert len(filtered) > 0
        assert (filtered["mmsi"] == 222).all()

    def test_all_segments_filtered_returns_empty(self):
        """If no segments meet thresholds, return empty DataFrame."""
        df = _two_vessel_positions()
        config = TrackConfig(
            min_points=100,  # impossibly high
            min_duration_seconds=0,
            min_distance_m=0,
        )
        segmented = detect_boundaries(df, config)
        filtered = filter_segments(segmented, config)
        assert len(filtered) == 0

    def test_passthrough_when_all_meet_thresholds(self):
        """All segments kept when thresholds are very low."""
        df = _two_vessel_positions()
        config = TrackConfig(min_points=1, min_duration_seconds=0, min_distance_m=0)
        segmented = detect_boundaries(df, config)
        filtered = filter_segments(segmented, config)
        assert len(filtered) == len(segmented)


# ---------------------------------------------------------------------------
# Track aggregation and schema conformance
# ---------------------------------------------------------------------------


class TestAggregateTracksSchema:
    """Tests for aggregate_tracks output conformance to tracks/v1 schema."""

    def test_required_columns_present(self):
        """All tracks/v1 required columns must be present."""
        tracks = _run_pipeline(_two_vessel_positions())
        for col in TRACK_REQUIRED:
            assert col in tracks.columns, f"Missing required column: {col}"

    def test_column_dtypes_match_schema(self):
        """Column dtypes match the tracks/v1 schema."""
        tracks = _run_pipeline(_two_vessel_positions())
        for col_name, expected_dtype in TRACK_SCHEMA.items():
            if col_name in tracks.columns:
                actual = tracks[col_name].dtype
                assert actual == expected_dtype, (
                    f"{col_name}: expected {expected_dtype}, got {actual}"
                )

    def test_one_row_per_segment(self):
        """Two vessels → two track rows."""
        tracks = _run_pipeline(_two_vessel_positions())
        assert len(tracks) == 2

    def test_track_id_is_deterministic(self):
        """Same input → same track_id."""
        tracks_a = _run_pipeline(_two_vessel_positions())
        tracks_b = _run_pipeline(_two_vessel_positions())
        assert tracks_a[TrackCol.TRACK_ID].to_list() == tracks_b[TrackCol.TRACK_ID].to_list()

    def test_track_id_changes_with_config(self):
        """Different config → different track_ids."""
        df = _two_vessel_positions()
        config_a = TrackConfig(
            gap_seconds=1800, min_points=1, min_duration_seconds=0, min_distance_m=0,
        )
        config_b = TrackConfig(
            gap_seconds=900, min_points=1, min_duration_seconds=0, min_distance_m=0,
        )
        tracks_a = _run_pipeline(df, config=config_a)
        tracks_b = _run_pipeline(df, config=config_b)
        # IDs should differ because config_hash changed.
        ids_a = set(tracks_a[TrackCol.TRACK_ID].to_list())
        ids_b = set(tracks_b[TrackCol.TRACK_ID].to_list())
        assert ids_a != ids_b

    def test_temporal_bounds_correct(self):
        """start_time <= end_time for every track."""
        tracks = _run_pipeline(_two_vessel_positions())
        assert (tracks[TrackCol.START_TIME] <= tracks[TrackCol.END_TIME]).all()

    def test_spatial_bounds_correct(self):
        """bbox_west <= bbox_east and bbox_south <= bbox_north."""
        tracks = _run_pipeline(_two_vessel_positions())
        assert (tracks[TrackCol.BBOX_WEST] <= tracks[TrackCol.BBOX_EAST]).all()
        assert (tracks[TrackCol.BBOX_SOUTH] <= tracks[TrackCol.BBOX_NORTH]).all()

    def test_point_count_matches_input(self):
        """point_count per track matches number of input positions."""
        tracks = _run_pipeline(_two_vessel_positions())
        # Vessel 111: 5 points, vessel 222: 4 points.
        counts = tracks.sort(TrackCol.MMSI)[TrackCol.POINT_COUNT].to_list()
        assert counts == [5, 4]

    def test_source_and_provenance(self):
        """source and record_provenance are set correctly."""
        tracks = _run_pipeline(_two_vessel_positions())
        assert (tracks[TrackCol.SOURCE] == "noaa").all()
        assert (tracks[TrackCol.RECORD_PROVENANCE] == "noaa:tracks").all()

    def test_sorted_by_mmsi_start_time(self):
        """Output is sorted by (mmsi, start_time)."""
        tracks = _run_pipeline(_two_vessel_positions())
        mmsis = tracks[TrackCol.MMSI].to_list()
        assert mmsis == sorted(mmsis)

    def test_mean_speed_from_sog(self):
        """mean_speed is derived from SOG when available."""
        tracks = _run_pipeline(_two_vessel_positions())
        # All SOG values are 5.0 or 3.0 — mean_speed should reflect that.
        assert tracks[TrackCol.MEAN_SPEED].is_not_null().all()
        for row in tracks.iter_rows(named=True):
            assert row[TrackCol.MEAN_SPEED] > 0


# ---------------------------------------------------------------------------
# Geometry encoding
# ---------------------------------------------------------------------------


class TestGeometryEncoding:
    """Tests for WKB LineString encoding and timestamp offsets."""

    def test_include_geometry_adds_columns(self):
        """include_geometry=True adds geometry_wkb and timestamp_offsets_ms."""
        df = _two_vessel_positions()
        config = TrackConfig(
            min_points=1, min_duration_seconds=0, min_distance_m=0,
            include_geometry=True,
        )
        segmented = detect_boundaries(df, config)
        filtered = filter_segments(segmented, config)
        tracks = aggregate_tracks(filtered, config, source="noaa")
        assert TrackCol.GEOMETRY_WKB in tracks.columns
        assert TrackCol.TIMESTAMP_OFFSETS_MS in tracks.columns
        # All tracks should have geometry (>= 2 points each).
        assert tracks[TrackCol.GEOMETRY_WKB].is_not_null().all()

    def test_wkb_linestring_structure(self):
        """WKB bytes have correct header and point count."""
        lats = [40.0, 40.01, 40.02]
        lons = [-74.0, -74.01, -74.02]
        wkb = _encode_wkb_linestring(lats, lons)
        assert wkb is not None
        # Parse header.
        byte_order, geom_type, n_points = struct.unpack_from("<BII", wkb, 0)
        assert byte_order == 1  # little-endian
        assert geom_type == 2  # LineString
        assert n_points == 3
        # Total size: 9 (header) + 3 * 16 (points) = 57 bytes.
        assert len(wkb) == 9 + 3 * 16

    def test_wkb_coordinate_order(self):
        """WKB stores coordinates as (lon, lat) per OGC spec."""
        wkb = _encode_wkb_linestring([40.0, 41.0], [-74.0, -73.0])
        # Skip 9-byte header, read first point.
        x, y = struct.unpack_from("<dd", wkb, 9)
        assert x == -74.0  # lon first
        assert y == 40.0   # lat second

    def test_wkb_minimum_two_points(self):
        """Exactly 2 points is the minimum valid LineString."""
        wkb = _encode_wkb_linestring([40.0, 41.0], [-74.0, -73.0])
        assert wkb is not None
        _, _, n_points = struct.unpack_from("<BII", wkb, 0)
        assert n_points == 2
        assert len(wkb) == 9 + 2 * 16

    def test_wkb_returns_none_for_single_point(self):
        """WKB LineString requires >= 2 points."""
        assert _encode_wkb_linestring([40.0], [-74.0]) is None
        assert _encode_wkb_linestring([], []) is None

    def test_timestamp_offsets_values(self):
        """Offsets are milliseconds from segment start."""
        timestamps = [_ts(0), _ts(1), _ts(5), _ts(10)]
        offsets = _compute_timestamp_offsets(timestamps)
        assert offsets == [0, 60_000, 300_000, 600_000]

    def test_timestamp_offsets_empty(self):
        assert _compute_timestamp_offsets([]) == []


# ---------------------------------------------------------------------------
# TrackConfig and parse_track_args
# ---------------------------------------------------------------------------


class TestTrackConfig:
    def test_default_config_is_valid(self):
        config = TrackConfig()
        assert config.validate() == []

    def test_config_hash_deterministic(self):
        a = TrackConfig().config_hash()
        b = TrackConfig().config_hash()
        assert a == b

    def test_config_hash_differs_for_different_params(self):
        a = TrackConfig(gap_seconds=1800).config_hash()
        b = TrackConfig(gap_seconds=900).config_hash()
        assert a != b

    def test_refresh_does_not_affect_hash(self):
        a = TrackConfig(refresh=False).config_hash()
        b = TrackConfig(refresh=True).config_hash()
        assert a == b

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="gap_seconds"):
            TrackConfig(gap_seconds=0)
        with pytest.raises(ValueError, match="min_points"):
            TrackConfig(min_points=0)

    def test_parse_track_args_durations(self):
        config = parse_track_args(gap="15m", min_duration="1h")
        assert config.gap_seconds == 900
        assert config.min_duration_seconds == 3600

    def test_parse_track_args_seconds(self):
        config = parse_track_args(gap="90s")
        assert config.gap_seconds == 90


# ---------------------------------------------------------------------------
# Integration: derive → viz pipeline
# ---------------------------------------------------------------------------


class TestDerivationToVizIntegration:
    """End-to-end: positions → tracks → viz layer preparation."""

    def test_prepare_tracks_accepts_derived_output(self):
        """Derived tracks feed directly into prepare_tracks."""
        tracks = _run_pipeline(_two_vessel_positions())
        result = prepare_tracks(tracks)
        assert len(result) == 2
        # Has all the required track columns.
        for col in TRACK_REQUIRED:
            assert col in result.columns

    def test_prepare_tracks_viewport_clips_derived(self):
        """Viewport clipping works on derived tracks."""
        tracks = _run_pipeline(_two_vessel_positions())
        # Viewport covers only vessel 111's area (lat ~40, lon ~-74).
        viewport = Viewport(west=-75.0, south=39.0, east=-73.0, north=41.0)
        result = prepare_tracks(tracks, viewport=viewport)
        assert len(result) == 1
        assert result[TrackCol.MMSI].to_list() == [111]

    def test_prepare_trips_requires_geometry(self):
        """prepare_trips returns empty when tracks lack geometry."""
        tracks = _run_pipeline(_two_vessel_positions(), include_geometry=False)
        result = prepare_trips(tracks)
        assert len(result) == 0
        assert _TRIP_PROGRESS in result.columns

    def test_prepare_trips_with_geometry(self):
        """prepare_trips works when tracks have geometry."""
        tracks = _run_pipeline(_two_vessel_positions(), include_geometry=True)
        result = prepare_trips(tracks)
        assert len(result) == 2
        assert _TRIP_PROGRESS in result.columns
        assert result[_TRIP_PROGRESS].is_not_null().all()
        # Progress values in [0, 1].
        assert result[_TRIP_PROGRESS].min() >= 0.0
        assert result[_TRIP_PROGRESS].max() <= 1.0

    def test_prepare_positions_with_source_data(self):
        """Positions used for derivation also work with prepare_positions."""
        positions = _two_vessel_positions()
        result = prepare_positions(positions)
        assert len(result) == 9  # all positions

    def test_prepare_density_with_source_data(self):
        """Density preparation works with the positions used for derivation."""
        positions = _two_vessel_positions()
        result = prepare_density(positions)
        assert result["count"].sum() == 9
        assert "h3_index" in result.columns

    def test_gapped_tracks_through_viz(self):
        """Gapped positions → 2 tracks → viewport can select one."""
        tracks = _run_pipeline(_gapped_positions())
        assert len(tracks) == 2

        # First segment is around lat 40.0-40.03, second around 40.10-40.12.
        # Viewport that covers only the second segment.
        viewport = Viewport(west=-75.0, south=40.05, east=-73.0, north=41.0)
        result = prepare_tracks(tracks, viewport=viewport)
        assert len(result) == 1
        assert result[TrackCol.BBOX_SOUTH][0] >= 40.05


# ---------------------------------------------------------------------------
# make_track_id contract
# ---------------------------------------------------------------------------


class TestMakeTrackId:
    def test_format(self):
        tid = make_track_id(111, 1718409600000000, "noaa", "abc123")
        assert isinstance(tid, str)
        assert len(tid) == 16
        assert all(c in "0123456789abcdef" for c in tid)

    def test_deterministic(self):
        a = make_track_id(111, 1718409600000000, "noaa", "abc")
        b = make_track_id(111, 1718409600000000, "noaa", "abc")
        assert a == b

    def test_differs_with_different_config(self):
        a = make_track_id(111, 1718409600000000, "noaa", "abc")
        b = make_track_id(111, 1718409600000000, "noaa", "xyz")
        assert a != b
