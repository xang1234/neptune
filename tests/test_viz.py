"""Tests for viz — viewport-aware layer preparation.

Covers: positions, tracks, trips, and density layer preparation,
viewport clipping, sampling, and edge cases.
"""

from __future__ import annotations

import polars as pl
import pytest

from neptune_ais.viz import (
    Viewport,
    _TRIP_PROGRESS,
    prepare_density,
    prepare_events,
    prepare_positions,
    prepare_tracks,
    prepare_trips,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_positions(n: int = 100) -> pl.DataFrame:
    """Generate *n* synthetic position rows spread across a bbox."""
    import random
    from datetime import datetime, timedelta, timezone

    rng = random.Random(42)
    base = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i) for i in range(n)]

    return pl.DataFrame({
        "mmsi": pl.Series([rng.choice([111, 222, 333]) for _ in range(n)], dtype=pl.Int64),
        "timestamp": pl.Series(timestamps, dtype=pl.Datetime("us", "UTC")),
        "lat": [40.0 + i * 0.01 for i in range(n)],
        "lon": [-74.0 + i * 0.01 for i in range(n)],
        "sog": [10.0] * n,
        "source": ["noaa"] * n,
        "record_provenance": ["noaa:direct"] * n,
        "qc_severity": ["ok"] * n,
    })


def _sample_tracks(n: int = 10, *, with_geometry: bool = False) -> pl.DataFrame:
    """Generate *n* synthetic track rows."""
    from datetime import datetime, timezone

    start_times = [
        datetime(2024, 6, 15, i, 0, 0, tzinfo=timezone.utc) for i in range(n)
    ]
    end_times = [
        datetime(2024, 6, 15, i, 30, 0, tzinfo=timezone.utc) for i in range(n)
    ]
    data = {
        "track_id": [f"t{i:016d}" for i in range(n)],
        "mmsi": pl.Series([111 + i % 3 for i in range(n)], dtype=pl.Int64),
        "start_time": pl.Series(start_times, dtype=pl.Datetime("us", "UTC")),
        "end_time": pl.Series(end_times, dtype=pl.Datetime("us", "UTC")),
        "point_count": [50] * n,
        "distance_m": [5000.0 + i * 100 for i in range(n)],
        "duration_s": [1800.0] * n,
        "mean_speed": [5.0] * n,
        "max_speed": [8.0] * n,
        # Tracks spread from lat 40–41, lon -74 to -73
        "bbox_west": [-74.0 + i * 0.05 for i in range(n)],
        "bbox_south": [40.0 + i * 0.05 for i in range(n)],
        "bbox_east": [-73.9 + i * 0.05 for i in range(n)],
        "bbox_north": [40.1 + i * 0.05 for i in range(n)],
        "source": ["noaa"] * n,
        "record_provenance": ["noaa:tracks"] * n,
    }

    if with_geometry:
        import struct

        wkb_list = []
        offsets_list = []
        for i in range(n):
            # Minimal 2-point LineString.
            lat1, lon1 = 40.0 + i * 0.05, -74.0 + i * 0.05
            lat2, lon2 = 40.1 + i * 0.05, -73.9 + i * 0.05
            wkb = struct.pack("<BII", 1, 2, 2)
            wkb += struct.pack("<dd", lon1, lat1)
            wkb += struct.pack("<dd", lon2, lat2)
            wkb_list.append(wkb)
            offsets_list.append([0, 1800000])  # 0ms to 30min
        data["geometry_wkb"] = wkb_list
        data["timestamp_offsets_ms"] = offsets_list

    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# Viewport validation
# ---------------------------------------------------------------------------


class TestViewport:
    def test_valid_viewport(self):
        v = Viewport(west=-74.0, south=40.0, east=-73.0, north=41.0)
        assert v.west == -74.0

    def test_invalid_lat_range(self):
        with pytest.raises(ValueError, match="latitude"):
            Viewport(west=-74.0, south=41.0, east=-73.0, north=40.0)

    def test_invalid_lon_range(self):
        with pytest.raises(ValueError, match="longitude"):
            Viewport(west=-200.0, south=40.0, east=-73.0, north=41.0)


# ---------------------------------------------------------------------------
# Positions layer
# ---------------------------------------------------------------------------


class TestPreparePositions:
    def test_passthrough_no_viewport(self):
        df = _sample_positions(50)
        result = prepare_positions(df)
        assert len(result) == 50

    def test_viewport_clipping(self):
        df = _sample_positions(100)
        viewport = Viewport(west=-74.0, south=40.0, east=-73.5, north=40.5)
        result = prepare_positions(df, viewport=viewport)
        # All returned rows must be within the viewport.
        assert (result["lat"] >= 40.0).all()
        assert (result["lat"] <= 40.5).all()
        assert (result["lon"] >= -74.0).all()
        assert (result["lon"] <= -73.5).all()
        assert len(result) < 100

    def test_sampling(self):
        df = _sample_positions(100)
        result = prepare_positions(df, max_points=20)
        assert len(result) == 20

    def test_viewport_and_sampling(self):
        df = _sample_positions(100)
        viewport = Viewport(west=-74.0, south=40.0, east=-73.0, north=41.0)
        result = prepare_positions(df, viewport=viewport, max_points=5)
        assert len(result) <= 5

    def test_lazyframe_input(self):
        df = _sample_positions(50)
        result = prepare_positions(df.lazy())
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 50

    def test_empty_after_clip(self):
        df = _sample_positions(50)
        # Viewport far from data.
        viewport = Viewport(west=10.0, south=50.0, east=11.0, north=51.0)
        result = prepare_positions(df, viewport=viewport)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tracks layer
# ---------------------------------------------------------------------------


class TestPrepareTracks:
    def test_passthrough_no_viewport(self):
        df = _sample_tracks(10)
        result = prepare_tracks(df)
        assert len(result) == 10

    def test_viewport_clips_by_bbox_intersection(self):
        df = _sample_tracks(10)
        # Viewport that only intersects the first few tracks.
        viewport = Viewport(west=-74.1, south=39.9, east=-73.8, north=40.15)
        result = prepare_tracks(df, viewport=viewport)
        assert 0 < len(result) < 10

    def test_sampling(self):
        df = _sample_tracks(10)
        result = prepare_tracks(df, max_tracks=3)
        assert len(result) == 3

    def test_lazyframe_input(self):
        df = _sample_tracks(5)
        result = prepare_tracks(df.lazy())
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Trips layer
# ---------------------------------------------------------------------------


class TestPrepareTrips:
    def test_with_geometry(self):
        df = _sample_tracks(5, with_geometry=True)
        result = prepare_trips(df)
        assert len(result) == 5
        assert _TRIP_PROGRESS in result.columns
        # Trip progress should be in [0, 1].
        assert result[_TRIP_PROGRESS].min() >= 0.0
        assert result[_TRIP_PROGRESS].max() <= 1.0

    def test_without_geometry_returns_empty(self):
        df = _sample_tracks(5, with_geometry=False)
        result = prepare_trips(df)
        assert len(result) == 0
        assert _TRIP_PROGRESS in result.columns

    def test_without_geometry_lazyframe_returns_empty(self):
        df = _sample_tracks(5, with_geometry=False)
        result = prepare_trips(df.lazy())
        assert len(result) == 0
        assert _TRIP_PROGRESS in result.columns

    def test_viewport_clipping(self):
        df = _sample_tracks(10, with_geometry=True)
        viewport = Viewport(west=-74.1, south=39.9, east=-73.8, north=40.15)
        result = prepare_trips(df, viewport=viewport)
        assert len(result) < 10

    def test_trip_progress_normalization(self):
        df = _sample_tracks(5, with_geometry=True)
        # Override duration_s to have varying values.
        df = df.with_columns(
            pl.Series("duration_s", [600.0, 1200.0, 1800.0, 900.0, 300.0])
        )
        result = prepare_trips(df)
        # The longest track (1800s) should have progress 1.0.
        assert result[_TRIP_PROGRESS].max() == pytest.approx(1.0)
        # The shortest (300s) should be 300/1800.
        assert result[_TRIP_PROGRESS].min() == pytest.approx(300.0 / 1800.0)


# ---------------------------------------------------------------------------
# Density layer
# ---------------------------------------------------------------------------


class TestPrepareDensity:
    def test_basic_density(self):
        df = _sample_positions(50)
        result = prepare_density(df)
        assert "h3_index" in result.columns
        assert "count" in result.columns
        assert "center_lat" in result.columns
        assert "center_lon" in result.columns
        # Total count should equal input row count.
        assert result["count"].sum() == 50

    def test_viewport_clipping(self):
        df = _sample_positions(100)
        viewport = Viewport(west=-74.0, south=40.0, east=-73.5, north=40.5)
        clipped = prepare_density(df, viewport=viewport)
        full = prepare_density(df)
        assert clipped["count"].sum() <= full["count"].sum()

    def test_empty_result(self):
        df = _sample_positions(10)
        viewport = Viewport(west=10.0, south=50.0, east=11.0, north=51.0)
        result = prepare_density(df, viewport=viewport)
        assert len(result) == 0
        assert result.columns == ["h3_index", "count", "center_lat", "center_lon"]

    def test_sorted_descending_by_count(self):
        df = _sample_positions(100)
        result = prepare_density(df)
        counts = result["count"].to_list()
        assert counts == sorted(counts, reverse=True)

    def test_sampling_before_binning(self):
        df = _sample_positions(100)
        result = prepare_density(df, max_points=20)
        assert result["count"].sum() == 20


# ---------------------------------------------------------------------------
# Events layer
# ---------------------------------------------------------------------------


def _sample_events(n: int = 5) -> pl.DataFrame:
    """Generate *n* synthetic event rows."""
    from datetime import datetime, timedelta, timezone

    base = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
    types = ["port_call", "eez_crossing", "encounter", "loitering", "port_call"]
    return pl.DataFrame({
        "event_id": [f"evt_{i:03d}" for i in range(n)],
        "event_type": types[:n],
        "mmsi": pl.Series([111 + i % 3 for i in range(n)], dtype=pl.Int64),
        "other_mmsi": pl.Series(
            [None, None, 222, None, None][:n], dtype=pl.Int64
        ),
        "start_time": pl.Series(
            [base + timedelta(hours=i) for i in range(n)],
            dtype=pl.Datetime("us", "UTC"),
        ),
        "end_time": pl.Series(
            [base + timedelta(hours=i, minutes=30) for i in range(n)],
            dtype=pl.Datetime("us", "UTC"),
        ),
        "lat": [40.0 + i * 0.5 for i in range(n)],
        "lon": [-74.0 + i * 0.5 for i in range(n)],
        "geometry_wkb": pl.Series([None] * n, dtype=pl.Binary),
        "confidence_score": [0.9, 0.5, 0.8, 0.3, 0.7][:n],
        "source": ["noaa"] * n,
        "record_provenance": ["noaa:detector/0.1.0[positions]"] * n,
    })


class TestPrepareEvents:
    def test_passthrough_no_filters(self):
        df = _sample_events(5)
        result = prepare_events(df)
        assert len(result) == 5

    def test_viewport_clipping(self):
        df = _sample_events(5)
        viewport = Viewport(west=-74.5, south=39.5, east=-73.0, north=40.5)
        result = prepare_events(df, viewport=viewport)
        assert len(result) < 5
        assert (result["lat"] >= 39.5).all()
        assert (result["lat"] <= 40.5).all()

    def test_event_type_filter(self):
        df = _sample_events(5)
        result = prepare_events(df, event_type="port_call")
        assert len(result) == 2
        assert (result["event_type"] == "port_call").all()

    def test_min_confidence_filter(self):
        df = _sample_events(5)
        result = prepare_events(df, min_confidence=0.7)
        assert len(result) == 3  # 0.9, 0.8, 0.7
        assert (result["confidence_score"] >= 0.7).all()

    def test_combined_filters(self):
        df = _sample_events(5)
        # Narrow viewport to only include the first port_call (lat=40, lon=-74).
        viewport = Viewport(west=-75.0, south=39.0, east=-73.0, north=41.0)
        result = prepare_events(
            df, viewport=viewport, event_type="port_call", min_confidence=0.7
        )
        # Only the first port_call (lat=40, conf=0.9) is inside this viewport.
        assert len(result) == 1
        assert result["confidence_score"][0] == 0.9

    def test_sampling(self):
        df = _sample_events(5)
        result = prepare_events(df, max_events=2)
        assert len(result) == 2

    def test_lazyframe_input(self):
        df = _sample_events(5)
        result = prepare_events(df.lazy())
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 5

    def test_empty_after_filters(self):
        df = _sample_events(5)
        result = prepare_events(df, event_type="fishing")
        assert len(result) == 0
