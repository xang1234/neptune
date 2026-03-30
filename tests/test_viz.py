"""Tests for viz — viewport-aware layer preparation.

Covers: positions, tracks, trips, and density layer preparation,
viewport clipping, sampling, and edge cases.
"""

from __future__ import annotations

import polars as pl
import pytest

from neptune_ais.viz import (
    DashboardConfig,
    InfrastructurePoint,
    Viewport,
    _TRIP_PROGRESS,
    _auto_view,
    _build_trips,
    _safe_json_embed,
    _validate_track_geometry,
    generate_dashboard,
    prepare_density,
    prepare_events,
    prepare_positions,
    prepare_tracks,
    prepare_trips,
)
from neptune_ais.derive.crossings import GateLine


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
        assert (result["lon"] >= -74.5).all()
        assert (result["lon"] <= -73.0).all()

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


# ---------------------------------------------------------------------------
# DashboardConfig
# ---------------------------------------------------------------------------


class TestDashboardConfig:
    def test_defaults(self):
        cfg = DashboardConfig(title="Test")
        assert cfg.title == "Test"
        assert cfg.gate is None
        assert cfg.event_date is None
        assert cfg.speed == 21600.0
        assert cfg.infrastructure == []

    def test_with_gate(self):
        gate = GateLine("G", (0.0, 0.0), (1.0, 1.0))
        cfg = DashboardConfig(title="T", gate=gate, event_date="2026-03-01")
        assert cfg.gate is not None
        assert cfg.event_date == "2026-03-01"

    def test_with_infrastructure(self):
        infra = [InfrastructurePoint("Port", 25.0, 55.0, "port")]
        cfg = DashboardConfig(title="T", infrastructure=infra)
        assert len(cfg.infrastructure) == 1
        assert cfg.infrastructure[0].name == "Port"


class TestInfrastructurePoint:
    def test_construction(self):
        p = InfrastructurePoint("Refinery", 26.0, 56.0, "refinery")
        assert p.name == "Refinery"
        assert p.kind == "refinery"

    def test_default_kind(self):
        p = InfrastructurePoint("Harbor", 10.0, 20.0)
        assert p.kind == "port"


# ---------------------------------------------------------------------------
# _build_trips helper
# ---------------------------------------------------------------------------


def _synthetic_tracks_with_geometry() -> pl.DataFrame:
    """Create synthetic tracks DataFrame with geometry for testing."""
    import struct
    from datetime import datetime, timezone

    def _encode_wkb(coords: list[tuple[float, float]]) -> bytes:
        """Encode [[lon, lat], ...] as WKB LineString."""
        n = len(coords)
        buf = struct.pack("<BII", 1, 2, n)
        for lon, lat in coords:
            buf += struct.pack("<dd", lon, lat)
        return buf

    ts1 = datetime(2026, 2, 25, 0, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 2, 25, 1, 0, 0, tzinfo=timezone.utc)
    ts3 = datetime(2026, 2, 25, 2, 0, 0, tzinfo=timezone.utc)
    ts4 = datetime(2026, 2, 25, 3, 0, 0, tzinfo=timezone.utc)

    return pl.DataFrame({
        "track_id": ["t1", "t2"],
        "mmsi": [111, 222],
        "start_time": [ts1, ts3],
        "end_time": [ts2, ts4],
        "point_count": [3, 3],
        "distance_m": [1000.0, 2000.0],
        "duration_s": [3600.0, 3600.0],
        "mean_speed": [1.0, 2.0],
        "max_speed": [2.0, 3.0],
        "bbox_west": [55.0, 56.0],
        "bbox_south": [25.0, 26.0],
        "bbox_east": [56.0, 57.0],
        "bbox_north": [26.0, 27.0],
        "geometry_wkb": [
            _encode_wkb([(55.0, 25.0), (55.5, 25.5), (56.0, 26.0)]),
            _encode_wkb([(56.0, 26.0), (56.5, 26.5), (57.0, 27.0)]),
        ],
        "timestamp_offsets_ms": [
            [0, 1800000, 3600000],
            [0, 1800000, 3600000],
        ],
        "source": ["noaa", "noaa"],
    })


class TestBuildTrips:
    def test_basic_extraction(self):
        df = _synthetic_tracks_with_geometry()
        trips, colors, max_time, global_start = _build_trips(df)
        assert len(trips) == 2
        assert len(colors) == 2
        assert max_time > 0
        assert global_start is not None

    def test_trip_has_mmsi(self):
        df = _synthetic_tracks_with_geometry()
        trips, _, _, _ = _build_trips(df)
        assert trips[0]["mmsi"] in (111, 222)

    def test_trip_has_path_and_timestamps(self):
        df = _synthetic_tracks_with_geometry()
        trips, _, _, _ = _build_trips(df)
        for trip in trips:
            assert len(trip["path"]) == 3
            assert len(trip["timestamps"]) == 3
            assert len(trip["color"]) == 3


class TestAutoView:
    def test_computes_center_and_zoom(self):
        df = _synthetic_tracks_with_geometry()
        lat, lon, zoom = _auto_view(df)
        assert 25.0 < lat < 27.0
        assert 55.0 < lon < 57.0
        assert isinstance(zoom, int)


class TestSafeJsonEmbed:
    def test_escapes_script_close(self):
        result = _safe_json_embed({"name": "</script><script>alert(1)"})
        assert "</script>" not in result
        assert r"\u003c" in result

    def test_escapes_html_comment(self):
        result = _safe_json_embed({"x": "<!--comment-->"})
        assert "<!--" not in result

    def test_no_raw_angle_brackets(self):
        result = _safe_json_embed({"a": "<b>bold</b>"})
        assert "<b>" not in result
        assert r"\u003c" in result

    def test_normal_json_unchanged(self):
        result = _safe_json_embed({"mmsi": 123, "lat": 25.5})
        assert '"mmsi": 123' in result


class TestValidateTrackGeometry:
    def test_valid_tracks(self):
        df = _synthetic_tracks_with_geometry()
        result = _validate_track_geometry(df)
        assert len(result) == 2

    def test_missing_columns_raises(self):
        df = pl.DataFrame({"mmsi": [111], "foo": ["bar"]})
        with pytest.raises(ValueError, match="geometry_wkb"):
            _validate_track_geometry(df)


# ---------------------------------------------------------------------------
# generate_dashboard
# ---------------------------------------------------------------------------


class TestGenerateDashboard:
    def test_generates_html_file(self, tmp_path):
        df = _synthetic_tracks_with_geometry()
        cfg = DashboardConfig(title="Test Dashboard")
        out = generate_dashboard(df, config=cfg, output=str(tmp_path / "test.html"))
        assert out.endswith("test.html")
        import pathlib
        assert pathlib.Path(out).exists()
        html = pathlib.Path(out).read_text()
        assert "Test Dashboard" in html
        assert "deck.gl" in html

    def test_with_gate(self, tmp_path):
        df = _synthetic_tracks_with_geometry()
        gate = GateLine("Test Gate", (25.5, 55.5), (26.5, 55.5))
        cfg = DashboardConfig(title="Gate Test", gate=gate)
        out = generate_dashboard(df, config=cfg, output=str(tmp_path / "gate.html"))
        html = open(out).read()
        assert "Test Gate" in html or "gate" in html.lower()
        assert "HAS_GATE = true" in html

    def test_no_gate_mode(self, tmp_path):
        df = _synthetic_tracks_with_geometry()
        cfg = DashboardConfig(title="No Gate")
        out = generate_dashboard(df, config=cfg, output=str(tmp_path / "nogate.html"))
        html = open(out).read()
        assert "HAS_GATE = false" in html
        assert "no-gate" in html

    def test_with_vessels(self, tmp_path):
        from datetime import datetime, timezone
        df = _synthetic_tracks_with_geometry()
        vessels = pl.DataFrame({
            "mmsi": [111, 222],
            "vessel_name": ["MV Test", "SS Demo"],
            "ship_type": ["CARGO", "TANKER"],
            "flag": ["PA", "LR"],
            "imo": ["1234567", "7654321"],
            "length": [200.0, 150.0],
            "beam": [30.0, 25.0],
            "first_seen": [datetime(2026, 1, 1, tzinfo=timezone.utc)] * 2,
            "last_seen": [datetime(2026, 3, 1, tzinfo=timezone.utc)] * 2,
            "source": ["noaa", "noaa"],
        })
        out = generate_dashboard(
            df, vessels=vessels, config=DashboardConfig(title="V"),
            output=str(tmp_path / "v.html"),
        )
        html = open(out).read()
        assert "MV Test" in html
        assert "CARGO" in html

    def test_xss_protection(self, tmp_path):
        """Vessel names with </script> must not break the HTML."""
        from datetime import datetime, timezone
        df = _synthetic_tracks_with_geometry()
        vessels = pl.DataFrame({
            "mmsi": [111],
            "vessel_name": ["Evil</script><script>alert(1)"],
            "ship_type": ["CARGO"],
            "flag": ["XX"],
            "imo": ["0"],
            "length": [0.0],
            "beam": [0.0],
            "first_seen": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "last_seen": [datetime(2026, 3, 1, tzinfo=timezone.utc)],
            "source": ["noaa"],
        })
        out = generate_dashboard(
            df, vessels=vessels, config=DashboardConfig(title="XSS"),
            output=str(tmp_path / "xss.html"),
        )
        html = open(out).read()
        # All '<' in embedded JSON must be escaped as \u003c so the HTML
        # parser can't see </script> or <!-- inside the <script> block.
        json_start = html.index("const TRIPS =")
        json_end = html.index("</script>", json_start)
        json_block = html[json_start:json_end]
        assert "</script>" not in json_block
        assert r"\u003c" in json_block

    def test_max_tracks_downsampling(self, tmp_path):
        df = _synthetic_tracks_with_geometry()
        cfg = DashboardConfig(title="Small")
        out = generate_dashboard(
            df, config=cfg, output=str(tmp_path / "small.html"), max_tracks=1,
        )
        html = open(out).read()
        assert "Showing" in html or "showing" in html.lower() or "1 of 2" in html

    def test_with_infrastructure(self, tmp_path):
        df = _synthetic_tracks_with_geometry()
        infra = [InfrastructurePoint("Test Port", 25.5, 55.5)]
        cfg = DashboardConfig(title="Infra", infrastructure=infra)
        out = generate_dashboard(df, config=cfg, output=str(tmp_path / "infra.html"))
        html = open(out).read()
        assert "Test Port" in html
