"""Tests for derive.crossings — gate crossing detection and reversals."""

from __future__ import annotations

import pytest

from neptune_ais.derive.crossings import (
    GateLine,
    _segments_intersect,
    detect_gate_crossings,
    detect_reversals,
)


# ---------------------------------------------------------------------------
# GateLine
# ---------------------------------------------------------------------------


class TestGateLine:
    def test_valid_construction(self):
        g = GateLine("Test", (26.5, 56.2), (26.2, 56.4))
        assert g.name == "Test"
        assert g.point_a == (26.5, 56.2)
        assert g.point_b == (26.2, 56.4)

    def test_identical_points_rejected(self):
        with pytest.raises(ValueError, match="must differ"):
            GateLine("Bad", (10.0, 20.0), (10.0, 20.0))

    def test_invalid_latitude(self):
        with pytest.raises(ValueError, match="latitude"):
            GateLine("Bad", (95.0, 0.0), (0.0, 0.0))

    def test_invalid_longitude(self):
        with pytest.raises(ValueError, match="longitude"):
            GateLine("Bad", (0.0, 200.0), (0.0, 0.0))

    def test_frozen(self):
        g = GateLine("Test", (0.0, 0.0), (1.0, 1.0))
        with pytest.raises(AttributeError):
            g.name = "Changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Segment intersection
# ---------------------------------------------------------------------------


class TestSegmentsIntersect:
    def test_perpendicular_crossing(self):
        # Horizontal segment crossing vertical segment.
        hit, t = _segments_intersect((0, -1), (0, 1), (-1, 0), (1, 0))
        assert hit is True
        assert abs(t - 0.5) < 1e-9

    def test_no_intersection_parallel(self):
        hit, _ = _segments_intersect((0, 0), (1, 0), (0, 1), (1, 1))
        assert hit is False

    def test_no_intersection_non_overlapping(self):
        # Segments that would intersect if extended but don't actually overlap.
        hit, _ = _segments_intersect((0, 0), (1, 0), (2, -1), (2, 1))
        assert hit is False

    def test_t_intersection_at_endpoint(self):
        hit, t = _segments_intersect((0, 0), (2, 0), (1, -1), (1, 1))
        assert hit is True
        assert abs(t - 0.5) < 1e-9

    def test_parametric_t_value(self):
        # Segment from (0,0) to (4,0) crossing a vertical at x=1.
        hit, t = _segments_intersect((0, 0), (4, 0), (1, -1), (1, 1))
        assert hit is True
        assert abs(t - 0.25) < 1e-9

    def test_collinear_segments(self):
        hit, _ = _segments_intersect((0, 0), (2, 0), (1, 0), (3, 0))
        assert hit is False  # denom ~0 → treated as parallel

    def test_diagonal_crossing(self):
        hit, t = _segments_intersect((0, 0), (2, 2), (0, 2), (2, 0))
        assert hit is True
        assert abs(t - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Gate crossing detection
# ---------------------------------------------------------------------------


def _make_trip(
    mmsi: int,
    path: list[list[float]],
    timestamps: list[float],
) -> dict:
    """Create a minimal trip dict."""
    return {
        "mmsi": mmsi,
        "path": path,
        "timestamps": timestamps,
        "color": [0, 200, 255],
    }


class TestDetectGateCrossings:
    # Gate is a vertical line at lon=0, from lat -1 to lat 1.
    GATE = GateLine("Test Gate", (-1.0, 0.0), (1.0, 0.0))

    def test_single_crossing_inbound(self):
        # Vessel moves from lon=-1 to lon=1 (left to right facing south→north).
        trip = _make_trip(111, [[-1, 0], [1, 0]], [0.0, 10.0])
        crossings = detect_gate_crossings([trip], self.GATE)
        assert len(crossings) == 1
        assert crossings[0]["mmsi"] == 111
        assert crossings[0]["direction"] == "inbound"
        assert abs(crossings[0]["timestamp_s"] - 5.0) < 0.1
        assert abs(crossings[0]["lon"] - 0.0) < 0.01

    def test_single_crossing_outbound(self):
        # Vessel moves from lon=1 to lon=-1 (right to left).
        trip = _make_trip(222, [[1, 0], [-1, 0]], [0.0, 10.0])
        crossings = detect_gate_crossings([trip], self.GATE)
        assert len(crossings) == 1
        assert crossings[0]["direction"] == "outbound"

    def test_no_crossing(self):
        # Vessel stays on one side.
        trip = _make_trip(333, [[-2, 0], [-1, 0]], [0.0, 10.0])
        crossings = detect_gate_crossings([trip], self.GATE)
        assert len(crossings) == 0

    def test_multiple_crossings(self):
        # Vessel crosses gate twice (out and back).
        trip = _make_trip(
            444,
            [[-1, 0], [1, 0], [-1, 0]],
            [0.0, 10.0, 20.0],
        )
        crossings = detect_gate_crossings([trip], self.GATE)
        assert len(crossings) == 2
        assert crossings[0]["direction"] == "inbound"
        assert crossings[1]["direction"] == "outbound"

    def test_timestamp_interpolation(self):
        # Gate at lon=0; vessel goes from lon=-2 to lon=2 in 100s.
        # Should cross at t=50.
        trip = _make_trip(555, [[-2, 0], [2, 0]], [0.0, 100.0])
        crossings = detect_gate_crossings([trip], self.GATE)
        assert len(crossings) == 1
        assert abs(crossings[0]["timestamp_s"] - 50.0) < 0.5

    def test_multiple_vessels(self):
        trips = [
            _make_trip(111, [[-1, 0], [1, 0]], [0.0, 10.0]),
            _make_trip(222, [[1, 0], [-1, 0]], [5.0, 15.0]),
        ]
        crossings = detect_gate_crossings(trips, self.GATE)
        assert len(crossings) == 2
        mmsis = {c["mmsi"] for c in crossings}
        assert mmsis == {111, 222}

    def test_sorted_by_time(self):
        trips = [
            _make_trip(222, [[1, 0], [-1, 0]], [20.0, 30.0]),
            _make_trip(111, [[-1, 0], [1, 0]], [0.0, 10.0]),
        ]
        crossings = detect_gate_crossings(trips, self.GATE)
        assert crossings[0]["timestamp_s"] <= crossings[1]["timestamp_s"]

    def test_empty_trips(self):
        crossings = detect_gate_crossings([], self.GATE)
        assert crossings == []


# ---------------------------------------------------------------------------
# Reversal detection
# ---------------------------------------------------------------------------


class TestDetectReversals:
    def test_simple_reversal(self):
        crossings = [
            {"mmsi": 111, "timestamp_s": 100.0, "direction": "inbound",
             "lat": 0.0, "lon": 0.0},
            {"mmsi": 111, "timestamp_s": 200.0, "direction": "outbound",
             "lat": 0.0, "lon": 0.0},
        ]
        revs = detect_reversals(crossings, window_hours=1.0)
        assert len(revs) == 1
        assert revs[0]["mmsi"] == 111
        assert "inbound" in revs[0]["direction_sequence"]
        assert "outbound" in revs[0]["direction_sequence"]

    def test_outside_window(self):
        crossings = [
            {"mmsi": 111, "timestamp_s": 0.0, "direction": "inbound",
             "lat": 0.0, "lon": 0.0},
            {"mmsi": 111, "timestamp_s": 100000.0, "direction": "outbound",
             "lat": 0.0, "lon": 0.0},
        ]
        revs = detect_reversals(crossings, window_hours=1.0)
        assert len(revs) == 0

    def test_same_direction_not_reversal(self):
        crossings = [
            {"mmsi": 111, "timestamp_s": 100.0, "direction": "inbound",
             "lat": 0.0, "lon": 0.0},
            {"mmsi": 111, "timestamp_s": 200.0, "direction": "inbound",
             "lat": 0.0, "lon": 0.0},
        ]
        revs = detect_reversals(crossings, window_hours=48.0)
        assert len(revs) == 0

    def test_different_vessels_not_confused(self):
        crossings = [
            {"mmsi": 111, "timestamp_s": 100.0, "direction": "inbound",
             "lat": 0.0, "lon": 0.0},
            {"mmsi": 222, "timestamp_s": 200.0, "direction": "outbound",
             "lat": 0.0, "lon": 0.0},
        ]
        revs = detect_reversals(crossings, window_hours=48.0)
        assert len(revs) == 0

    def test_multiple_reversals(self):
        crossings = [
            {"mmsi": 111, "timestamp_s": 0.0, "direction": "inbound",
             "lat": 0.0, "lon": 0.0},
            {"mmsi": 111, "timestamp_s": 100.0, "direction": "outbound",
             "lat": 0.0, "lon": 0.0},
            {"mmsi": 111, "timestamp_s": 200.0, "direction": "inbound",
             "lat": 0.0, "lon": 0.0},
        ]
        revs = detect_reversals(crossings, window_hours=48.0)
        assert len(revs) == 2

    def test_empty_crossings(self):
        revs = detect_reversals([], window_hours=48.0)
        assert revs == []
