"""Crossings — gate/chokepoint crossing detection.

Detects when vessel tracks cross a geographic gate line (chokepoint),
classifies crossing direction (inbound/outbound), and identifies
same-hull reversals (vessels that cross in both directions within a
time window).

Uses pure-Python parametric line-segment intersection — no spatial
library dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# GateLine — geographic chokepoint definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateLine:
    """Geographic line segment that vessels cross (chokepoint / gate).

    Defined by two WGS-84 endpoints. Crossing direction is determined
    by the cross-product of the gate vector (A → B) with the vessel's
    movement vector at the crossing point:

    - **Inbound**: vessel crosses from left to right when facing A → B.
    - **Outbound**: vessel crosses from right to left.

    Args:
        name: Human-readable gate name (e.g., "Strait of Hormuz").
        point_a: ``(lat, lon)`` tuple for one endpoint.
        point_b: ``(lat, lon)`` tuple for the other endpoint.
    """

    name: str
    point_a: tuple[float, float]  # (lat, lon)
    point_b: tuple[float, float]  # (lat, lon)

    def __post_init__(self) -> None:
        for label, pt in [("point_a", self.point_a), ("point_b", self.point_b)]:
            if not (-90 <= pt[0] <= 90):
                raise ValueError(f"{label} latitude {pt[0]} out of range [-90, 90]")
            if not (-180 <= pt[1] <= 180):
                raise ValueError(f"{label} longitude {pt[1]} out of range [-180, 180]")
        if self.point_a == self.point_b:
            raise ValueError("point_a and point_b must differ")


# ---------------------------------------------------------------------------
# Line-segment intersection
# ---------------------------------------------------------------------------


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> tuple[bool, float]:
    """Test whether two line segments intersect.

    Uses the parametric form: each segment is parameterized as
    ``P(t) = p1 + t*(p2-p1)`` and ``Q(u) = q1 + u*(q2-q1)``.
    Intersection requires ``0 <= t <= 1`` and ``0 <= u <= 1``.

    Args:
        p1: Start of segment 1 (vessel) as ``(lon, lat)``.
        p2: End of segment 1 as ``(lon, lat)``.
        q1: Start of segment 2 (gate) as ``(lon, lat)``.
        q2: End of segment 2 as ``(lon, lat)``.

    Returns:
        ``(intersects, t)`` where *t* is the parametric position
        along segment 1 in ``[0, 1]``, useful for interpolating
        the crossing timestamp.
    """
    dx1 = p2[0] - p1[0]
    dy1 = p2[1] - p1[1]
    dx2 = q2[0] - q1[0]
    dy2 = q2[1] - q1[1]

    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-12:
        return False, 0.0  # parallel or collinear

    dx3 = q1[0] - p1[0]
    dy3 = q1[1] - p1[1]

    t = (dx3 * dy2 - dy3 * dx2) / denom
    u = (dx3 * dy1 - dy3 * dx1) / denom

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return True, t
    return False, 0.0


# ---------------------------------------------------------------------------
# Gate crossing detection
# ---------------------------------------------------------------------------


def detect_gate_crossings(
    trips: list[dict],
    gate: GateLine,
) -> list[dict]:
    """Detect all crossings of a gate line by vessel tracks.

    Walks each track's coordinate pairs, tests for intersection with
    the gate line, and classifies direction using the cross-product.

    Args:
        trips: List of trip dicts as produced by ``viz._build_trips()``.
            Each must have ``path`` (``[[lon, lat], ...]``),
            ``timestamps`` (seconds), and ``mmsi`` (int) keys.
        gate: The gate line to test crossings against.

    Returns:
        List of crossing dicts, each with keys:

        - ``mmsi`` (int): Vessel identifier.
        - ``timestamp_s`` (float): Crossing time in seconds (same
          epoch as trip timestamps).
        - ``direction`` (str): ``"inbound"`` or ``"outbound"``.
        - ``lat`` (float): Crossing latitude.
        - ``lon`` (float): Crossing longitude.
    """
    # Gate endpoints in (lon, lat) order to match path coordinate order.
    ga = (gate.point_a[1], gate.point_a[0])
    gb = (gate.point_b[1], gate.point_b[0])

    # Gate direction vector for cross-product.
    gate_dx = gb[0] - ga[0]
    gate_dy = gb[1] - ga[1]

    crossings: list[dict] = []

    for trip in trips:
        path = trip["path"]
        timestamps = trip["timestamps"]
        mmsi = trip["mmsi"]

        for i in range(len(path) - 1):
            p1 = (path[i][0], path[i][1])
            p2 = (path[i + 1][0], path[i + 1][1])

            hit, t = _segments_intersect(p1, p2, ga, gb)
            if not hit:
                continue

            # Interpolate crossing position.
            cx = p1[0] + t * (p2[0] - p1[0])
            cy = p1[1] + t * (p2[1] - p1[1])

            # Interpolate crossing timestamp.
            ct = timestamps[i] + t * (timestamps[i + 1] - timestamps[i])

            # Determine direction via cross-product of gate vector
            # with vessel movement vector.
            vessel_dx = p2[0] - p1[0]
            vessel_dy = p2[1] - p1[1]
            cross = gate_dx * vessel_dy - gate_dy * vessel_dx

            crossings.append({
                "mmsi": mmsi,
                "timestamp_s": round(ct, 3),
                "direction": "inbound" if cross < 0 else "outbound",
                "lat": round(cy, 6),
                "lon": round(cx, 6),
            })

    # detect_reversals requires crossings sorted by timestamp_s.
    crossings.sort(key=lambda c: c["timestamp_s"])
    return crossings


# ---------------------------------------------------------------------------
# Same-hull reversal detection
# ---------------------------------------------------------------------------


def detect_reversals(
    crossings: list[dict],
    window_hours: float = 48.0,
) -> list[dict]:
    """Find vessels that cross the gate in both directions within a time window.

    A reversal is when a vessel crosses inbound then outbound (or vice
    versa) within ``window_hours``. This is a key intelligence signal
    for round-trip traffic through chokepoints.

    Args:
        crossings: Crossing dicts from :func:`detect_gate_crossings`,
            assumed sorted by ``timestamp_s``.
        window_hours: Maximum hours between first crossing and reversal.
            Default 48.

    Returns:
        List of reversal dicts, each with keys:

        - ``mmsi`` (int): Vessel identifier.
        - ``first_crossing_s`` (float): Timestamp of the first crossing.
        - ``reversal_s`` (float): Timestamp of the direction change.
        - ``direction_sequence`` (str): e.g. ``"inbound→outbound"``.
    """
    window_s = window_hours * 3600.0

    # Group crossings by MMSI, preserving time order.
    by_mmsi: dict[int, list[dict]] = {}
    for c in crossings:
        by_mmsi.setdefault(c["mmsi"], []).append(c)

    reversals: list[dict] = []

    for mmsi, vessel_crossings in by_mmsi.items():
        for i in range(len(vessel_crossings) - 1):
            c1 = vessel_crossings[i]
            c2 = vessel_crossings[i + 1]
            if c1["direction"] != c2["direction"]:
                dt = c2["timestamp_s"] - c1["timestamp_s"]
                if dt <= window_s:
                    reversals.append({
                        "mmsi": mmsi,
                        "first_crossing_s": c1["timestamp_s"],
                        "reversal_s": c2["timestamp_s"],
                        "direction_sequence": (
                            f"{c1['direction']}\u2192{c2['direction']}"
                        ),
                    })

    reversals.sort(key=lambda r: r["first_crossing_s"])
    return reversals
