#!/usr/bin/env python3
"""Animated strait-crossing GIF from AIS data.

Renders a dark-map timelapse of vessel traffic around a strait: vessels are
drawn as direction-oriented arrows colored by type (stationary vessels as
dots), with fading trails, dashed rings around vessels actively crossing a
user-defined gate line (color-coded by direction), a running UTC clock, and
a per-day crossing-count panel.

Usage:
    python scripts/generate_crossings_gif.py                     # demo: Straits of Florida
    python scripts/generate_crossings_gif.py \
        --bbox -84.8,21.7,-79.2,27.8 \
        --gate -81.8,24.50,-81.8,23.20 \
        --date 2024-06-15 --days 2 \
        --title "Straits of Florida" \
        --output assets/crossings_florida.gif
    python scripts/generate_crossings_gif.py --self-test

Output: an animated GIF (default ``assets/crossings_florida.gif``).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import urllib.request
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("crossings_gif")

# ---------------------------------------------------------------------------
# Theme — neutral dark palette. Categorical slots validated for the dark
# surface (dataviz six-checks); pleasure craft is deliberately the one
# low-chroma slot so the dominant class recedes and commercial traffic pops.
# ---------------------------------------------------------------------------
SURFACE = "#10141a"
LAND = "#1b222b"
COAST = "#323c48"
GRID = "#182029"
INK = "#ffffff"
INK_2 = "#c3c2b7"
INK_MUTED = "#898781"
PANEL_EDGE = (1.0, 1.0, 1.0, 0.12)

CATEGORIES = [  # (key, legend label, hex)
    ("cargo", "Cargo", "#3987e5"),
    ("tanker", "Tankers", "#d95926"),
    ("passenger", "Passenger", "#9085e9"),
    ("tug", "Tugs & special craft", "#199e70"),
    ("hsc", "High-speed craft", "#c98500"),
    ("fishing", "Fishing", "#008300"),
    ("pleasure", "Pleasure craft", "#56789c"),
    ("other", "Other", "#d55181"),
]
CAT_INDEX = {k: i for i, (k, _, _) in enumerate(CATEGORIES)}
RING_COLORS = ["#e8e6df", "#6c7788"]  # direction 0 / 1 crossing rings

LAND_TOPO_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2/land-50m.json"
LAND_CACHE = Path.home() / ".neptune" / "viz_assets" / "land_50m.json"


def categorize(ship_type: str | None) -> str:
    """Map an AIS ship-type code (NOAA numeric) to a legend category."""
    try:
        code = int(ship_type)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "other"
    if 70 <= code <= 79:
        return "cargo"
    if 80 <= code <= 89:
        return "tanker"
    if 60 <= code <= 69:
        return "passenger"
    if 31 <= code <= 35 or 50 <= code <= 55:
        return "tug"
    if 40 <= code <= 49:
        return "hsc"
    if code == 30:
        return "fishing"
    if code in (36, 37):
        return "pleasure"
    return "other"


def _format_date_label(days: list[_date]) -> str:
    """Format a single date or date range without platform-specific directives."""
    if len(days) > 1:
        return f"{days[0].day}–{days[-1].day} {days[-1].strftime('%b %Y')}"
    return f"{days[0].day} {days[0].strftime('%b %Y')}"


# ---------------------------------------------------------------------------
# TopoJSON land decoding (world-atlas land-50m, quantized delta-encoded arcs)
# ---------------------------------------------------------------------------
def decode_topojson_rings(topo: dict, object_name: str) -> list[list[tuple[float, float]]]:
    """Decode all polygon rings of one TopoJSON object to lon/lat point lists."""
    scale = topo["transform"]["scale"]
    translate = topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in arc:
            x += dx
            y += dy
            pts.append((x * scale[0] + translate[0], y * scale[1] + translate[1]))
        arcs.append(pts)

    def ring(arc_idxs: list[int]) -> list[tuple[float, float]]:
        pts: list[tuple[float, float]] = []
        for i in arc_idxs:
            seg = arcs[i] if i >= 0 else arcs[~i][::-1]
            pts.extend(seg if not pts else seg[1:])
        return pts

    rings: list[list[tuple[float, float]]] = []
    obj = topo["objects"][object_name]
    geoms = obj["geometries"] if obj["type"] == "GeometryCollection" else [obj]
    for g in geoms:
        if g["type"] == "Polygon":
            polys = [g["arcs"]]
        elif g["type"] == "MultiPolygon":
            polys = g["arcs"]
        else:
            continue
        for poly in polys:
            rings.extend(ring(r) for r in poly)
    return rings


def load_land_rings() -> list[list[tuple[float, float]]]:
    if not LAND_CACHE.exists():
        log.info("Downloading land TopoJSON to %s ...", LAND_CACHE)
        LAND_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(LAND_TOPO_URL, timeout=60) as resp:
            LAND_CACHE.write_bytes(resp.read())
    return decode_topojson_rings(json.loads(LAND_CACHE.read_text()), "land")


# ---------------------------------------------------------------------------
# Gate-crossing detection
# ---------------------------------------------------------------------------
def detect_crossings(
    ts: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    gate: tuple[float, float, float, float],
    max_gap_s: float = 1800.0,
    dedupe_s: float = 3600.0,
) -> list[tuple[float, int]]:
    """Return [(unix_time, direction)] where a track crosses the gate segment.

    ``direction`` is 0 if motion has a positive component along the gate
    normal (oriented toward +x, or +y for near-E/W gates), else 1.
    Consecutive crossings within ``dedupe_s`` collapse to the first.
    """
    gx0, gy0, gx1, gy1 = gate
    gvx, gvy = gx1 - gx0, gy1 - gy0
    nx, ny = gvy, -gvx  # normal
    if (abs(nx) >= abs(ny) and nx < 0) or (abs(ny) > abs(nx) and ny < 0):
        nx, ny = -nx, -ny

    p0x, p0y, p1x, p1y = x[:-1], y[:-1], x[1:], y[1:]
    ok = (ts[1:] - ts[:-1]) <= max_gap_s
    # side of gate line for each endpoint
    d0 = gvx * (p0y - gy0) - gvy * (p0x - gx0)
    d1 = gvx * (p1y - gy0) - gvy * (p1x - gx0)
    # side of track segment for each gate endpoint
    svx, svy = p1x - p0x, p1y - p0y
    e0 = svx * (gy0 - p0y) - svy * (gx0 - p0x)
    e1 = svx * (gy1 - p0y) - svy * (gx1 - p0x)
    hit = ok & (np.sign(d0) != np.sign(d1)) & (np.sign(e0) != np.sign(e1))

    events: list[tuple[float, int]] = []
    for i in np.flatnonzero(hit):
        frac = abs(d0[i]) / (abs(d0[i]) + abs(d1[i]) + 1e-12)
        t = ts[i] + frac * (ts[i + 1] - ts[i])
        direction = 0 if (svx[i] * nx + svy[i] * ny) > 0 else 1
        if events and t - events[-1][0] < dedupe_s:
            continue
        events.append((float(t), direction))
    return events


def direction_labels(gate: tuple[float, float, float, float]) -> tuple[str, str]:
    """Human labels for direction 0/1 given the gate orientation."""
    gx0, gy0, gx1, gy1 = gate
    nx, ny = gy1 - gy0, -(gx1 - gx0)
    if abs(nx) >= abs(ny):  # near-N/S gate -> crossings run E/W
        return ("W → E", "E → W")
    return ("S → N", "N → S")


# ---------------------------------------------------------------------------
# Track interpolation onto the frame grid
# ---------------------------------------------------------------------------
def interpolate_tracks(
    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]],
    frame_ts: np.ndarray,
    max_gap_s: float = 1800.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate per-vessel (ts, x, y, cat) fixes onto frame timestamps.

    Returns X, Y (n_vessels x n_frames, NaN outside coverage/gaps) and the
    per-vessel category index array.
    """
    n_v, n_f = len(groups), len(frame_ts)
    X = np.full((n_v, n_f), np.nan)
    Y = np.full((n_v, n_f), np.nan)
    cats = np.zeros(n_v, dtype=int)
    for vi, (ts, x, y, cat) in enumerate(groups):
        cats[vi] = cat
        if len(ts) < 2:
            continue
        j = np.searchsorted(ts, frame_ts, side="right") - 1
        valid = (j >= 0) & (j < len(ts) - 1)
        jj = np.clip(j, 0, len(ts) - 2)
        gap_ok = (ts[jj + 1] - ts[jj]) <= max_gap_s
        use = valid & gap_ok
        w = (frame_ts - ts[jj]) / np.maximum(ts[jj + 1] - ts[jj], 1e-9)
        X[vi, use] = (x[jj] * (1 - w) + x[jj + 1] * w)[use]
        Y[vi, use] = (y[jj] * (1 - w) + y[jj + 1] * w)[use]
    return X, Y, cats


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_gif(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    cats: np.ndarray,
    frame_ts: np.ndarray,
    events: list[list[tuple[float, int]]],
    day_counts: list[tuple[str, str, int]],
    land_rings: list[list[tuple[float, float]]],
    bbox: tuple[float, float, float, float],
    gate: tuple[float, float, float, float],
    cos0: float,
    title: str,
    subtitle: str,
    date_label: str,
    output: str,
    width: int,
    fps: int,
    show_gate: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    w_lon, s_lat, e_lon, n_lat = bbox
    x0, x1 = w_lon * cos0, e_lon * cos0
    y0, y1 = s_lat, n_lat
    height = int(round(width * (y1 - y0) / (x1 - x0)))
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(SURFACE)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")

    # graticule
    for lon in range(math.ceil(w_lon), math.floor(e_lon) + 1):
        ax.axvline(lon * cos0, color=GRID, lw=0.6, zorder=1)
    for lat in range(math.ceil(s_lat), math.floor(n_lat) + 1):
        ax.axhline(lat, color=GRID, lw=0.6, zorder=1)

    # land (single nonzero-winding path; preserved ring orientation keeps lakes water-colored)
    verts: list[tuple[float, float]] = []
    codes: list[int] = []
    for ring in land_rings:
        rx = [p[0] for p in ring]
        if max(rx) < w_lon - 5 or min(rx) > e_lon + 5:
            continue
        ry = [p[1] for p in ring]
        if max(ry) < s_lat - 5 or min(ry) > n_lat + 5:
            continue
        pts = [(px * cos0, py) for px, py in ring]
        verts.extend(pts)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 2) + [MplPath.CLOSEPOLY])
    if verts:
        land_patch = PathPatch(
            MplPath(verts, codes), facecolor=LAND, edgecolor=COAST, lw=0.8, zorder=2
        )
        ax.add_patch(land_patch)

    if show_gate:
        ax.plot(
            [gate[0], gate[2]], [gate[1], gate[3]],
            color=INK_MUTED, lw=0.9, ls=(0, (5, 4)), alpha=0.55, zorder=3,
        )

    cat_rgb = np.array([matplotlib.colors.to_rgb(c) for _, _, c in CATEGORIES])
    v_rgb = cat_rgb[cats]
    small = np.isin(cats, [CAT_INDEX["pleasure"], CAT_INDEX["other"]])
    tri_size = np.where(small, 0.006, 0.009) * (x1 - x0)  # ponytail: size by class, not vessel length

    trails = LineCollection([], linewidths=1.0, zorder=4, capstyle="round")
    ax.add_collection(trails)
    heads = PolyCollection([], zorder=6)
    ax.add_collection(heads)
    dots = ax.scatter([], [], s=3.5, zorder=5, edgecolors="none")
    ring_scatters = [
        ax.scatter(
            [], [], s=340, facecolors="none", edgecolors=c,
            linewidths=1.3, linestyle=(0, (4, 3)), zorder=7, alpha=0.95,
        )
        for c in RING_COLORS
    ]

    # --- HUD (axes-fraction coordinates) ---
    def panel(xf, yf, wf, hf):
        p = FancyBboxPatch(
            (xf, yf), wf, hf, transform=ax.transAxes,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor=SURFACE, alpha=0.86, edgecolor=PANEL_EDGE, lw=1.0, zorder=9,
        )
        ax.add_patch(p)

    mono = {"family": "monospace"}
    ax.text(0.045, 0.972, "●", transform=ax.transAxes, color=INK_2, fontsize=5,
            va="top", zorder=10)
    ax.text(0.065, 0.978, "AIS TIMELAPSE", transform=ax.transAxes, color=INK_2,
            fontsize=7.5, va="top", zorder=10, **mono)
    ax.text(0.045, 0.952, title, transform=ax.transAxes, color=INK, fontsize=15,
            fontweight="bold", va="top", zorder=10)
    ax.text(0.045, 0.912, f"{subtitle}  ·  {date_label}", transform=ax.transAxes,
            color=INK_2, fontsize=9, va="top", zorder=10)

    for i, (_, label, color) in enumerate(CATEGORIES):
        yf = 0.862 - i * 0.026
        ax.scatter([0.052], [yf], transform=ax.transAxes, marker=(3, 0, 0),
                   s=26, color=color, zorder=10)
        ax.text(0.072, yf, label, transform=ax.transAxes, color=INK_2, fontsize=7.5,
                va="center", zorder=10)

    # clock + stats block, bottom center
    dir0, dir1 = direction_labels(gate)
    panel(0.28, 0.155, 0.44, 0.052)
    clock_txt = ax.text(0.5, 0.181, "", transform=ax.transAxes, color=INK, fontsize=12,
                        ha="center", va="center", zorder=10, **mono)

    n_days = len(day_counts)
    stats_w = max(0.44, 0.115 + 0.115 * n_days)
    sx = 0.5 - stats_w / 2
    panel(sx, 0.028, stats_w, 0.112)
    ax.text(sx + 0.018, 0.124, "VESSEL CROSSINGS", transform=ax.transAxes, color=INK_2,
            fontsize=7.5, va="center", zorder=10, **mono)
    for k, (color, lbl) in enumerate(zip(RING_COLORS, (dir0, dir1))):
        cx = sx + stats_w - 0.20 + k * 0.10
        ax.scatter([cx], [0.124], transform=ax.transAxes, s=60, facecolors="none",
                   edgecolors=color, linewidths=1.1, linestyle=(0, (3, 2)), zorder=10)
        ax.text(cx + 0.012, 0.124, lbl, transform=ax.transAxes, color=INK_MUTED,
                fontsize=6.5, va="center", zorder=10, **mono)
    col_w = stats_w / max(n_days, 1)
    for k, (wd, dm, cnt) in enumerate(day_counts):
        cx = sx + col_w * (k + 0.5)
        ax.text(cx, 0.098, f"{wd} {dm}", transform=ax.transAxes, color=INK_2,
                fontsize=7, ha="center", zorder=10, **mono)
        ax.text(cx, 0.062, str(cnt), transform=ax.transAxes, color=INK, fontsize=15,
                fontweight="bold", ha="center", zorder=10)
        ax.text(cx, 0.040, "crossings", transform=ax.transAxes, color=INK_MUTED,
                fontsize=6, ha="center", zorder=10)

    ax.text(0.045, 0.018, "DATA: NOAA MARINE CADASTRE (AIS)", transform=ax.transAxes,
            color=INK_MUTED, fontsize=6, zorder=10, **mono)
    ax.text(0.955, 0.018, "NEPTUNE AIS", transform=ax.transAxes, color=INK_2,
            fontsize=8, ha="right", zorder=10, fontweight="bold")

    # --- per-frame update ---
    trail_len = 9
    n_frames = len(frame_ts)
    ring_window = 1500.0  # seconds a crossing ring stays visible each side
    tri_u = np.array([[0.0, 1.0], [-0.42, -0.65], [0.42, -0.65]])

    # per-frame motion direction (unit vectors); NaN-safe
    dX = np.gradient(X, axis=1)
    dY = np.gradient(Y, axis=1)
    speed = np.hypot(dX, dY)
    moving = speed > 1e-4 * (x1 - x0) / 100

    writer = PillowWriter(fps=fps)
    t_start = time.time()
    with writer.saving(fig, output, dpi):
        for f in range(n_frames):
            segs, seg_colors = [], []
            for k in range(1, min(trail_len, f) + 1):
                a_x, a_y = X[:, f - k], Y[:, f - k]
                b_x, b_y = X[:, f - k + 1], Y[:, f - k + 1]
                m = np.isfinite(a_x) & np.isfinite(b_x)
                if not m.any():
                    continue
                s = np.stack(
                    [np.stack([a_x[m], a_y[m]], 1), np.stack([b_x[m], b_y[m]], 1)], 1
                )
                alpha = 0.38 * (1 - (k - 1) / trail_len)
                c = np.concatenate(
                    [v_rgb[m], np.full((m.sum(), 1), alpha)], axis=1
                )
                segs.append(s)
                seg_colors.append(c)
            if segs:
                trails.set_segments(list(np.concatenate(segs)))
                trails.set_color(np.concatenate(seg_colors))
            else:
                trails.set_segments([])

            px, py = X[:, f], Y[:, f]
            alive = np.isfinite(px)
            mv = alive & moving[:, f]
            st = alive & ~moving[:, f]

            if mv.any():
                ux = dX[mv, f] / np.maximum(speed[mv, f], 1e-12)
                uy = dY[mv, f] / np.maximum(speed[mv, f], 1e-12)
                sz = tri_size[mv]
                # rotate template: dir=(ux,uy), perp=(uy,-ux)
                vx = (tri_u[None, :, 0] * uy[:, None] + tri_u[None, :, 1] * ux[:, None])
                vy = (-tri_u[None, :, 0] * ux[:, None] + tri_u[None, :, 1] * uy[:, None])
                polys = np.stack(
                    [px[mv][:, None] + vx * sz[:, None], py[mv][:, None] + vy * sz[:, None]],
                    axis=2,
                )
                heads.set_verts(list(polys))
                heads.set_facecolor(v_rgb[mv])
            else:
                heads.set_verts([])
            dots.set_offsets(np.stack([px[st], py[st]], 1) if st.any() else np.empty((0, 2)))
            dots.set_facecolor(
                np.concatenate([v_rgb[st], np.full((st.sum(), 1), 0.75)], 1)
                if st.any() else np.empty((0, 4))
            )

            t_now = frame_ts[f]
            for d in (0, 1):
                pts = [
                    (px[vi], py[vi])
                    for vi, evs in enumerate(events)
                    if np.isfinite(px[vi])
                    for (te, ed) in evs
                    if ed == d and abs(te - t_now) <= ring_window
                ]
                ring_scatters[d].set_offsets(np.array(pts) if pts else np.empty((0, 2)))

            dt = datetime.fromtimestamp(t_now, tz=timezone.utc)
            clock_txt.set_text(dt.strftime("%a %d %b  %H:%M UTC").upper())

            writer.grab_frame()
            if f % 40 == 0:
                log.info("frame %d/%d (%.1fs)", f, n_frames, time.time() - t_start)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_floats(s: str, n: int, name: str) -> tuple[float, ...]:
    parts = [float(p) for p in s.split(",")]
    if len(parts) != n:
        raise argparse.ArgumentTypeError(f"{name} needs {n} comma-separated numbers")
    return tuple(parts)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bbox", default="-84.8,21.7,-79.2,27.8",
                   help="Map bounds: W,S,E,N (lon/lat degrees).")
    p.add_argument("--gate", default="-81.8,24.50,-81.8,23.20",
                   help="Crossing gate segment: lon1,lat1,lon2,lat2.")
    p.add_argument("--date", default="2024-06-15", help="Start date (YYYY-MM-DD).")
    p.add_argument("--days", type=int, default=2, help="Number of days. Default: 2.")
    p.add_argument("--source", default="noaa", help="AIS source. Default: noaa.")
    p.add_argument("--title", default="Straits of Florida")
    p.add_argument("--subtitle", default="Vessel crossings")
    p.add_argument("--output", default=str(_REPO_ROOT / "assets" / "crossings_florida.gif"))
    p.add_argument("--width", type=int, default=640, help="GIF width px. Default: 640.")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--seconds", type=float, default=20.0, help="GIF duration.")
    p.add_argument("--max-vessels", type=int, default=4000,
                   help="Keep the N vessels with the most fixes.")
    p.add_argument("--hide-gate", action="store_true", help="Do not draw the gate line.")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def self_test() -> int:
    # gate crossing: eastbound track over a N-S gate
    ts = np.array([0.0, 600.0, 1200.0])
    x = np.array([-1.0, 1.0, 2.0])
    y = np.array([0.5, 0.5, 0.5])
    ev = detect_crossings(ts, x, y, gate=(0.0, 0.0, 0.0, 1.0))
    assert len(ev) == 1 and ev[0][1] == 0, ev
    assert abs(ev[0][0] - 300.0) < 1e-6, ev
    # westbound + jitter dedupe: back-and-forth within an hour counts once
    ts2 = np.array([0.0, 600.0, 1200.0, 1800.0])
    x2 = np.array([1.0, -0.1, 0.1, -1.0])
    ev2 = detect_crossings(ts2, x2, np.full(4, 0.5), gate=(0.0, 0.0, 0.0, 1.0))
    assert len(ev2) == 1 and ev2[0][1] == 1, ev2
    # gap guard: crossing across a 2h data hole is ignored
    ev3 = detect_crossings(
        np.array([0.0, 7200.0]), np.array([-1.0, 1.0]), np.array([0.5, 0.5]),
        gate=(0.0, 0.0, 0.0, 1.0),
    )
    assert ev3 == [], ev3
    # topojson decode: one square arc, quantized
    topo = {
        "transform": {"scale": [1.0, 1.0], "translate": [10.0, 20.0]},
        "arcs": [[[0, 0], [2, 0], [0, 2], [-2, 0], [0, -2]]],
        "objects": {"land": {"type": "Polygon", "arcs": [[0]]}},
    }
    rings = decode_topojson_rings(topo, "land")
    assert rings == [[(10.0, 20.0), (12.0, 20.0), (12.0, 22.0), (10.0, 22.0), (10.0, 20.0)]]
    # direction labels flip with gate orientation
    assert direction_labels((0, 0, 0, 1))[0].startswith("W")
    assert direction_labels((0, 0, 1, 0))[0].startswith("S")
    print("self-test OK")
    return 0


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.self_test:
        return self_test()

    import polars as pl

    from neptune_ais import Neptune
    from neptune_ais.datasets.positions import Col as C

    bbox = _parse_floats(args.bbox, 4, "--bbox")
    gate_ll = _parse_floats(args.gate, 4, "--gate")
    w_lon, s_lat, e_lon, n_lat = bbox
    cos0 = math.cos(math.radians((s_lat + n_lat) / 2))

    start = _date.fromisoformat(args.date)
    days = [start + timedelta(days=i) for i in range(args.days)]
    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = start_dt + timedelta(days=args.days)

    log.info("Loading %s positions for %s day(s) from %s ...", args.source, args.days, start)
    neptune = Neptune(dates=(days[0].isoformat(), days[-1].isoformat()), sources=[args.source])
    neptune.download()
    df = (
        neptune.positions()
        .filter(
            (pl.col(C.LON) >= w_lon) & (pl.col(C.LON) <= e_lon)
            & (pl.col(C.LAT) >= s_lat) & (pl.col(C.LAT) <= n_lat)
            & (pl.col(C.TIMESTAMP) >= pl.lit(start_dt).cast(pl.Datetime("us", "UTC")))
            & (pl.col(C.TIMESTAMP) < pl.lit(end_dt).cast(pl.Datetime("us", "UTC")))
        )
        .select([C.MMSI, C.TIMESTAMP, C.LAT, C.LON, C.SHIP_TYPE])
        .collect()
        .sort([C.MMSI, C.TIMESTAMP])
    )
    if len(df) == 0:
        log.error("No positions in bbox/window.")
        return 1
    log.info("%s positions, %s vessels", f"{len(df):,}", df[C.MMSI].n_unique())

    # dominant ship_type per vessel, transformed coords
    df = df.with_columns(
        (pl.col(C.LON) * cos0).alias("x"),
        pl.col(C.LAT).alias("y"),
        (pl.col(C.TIMESTAMP).dt.epoch(time_unit="ms") / 1000.0).alias("t"),
    )
    gate = (gate_ll[0] * cos0, gate_ll[1], gate_ll[2] * cos0, gate_ll[3])

    groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
    all_events: list[list[tuple[float, int]]] = []
    parts = df.partition_by(C.MMSI, maintain_order=True)
    parts.sort(key=len, reverse=True)
    parts = parts[: args.max_vessels]
    for g in parts:
        ts = g["t"].to_numpy()
        x = g["x"].to_numpy()
        y = g["y"].to_numpy()
        cat = categorize(g[C.SHIP_TYPE].mode().first())
        groups.append((ts, x, y, CAT_INDEX[cat]))
        all_events.append(detect_crossings(ts, x, y, gate))

    n_frames = int(args.seconds * args.fps)
    frame_ts = np.linspace(start_dt.timestamp(), end_dt.timestamp(), n_frames)
    X, Y, cats = interpolate_tracks(groups, frame_ts)

    flat = [(t, d) for evs in all_events for (t, d) in evs]
    day_counts = []
    for d in days:
        d0 = datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc).timestamp()
        cnt = sum(1 for t, _ in flat if d0 <= t < d0 + 86400)
        day_counts.append((d.strftime("%a").upper(), d.strftime("%d %b").upper(), cnt))
    log.info("Crossings per day: %s", [(dm, c) for _, dm, c in day_counts])

    date_label = _format_date_label(days)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    log.info("Rendering %d frames -> %s", n_frames, args.output)
    render_gif(
        X=X, Y=Y, cats=cats, frame_ts=frame_ts, events=all_events,
        day_counts=day_counts, land_rings=load_land_rings(), bbox=bbox, gate=gate,
        cos0=cos0, title=args.title, subtitle=args.subtitle, date_label=date_label,
        output=args.output, width=args.width, fps=args.fps, show_gate=not args.hide_gate,
    )
    import shutil
    import subprocess

    if shutil.which("gifsicle"):  # ponytail: optional post-pass, halves file size
        subprocess.run(
            ["gifsicle", "-O3", "--lossy=40", "--colors", "128",
             args.output, "-o", args.output],
            check=False,
        )
    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    log.info("Wrote %s (%.1f MB)", args.output, size_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
