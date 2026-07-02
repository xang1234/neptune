#!/usr/bin/env python3
"""Generate the cinematic D3 AIS timelapse HTML from NOAA data.

Two layouts replicating the reference videos:

- ``--layout corridors`` (default): three stacked port panels
  (New York, Los Angeles, Houston) — replica of ais_timelapse2.mp4.
- ``--layout single``: one framed map of the Gulf of Mexico with a
  live clock, city labels and scale bar — replica of ais_timelapse1.mp4.

Usage:
    python scripts/generate_ais_timelapse_d3.py
    python scripts/generate_ais_timelapse_d3.py --layout single
    python scripts/generate_ais_timelapse_d3.py --date 2024-06-15 --window 48h

Output: ``ais_timelapse_d3_<layout>.html`` at the repo root by default.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from neptune_ais import Neptune  # noqa: E402
from neptune_ais.datasets.positions import Col as PosCol  # noqa: E402
from neptune_ais.viz import generate_timelapse_d3  # noqa: E402


CORRIDOR_PANELS: list[dict] = [
    {
        "label": "New York",
        "sea": "Hudson / Kill Van Kull",
        "bbox": (-74.30, 40.42, -73.70, 40.75),
    },
    {
        "label": "Los Angeles",
        "sea": "San Pedro Bay",
        "bbox": (-118.55, 33.55, -117.95, 33.88),
    },
    {
        "label": "Houston",
        "sea": "Galveston Bay",
        "bbox": (-95.15, 29.20, -94.45, 29.78),
    },
]

SINGLE_PANEL: dict = {
    "label": "ALL VESSELS",
    "sea": "GULF OF MEXICO",
    "bbox": (-95.6, 27.4, -88.6, 30.2),
    "cities": [
        {"name": "Galveston", "lat": 29.30, "lon": -94.80, "anchor": "right"},
        {"name": "Port Fourchon", "lat": 29.11, "lon": -90.20},
        {"name": "New Orleans", "lat": 29.93, "lon": -90.08, "anchor": "right"},
    ],
}


def _parse_window(s: str) -> timedelta:
    s = s.strip().lower()
    if s.endswith("h"):
        return timedelta(hours=float(s[:-1]))
    if s.endswith("d"):
        return timedelta(days=float(s[:-1]))
    if s.endswith("m"):
        return timedelta(minutes=float(s[:-1]))
    raise argparse.ArgumentTypeError(f"Window must end with h/d/m (got {s!r})")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--layout", choices=["corridors", "single"], default="corridors")
    p.add_argument("--date", default="2024-06-15", help="Start date (YYYY-MM-DD).")
    p.add_argument(
        "--window", type=_parse_window, default=timedelta(hours=48),
        help="Time window (e.g. 48h, 1d). Default: 48h.",
    )
    p.add_argument(
        "--duration", type=float, default=15.0,
        help="Animation loop duration in seconds. Default: 15.",
    )
    p.add_argument(
        "--max-points", type=int, default=150_000,
        help="Maximum positions per panel after decimation. Default: 150000.",
    )
    p.add_argument("--output", default=None, help="Output HTML path.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("timelapse_d3")

    output = args.output or str(
        _REPO_ROOT / f"ais_timelapse_d3_{args.layout}.html"
    )

    n_days = max(1, int((args.window.total_seconds() + 86399) // 86400))
    start = _date.fromisoformat(args.date)
    days = [start + timedelta(days=i) for i in range(n_days)]

    log.info("Loading NOAA positions for %s day(s) starting %s …", n_days, start)
    t0 = time.time()
    neptune = Neptune(
        dates=(days[0].isoformat(), days[-1].isoformat()),
        sources=["noaa"],
    )
    neptune.download()
    positions_all = neptune.positions().collect()
    log.info("Loaded %s positions in %.1fs", f"{len(positions_all):,}", time.time() - t0)
    if len(positions_all) == 0:
        log.error("No positions returned from NOAA.")
        return 1

    start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = start_dt + args.window
    positions_all = positions_all.filter(
        (pl.col(PosCol.TIMESTAMP) >= pl.lit(start_dt).cast(pl.Datetime("us", "UTC")))
        & (pl.col(PosCol.TIMESTAMP) < pl.lit(end_dt).cast(pl.Datetime("us", "UTC")))
    )
    log.info("After window trim: %s positions", f"{len(positions_all):,}")

    date_label = f"{days[0].strftime('%b %d')}-{days[-1].strftime('%d %Y')}"
    specs = CORRIDOR_PANELS if args.layout == "corridors" else [SINGLE_PANEL]

    panels_input: list[dict] = []
    for spec in specs:
        w, s, e, n = spec["bbox"]
        panel_df = positions_all.filter(
            (pl.col(PosCol.LON) >= w) & (pl.col(PosCol.LON) <= e)
            & (pl.col(PosCol.LAT) >= s) & (pl.col(PosCol.LAT) <= n)
        )
        if len(panel_df) == 0:
            log.warning("Panel %r has 0 positions — skipping.", spec["label"])
            continue
        log.info("Panel %-14s positions=%s", spec["label"], f"{len(panel_df):,}")
        panels_input.append({**spec, "daterange": date_label, "positions": panel_df})

    if not panels_input:
        log.error("No panels had any positions. Aborting.")
        return 1

    if args.layout == "corridors":
        title = "US shipping corridors"
        eyebrow = "VESSEL MOVEMENT — AIS TIMELAPSE"
    else:
        title = f"{SINGLE_PANEL['label']} - {SINGLE_PANEL['sea']}"
        eyebrow = "VESSEL MOVEMENT - AIS TIMELAPSE"

    log.info("Rendering D3 timelapse HTML …")
    t0 = time.time()
    out = generate_timelapse_d3(
        panels=panels_input,
        title=title,
        eyebrow=eyebrow,
        output=output,
        max_points_per_panel=args.max_points,
        total_duration_sec=args.duration,
    )
    size_mb = Path(out).stat().st_size / 1024 / 1024
    log.info("Wrote %s (%.1f MB) in %.1fs", out, size_mb, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
