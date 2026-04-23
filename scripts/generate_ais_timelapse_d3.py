#!/usr/bin/env python3
"""Generate the D3.js + Canvas2D AIS timelapse HTML from NOAA data.

Replica of ``ais_timelapse2.mp4`` rendered with D3.js. Downloads a
48-hour slice of NOAA Marine Cadastre AIS for three iconic US port
regions (New York/New Jersey, Los Angeles/Long Beach, Houston Ship
Channel) and writes a self-contained HTML file.

Usage:
    python scripts/generate_ais_timelapse_d3.py                        # defaults
    python scripts/generate_ais_timelapse_d3.py --date 2024-06-15
    python scripts/generate_ais_timelapse_d3.py --output my.html
    python scripts/generate_ais_timelapse_d3.py --window 24h --bin-minutes 20

Output: ``ais_timelapse_d3.html`` at the repo root by default, ~5–15 MB
depending on vessel density.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

import polars as pl

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from neptune_ais import Neptune  # noqa: E402
from neptune_ais.datasets.positions import Col as PosCol  # noqa: E402
from neptune_ais.viz import generate_timelapse_d3  # noqa: E402


# Three US port chokepoints — bboxes sized so that each panel's 4:5
# aspect ratio frames the corridor naturally. Coordinates are
# ``(west, south, east, north)``.
DEFAULT_PANELS: list[dict] = [
    {
        "label": "New York",
        "sea": "Hudson / Kill Van Kull",
        "bbox": (-74.30, 40.48, -73.86, 40.90),
    },
    {
        "label": "Los Angeles",
        "sea": "San Pedro Bay",
        "bbox": (-118.40, 33.60, -117.88, 33.88),
    },
    {
        "label": "Houston",
        "sea": "Galveston Bay",
        "bbox": (-95.35, 29.18, -94.55, 29.90),
    },
]


def _parse_window(s: str) -> timedelta:
    """Parse a window spec like ``'48h'`` or ``'2d'`` into a timedelta."""
    s = s.strip().lower()
    if s.endswith("h"):
        return timedelta(hours=float(s[:-1]))
    if s.endswith("d"):
        return timedelta(days=float(s[:-1]))
    if s.endswith("m"):
        return timedelta(minutes=float(s[:-1]))
    raise argparse.ArgumentTypeError(
        f"Window must end with h/d/m (got {s!r})"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--date",
        default="2024-06-15",
        help="Start date (YYYY-MM-DD). Default: 2024-06-15.",
    )
    p.add_argument(
        "--window",
        type=_parse_window,
        default=timedelta(hours=48),
        help="Time window length (e.g. 48h, 1d). Default: 48h.",
    )
    p.add_argument(
        "--bin-minutes",
        type=int,
        default=30,
        help="Animation bin size in minutes. Default: 30.",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Animation loop duration in seconds. Default: 15.",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=120_000,
        help="Maximum positions per panel after sampling. Default: 120000.",
    )
    p.add_argument(
        "--output",
        default=str(_REPO_ROOT / "ais_timelapse_d3.html"),
        help="Output HTML path. Default: <repo>/ais_timelapse_d3.html",
    )
    p.add_argument(
        "--no-us-states",
        action="store_true",
        help="Skip embedding the us-atlas states layer (smaller HTML).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging (shows NOAA download progress).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("timelapse_d3")

    # Determine the union bbox covering all three panels so we can do
    # a single NOAA download + filter rather than three separate fetches.
    # NOAA serves daily CSV-in-ZIP — the adapter caches by date and
    # filters in-process, so one download covers all panels cheaply.
    n_days = max(1, int((args.window.total_seconds() + 86399) // 86400))
    from datetime import date as _date
    start = _date.fromisoformat(args.date)
    days = [start + timedelta(days=i) for i in range(n_days)]

    log.info(
        "Loading NOAA positions for %s days starting %s …",
        n_days, start.isoformat(),
    )
    t0 = time.time()
    neptune = Neptune(
        dates=(days[0].isoformat(), days[-1].isoformat()),
        sources=["noaa"],
    )
    neptune.download()
    positions_all = neptune.positions().collect()
    log.info(
        "Loaded %s positions in %.1fs",
        f"{len(positions_all):,}", time.time() - t0,
    )

    if len(positions_all) == 0:
        log.error(
            "No positions returned from NOAA. Check network access and "
            "that the date is after 2009-01-01.",
        )
        return 1

    # Trim to the requested window (NOAA delivers whole days; we keep
    # only the first ``window`` hours from ``start``). Positions are
    # UTC-aware Datetime; cast our literals to match.
    from datetime import datetime as _datetime, timezone as _tz
    start_dt = _datetime.combine(start, _datetime.min.time(), tzinfo=_tz.utc)
    end_dt = start_dt + args.window
    positions_all = positions_all.filter(
        (pl.col(PosCol.TIMESTAMP) >= pl.lit(start_dt).cast(pl.Datetime("us", "UTC")))
        & (pl.col(PosCol.TIMESTAMP) < pl.lit(end_dt).cast(pl.Datetime("us", "UTC")))
    )
    log.info("After window trim: %s positions", f"{len(positions_all):,}")

    # Build per-panel filtered DataFrames. Each panel's viewport is
    # applied inside ``generate_timelapse_d3`` via ``prepare_timelapse``,
    # but we pre-filter here to keep memory bounded.
    panels_input: list[dict] = []
    date_label = f"{days[0].strftime('%b %-d')}–{days[-1].strftime('%-d %Y')}"
    for spec in DEFAULT_PANELS:
        w, s, e, n = spec["bbox"]
        panel_df = positions_all.filter(
            (pl.col(PosCol.LON) >= w) & (pl.col(PosCol.LON) <= e)
            & (pl.col(PosCol.LAT) >= s) & (pl.col(PosCol.LAT) <= n)
        )
        if len(panel_df) == 0:
            log.warning(
                "Panel %r has 0 positions in bbox %s — widen the bbox "
                "or pick a busier date.",
                spec["label"], spec["bbox"],
            )
            continue
        log.info(
            "Panel %-14s bbox=%s positions=%s",
            spec["label"], spec["bbox"], f"{len(panel_df):,}",
        )
        panels_input.append({
            **spec,
            "daterange": date_label,
            "positions": panel_df,
        })

    if not panels_input:
        log.error("No panels had any positions. Aborting.")
        return 1

    log.info("Rendering D3 timelapse HTML …")
    t0 = time.time()
    out = generate_timelapse_d3(
        panels=panels_input,
        title="VESSEL MOVEMENT — AIS TIMELAPSE",
        eyebrow="US Pacific / Atlantic / Gulf Corridors",
        output=args.output,
        max_points_per_panel=args.max_points,
        bin_interval_minutes=args.bin_minutes,
        total_duration_sec=args.duration,
        include_us_states=not args.no_us_states,
    )
    size_mb = Path(out).stat().st_size / 1024 / 1024
    log.info(
        "Wrote %s (%.1f MB) in %.1fs",
        out, size_mb, time.time() - t0,
    )
    log.info("Open the file in a browser — no server required.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
