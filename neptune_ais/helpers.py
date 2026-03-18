"""Helpers — high-level maritime primitives.

Convenience functions for common maritime questions: latest positions,
port calls, encounters, loitering, EEZ crossings, density, vessel history,
and point-in-time snapshots.

Module role — convenience API
-----------------------------
**Owns:**
- Implementations of ``Neptune.latest_positions()``, ``port_calls()``,
  ``encounters()``, ``loitering()``, ``eez_crossings()``, ``density()``,
  ``vessel_history()``, and ``snapshot()``.
- These compose ``derive`` pipelines and ``datasets`` queries into
  user-friendly one-call methods.

**Does not own:**
- The derivation algorithms themselves — those live in ``derive``.
- Schema definitions — those live in ``datasets``.
- Spatial lookups — those live in ``geometry``.

**Import rule:** Helpers may import from ``datasets``, ``derive``, and
``geometry``. It must not import from ``adapters``, ``storage``,
``catalog``, or ``cli``.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from neptune_ais.datasets.positions import Col as PosCol


def latest_positions(positions: pl.LazyFrame) -> pl.LazyFrame:
    """Return the most recent position per vessel.

    For each MMSI, returns the row with the latest timestamp. This
    answers the "where are my vessels right now?" question.

    Args:
        positions: A Polars LazyFrame of positions (from
            ``Neptune.positions()``).

    Returns:
        A LazyFrame with one row per vessel, sorted by MMSI.
    """
    return (
        positions
        .group_by(PosCol.MMSI)
        .agg(pl.all().sort_by(PosCol.TIMESTAMP).last())
        .sort(PosCol.MMSI)
    )


def snapshot(
    positions: pl.LazyFrame,
    when: datetime | str,
) -> pl.LazyFrame:
    """Return the closest position per vessel to a given timestamp.

    For each MMSI, finds the position with the smallest absolute
    time difference from ``when``. This answers "where were my
    vessels at time T?"

    Args:
        positions: A Polars LazyFrame of positions.
        when: Target timestamp (datetime or ISO-8601 string).

    Returns:
        A LazyFrame with one row per vessel, sorted by MMSI.
    """
    if isinstance(when, str):
        when_expr = pl.lit(when).str.to_datetime(time_unit="us", time_zone="UTC")
    else:
        when_expr = pl.lit(when)

    return (
        positions
        .with_columns(
            (pl.col(PosCol.TIMESTAMP) - when_expr)
            .abs()
            .alias("_time_diff")
        )
        .group_by(PosCol.MMSI)
        .agg(pl.all().sort_by("_time_diff").first())
        .drop("_time_diff")
        .sort(PosCol.MMSI)
    )


def vessel_history(
    mmsi: int,
    *,
    positions: pl.LazyFrame,
    tracks: pl.LazyFrame | None = None,
    events: pl.LazyFrame | None = None,
) -> dict[str, pl.LazyFrame]:
    """Return all data for a single vessel.

    Filters positions, tracks, and events to a single MMSI and
    returns them as a dict of LazyFrames. This answers "show me
    everything about vessel X."

    Args:
        mmsi: The vessel MMSI to look up.
        positions: Positions LazyFrame (required).
        tracks: Tracks LazyFrame (optional).
        events: Events LazyFrame (optional).

    Returns:
        A dict with keys ``"positions"``, and optionally ``"tracks"``
        and ``"events"``, each containing a filtered LazyFrame for
        the requested MMSI.
    """
    result: dict[str, pl.LazyFrame] = {
        "positions": positions.filter(pl.col(PosCol.MMSI) == mmsi),
    }

    if tracks is not None:
        from neptune_ais.datasets.tracks import Col as TrackCol
        result["tracks"] = tracks.filter(pl.col(TrackCol.MMSI) == mmsi)

    if events is not None:
        from neptune_ais.datasets.events import Col as EventCol
        result["events"] = events.filter(
            (pl.col(EventCol.MMSI) == mmsi)
            | (pl.col(EventCol.OTHER_MMSI) == mmsi)
        )

    return result
