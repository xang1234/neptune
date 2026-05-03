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

**Import rule:** Helpers may import from ``datasets``, ``derive``,
``geometry``, and ``ports``. It must not import from ``adapters``,
``storage``, ``catalog``, or ``cli``.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

from neptune_ais.datasets.positions import Col as PosCol

if TYPE_CHECKING:
    from neptune_ais.derive.events import EEZCrossingConfig, PortCallConfig
    from neptune_ais.geometry.boundaries import BoundaryRegistry
    from neptune_ais.ports._index import PortIndex


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


def enrich_port_calls(
    events: pl.DataFrame,
    port_index: PortIndex | None = None,
) -> pl.DataFrame:
    """Add port metadata columns to port-call events.

    Joins each port-call event's (lat, lon) against the WPI port
    dictionary to add: ``unlocode``, ``country_code``, ``country_name``,
    ``harbor_size``, ``shelter_quality``, ``channel_depth_m``,
    ``has_pilotage``, ``has_cranes``, ``has_fuel``.

    Args:
        events: A port-call events DataFrame (from ``detect_port_calls()``
            or ``Neptune.events(kind="port_call")``).
        port_index: Optional PortIndex. If None, loads the default
            singleton from ``neptune_ais.ports``.

    Returns:
        The events DataFrame with additional port metadata columns.
        Events that don't match any port retain null metadata columns.
    """
    if port_index is None:
        from neptune_ais.ports import index
        port_index = index()

    if len(events) == 0:
        return events

    from neptune_ais.ports._registry_bridge import vectorized_port_lookup

    port_bounds = port_index.ports.select(
        "wpi_number", "name", "lat", "lon",
        "bbox_west", "bbox_south", "bbox_east", "bbox_north",
    )
    matched_names = vectorized_port_lookup(events, port_bounds)

    # Port names can be duplicated (e.g., "Georgetown" appears 5x),
    # so resolve to wpi_number (unique) for the join key.
    name_to_wpi: dict[str, int] = {}
    for row in port_bounds.select("name", "wpi_number").iter_rows():
        name_to_wpi.setdefault(row[0], row[1])  # first match wins

    wpi_numbers = [
        name_to_wpi.get(n) if n is not None else None
        for n in matched_names.to_list()
    ]

    enrich_cols = [
        "wpi_number", "name", "unlocode", "country_code", "country_name",
        "harbor_size", "shelter_quality", "channel_depth_m",
        "has_pilotage", "has_cranes", "has_fuel",
    ]
    port_meta = port_index.ports.select(enrich_cols)

    enriched = (
        events
        .with_columns(pl.Series("_wpi", wpi_numbers, dtype=pl.Int64))
        .join(port_meta, left_on="_wpi", right_on="wpi_number", how="left")
        .rename({"name": "port_name"})
        .drop("_wpi")
    )

    return enriched


def port_calls(
    positions: pl.LazyFrame,
    *,
    registry: BoundaryRegistry | None = None,
    dataset_name: str = "builtin_ports",
    config: PortCallConfig | None = None,
    source: str = "",
    enrich: bool = True,
    port_index: PortIndex | None = None,
) -> pl.DataFrame:
    """Detect port calls with zero-config boundary loading.

    When ``registry`` is None (the default), automatically loads the
    built-in World Port Index and uses the fast vectorized spatial
    join to assign positions to ports. When a ``registry`` is provided,
    falls back to ``BoundaryRegistry.lookup_column()``.

    Args:
        positions: A Polars LazyFrame of positions (from
            ``Neptune.positions()``). Collected internally.
        registry: Optional BoundaryRegistry with pre-registered port
            boundaries. If None, loads built-in WPI data automatically.
        dataset_name: Dataset name to look up when using ``registry``.
            Ignored in the zero-config path.
        config: Port-call detection configuration. Uses defaults if None.
        source: Source identifier for provenance.
        enrich: If True (default), join port metadata (UNLOCODE, country,
            facilities) onto the detected events.
        port_index: Optional PortIndex for the zero-config path and
            enrichment. If None, loads the default singleton.

    Returns:
        A DataFrame of port-call events. When ``enrich=True``, includes
        additional columns: ``port_name``, ``unlocode``, ``country_code``,
        ``country_name``, ``harbor_size``, ``shelter_quality``,
        ``channel_depth_m``, ``has_pilotage``, ``has_cranes``, ``has_fuel``.
    """
    from neptune_ais.derive.events import detect_port_calls

    if port_index is None and (registry is None or enrich):
        from neptune_ais.ports import index
        port_index = index()

    pos_df = positions.collect()

    if registry is None:
        from neptune_ais.ports._registry_bridge import vectorized_port_lookup

        port_bounds = port_index.ports.select(
            "name", "lat", "lon",
            "bbox_west", "bbox_south", "bbox_east", "bbox_north",
        )
        port_regions = vectorized_port_lookup(pos_df, port_bounds)
    else:
        port_regions = registry.lookup_column(
            pos_df, dataset_name, lat_col="lat", lon_col="lon",
        )

    events = detect_port_calls(
        pos_df, port_regions, config=config, source=source,
    )

    if enrich and len(events) > 0:
        events = enrich_port_calls(events, port_index)

    return events


def eez_crossings(
    positions: pl.LazyFrame,
    *,
    registry: BoundaryRegistry | None = None,
    dataset_name: str = "builtin_eez",
    config: EEZCrossingConfig | None = None,
    source: str = "",
    port_index: PortIndex | None = None,
) -> pl.DataFrame:
    """Detect EEZ crossings with zero-config boundary loading.

    When ``registry`` is None (the default), automatically loads the
    built-in EEZ metadata and uses vectorized bbox matching to assign
    positions to EEZ regions. When a ``registry`` is provided, falls
    back to ``BoundaryRegistry.lookup_column()``.

    Args:
        positions: A Polars LazyFrame of positions (from
            ``Neptune.positions()``). Collected internally.
        registry: Optional BoundaryRegistry with pre-registered EEZ
            boundaries. If None, loads built-in EEZ metadata (bbox-only).
            For polygon-level accuracy, pass a registry with EEZ
            polygon data loaded.
        dataset_name: Dataset name to look up when using ``registry``.
            Ignored in the zero-config path.
        config: EEZ-crossing detection configuration. Uses defaults
            if None.
        source: Source identifier for provenance.
        port_index: Optional PortIndex for the zero-config path. If
            None, loads the default singleton.

    Returns:
        A DataFrame of EEZ-crossing events conforming to the
        ``events/v1`` schema with ``event_type = "eez_crossing"``.
    """
    from neptune_ais.derive.events import detect_eez_crossings

    pos_df = positions.collect()

    if registry is None:
        if port_index is None:
            from neptune_ais.ports import index
            port_index = index()

        from neptune_ais.ports._registry_bridge import vectorized_port_lookup

        eez_bounds = port_index.eez.select(
            "name",
            "bbox_west", "bbox_south", "bbox_east", "bbox_north",
        )
        eez_regions = vectorized_port_lookup(pos_df, eez_bounds)
    else:
        eez_regions = registry.lookup_column(
            pos_df, dataset_name, lat_col="lat", lon_col="lon",
        )

    return detect_eez_crossings(
        pos_df, eez_regions, config=config, source=source,
    )
