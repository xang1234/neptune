"""CLI main — top-level command group and entry point.

Registers all subcommands and serves as the ``neptune`` console script.
"""

from __future__ import annotations

import click

from neptune_ais import __version__


@click.group()
@click.version_option(version=__version__, prog_name="neptune")
def cli() -> None:
    """Neptune AIS — Open AIS data platform for Python."""


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--source", "-s", multiple=True, help="Source(s) to download from.")
@click.option("--date", "-d", "date_str", help="Single date (YYYY-MM-DD).")
@click.option("--start", "start_str", help="Start date for range (YYYY-MM-DD).")
@click.option("--end", "end_str", help="End date for range (YYYY-MM-DD).")
@click.option("--cache-dir", type=click.Path(), help="Override store root.")
@click.option("--overwrite", is_flag=True, help="Re-download existing data.")
def download(
    source: tuple[str, ...],
    date_str: str | None,
    start_str: str | None,
    end_str: str | None,
    cache_dir: str | None,
    overwrite: bool,
) -> None:
    """Download and ingest AIS data."""
    from neptune_ais.api import Neptune

    dates = _resolve_dates(date_str, start_str, end_str)
    sources = list(source) if source else None

    n = Neptune(
        dates,
        sources=sources,
        cache_dir=cache_dir,
        overwrite=overwrite,
    )

    written = n.download()

    if written:
        click.echo(f"Downloaded {len(written)} partition(s):")
        for key in written:
            click.echo(f"  {key}")
    else:
        click.echo("No partitions written.")


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--dataset", help="Filter by dataset name.")
@click.option("--cache-dir", type=click.Path(), help="Override store root.")
def inventory(dataset: str | None, cache_dir: str | None) -> None:
    """Show inventory of stored datasets."""
    from neptune_ais.catalog import CatalogRegistry

    store = _resolve_store(cache_dir)
    registry = CatalogRegistry(store)
    registry.scan()

    items = registry.inventory(dataset)
    if not items:
        click.echo("No data found.")
        return

    for inv in items:
        click.echo(f"\n{inv.dataset}:")
        click.echo(f"  Sources:     {', '.join(inv.sources)}")
        click.echo(f"  Date range:  {inv.date_range[0]} → {inv.date_range[1]}" if inv.date_range else "  Date range:  (none)")
        click.echo(f"  Partitions:  {inv.partition_count}")
        click.echo(f"  Total rows:  {inv.total_rows:,}")
        click.echo(f"  Versions:    {', '.join(inv.schema_versions)}")
        if inv.has_mixed_versions:
            click.echo("  ⚠ Mixed schema versions detected")


# ---------------------------------------------------------------------------
# qc
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--dataset", default="positions", help="Dataset to report on.")
@click.option("--source", "-s", help="Filter by source.")
@click.option("--date", "-d", "date_str", help="Single date (YYYY-MM-DD).")
@click.option("--start", "start_str", help="Start date.")
@click.option("--end", "end_str", help="End date.")
@click.option("--cache-dir", type=click.Path(), help="Override store root.")
def qc(
    dataset: str,
    source: str | None,
    date_str: str | None,
    start_str: str | None,
    end_str: str | None,
    cache_dir: str | None,
) -> None:
    """Show quality report for stored data."""
    from neptune_ais.catalog import CatalogRegistry

    store = _resolve_store(cache_dir)
    registry = CatalogRegistry(store)
    registry.scan()

    date_from, date_to = _resolve_date_range(date_str, start_str, end_str)

    report = registry.quality_report(
        dataset, source=source, date_from=date_from, date_to=date_to,
    )

    click.echo(f"Quality report: {dataset}")
    click.echo(f"  Partitions:  {report.partitions_scanned}")
    click.echo(f"  Total rows:  {report.total_rows:,}")
    click.echo(f"  Rows OK:     {report.rows_ok:,} ({report.ok_rate:.1%})")
    click.echo(f"  Warnings:    {report.rows_warning:,} ({report.warning_rate:.1%})")
    click.echo(f"  Errors:      {report.rows_error:,} ({report.error_rate:.1%})")
    click.echo(f"  Dropped:     {report.rows_dropped:,} ({report.drop_rate:.1%})")
    if report.checks_applied:
        click.echo(f"  Checks:      {', '.join(report.checks_applied)}")


# ---------------------------------------------------------------------------
# sql
# ---------------------------------------------------------------------------


@cli.command(name="sql")
@click.argument("query")
@click.option("--date", "-d", "date_str", help="Single date (YYYY-MM-DD).")
@click.option("--start", "start_str", help="Start date.")
@click.option("--end", "end_str", help="End date.")
@click.option("--source", "-s", multiple=True, help="Source(s).")
@click.option("--merge", "-m", "merge_mode", default="best", help="Merge mode: union, best, prefer:<source>.")
@click.option("--cache-dir", type=click.Path(), help="Override store root.")
def sql_cmd(
    query: str,
    date_str: str | None,
    start_str: str | None,
    end_str: str | None,
    source: tuple[str, ...],
    merge_mode: str,
    cache_dir: str | None,
) -> None:
    """Execute a SQL query over stored datasets."""
    from neptune_ais.api import Neptune

    dates = _resolve_dates(date_str, start_str, end_str)
    sources = list(source) if source else None

    n = Neptune(dates, sources=sources, merge=merge_mode, cache_dir=cache_dir)
    result = n.sql(query)
    click.echo(result)


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--cache-dir", type=click.Path(), help="Override store root.")
def health(cache_dir: str | None) -> None:
    """Check catalog health and report issues."""
    from neptune_ais.catalog import CatalogRegistry

    store = _resolve_store(cache_dir)
    registry = CatalogRegistry(store)
    count = registry.scan()

    click.echo(f"Catalog: {count} manifest(s) loaded")

    warnings = registry.check_health()
    if warnings:
        click.echo(f"\n{len(warnings)} issue(s) found:")
        for w in warnings:
            click.echo(f"  ⚠ {w}")
    else:
        click.echo("No issues found.")


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("source_id", required=False)
@click.option("--backfill", "filter_backfill", is_flag=True, default=False, help="Only sources supporting backfill.")
@click.option("--streaming", "filter_streaming", is_flag=True, default=False, help="Only sources supporting streaming.")
@click.option("--open", "filter_open", is_flag=True, default=False, help="Only open-data sources (no auth).")
@click.option("--compare", "do_compare", is_flag=True, default=False, help="Side-by-side capability comparison.")
def sources(
    source_id: str | None,
    filter_backfill: bool,
    filter_streaming: bool,
    filter_open: bool,
    do_compare: bool,
) -> None:
    """List available sources or show details for a specific source.

    Use filters to narrow results: --backfill, --streaming, --open.
    Use --compare for a side-by-side capability matrix.
    """
    from neptune_ais.adapters import registry

    registry.load_all_adapters()

    if source_id:
        # Detailed view for one source.
        try:
            caps = registry.info(source_id)
        except KeyError as e:
            raise click.ClickException(str(e))

        click.echo(f"\n{caps.source_id} — {caps.provider}")
        click.echo(f"  {caps.description}")
        click.echo(f"  Coverage:    {caps.coverage}")
        click.echo(f"  History:     {caps.history_start or 'unknown'}")
        click.echo(f"  Format:      {caps.delivery_format}")
        click.echo(f"  Datasets:    {', '.join(caps.datasets_provided)}")
        click.echo(f"  Backfill:    {'yes' if caps.supports_backfill else 'no'}")
        click.echo(f"  Streaming:   {'yes' if caps.supports_streaming else 'no'}")
        click.echo(f"  Server bbox: {'yes' if caps.supports_server_side_bbox else 'no'}")
        click.echo(f"  Auth:        {caps.auth_scheme or 'none'}")
        click.echo(f"  Latency:     {caps.expected_latency or 'unknown'}")
        click.echo(f"  Daily rows:  {caps.typical_daily_rows or 'unknown'}")
        click.echo(f"  License:     {caps.license_requirements or 'unknown'}")
        if caps.known_quirks:
            click.echo(f"  Quirks:")
            for q in caps.known_quirks:
                click.echo(f"    - {q}")
        return

    # Apply filters.
    has_filters = filter_backfill or filter_streaming or filter_open
    if has_filters:
        all_caps = registry.find_sources(
            backfill=filter_backfill or None,
            streaming=filter_streaming or None,
            auth=False if filter_open else None,
        )
    else:
        all_caps = registry.catalog()

    if not all_caps:
        click.echo("No sources match the given filters." if has_filters else "No sources registered.")
        return

    if do_compare:
        # Side-by-side comparison using registry.compare().
        source_ids = [c.source_id for c in all_caps]
        summaries = registry.compare(*source_ids)
        keys = list(summaries[0].keys())

        click.echo(f"\n{'Capability':<20}", nl=False)
        for sid in source_ids:
            click.echo(f" {sid:<14}", nl=False)
        click.echo()
        click.echo("-" * (20 + 15 * len(source_ids)))

        for key in keys:
            if key == "source":
                continue  # already in header
            click.echo(f"{key:<20}", nl=False)
            for s in summaries:
                val = s.get(key, "")
                if len(val) > 13:
                    val = val[:12] + "\u2026"
                click.echo(f" {val:<14}", nl=False)
            click.echo()
    else:
        # Summary table.
        click.echo(f"\n{'Source':<12} {'Provider':<30} {'Coverage':<30} {'History':<12} {'Format'}")
        click.echo("-" * 100)
        for caps in all_caps:
            click.echo(
                f"{caps.source_id:<12} "
                f"{caps.provider:<30} "
                f"{caps.coverage:<30} "
                f"{(caps.history_start or '?'):<12} "
                f"{caps.delivery_format}"
            )


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--date", "-d", "date_str", help="Single date (YYYY-MM-DD).")
@click.option("--start", "start_str", help="Start date.")
@click.option("--end", "end_str", help="End date.")
@click.option("--source", "-s", multiple=True, help="Source(s).")
@click.option("--merge", "-m", "merge_mode", default="best", help="Merge mode.")
@click.option("--cache-dir", type=click.Path(), help="Override store root.")
def fusion(
    date_str: str | None,
    start_str: str | None,
    end_str: str | None,
    source: tuple[str, ...],
    merge_mode: str,
    cache_dir: str | None,
) -> None:
    """Show fusion configuration and multi-source breakdown."""
    from neptune_ais.api import Neptune

    dates = _resolve_dates(date_str, start_str, end_str)
    sources = list(source) if source else None

    n = Neptune(dates, sources=sources, merge=merge_mode, cache_dir=cache_dir)
    info = n.fusion_info()

    click.echo(f"\nFusion: {info['fusion']['mode']} mode")
    click.echo(f"  Sources:     {', '.join(info['sources'])}")
    click.echo(f"  Dates:       {info['dates']['from']} → {info['dates']['to']} ({info['dates']['count']} day(s))")
    click.echo(f"  Precedence:  {', '.join(info['fusion']['source_precedence'])}")
    click.echo(f"  Tolerance:   {info['fusion']['timestamp_tolerance_seconds']}s time, {info['fusion']['coordinate_tolerance_degrees']}° coord")

    if "prefer_source" in info["fusion"]:
        click.echo(f"  Prefer:      {info['fusion']['prefer_source']}")
    if "field_precedence" in info["fusion"]:
        for field, order in info["fusion"]["field_precedence"].items():
            click.echo(f"  Field {field}: {' > '.join(order)}")

    click.echo(f"\nPer source:")
    for sd in info["per_source"]:
        click.echo(f"  {sd['source']:<12} {sd['partitions']} partition(s), {sd['rows']:,} rows")
    click.echo(f"  {'Total':<12} {info['total_partitions']} partition(s), {info['total_rows_before_fusion']:,} rows")

    if info["multi_source"]:
        click.echo(f"\n  Multi-source fusion will be applied on query.")
    else:
        click.echo(f"\n  Single source — no fusion needed.")


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--kind", "-k", help="Filter by event type (e.g. port_call).")
@click.option("--min-confidence", type=float, help="Minimum confidence score (0.0-1.0).")
@click.option("--date", "-d", "date_str", help="Single date (YYYY-MM-DD).")
@click.option("--start", "start_str", help="Start date.")
@click.option("--end", "end_str", help="End date.")
@click.option("--source", "-s", multiple=True, help="Source(s).")
@click.option("--mmsi", type=int, multiple=True, help="Filter by MMSI.")
@click.option("--limit", "-n", "row_limit", type=int, help="Max rows to display.")
@click.option("--cache-dir", type=click.Path(), help="Override store root.")
def events(
    kind: str | None,
    min_confidence: float | None,
    date_str: str | None,
    start_str: str | None,
    end_str: str | None,
    source: tuple[str, ...],
    mmsi: tuple[int, ...],
    row_limit: int | None,
    cache_dir: str | None,
) -> None:
    """Query stored events."""
    from neptune_ais.api import Neptune

    dates = _resolve_dates(date_str, start_str, end_str)
    sources = list(source) if source else None
    mmsi_list = list(mmsi) if mmsi else None

    n = Neptune(dates, sources=sources, mmsi=mmsi_list, cache_dir=cache_dir)
    lf = n.events(kind=kind, min_confidence=min_confidence)

    if row_limit is not None:
        # Push limit into the lazy plan so Polars can short-circuit I/O.
        lf = lf.head(row_limit)

    df = lf.collect()
    if len(df) == 0:
        click.echo("No events found.")
        return

    click.echo(f"Events: {len(df)} row(s)")
    click.echo(df)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--dataset", default="positions", help="Dataset to inspect.")
@click.option("--date", "-d", "date_str", help="Single date (YYYY-MM-DD).")
@click.option("--start", "start_str", help="Start date.")
@click.option("--end", "end_str", help="End date.")
@click.option("--cache-dir", type=click.Path(), help="Override store root.")
def provenance(
    dataset: str,
    date_str: str | None,
    start_str: str | None,
    end_str: str | None,
    cache_dir: str | None,
) -> None:
    """Show provenance summary for stored data."""
    from neptune_ais.catalog import CatalogRegistry

    store = _resolve_store(cache_dir)
    registry = CatalogRegistry(store)
    registry.scan()

    date_from, date_to = _resolve_date_range(date_str, start_str, end_str)

    prov = registry.provenance(
        dataset, date_from=date_from, date_to=date_to,
    )

    click.echo(f"Provenance: {dataset}")
    click.echo(f"  Partitions:       {prov.partitions_scanned}")
    click.echo(f"  Schema versions:  {', '.join(prov.schema_versions) or 'none'}")
    click.echo(f"  Adapter versions: {', '.join(prov.adapter_versions) or 'none'}")
    click.echo(f"  Raw policies:     {', '.join(prov.raw_policies) or 'none'}")
    click.echo(f"  Raw artifacts:    {prov.total_raw_artifacts}")
    click.echo(f"    With local:     {prov.artifacts_with_local_copy}")
    click.echo(f"    Without local:  {prov.artifacts_without_local_copy}")
    click.echo(f"  Can rebuild:      {'yes' if prov.can_rebuild_locally else 'no'}")
    if prov.has_mixed_versions:
        click.echo(f"  ⚠ Mixed versions detected")


# ---------------------------------------------------------------------------
# promote
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("source")
@click.option("--landing-dir", type=click.Path(file_okay=False), required=True, help="Landing directory with Parquet files.")
@click.option("--cache-dir", type=click.Path(), help="Override store root.")
@click.option("--dataset", default="positions", help="Dataset name.")
@click.option("--cleanup", is_flag=True, help="Delete landing files after promotion.")
def promote(
    source: str,
    landing_dir: str,
    cache_dir: str | None,
    dataset: str,
    cleanup: bool,
) -> None:
    """Promote landed stream data into canonical partitions."""
    from neptune_ais.sinks import promote_landing

    store = _resolve_store(cache_dir)

    results = promote_landing(
        landing_dir, store, source=source, dataset=dataset, cleanup=cleanup,
    )

    if not results:
        click.echo("No data to promote.")
        return

    total = sum(r.record_count for r in results)
    click.echo(f"Promoted {total:,} rows across {len(results)} date partition(s):")
    for r in results:
        click.echo(f"  {r.date}: {r.record_count:,} rows → {len(r.shard_files)} shard(s)")

    if cleanup:
        click.echo("Landing files cleaned up.")


# ---------------------------------------------------------------------------
# ports — World Port Index command group
# ---------------------------------------------------------------------------


@cli.group()
def ports() -> None:
    """World Port Index — search, inspect, and derive port boundaries."""


@ports.command()
@click.argument("query")
@click.option("--limit", "-n", default=20, show_default=True, help="Max results.")
def search(query: str, limit: int) -> None:
    """Full-text search over port names and UNLOCODEs."""
    from neptune_ais.ports import index

    pi = index()
    results = pi.search(query, limit=limit)

    if len(results) == 0:
        click.echo(f"No ports matching '{query}'.")
        return

    click.echo(f"Found {len(results)} port(s) matching '{query}':\n")
    for row in results.iter_rows(named=True):
        unlocode = row.get("unlocode") or ""
        size = row.get("harbor_size") or "?"
        click.echo(
            f"  {row['name']:<30} {unlocode:<8} "
            f"{row.get('country_code', ''):<4} "
            f"size={size}  ({row['lat']:.2f}, {row['lon']:.2f})"
        )


@ports.command()
@click.argument("lat", type=float)
@click.argument("lon", type=float)
@click.option("--radius", "-r", default=50.0, show_default=True, help="Search radius in km.")
@click.option("--limit", "-n", default=10, show_default=True, help="Max results.")
def near(lat: float, lon: float, radius: float, limit: int) -> None:
    """Find ports near a coordinate (lat lon)."""
    from neptune_ais.ports import index

    pi = index()
    results = pi.near(lat, lon, radius_km=radius, limit=limit)

    if len(results) == 0:
        click.echo(f"No ports within {radius} km of ({lat}, {lon}).")
        return

    click.echo(f"Ports within {radius} km of ({lat}, {lon}):\n")
    for row in results.iter_rows(named=True):
        dist = row.get("distance_km", 0)
        unlocode = row.get("unlocode") or ""
        click.echo(
            f"  {dist:6.1f} km  {row['name']:<30} {unlocode:<8} "
            f"{row.get('country_code', '')}"
        )


@ports.command()
@click.argument("identifier")
def info(identifier: str) -> None:
    """Show detailed port card by UNLOCODE, WPI number, or name.

    Examples: neptune ports info NLRTM, neptune ports info 56850
    """
    from neptune_ais.ports import index

    pi = index()
    port = pi.get(identifier)

    if port is None:
        raise click.ClickException(f"No port found for '{identifier}'.")

    click.echo(f"\n{port.name}")
    click.echo(f"  WPI number:    {port.wpi_number}")
    click.echo(f"  UNLOCODE:      {port.unlocode or '—'}")
    click.echo(f"  Country:       {port.country_name} ({port.country_code})")
    click.echo(f"  Coordinates:   {port.lat:.4f}, {port.lon:.4f}")
    click.echo(f"  Harbor size:   {port.harbor_size}")
    click.echo(f"  Harbor type:   {port.harbor_type}")
    click.echo(f"  Shelter:       {port.shelter_quality}")
    click.echo(f"  Channel depth: {port.channel_depth_m or '—'} m")
    click.echo(f"  Anchorage:     {port.anchorage_depth_m or '—'} m")
    click.echo(f"  Cargo pier:    {port.cargo_pier_depth_m or '—'} m")
    click.echo(f"  Max vessel:    {port.max_vessel_length_m or '—'} m")
    click.echo(f"  Tide range:    {port.tide_range_m or '—'} m")
    click.echo(f"  Pilotage:      {'yes' if port.has_pilotage else 'no'}")
    click.echo(f"  Tugs:          {'yes' if port.has_tugs else 'no'}")
    click.echo(f"  Fuel:          {'yes' if port.has_fuel else 'no'}")
    click.echo(f"  Drydock:       {'yes' if port.has_drydock else 'no'}")
    click.echo(f"  Cranes:        {'yes' if port.has_cranes else 'no'}")


@ports.command()
@click.argument("country_code")
def country(country_code: str) -> None:
    """List all ports in a country (ISO alpha-2 code)."""
    from neptune_ais.ports import index

    pi = index()
    results = pi.by_country(country_code)

    if len(results) == 0:
        click.echo(f"No ports found for country '{country_code}'.")
        return

    click.echo(f"Ports in {country_code.upper()} ({len(results)}):\n")
    for row in results.iter_rows(named=True):
        unlocode = row.get("unlocode") or ""
        size = row.get("harbor_size") or "?"
        click.echo(
            f"  {row['name']:<30} {unlocode:<8} size={size}  "
            f"({row['lat']:.2f}, {row['lon']:.2f})"
        )


@ports.command()
@click.option("--date", "-d", "date_str", help="Single date (YYYY-MM-DD).")
@click.option("--start", "start_str", help="Start date.")
@click.option("--end", "end_str", help="End date.")
@click.option("--source", "-s", multiple=True, help="Source(s).")
@click.option("--cache-dir", type=click.Path(), help="Override store root.")
def derive(
    date_str: str | None,
    start_str: str | None,
    end_str: str | None,
    source: tuple[str, ...],
    cache_dir: str | None,
) -> None:
    """Derive port polygons from ingested AIS data (Tier 2)."""
    from neptune_ais.api import Neptune

    dates = _resolve_dates(date_str, start_str, end_str)
    sources = list(source) if source else None

    n = Neptune(dates, sources=sources, cache_dir=cache_dir)
    click.echo("Deriving port polygons from AIS positions...")
    result = n.derive_port_polygons()

    if len(result) == 0:
        click.echo("No polygons derived (insufficient low-speed positions near ports).")
        return

    n_ports = result["port_name"].n_unique()
    click.echo(f"Derived {len(result)} zone(s) across {n_ports} port(s).")

    high_conf = result.filter(result["confidence"] >= 0.7)
    if len(high_conf) > 0:
        click.echo(f"  High confidence (>=0.7): {len(high_conf)} zone(s)")


@ports.command(name="download")
def download_eez() -> None:
    """Pre-download EEZ polygon data for offline use."""
    from neptune_ais.ports._loader import load_eez_polygons

    click.echo("Downloading EEZ polygon data...")
    try:
        df = load_eez_polygons()
        click.echo(f"EEZ polygons ready: {len(df)} region(s).")
    except Exception as e:
        raise click.ClickException(str(e))


@ports.command()
@click.option("--format", "fmt", default="geojson", type=click.Choice(["geojson"]), help="Output format.")
@click.option("--output", "-o", "output_path", default="ports.geojson", show_default=True, help="Output file path.")
@click.option("--min-confidence", type=float, default=0.0, help="Min polygon confidence.")
@click.option("--cache-dir", type=click.Path(), help="Override store root.")
def export(
    fmt: str,
    output_path: str,
    min_confidence: float,
    cache_dir: str | None,
) -> None:
    """Export port polygons as GeoJSON."""
    try:
        import shapely
    except ImportError:
        raise click.ClickException(
            "Export requires shapely. Install with: pip install neptune-ais[geo]"
        )

    from pathlib import Path
    from shapely.geometry import box, mapping
    from neptune_ais.ports import index
    from neptune_ais.storage import StoreLayer

    pi = index()

    store = _resolve_store(cache_dir)
    derived_dir = store / StoreLayer.DERIVED.value / "port_polygons"
    derived_polygons = None
    if derived_dir.exists():
        import polars as pl
        parquet_files = sorted(
            derived_dir.glob("*.parquet"),
            key=lambda p: p.stat().st_mtime,
        )
        if parquet_files:
            derived_polygons = pl.read_parquet(parquet_files[-1])
            if min_confidence > 0:
                derived_polygons = derived_polygons.filter(
                    pl.col("confidence") >= min_confidence
                )
            click.echo(f"Using derived polygons: {len(derived_polygons)} zone(s)")

    features: list[dict] = []
    tier2_names: set[str] = set()

    if derived_polygons is not None and len(derived_polygons) > 0:
        for row in derived_polygons.iter_rows(named=True):
            geom = shapely.from_wkb(row["geometry_wkb"])
            tier2_names.add(row["port_name"])
            features.append({
                "type": "Feature",
                "properties": {
                    "name": row["port_name"],
                    "zone_id": row.get("zone_id", ""),
                    "confidence": row.get("confidence"),
                    "source": "tier2_derived",
                },
                "geometry": mapping(geom),
            })

    for row in pi.ports.iter_rows(named=True):
        if row["name"] in tier2_names:
            continue
        w, s, e, n = row["bbox_west"], row["bbox_south"], row["bbox_east"], row["bbox_north"]
        if any(v is None for v in (w, s, e, n)):
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "name": row["name"],
                "unlocode": row.get("unlocode") or "",
                "harbor_size": row.get("harbor_size") or "",
                "source": "tier1_bbox",
            },
            "geometry": mapping(box(w, s, e, n)),
        })

    import json
    geojson = {"type": "FeatureCollection", "features": features}
    out = Path(output_path)
    out.write_text(json.dumps(geojson))
    click.echo(f"Exported {len(features)} port(s) to {out}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_store(cache_dir: str | None):
    """Resolve --cache-dir into a store root Path."""
    from pathlib import Path
    from neptune_ais.storage import DEFAULT_STORE_ROOT
    return Path(cache_dir) if cache_dir else DEFAULT_STORE_ROOT


def _resolve_dates(
    date_str: str | None,
    start_str: str | None,
    end_str: str | None,
) -> str | tuple[str, str]:
    """Resolve CLI date arguments into a dates value for Neptune."""
    if date_str:
        return date_str
    if start_str and end_str:
        return (start_str, end_str)
    if start_str:
        return start_str
    if end_str:
        raise click.UsageError("--end requires --start")
    raise click.UsageError("Provide --date or --start/--end")


def _resolve_date_range(
    date_str: str | None,
    start_str: str | None,
    end_str: str | None,
) -> tuple[str | None, str | None]:
    """Resolve CLI date arguments into (date_from, date_to)."""
    if date_str:
        return date_str, date_str
    return start_str, end_str
