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
    from neptune_ais.storage import DEFAULT_STORE_ROOT
    from pathlib import Path

    store = Path(cache_dir) if cache_dir else DEFAULT_STORE_ROOT
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
    from neptune_ais.storage import DEFAULT_STORE_ROOT
    from pathlib import Path

    store = Path(cache_dir) if cache_dir else DEFAULT_STORE_ROOT
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
    from neptune_ais.storage import DEFAULT_STORE_ROOT
    from pathlib import Path

    store = Path(cache_dir) if cache_dir else DEFAULT_STORE_ROOT
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
def sources(source_id: str | None) -> None:
    """List available sources or show details for a specific source."""
    # Trigger adapter registration.
    from neptune_ais.adapters import dma as _dma  # noqa: F401
    from neptune_ais.adapters import noaa as _noaa  # noqa: F401
    from neptune_ais.adapters import registry

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
    else:
        # Summary table of all sources.
        all_caps = registry.catalog()
        if not all_caps:
            click.echo("No sources registered.")
            return

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

    df = lf.collect()
    if len(df) == 0:
        click.echo("No events found.")
        return

    if row_limit:
        df = df.head(row_limit)

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
    from neptune_ais.storage import DEFAULT_STORE_ROOT
    from pathlib import Path

    store = Path(cache_dir) if cache_dir else DEFAULT_STORE_ROOT
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
# Helpers
# ---------------------------------------------------------------------------


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
