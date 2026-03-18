"""Neptune — unified dataset object and user-facing API.

This module contains the primary ``Neptune`` class that provides access to
archival AIS datasets across one or more dates and sources.

Module role — orchestration layer
---------------------------------
``api`` is the coordination point that wires subsystems together. It is the
only module that is allowed to import broadly across the package.

**Owns:**
- The ``Neptune`` class and its public method signatures.
- Orchestration of the fetch → normalize → QC → store → catalog pipeline.
- Delegation to ``fusion`` for multi-source merges.
- Delegation to ``derive`` for track/event/density computation.
- Delegation to ``sql`` and ``viz`` for DuckDB and map access.
- Delegation to ``helpers`` for high-level maritime convenience methods.

**Import rule:** ``api`` may import from any sibling module or subpackage.
No other module should import from ``api`` (to prevent circular deps). The
only exception is ``cli``, which constructs ``Neptune`` instances.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from neptune_ais.catalog import (
    BBox,
    CatalogRegistry,
    Manifest,
    QCSummary,
    RawArtifact as CatalogRawArtifact,
    WriteStatus,
    current_schema_version,
)
from neptune_ais.storage import (
    DEFAULT_STORE_ROOT,
    PartitionWriter,
    RawPolicy,
    raw_partition_path,
    shard_filename,
    SORT_ORDER_POSITIONS,
    SORT_ORDER_VESSELS,
    DEFAULT_MAX_ROWS_PER_SHARD,
    DEFAULT_ROW_GROUP_SIZE,
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_WRITE_STATISTICS,
)

logger = logging.getLogger(__name__)

# Default source list when user doesn't specify.
DEFAULT_SOURCES = ["noaa"]


def _parse_dates(dates) -> list[date]:
    """Normalize the dates argument into a list of date objects.

    Accepts:
    - A single date string: "2024-06-15"
    - A single date object
    - A tuple of (start, end) for a date range
    - A list of date strings or date objects
    """
    if isinstance(dates, str):
        return [date.fromisoformat(dates)]
    if isinstance(dates, date):
        return [dates]
    if isinstance(dates, tuple) and len(dates) == 2:
        start = date.fromisoformat(dates[0]) if isinstance(dates[0], str) else dates[0]
        end = date.fromisoformat(dates[1]) if isinstance(dates[1], str) else dates[1]
        result = []
        current = start
        while current <= end:
            result.append(current)
            current += timedelta(days=1)
        return result
    if isinstance(dates, list):
        return [
            date.fromisoformat(d) if isinstance(d, str) else d
            for d in dates
        ]
    raise TypeError(f"Unsupported dates type: {type(dates)}")


class Neptune:
    """Open Neptune datasets over one or more dates and sources.

    This is the primary user-facing object. It provides:
    - Polars LazyFrame access to canonical datasets.
    - DuckDB SQL access over the same catalog.
    - Inventory, provenance, and quality report surfaces.
    - Orchestration of the download → normalize → QC → store pipeline.

    Usage::

        from neptune_ais import Neptune

        n = Neptune("2024-06-15", sources=["noaa"])
        n.download()
        positions = n.positions()  # polars.LazyFrame
    """

    def __init__(
        self,
        dates,
        *,
        sources: list[str] | None = None,
        merge: str = "best",
        bbox: tuple[float, float, float, float] | None = None,
        mmsi: list[int] | None = None,
        cache_dir: str | Path | None = None,
        raw_policy: str | RawPolicy = "metadata",
        overwrite: bool = False,
    ) -> None:
        from neptune_ais.fusion import FusionConfig, parse_merge_arg

        self._dates = _parse_dates(dates)
        self._sources = sources or DEFAULT_SOURCES
        self._fusion_config: FusionConfig = (
            merge if isinstance(merge, FusionConfig)
            else parse_merge_arg(merge, self._sources)
        )
        self._bbox = bbox
        self._mmsi = mmsi
        self._store_root = Path(cache_dir) if cache_dir else DEFAULT_STORE_ROOT
        self._raw_policy = (
            RawPolicy(raw_policy) if isinstance(raw_policy, str) else raw_policy
        )
        self._overwrite = overwrite

        # Catalog registry — loaded lazily on first access.
        self._registry: CatalogRegistry | None = None

    # --- Internal helpers ---

    def _get_registry(self) -> CatalogRegistry:
        """Return the catalog registry, scanning on first access."""
        if self._registry is None:
            self._registry = CatalogRegistry(self._store_root)
            self._registry.scan()
        return self._registry

    @property
    def _date_range(self) -> tuple[str | None, str | None]:
        """Return (date_from, date_to) for the configured dates."""
        if not self._dates:
            return None, None
        return min(self._dates).isoformat(), max(self._dates).isoformat()

    def _dataset_files(self, dataset: str) -> list[Path]:
        """Return all Parquet files for a dataset across configured sources."""
        registry = self._get_registry()
        date_from, date_to = self._date_range
        files: list[Path] = []
        for source_id in self._sources:
            files.extend(
                registry.parquet_files(
                    dataset,
                    source=source_id,
                    date_from=date_from,
                    date_to=date_to,
                )
            )
        return files

    def _rescan(self) -> None:
        """Force a re-scan of the catalog after new data is written."""
        if self._registry is not None:
            self._registry.scan()

    # --- Download / ingest pipeline ---

    def download(self) -> list[str]:
        """Download and ingest data for the configured dates and sources.

        Executes the full pipeline: fetch → normalize → QC → store → catalog.

        Returns a list of partition keys that were written (dataset/source/date).
        """
        # Import adapters lazily to trigger auto-registration.
        from neptune_ais.adapters import dma as _dma  # noqa: F401
        from neptune_ais.adapters import noaa as _noaa  # noqa: F401
        from neptune_ais.adapters.base import FetchSpec
        from neptune_ais.adapters.registry import get_adapter

        written: list[str] = []

        for source_id in self._sources:
            adapter = get_adapter(source_id)

            for target_date in self._dates:
                partition_key = f"positions/{source_id}/{target_date}"
                logger.info("Processing %s", partition_key)

                try:
                    # 1. Fetch raw artifacts.
                    raw_dir = self._store_root / raw_partition_path(
                        source_id, target_date.isoformat()
                    )
                    # Set download dir on adapter if it supports it.
                    if hasattr(adapter, '_download_dir'):
                        adapter._download_dir = raw_dir

                    spec = FetchSpec(
                        date=target_date,
                        bbox=self._bbox,
                        overwrite=self._overwrite,
                    )
                    artifacts = adapter.fetch_raw(spec)

                    # 2. Normalize to canonical positions.
                    positions_df = adapter.normalize_positions(artifacts)

                    # 3. Add pipeline-generated columns.
                    ingest_id = str(uuid.uuid4())
                    positions_df = positions_df.with_columns(
                        pl.lit(artifacts[0].filename).alias("source_file"),
                        pl.lit(ingest_id).alias("ingest_id"),
                        pl.lit("ok").alias("qc_severity"),
                        pl.lit(f"{source_id}:direct").alias("record_provenance"),
                    )

                    # 4. Sort for optimal Parquet layout.
                    sort_cols = [
                        c for c in SORT_ORDER_POSITIONS
                        if c in positions_df.columns
                    ]
                    if sort_cols:
                        positions_df = positions_df.sort(sort_cols)

                    # 5. Write to canonical store via PartitionWriter.
                    date_str = target_date.isoformat()
                    writer = PartitionWriter(
                        self._store_root, "positions", source_id, date_str
                    )

                    try:
                        staging = writer.prepare()
                    except Exception:
                        # Stale staging — abort and retry.
                        writer.abort()
                        staging = writer.prepare()

                    # Shard if necessary.
                    n_rows = len(positions_df)
                    shard_files: list[str] = []

                    if n_rows <= DEFAULT_MAX_ROWS_PER_SHARD:
                        shard_file = shard_filename(0)
                        positions_df.write_parquet(
                            staging / shard_file,
                            compression=PARQUET_COMPRESSION,
                            compression_level=PARQUET_COMPRESSION_LEVEL,
                            row_group_size=DEFAULT_ROW_GROUP_SIZE,
                            statistics=PARQUET_WRITE_STATISTICS,
                        )
                        shard_files.append(shard_file)
                    else:
                        shard_idx = 0
                        for offset in range(0, n_rows, DEFAULT_MAX_ROWS_PER_SHARD):
                            shard_df = positions_df.slice(
                                offset, DEFAULT_MAX_ROWS_PER_SHARD
                            )
                            shard_file = shard_filename(shard_idx)
                            shard_df.write_parquet(
                                staging / shard_file,
                                compression=PARQUET_COMPRESSION,
                                compression_level=PARQUET_COMPRESSION_LEVEL,
                                row_group_size=DEFAULT_ROW_GROUP_SIZE,
                                statistics=PARQUET_WRITE_STATISTICS,
                            )
                            shard_files.append(shard_file)
                            shard_idx += 1

                    # 6. Validate.
                    writer.validate(expected_files=shard_files)

                    # 7. Build manifest.
                    catalog_artifacts = [
                        CatalogRawArtifact(
                            source_url=a.source_url,
                            filename=a.filename,
                            content_hash=a.content_hash,
                            size_bytes=a.size_bytes,
                            fetch_timestamp=a.fetch_timestamp,
                            content_type=a.content_type,
                            local_path=(
                                a.local_path
                                if self._raw_policy == RawPolicy.FULL
                                else None
                            ),
                            headers=a.headers,
                        )
                        for a in artifacts
                    ]

                    # Compute stats for manifest.
                    lat_col = "lat" if "lat" in positions_df.columns else None
                    lon_col = "lon" if "lon" in positions_df.columns else None
                    bbox = None
                    if lat_col and lon_col:
                        bbox = BBox(
                            west=positions_df[lon_col].min(),
                            south=positions_df[lat_col].min(),
                            east=positions_df[lon_col].max(),
                            north=positions_df[lat_col].max(),
                        )

                    ts_col = "timestamp"
                    min_ts = positions_df[ts_col].min()
                    max_ts = positions_df[ts_col].max()

                    distinct_mmsi = positions_df["mmsi"].n_unique()

                    manifest = Manifest(
                        dataset="positions",
                        source=source_id,
                        date=date_str,
                        schema_version=current_schema_version("positions"),
                        adapter_version=getattr(
                            adapter, "ADAPTER_VERSION",
                            f"{source_id}/unknown",
                        ),
                        transform_version="normalize/0.1.0",
                        files=shard_files,
                        raw_artifacts=catalog_artifacts,
                        raw_policy=self._raw_policy,
                        record_count=n_rows,
                        distinct_mmsi_count=distinct_mmsi,
                        min_timestamp=min_ts,
                        max_timestamp=max_ts,
                        bbox=bbox,
                        qc_summary=QCSummary(
                            total_rows=n_rows,
                            rows_ok=n_rows,
                            rows_warning=0,
                            rows_error=0,
                            rows_dropped=0,
                        ),
                        write_status=WriteStatus.COMMITTED,
                    )

                    # 8. Commit.
                    writer.commit(
                        manifest_json=manifest.model_dump_json(indent=2)
                    )

                    written.append(partition_key)
                    logger.info(
                        "Wrote %s: %d rows, %d MMSIs",
                        partition_key, n_rows, distinct_mmsi,
                    )

                except Exception:
                    logger.exception("Failed to process %s", partition_key)
                    # Clean up staging if it was created.
                    if "writer" in locals():
                        writer.abort()

        # Re-scan catalog after writes.
        self._rescan()
        return written

    # --- Polars lazy accessors ---

    def positions(self, *, raw: bool = False) -> pl.LazyFrame:
        """Return positions as a Polars LazyFrame.

        Scans all committed positions partitions matching the configured
        dates and sources.
        """
        from neptune_ais.fusion import MergeMode, merge as fusion_merge

        registry = self._get_registry()
        date_from, date_to = self._date_range

        # Collect files per source for potential fusion.
        per_source_files: dict[str, list[Path]] = {}
        for source_id in self._sources:
            source_files = registry.parquet_files(
                "positions",
                source=source_id,
                date_from=date_from,
                date_to=date_to,
            )
            if source_files:
                per_source_files[source_id] = source_files

        if not per_source_files:
            logger.warning("No positions data found for the configured scope")
            from neptune_ais.datasets.positions import SCHEMA
            return pl.DataFrame(schema=SCHEMA).lazy()

        # If multiple sources and merge mode is not UNION, apply fusion.
        if (
            len(per_source_files) > 1
            and self._fusion_config.mode != MergeMode.UNION
        ):
            # Read each source eagerly for fusion (dedup requires materialized data).
            frames: dict[str, pl.DataFrame] = {}
            for source_id, source_files in per_source_files.items():
                frames[source_id] = pl.read_parquet(
                    source_files,
                    allow_missing_columns=True,
                )
            merged = fusion_merge(frames, self._fusion_config)
            lf = merged.lazy()
        else:
            # Single source or UNION mode — scan lazily.
            all_files = [f for files in per_source_files.values() for f in files]
            lf = pl.scan_parquet(
                all_files,
                missing_columns="insert",
                extra_columns="ignore",
            )
            # Tag union provenance if multi-source UNION.
            if len(per_source_files) > 1 and self._fusion_config.tag_provenance:
                lf = lf.with_columns(
                    (pl.col("source") + pl.lit(":union")).alias("record_provenance"),
                )

        # Apply filters.
        if self._bbox:
            west, south, east, north = self._bbox
            lf = lf.filter(
                (pl.col("lat") >= south)
                & (pl.col("lat") <= north)
                & (pl.col("lon") >= west)
                & (pl.col("lon") <= east)
            )

        if self._mmsi:
            lf = lf.filter(pl.col("mmsi").is_in(self._mmsi))

        return lf

    def tracks(
        self,
        *,
        gap: str = "30m",
        min_points: int = 3,
        min_duration: str = "5m",
        min_distance_m: float = 100.0,
        generalize: str = "0",
        include_geometry: bool = False,
        refresh: bool = False,
        as_geo: bool = False,
        as_movingpandas: bool = False,
    ):
        """Derive and return tracks.

        Runs the full track derivation pipeline on positions data:
        sort → detect boundaries → filter segments → aggregate.

        Results are cached in the derived store. Subsequent calls with
        the same parameters and unchanged upstream data return cached
        results without recomputation.

        Args:
            gap: Max observation gap before segment break (e.g. "30m").
            min_points: Minimum positions per segment.
            min_duration: Minimum segment duration (e.g. "5m").
            min_distance_m: Minimum segment distance in meters.
            generalize: Douglas-Peucker tolerance (e.g. "1m", "0" to disable).
            include_geometry: Whether to include WKB geometry and timestamp offsets.
            refresh: If True, recompute even if cached.
        """
        from neptune_ais.derive.tracks import (
            TrackConfig,
            aggregate_tracks,
            detect_boundaries,
            filter_segments,
            parse_track_args,
        )
        from neptune_ais.datasets.tracks import SCHEMA

        config = parse_track_args(
            gap=gap,
            min_points=min_points,
            min_duration=min_duration,
            min_distance_m=min_distance_m,
            generalize=generalize,
            include_geometry=include_geometry,
            refresh=refresh,
        )

        # Check for cached tracks in the derived store.
        derived_files = self._dataset_files("tracks")
        if derived_files and not config.refresh:
            logger.info("Using cached tracks (%d file(s))", len(derived_files))
            lf = pl.scan_parquet(
                derived_files,
                missing_columns="insert",
                extra_columns="ignore",
            )
            if as_geo:
                from neptune_ais.geometry.bridges import tracks_to_geodataframe
                return tracks_to_geodataframe(lf)
            return lf

        # Compute tracks from positions.
        positions = self.positions().collect()
        if len(positions) == 0:
            return pl.DataFrame(schema=SCHEMA).lazy()

        positions = positions.sort(["mmsi", "timestamp"])

        # Run pipeline per source.
        all_tracks: list[pl.DataFrame] = []
        for source_id in positions["source"].unique().to_list():
            source_df = positions.filter(pl.col("source") == source_id)
            segmented = detect_boundaries(source_df, config)
            filtered = filter_segments(segmented, config)
            if len(filtered) > 0:
                tracks_df = aggregate_tracks(filtered, config, source=source_id)
                all_tracks.append(tracks_df)

        if not all_tracks:
            return pl.DataFrame(schema=SCHEMA).lazy()

        result = pl.concat(all_tracks) if len(all_tracks) > 1 else all_tracks[0]

        logger.info(
            "Derived %d tracks from %d positions",
            len(result),
            len(positions),
        )

        # Optional geometry conversions.
        if as_geo:
            from neptune_ais.geometry.bridges import tracks_to_geodataframe
            return tracks_to_geodataframe(result)
        if as_movingpandas:
            from neptune_ais.geometry.bridges import tracks_to_movingpandas
            return tracks_to_movingpandas(result, positions)

        return result.lazy()

    def events(
        self,
        *,
        kind: str | None = None,
        min_confidence: float | None = None,
    ) -> pl.LazyFrame:
        """Return events as a Polars LazyFrame.

        Scans all committed event partitions matching the configured
        dates and sources. Events are produced by detector pipelines
        and stored in the derived event store.

        Args:
            kind: Filter by event type (e.g. ``"port_call"``).
                If None, returns all event types.
            min_confidence: Minimum confidence score (0.0–1.0).
                If set, filters out events below this threshold.

        Returns:
            A Polars LazyFrame over the events dataset. Returns an
            empty LazyFrame with the correct schema if no events
            are found.
        """
        from neptune_ais.datasets.events import Col as EventCol, SCHEMA

        files = self._dataset_files("events")
        if not files:
            return pl.DataFrame(schema=SCHEMA).lazy()

        lf = pl.scan_parquet(
            files,
            missing_columns="insert",
            extra_columns="ignore",
        )

        if kind is not None:
            lf = lf.filter(pl.col(EventCol.EVENT_TYPE) == kind)

        if min_confidence is not None:
            lf = lf.filter(
                pl.col(EventCol.CONFIDENCE_SCORE) >= min_confidence
            )

        if self._bbox:
            west, south, east, north = self._bbox
            lf = lf.filter(
                (pl.col(EventCol.LAT) >= south)
                & (pl.col(EventCol.LAT) <= north)
                & (pl.col(EventCol.LON) >= west)
                & (pl.col(EventCol.LON) <= east)
            )

        if self._mmsi:
            lf = lf.filter(pl.col(EventCol.MMSI).is_in(self._mmsi))

        return lf

    def vessels(self) -> pl.LazyFrame:
        """Return vessels as a Polars LazyFrame."""
        files = self._dataset_files("vessels")
        if not files:
            from neptune_ais.datasets.vessels import SCHEMA
            return pl.DataFrame(schema=SCHEMA).lazy()

        return pl.scan_parquet(
            files,
            missing_columns="insert",
            extra_columns="ignore",
        )

    # --- DuckDB access ---

    def duckdb(self):
        """Return a DuckDB connection with cataloged datasets registered.

        The connection has views for each dataset (positions, vessels,
        tracks, events) pointing to the same Parquet files that Polars
        would scan.
        """
        import duckdb

        con = duckdb.connect()

        for dataset_name in ("positions", "vessels", "tracks", "events"):
            files = self._dataset_files(dataset_name)
            if files:
                file_list = ", ".join(f"'{f}'" for f in files)
                con.execute(
                    f"CREATE VIEW {dataset_name} AS "
                    f"SELECT * FROM read_parquet([{file_list}])"
                )

        return con

    def sql(self, query: str):
        """Execute a SQL query over the cataloged datasets.

        Returns a DuckDB relation (call .df() for pandas, .pl() for polars).
        """
        con = self.duckdb()
        return con.sql(query)

    # --- Helper APIs ---

    def latest_positions(self) -> pl.LazyFrame:
        """Return the most recent position per vessel.

        For each MMSI in the configured scope, returns the single
        row with the latest timestamp.
        """
        from neptune_ais.helpers import latest_positions
        return latest_positions(self.positions())

    def snapshot(self, when: str | datetime) -> pl.LazyFrame:
        """Return the closest position per vessel to a timestamp.

        Args:
            when: Target timestamp (datetime or ISO-8601 string).
        """
        from neptune_ais.helpers import snapshot
        return snapshot(self.positions(), when)

    def vessel_history(self, mmsi: int) -> dict[str, pl.LazyFrame]:
        """Return all data for a single vessel.

        Returns a dict with ``"positions"`` and optionally
        ``"tracks"`` and ``"events"`` LazyFrames filtered to the
        given MMSI.
        """
        from neptune_ais.helpers import vessel_history

        tracks = None
        tracks_files = self._dataset_files("tracks")
        if tracks_files:
            tracks = pl.scan_parquet(
                tracks_files,
                missing_columns="insert",
                extra_columns="ignore",
            )

        events_lf = None
        events_files = self._dataset_files("events")
        if events_files:
            events_lf = pl.scan_parquet(
                events_files,
                missing_columns="insert",
                extra_columns="ignore",
            )

        return vessel_history(
            mmsi,
            positions=self.positions(),
            tracks=tracks,
            events=events_lf,
        )

    # --- Inspection surfaces ---

    def inventory(self):
        """Return inventory summaries for the configured scope."""
        registry = self._get_registry()
        return registry.inventory()

    def provenance(self, dataset: str = "positions"):
        """Return provenance summary for a dataset in the configured scope."""
        date_from, date_to = self._date_range
        return self._get_registry().provenance(
            dataset, date_from=date_from, date_to=date_to,
        )

    def quality_report(self, dataset: str = "positions"):
        """Return quality report for a dataset in the configured scope."""
        date_from, date_to = self._date_range
        return self._get_registry().quality_report(
            dataset, date_from=date_from, date_to=date_to,
        )

    def fusion_info(self) -> dict:
        """Return information about the configured fusion behavior.

        Provides a combined view of:
        - Fusion configuration (mode, precedence, tolerances)
        - Multi-source inventory (which sources have data)
        - Per-source partition and row counts

        This is the primary inspection surface for understanding how
        multi-source results are assembled.
        """
        from neptune_ais.fusion import MergeMode

        registry = self._get_registry()
        date_from, date_to = self._date_range

        # Per-source breakdown.
        source_details: list[dict] = []
        total_partitions = 0
        total_rows = 0

        for source_id in self._sources:
            parts = registry.partitions(
                "positions",
                source=source_id,
                date_from=date_from,
                date_to=date_to,
            )
            rows = sum(p.record_count for p in parts)
            source_details.append({
                "source": source_id,
                "partitions": len(parts),
                "rows": rows,
            })
            total_partitions += len(parts)
            total_rows += rows

        # Fusion config summary.
        fc = self._fusion_config
        config_summary = {
            "mode": fc.mode.value,
            "source_precedence": fc.source_precedence or self._sources,
            "timestamp_tolerance_seconds": fc.timestamp_tolerance_seconds,
            "coordinate_tolerance_degrees": fc.coordinate_tolerance_degrees,
        }
        if fc.mode == MergeMode.PREFER:
            config_summary["prefer_source"] = fc.prefer_source
        if fc.field_precedence:
            config_summary["field_precedence"] = fc.field_precedence
        if fc.source_confidence_weights:
            config_summary["source_confidence_weights"] = fc.source_confidence_weights

        return {
            "sources": self._sources,
            "dates": {
                "from": date_from,
                "to": date_to,
                "count": len(self._dates),
            },
            "fusion": config_summary,
            "per_source": source_details,
            "total_partitions": total_partitions,
            "total_rows_before_fusion": total_rows,
            "multi_source": len(self._sources) > 1,
        }
