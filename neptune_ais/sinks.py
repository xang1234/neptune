"""Stream sinks — durable output targets for live AIS data.

Implements the ``StreamSink`` protocol for Parquet and DuckDB landing,
giving ``NeptuneStream.run_sink()`` concrete output paths.

Module role — separate from stream lifecycle
---------------------------------------------
**Owns:**
- ``ParquetSink`` — rolling Parquet files in a landing directory.
- ``DuckDBSink`` — append-only table in a DuckDB database.

**Does not own:**
- Stream lifecycle or backpressure — that is ``stream``'s job.
- Canonical partitioning or promotion — that is ``storage``/``catalog``.
- Schema definitions — uses ``datasets``.

**Import rule:** Sinks may import from ``storage`` (constants), ``datasets``
(schema), and ``stream`` (protocol, compactor). Must not import from
``adapters``, ``derive``, ``geometry``, ``viz``, ``helpers``, or ``cli``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from neptune_ais.storage import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_WRITE_STATISTICS,
    DEFAULT_ROW_GROUP_SIZE,
    SORT_ORDER_POSITIONS,
)
from neptune_ais.stream import (
    CompactionConfig,
    DEDUP_KEY_FIELDS,
    StreamCompactor,
    StreamSink,
)

logger = logging.getLogger(__name__)

_VALID_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _first_non_none(messages: list[dict[str, Any]], col: str) -> Any:
    """Return the first non-None value for *col* across *messages*."""
    for msg in messages:
        val = msg.get(col)
        if val is not None:
            return val
    return None


# ---------------------------------------------------------------------------
# Parquet sink
# ---------------------------------------------------------------------------


class ParquetSink:
    """Append-only Parquet sink for stream landing data.

    Writes batches of position messages as Parquet files in a landing
    directory.  Each flush produces one file, named with a monotonic
    batch counter and UTC timestamp for ordering.

    File layout::

        {landing_dir}/{source}/landing-{YYYYMMDD_HHMMSS}-{batch:04d}.parquet

    The sink optionally compacts messages before writing (dedup + sort)
    using ``StreamCompactor``.

    Args:
        landing_dir: Root directory for landed data.
        source: Source identifier (used in path and provenance).
        compact: Whether to compact (dedup + sort) before writing.
            Default True.
        compaction_config: Configuration for the compactor. None uses
            defaults.
    """

    def __init__(
        self,
        landing_dir: str | Path,
        source: str = "stream",
        *,
        compact: bool = True,
        compaction_config: CompactionConfig | None = None,
    ) -> None:
        self._landing_dir = Path(landing_dir) / source
        self._source = source
        self._buffer: list[dict[str, Any]] = []
        self._batch_count = 0
        self._rows_written = 0
        self._compact = compact
        self._compactor = StreamCompactor(compaction_config) if compact else None
        self._sort_cols: list[str] | None = None  # cached after first flush
        self._dir_created = False

    @property
    def rows_written(self) -> int:
        """Total rows written across all flushes."""
        return self._rows_written

    @property
    def batch_count(self) -> int:
        """Number of Parquet files written."""
        return self._batch_count

    async def write(self, messages: list[dict[str, Any]]) -> None:
        """Buffer messages for the next flush."""
        self._buffer.extend(messages)

    async def flush(self) -> None:
        """Write buffered messages to a new Parquet file."""
        if not self._buffer:
            return

        # Swap buffer before processing but restore on failure to avoid
        # data loss if the write raises.
        messages, self._buffer = self._buffer, []

        try:
            if self._compact and self._compactor is not None:
                self._compactor.add(messages)
                messages = self._compactor.compact()

            if not messages:
                return  # compaction removed all duplicates

            df = pl.DataFrame(messages)

            # Sort by mmsi, timestamp for efficient downstream reads.
            # Cache sort_cols on first flush: column presence is fixed per stream.
            if self._sort_cols is None:
                self._sort_cols = [c for c in SORT_ORDER_POSITIONS if c in df.columns]
            if self._sort_cols:
                df = df.sort(self._sort_cols)

            if not self._dir_created:
                self._landing_dir.mkdir(parents=True, exist_ok=True)
                self._dir_created = True
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"landing-{ts}-{self._batch_count:04d}.parquet"
            filepath = self._landing_dir / filename

            df.write_parquet(
                filepath,
                compression=PARQUET_COMPRESSION,
                compression_level=PARQUET_COMPRESSION_LEVEL,
                row_group_size=DEFAULT_ROW_GROUP_SIZE,
                statistics=PARQUET_WRITE_STATISTICS,
            )

            self._batch_count += 1
            self._rows_written += len(df)
            logger.debug(
                "ParquetSink: wrote %d rows to %s", len(df), filepath,
            )
        except Exception:
            # Restore buffer so data is not lost on transient failures.
            self._buffer = messages + self._buffer
            raise

    async def close(self) -> None:
        """Flush remaining data and log summary."""
        await self.flush()
        logger.info(
            "ParquetSink closed: %d batches, %d total rows to %s",
            self._batch_count, self._rows_written, self._landing_dir,
        )


# ---------------------------------------------------------------------------
# DuckDB sink
# ---------------------------------------------------------------------------


class DuckDBSink:
    """Append-only DuckDB sink for stream landing data.

    Inserts batches of position messages into a DuckDB table.  Uses
    an in-memory database by default, or a persistent file for durable
    landing.

    The table is created on the first write with columns inferred from
    the message keys.

    Args:
        db_path: Path to DuckDB file, or ``":memory:"`` for in-memory.
        table_name: Target table name. Default ``"landing"``.
        source: Source identifier stored as a column for provenance.
        compact: Whether to compact (dedup + sort) before writing.
        compaction_config: Configuration for the compactor.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        table_name: str = "landing",
        source: str = "stream",
        *,
        compact: bool = True,
        compaction_config: CompactionConfig | None = None,
    ) -> None:
        import duckdb

        if not _VALID_IDENTIFIER.match(table_name):
            raise ValueError(
                f"table_name must be a valid SQL identifier, got {table_name!r}"
            )

        self._db_path = str(db_path)
        self._table_name = table_name
        self._source = source
        self._con = duckdb.connect(self._db_path)
        self._table_created = False
        self._insert_sql: str | None = None   # cached after table creation
        self._columns: list[str] | None = None  # cached after table creation
        self._buffer: list[dict[str, Any]] = []
        self._rows_written = 0
        self._compact = compact
        self._compactor = StreamCompactor(compaction_config) if compact else None

    @property
    def rows_written(self) -> int:
        """Total rows inserted across all flushes."""
        return self._rows_written

    @property
    def connection(self):
        """Return the underlying DuckDB connection for queries."""
        return self._con

    async def write(self, messages: list[dict[str, Any]]) -> None:
        """Buffer messages for the next flush."""
        self._buffer.extend(messages)

    async def flush(self) -> None:
        """Insert buffered messages into the DuckDB table."""
        if not self._buffer:
            return

        messages, self._buffer = self._buffer, []

        try:
            if self._compact and self._compactor is not None:
                self._compactor.add(messages)
                messages = self._compactor.compact()

            if not messages:
                return

            if not self._table_created:
                # Infer schema from first batch. For each column, scan
                # messages for the first non-None value to determine type.
                columns = list(messages[0].keys())
                # bool must precede int (bool subclasses int in Python).
                type_map = {bool: "BOOLEAN", str: "VARCHAR", int: "BIGINT", float: "DOUBLE"}
                col_defs = []
                for col in columns:
                    val = _first_non_none(messages, col)
                    sql_type = type_map.get(type(val), "VARCHAR") if val is not None else "VARCHAR"
                    col_defs.append(f'"{col}" {sql_type}')
                self._con.execute(
                    f"CREATE TABLE IF NOT EXISTS {self._table_name} "
                    f"({', '.join(col_defs)})"
                )
                self._columns = columns
                placeholders = ", ".join("?" for _ in columns)
                quoted_cols = ", ".join(f'"{c}"' for c in columns)
                self._insert_sql = (
                    f"INSERT INTO {self._table_name} ({quoted_cols}) "
                    f"VALUES ({placeholders})"
                )
                self._table_created = True

            # Parameterized batch insert using cached SQL and column order.
            columns = self._columns  # type: ignore[assignment]
            rows = [tuple(msg.get(c) for c in columns) for msg in messages]
            self._con.executemany(self._insert_sql, rows)

            self._rows_written += len(messages)
            logger.debug(
                "DuckDBSink: inserted %d rows into %s",
                len(messages), self._table_name,
            )
        except Exception:
            self._buffer = messages + self._buffer
            raise

    async def close(self) -> None:
        """Flush remaining data and close the connection."""
        await self.flush()
        logger.info(
            "DuckDBSink closed: %d total rows in %s (db: %s)",
            self._rows_written, self._table_name, self._db_path,
        )
        self._con.close()


# ---------------------------------------------------------------------------
# Promotion — landing data → canonical partitions
# ---------------------------------------------------------------------------

PROMOTION_VERSION = "promotion/landing-to-canonical/1.0.0"


@dataclass
class PromotionResult:
    """Result of promoting landing files for one date partition."""

    date: str
    source: str
    record_count: int
    files_promoted: int
    shard_files: list[str]
    landing_files: list[str]


def promote_landing(
    landing_dir: str | Path,
    store_root: str | Path,
    source: str,
    dataset: str = "positions",
    *,
    cleanup: bool = False,
) -> list[PromotionResult]:
    """Promote landed stream data into canonical partitions.

    Reads Parquet files from the landing directory, groups them by date
    (extracted from the ``timestamp`` column), deduplicates, sorts, and
    writes them as canonical partitions using the same atomic
    ``PartitionWriter`` protocol used by archival ingestion.

    Each promoted partition gets a manifest with provenance tracing
    back to the landing files and live source.

    Args:
        landing_dir: Root landing directory (contains ``{source}/`` subdirs).
        store_root: Neptune store root for canonical output.
        source: Source identifier (e.g. ``"aisstream"``).
        dataset: Dataset name. Default ``"positions"``.
        cleanup: If True, delete landing files after successful promotion.

    Returns:
        List of ``PromotionResult`` for each date partition promoted.
    """
    from neptune_ais.catalog import (
        BBox,
        Manifest,
        QCSummary,
        WriteStatus,
        current_schema_version,
    )
    from neptune_ais.storage import (
        DEFAULT_MAX_ROWS_PER_SHARD,
        PartitionWriter,
        PartitionWriteError,
        shard_filename,
    )

    landing_path = Path(landing_dir) / source
    store_path = Path(store_root)

    if not landing_path.exists():
        logger.info("No landing directory for source=%s", source)
        return []

    parquet_files = sorted(landing_path.glob("*.parquet"))
    if not parquet_files:
        logger.info("No landing files found in %s", landing_path)
        return []

    # Lazy scan avoids loading all files into memory at once.
    # Each date partition is collected separately.
    all_data = pl.scan_parquet(parquet_files).with_columns(
        pl.col("timestamp").cast(pl.String).str.slice(0, 10).alias("_date")
    )

    dates = sorted(
        all_data.select("_date").unique().collect()["_date"].to_list()
    )
    if not dates:
        return []

    # Shared Parquet write kwargs — single definition for all shards.
    _write_kwargs = dict(
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
        row_group_size=DEFAULT_ROW_GROUP_SIZE,
        statistics=PARQUET_WRITE_STATISTICS,
    )

    results: list[PromotionResult] = []

    for date_str in dates:
        # Collect only this date's data (predicate pushdown via lazy scan).
        partition_df = (
            all_data.filter(pl.col("_date") == date_str)
            .drop("_date")
            .collect()
        )

        # Dedup by position identity keys (same as streaming dedup).
        dedup_cols = [c for c in DEDUP_KEY_FIELDS if c in partition_df.columns]
        if dedup_cols:
            # All copies with identical (mmsi, timestamp, lat, lon) are
            # equivalent observations — keep any.
            partition_df = partition_df.unique(subset=dedup_cols, keep="any")

        # Sort for optimal Parquet layout.
        sort_cols = [c for c in SORT_ORDER_POSITIONS if c in partition_df.columns]
        if sort_cols:
            partition_df = partition_df.sort(sort_cols)

        n_rows = len(partition_df)

        # Stage → write shards → validate → commit.
        writer = PartitionWriter(store_path, dataset, source, date_str)
        try:
            staging = writer.prepare()
        except PartitionWriteError:
            writer.abort()
            staging = writer.prepare()

        shard_files: list[str] = []
        if n_rows <= DEFAULT_MAX_ROWS_PER_SHARD:
            sf = shard_filename(0)
            partition_df.write_parquet(staging / sf, **_write_kwargs)
            shard_files.append(sf)
        else:
            for shard_idx, offset in enumerate(
                range(0, n_rows, DEFAULT_MAX_ROWS_PER_SHARD)
            ):
                sf = shard_filename(shard_idx)
                partition_df.slice(
                    offset, DEFAULT_MAX_ROWS_PER_SHARD
                ).write_parquet(staging / sf, **_write_kwargs)
                shard_files.append(sf)

        writer.validate(expected_files=shard_files)

        # Build manifest with promotion provenance.
        manifest = Manifest(
            dataset=dataset,
            source=source,
            date=date_str,
            schema_version=current_schema_version(dataset),
            adapter_version=f"{source}/streaming",
            transform_version=PROMOTION_VERSION,
            files=shard_files,
            raw_artifacts=[],
            raw_policy="none",
            record_count=n_rows,
            distinct_mmsi_count=(
                partition_df["mmsi"].n_unique() if "mmsi" in partition_df.columns else 0
            ),
            min_timestamp=partition_df["timestamp"].min(),
            max_timestamp=partition_df["timestamp"].max(),
            bbox=BBox(
                west=float(partition_df["lon"].min()),
                south=float(partition_df["lat"].min()),
                east=float(partition_df["lon"].max()),
                north=float(partition_df["lat"].max()),
            ) if {"lat", "lon"}.issubset(partition_df.columns) else None,
            qc_summary=QCSummary(
                total_rows=n_rows,
                rows_ok=n_rows,
                rows_warning=0,
                rows_error=0,
                rows_dropped=0,
            ),
            write_status=WriteStatus.COMMITTED,
        )

        writer.commit(manifest_json=manifest.model_dump_json(indent=2))

        results.append(PromotionResult(
            date=date_str,
            source=source,
            record_count=n_rows,
            files_promoted=len(parquet_files),
            shard_files=shard_files,
            landing_files=[f.name for f in parquet_files],
        ))

        logger.info(
            "Promoted %d rows for %s/%s/%s → %d shards",
            n_rows, dataset, source, date_str, len(shard_files),
        )

    # Clean up landing files after successful promotion.
    if cleanup and results:
        for f in parquet_files:
            f.unlink()
        logger.info("Cleaned up %d landing files", len(parquet_files))

    return results
