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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from neptune_ais.storage import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_WRITE_STATISTICS,
    DEFAULT_ROW_GROUP_SIZE,
)
from neptune_ais.stream import (
    CompactionConfig,
    StreamCompactor,
    StreamSink,
)

logger = logging.getLogger(__name__)

# Columns present in raw streaming messages that map to the positions schema.
# Streaming messages are dicts with these keys; Parquet files will have them
# as columns.  Additional keys in the message are preserved as-is.
_LANDING_SORT_COLUMNS = ["mmsi", "timestamp"]


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

        messages = self._buffer
        self._buffer = []

        if self._compact and self._compactor is not None:
            self._compactor.add(messages)
            messages = self._compactor.compact()

        if not messages:
            return  # compaction removed all duplicates

        df = pl.DataFrame(messages)

        # Sort by mmsi, timestamp for efficient downstream reads.
        sort_cols = [c for c in _LANDING_SORT_COLUMNS if c in df.columns]
        if sort_cols:
            df = df.sort(sort_cols)

        self._landing_dir.mkdir(parents=True, exist_ok=True)
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

        self._db_path = str(db_path)
        self._table_name = table_name
        self._source = source
        self._con = duckdb.connect(self._db_path)
        self._table_created = False
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

        messages = self._buffer
        self._buffer = []

        if self._compact and self._compactor is not None:
            self._compactor.add(messages)
            messages = self._compactor.compact()

        if not messages:
            return

        columns = list(messages[0].keys())

        if not self._table_created:
            # Infer schema from first message and create table.
            type_map = {str: "VARCHAR", int: "BIGINT", float: "DOUBLE"}
            col_defs = []
            for col in columns:
                val = messages[0].get(col)
                sql_type = type_map.get(type(val), "VARCHAR")
                col_defs.append(f'"{col}" {sql_type}')
            self._con.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table_name} "
                f"({', '.join(col_defs)})"
            )
            self._table_created = True

        # Parameterized batch insert.
        placeholders = ", ".join("?" for _ in columns)
        quoted_cols = ", ".join(f'"{c}"' for c in columns)
        sql = (
            f"INSERT INTO {self._table_name} ({quoted_cols}) "
            f"VALUES ({placeholders})"
        )
        rows = [tuple(msg.get(c) for c in columns) for msg in messages]
        self._con.executemany(sql, rows)

        self._rows_written += len(messages)
        logger.debug(
            "DuckDBSink: inserted %d rows into %s",
            len(messages), self._table_name,
        )

    async def close(self) -> None:
        """Flush remaining data and close the connection."""
        await self.flush()
        logger.info(
            "DuckDBSink closed: %d total rows in %s (db: %s)",
            self._rows_written, self._table_name, self._db_path,
        )
        self._con.close()
