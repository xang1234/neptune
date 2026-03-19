"""Tests for ParquetSink and DuckDBSink stream landing.

Covers: write/flush/close lifecycle, file output, compaction
integration, DuckDB table creation, and StreamSink protocol
conformance.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from neptune_ais.sinks import DuckDBSink, ParquetSink
from neptune_ais.stream import StreamSink


def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _sample_message(
    mmsi: int = 111,
    ts: str = "2024-06-15T00:00:00",
    lat: float = 40.0,
    lon: float = -74.0,
) -> dict[str, Any]:
    return {
        "mmsi": mmsi,
        "timestamp": ts,
        "lat": lat,
        "lon": lon,
        "sog": 5.0,
        "source": "test",
    }


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# ParquetSink
# ---------------------------------------------------------------------------


class TestParquetSink:
    def test_protocol_conformance(self):
        """ParquetSink satisfies StreamSink protocol."""
        assert isinstance(ParquetSink("/tmp/test"), StreamSink)

    def test_write_and_flush(self, tmp_path):
        async def _test():
            sink = ParquetSink(tmp_path, source="ais")
            await sink.write([
                _sample_message(mmsi=1, ts="2024-01-01T00:00:00"),
                _sample_message(mmsi=2, ts="2024-01-01T00:01:00"),
            ])
            await sink.flush()

            assert sink.rows_written == 2
            assert sink.batch_count == 1

            files = list((tmp_path / "ais").glob("*.parquet"))
            assert len(files) == 1

            df = pl.read_parquet(files[0])
            assert len(df) == 2
            assert "mmsi" in df.columns
            assert "timestamp" in df.columns

        _run(_test())

    def test_multiple_flushes(self, tmp_path):
        async def _test():
            sink = ParquetSink(tmp_path, source="ais")
            await sink.write([_sample_message(mmsi=1, ts="2024-01-01T00:00:00")])
            await sink.flush()
            await sink.write([_sample_message(mmsi=2, ts="2024-01-01T00:01:00")])
            await sink.flush()

            assert sink.batch_count == 2
            assert sink.rows_written == 2
            files = list((tmp_path / "ais").glob("*.parquet"))
            assert len(files) == 2

        _run(_test())

    def test_flush_empty_noop(self, tmp_path):
        async def _test():
            sink = ParquetSink(tmp_path, source="ais")
            await sink.flush()
            assert sink.batch_count == 0
            assert sink.rows_written == 0

        _run(_test())

    def test_close_flushes_remaining(self, tmp_path):
        async def _test():
            sink = ParquetSink(tmp_path, source="ais")
            await sink.write([_sample_message()])
            await sink.close()

            assert sink.rows_written == 1
            files = list((tmp_path / "ais").glob("*.parquet"))
            assert len(files) == 1

        _run(_test())

    def test_sorts_by_mmsi_timestamp(self, tmp_path):
        async def _test():
            sink = ParquetSink(tmp_path, source="ais")
            await sink.write([
                _sample_message(mmsi=2, ts="2024-01-01T00:01:00"),
                _sample_message(mmsi=1, ts="2024-01-01T00:00:00"),
            ])
            await sink.flush()

            files = list((tmp_path / "ais").glob("*.parquet"))
            df = pl.read_parquet(files[0])
            assert df["mmsi"].to_list() == [1, 2]

        _run(_test())

    def test_compaction_deduplicates(self, tmp_path):
        """Compaction removes duplicates before writing."""
        async def _test():
            sink = ParquetSink(tmp_path, source="ais")
            msg = _sample_message(mmsi=1, ts="2024-01-01T00:00:00")
            await sink.write([msg, msg, msg])
            await sink.flush()

            files = list((tmp_path / "ais").glob("*.parquet"))
            df = pl.read_parquet(files[0])
            assert len(df) == 1

        _run(_test())

    def test_compaction_disabled(self, tmp_path):
        """With compact=False, duplicates are preserved."""
        async def _test():
            sink = ParquetSink(tmp_path, source="ais", compact=False)
            msg = _sample_message(mmsi=1, ts="2024-01-01T00:00:00")
            await sink.write([msg, msg, msg])
            await sink.flush()

            files = list((tmp_path / "ais").glob("*.parquet"))
            df = pl.read_parquet(files[0])
            assert len(df) == 3

        _run(_test())

    @pytest.mark.skipif(
        not _has_module("pyarrow.parquet"),
        reason="pyarrow not installed",
    )
    def test_parquet_uses_zstd(self, tmp_path):
        """Parquet files use ZSTD compression."""
        async def _test():
            sink = ParquetSink(tmp_path, source="ais")
            await sink.write([_sample_message()])
            await sink.flush()

            files = list((tmp_path / "ais").glob("*.parquet"))
            # Read metadata to verify compression.
            import pyarrow.parquet as pq
            meta = pq.read_metadata(files[0])
            col_meta = meta.row_group(0).column(0)
            assert "ZSTD" in col_meta.compression.upper()

        _run(_test())

    def test_file_naming(self, tmp_path):
        async def _test():
            sink = ParquetSink(tmp_path, source="myais")
            await sink.write([_sample_message()])
            await sink.flush()

            files = list((tmp_path / "myais").glob("*.parquet"))
            assert len(files) == 1
            name = files[0].name
            assert name.startswith("landing-")
            assert name.endswith("-0000.parquet")

        _run(_test())


# ---------------------------------------------------------------------------
# DuckDBSink
# ---------------------------------------------------------------------------


try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

skip_no_duckdb = pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb not installed")


@skip_no_duckdb
class TestDuckDBSink:
    def test_protocol_conformance(self):
        assert isinstance(DuckDBSink(), StreamSink)

    def test_write_and_flush(self):
        async def _test():
            sink = DuckDBSink(db_path=":memory:", table_name="test_landing")
            await sink.write([
                _sample_message(mmsi=1, ts="2024-01-01T00:00:00"),
                _sample_message(mmsi=2, ts="2024-01-01T00:01:00"),
            ])
            await sink.flush()

            assert sink.rows_written == 2
            result = sink.connection.execute(
                "SELECT count(*) FROM test_landing"
            ).fetchone()
            assert result[0] == 2

        _run(_test())

    def test_multiple_flushes_append(self):
        async def _test():
            sink = DuckDBSink(db_path=":memory:")
            await sink.write([_sample_message(mmsi=1, ts="2024-01-01T00:00:00")])
            await sink.flush()
            await sink.write([_sample_message(mmsi=2, ts="2024-01-01T00:01:00")])
            await sink.flush()

            result = sink.connection.execute(
                "SELECT count(*) FROM landing"
            ).fetchone()
            assert result[0] == 2

        _run(_test())

    def test_flush_empty_noop(self):
        async def _test():
            sink = DuckDBSink(db_path=":memory:")
            await sink.flush()
            assert sink.rows_written == 0
            # Table should not exist yet.
            tables = sink.connection.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
            assert len(tables) == 0

        _run(_test())

    def test_close_flushes_and_closes(self):
        async def _test():
            sink = DuckDBSink(db_path=":memory:")
            await sink.write([_sample_message()])
            await sink.close()
            assert sink.rows_written == 1

        _run(_test())

    def test_compaction_deduplicates(self):
        async def _test():
            sink = DuckDBSink(db_path=":memory:")
            msg = _sample_message(mmsi=1, ts="2024-01-01T00:00:00")
            await sink.write([msg, msg, msg])
            await sink.flush()

            result = sink.connection.execute(
                "SELECT count(*) FROM landing"
            ).fetchone()
            assert result[0] == 1

        _run(_test())

    def test_compaction_disabled(self):
        async def _test():
            sink = DuckDBSink(db_path=":memory:", compact=False)
            msg = _sample_message(mmsi=1, ts="2024-01-01T00:00:00")
            await sink.write([msg, msg, msg])
            await sink.flush()

            result = sink.connection.execute(
                "SELECT count(*) FROM landing"
            ).fetchone()
            assert result[0] == 3

        _run(_test())

    def test_queryable_after_flush(self):
        """Data is queryable via SQL immediately after flush."""
        async def _test():
            sink = DuckDBSink(db_path=":memory:", table_name="positions")
            await sink.write([
                _sample_message(mmsi=111, ts="2024-01-01T00:00:00"),
                _sample_message(mmsi=222, ts="2024-01-01T00:01:00"),
            ])
            await sink.flush()

            result = sink.connection.execute(
                "SELECT mmsi FROM positions ORDER BY mmsi"
            ).fetchall()
            assert [r[0] for r in result] == [111, 222]

        _run(_test())

    def test_persistent_db(self, tmp_path):
        """DuckDB file persists data across sink instances."""
        db_file = tmp_path / "test.duckdb"

        async def _test():
            sink = DuckDBSink(db_path=db_file, table_name="ais")
            await sink.write([_sample_message(mmsi=1, ts="2024-01-01T00:00:00")])
            await sink.close()

        _run(_test())

        # Reopen and verify data persisted.
        con = duckdb.connect(str(db_file))
        result = con.execute("SELECT count(*) FROM ais").fetchone()
        assert result[0] == 1
        con.close()

    def test_schema_matches_message_keys(self):
        """Table columns match the message dict keys."""
        async def _test():
            sink = DuckDBSink(db_path=":memory:")
            await sink.write([_sample_message()])
            await sink.flush()

            cols = sink.connection.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'landing' ORDER BY ordinal_position"
            ).fetchall()
            col_names = [c[0] for c in cols]
            assert "mmsi" in col_names
            assert "timestamp" in col_names
            assert "lat" in col_names
            assert "source" in col_names

        _run(_test())
