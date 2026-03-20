"""Soak and reliability validation for the full streaming path.

Tests the complete pipeline under sustained operation:
  stream → ingest → dedup → backpressure → sink → compaction → promotion

These tests exercise long-running invariants that unit tests miss:
restart recovery, cross-reconnection dedup, sustained throughput with
backpressure, health state transitions under load, and end-to-end
data integrity from ingest through canonical promotion.

Marked with ``pytest.mark.soak`` for selective execution.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from neptune_ais.sinks import ParquetSink, DuckDBSink, promote_landing
from neptune_ais.stream import (
    BackpressurePolicy,
    Checkpoint,
    NeptuneStream,
    StreamConfig,
    StreamHealth,
    StreamStats,
    load_checkpoint,
    run_with_reconnect,
    save_checkpoint,
)


def _run(coro):
    return asyncio.run(coro)


def _msg(mmsi: int, ts: str, lat: float = 40.0, lon: float = -74.0) -> dict[str, Any]:
    return {
        "mmsi": mmsi,
        "timestamp": ts,
        "lat": lat,
        "lon": lon,
        "sog": 5.0,
        "source": "soak",
    }


def _ts(i: int, date: str = "2024-01-01") -> str:
    """Generate a valid ISO timestamp from an integer index (0–86399)."""
    assert 0 <= i < 86400, f"_ts() index {i} out of valid range [0, 86400)"
    h, m, s = i // 3600, (i % 3600) // 60, i % 60
    return f"{date}T{h:02d}:{m:02d}:{s:02d}"


soak = pytest.mark.soak


# ---------------------------------------------------------------------------
# Sustained throughput with backpressure
# ---------------------------------------------------------------------------


class TestSustainedThroughput:
    """Verify the pipeline handles hundreds of messages without corruption."""

    @soak
    def test_high_volume_block_policy(self):
        """Block policy under sustained load: no data loss."""
        async def _test():
            config = StreamConfig(
                max_queue_size=10,
                backpressure=BackpressurePolicy.BLOCK,
            )
            async with NeptuneStream(config=config) as stream:
                n_messages = 200

                async def _produce():
                    for i in range(n_messages):
                        await stream.ingest(_msg(mmsi=i, ts=_ts(i)))

                async def _consume():
                    count = 0
                    async for _ in stream:
                        count += 1
                        if count >= n_messages:
                            break
                    return count

                producer = asyncio.create_task(_produce())
                consumed = await _consume()
                await producer

                assert consumed == n_messages
                assert stream.stats.messages_received == n_messages
                assert stream.stats.messages_delivered == n_messages
                assert stream.stats.messages_dropped == 0

        _run(_test())

    @soak
    def test_high_volume_drop_oldest_policy(self):
        """Drop-oldest under sustained load: stats are consistent."""
        async def _test():
            config = StreamConfig(
                max_queue_size=5,
                backpressure=BackpressurePolicy.DROP_OLDEST,
            )
            async with NeptuneStream(config=config) as stream:
                # Produce 100 messages without consuming (all go into queue + drops).
                for i in range(100):
                    await stream.ingest(
                        _msg(mmsi=i, ts=f"2024-01-01T00:{i:02d}:00")
                    )

                stats = stream.stats
                assert stats.messages_received == 100
                # delivered + dropped should equal received (minus deduped).
                assert stats.messages_delivered + stats.messages_dropped == (
                    stats.messages_received - stats.messages_deduplicated
                )
                # Queue holds exactly max_queue_size messages.
                assert stream._message_queue.qsize() == 5

        _run(_test())

    @soak
    def test_dedup_under_sustained_load(self):
        """Rolling dedup correctly handles window eviction over many messages."""
        async def _test():
            config = StreamConfig(dedup_window_size=50)
            async with NeptuneStream(config=config) as stream:
                # Send 100 unique messages with valid timestamps.
                for i in range(100):
                    await stream.ingest(_msg(mmsi=i, ts=_ts(i)))
                assert stream.stats.messages_delivered == 100
                assert stream.stats.messages_deduplicated == 0

                # Re-send mmsi=0 — it was evicted from the window (size 50).
                accepted = await stream.ingest(
                    _msg(mmsi=0, ts="2024-01-01T00:00:00")
                )
                assert accepted is True  # evicted, so accepted again

                # Re-send mmsi=99 (last message) — still in the window.
                accepted = await stream.ingest(
                    _msg(mmsi=99, ts="2024-01-01T00:01:39")
                )
                assert accepted is False  # still in window

        _run(_test())


# ---------------------------------------------------------------------------
# Restart and reconnect reliability
# ---------------------------------------------------------------------------


class TestRestartReliability:
    """Verify checkpoint persistence and reconnection recovery."""

    @soak
    def test_checkpoint_survives_multiple_reconnections(self, tmp_path):
        """Checkpoint accumulates correctly across multiple failures."""
        async def _test():
            call_count = 0

            async def connect():
                nonlocal call_count
                call_count += 1
                # Deliver some messages before failing.
                await stream.ingest(
                    _msg(mmsi=call_count, ts=f"2024-01-01T00:0{call_count}:00")
                )
                if call_count < 4:
                    raise ConnectionError(f"fail {call_count}")

            config = StreamConfig(
                source="soak-test",
                reconnect_delay_s=0.01,
                checkpoint_dir=str(tmp_path),
            )
            async with NeptuneStream(config=config) as stream:
                await run_with_reconnect(stream, connect, max_retries=5)

            assert call_count == 4
            assert stream.stats.reconnections == 3

            # Verify checkpoint was saved with accumulated counts.
            cp = load_checkpoint("soak-test", str(tmp_path))
            assert cp is not None
            assert cp.messages_total == 4
            assert cp.session_count == 1

        _run(_test())

    @soak
    def test_checkpoint_resume_from_prior_session(self, tmp_path):
        """A new session resumes from a prior checkpoint."""
        # Session 1: save a checkpoint.
        cp = Checkpoint(
            source="soak-test",
            last_timestamp="2024-01-01T00:00:00+00:00",
            messages_total=500,
            session_count=3,
        )
        save_checkpoint(cp, str(tmp_path))

        async def _test():
            async def connect():
                await stream.ingest(_msg(mmsi=1, ts="2024-01-01T01:00:00"))

            config = StreamConfig(
                source="soak-test",
                reconnect_delay_s=0.01,
                checkpoint_dir=str(tmp_path),
            )
            async with NeptuneStream(config=config) as stream:
                await run_with_reconnect(stream, connect, max_retries=0)

            # Verify session count incremented and total accumulated.
            loaded = load_checkpoint("soak-test", str(tmp_path))
            assert loaded is not None
            assert loaded.session_count == 4  # was 3, now +1
            assert loaded.messages_total == 501  # was 500, +1

        _run(_test())


# ---------------------------------------------------------------------------
# Health state transitions
# ---------------------------------------------------------------------------


class TestHealthTransitions:
    """Verify health state machine under sustained operation."""

    @soak
    def test_full_lifecycle_health(self):
        """DISCONNECTED → STALE → HEALTHY → LAGGING → STALE → DISCONNECTED."""
        async def _test():
            config = StreamConfig(
                lag_threshold_s=0.02,
                # Wide gap so asyncio.sleep overshoot can't push us past
                # stale_threshold before we assert LAGGING.  With a 480ms
                # safe window (0.02–0.5s) the test tolerates heavy CI load.
                stale_threshold_s=0.5,
            )
            stream = NeptuneStream(config=config)

            # Before start: DISCONNECTED.
            assert stream.health is StreamHealth.DISCONNECTED

            async with stream:
                # Running, no messages yet: HEALTHY (just started).
                assert stream.health is StreamHealth.HEALTHY

                # After message: HEALTHY.
                await stream.ingest(_msg(mmsi=1, ts="2024-01-01T00:00:00"))
                assert stream.health is StreamHealth.HEALTHY

                # Wait past lag threshold (0.02s) but well inside stale (0.5s).
                await asyncio.sleep(0.07)
                assert stream.health is StreamHealth.LAGGING

                # Wait past stale threshold (need ≥0.5s total elapsed).
                await asyncio.sleep(0.50)
                assert stream.health is StreamHealth.STALE

                # Recovery: new message restores HEALTHY.
                await stream.ingest(_msg(mmsi=2, ts="2024-01-01T00:01:00"))
                assert stream.health is StreamHealth.HEALTHY

            # After exit: DISCONNECTED.
            assert stream.health is StreamHealth.DISCONNECTED

        _run(_test())

    @soak
    def test_health_snapshot_consistency(self):
        """Snapshot health and lag_seconds are always consistent."""
        async def _test():
            config = StreamConfig(
                lag_threshold_s=0.01,
                # Wide stale window so a slow asyncio.sleep(0.05) can't
                # overshoot into STALE before we assert LAGGING.
                stale_threshold_s=0.20,
            )
            async with NeptuneStream(config=config) as stream:
                # Just-started snapshot: healthy, lag measures from start time.
                snap = stream.health_snapshot()
                assert snap["health"] == "healthy"
                assert snap["lag_seconds"] is not None
                assert snap["lag_seconds"] < 1.0

                # Healthy snapshot.
                await stream.ingest(_msg(mmsi=1, ts="2024-01-01T00:00:00"))
                snap = stream.health_snapshot()
                assert snap["health"] == "healthy"
                assert snap["lag_seconds"] is not None
                assert snap["lag_seconds"] < 0.1

                # Lagging snapshot: sleep past lag (0.01s), well inside stale (0.20s).
                await asyncio.sleep(0.05)
                snap = stream.health_snapshot()
                assert snap["health"] == "lagging"
                assert snap["lag_seconds"] >= 0.01

        _run(_test())


# ---------------------------------------------------------------------------
# End-to-end: stream → sink → promotion
# ---------------------------------------------------------------------------


try:
    from pydantic import BaseModel as _PydanticModel
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

skip_no_pydantic = pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

skip_no_duckdb = pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb not installed")


class TestEndToEndParquet:
    """Full pipeline: stream → ParquetSink → promote → canonical."""

    @soak
    @skip_no_pydantic
    def test_stream_to_canonical(self, tmp_path):
        """100 messages flow from stream through sink to canonical store."""
        landing = tmp_path / "landing"
        store = tmp_path / "store"

        async def _test():
            sink = ParquetSink(landing, source="soak")
            async with NeptuneStream() as stream:
                for i in range(100):
                    await stream.ingest(
                        _msg(mmsi=i % 20, ts=_ts(i, date="2024-06-15"))
                    )
                await stream.run_sink(sink, max_messages=100)

            assert sink.rows_written > 0

        _run(_test())

        # Promote landed data to canonical.
        results = promote_landing(landing, store, source="soak")
        assert len(results) >= 1
        total_promoted = sum(r.record_count for r in results)
        assert total_promoted == 100  # all unique (20 mmsi × different timestamps)

        # Verify canonical data is readable.
        canonical = store / "canonical" / "positions" / "source=soak"
        parquet_files = list(canonical.rglob("*.parquet"))
        assert len(parquet_files) >= 1
        df = pl.concat([pl.read_parquet(f) for f in parquet_files])
        assert len(df) == 100

    @soak
    @skip_no_pydantic
    def test_stream_to_canonical_with_duplicates(self, tmp_path):
        """Duplicates are removed at both dedup layers (stream + promotion)."""
        landing = tmp_path / "landing"
        store = tmp_path / "store"

        async def _test():
            sink = ParquetSink(landing, source="soak", compact=False)
            async with NeptuneStream() as stream:
                # Send 50 unique + 50 duplicates.
                for i in range(50):
                    await stream.ingest(
                        _msg(mmsi=i, ts=_ts(i, date="2024-06-15"))
                    )
                # Duplicates are caught by in-flight dedup.
                for i in range(50):
                    await stream.ingest(
                        _msg(mmsi=i, ts=_ts(i, date="2024-06-15"))
                    )
                await stream.run_sink(sink, max_messages=50)

            assert stream.stats.messages_deduplicated == 50
            assert sink.rows_written == 50

        _run(_test())

        results = promote_landing(landing, store, source="soak")
        total = sum(r.record_count for r in results)
        assert total == 50  # no additional duplicates after promotion dedup

    @soak
    @skip_no_pydantic
    def test_promotion_manifest_integrity(self, tmp_path):
        """Promoted manifest has correct provenance and statistics."""
        landing = tmp_path / "landing"
        store = tmp_path / "store"

        async def _test():
            sink = ParquetSink(landing, source="soak")
            async with NeptuneStream() as stream:
                for i in range(10):
                    await stream.ingest(
                        _msg(mmsi=i, ts=f"2024-06-15T00:0{i}:00", lat=40.0 + i * 0.1, lon=-74.0 + i * 0.1)
                    )
                await stream.run_sink(sink, max_messages=10)

        _run(_test())

        results = promote_landing(landing, store, source="soak")
        assert len(results) == 1

        manifest_path = store / "manifests" / "positions" / "source=soak" / "date=2024-06-15.json"
        manifest = json.loads(manifest_path.read_text())

        assert manifest["dataset"] == "positions"
        assert manifest["source"] == "soak"
        assert manifest["record_count"] == 10
        assert manifest["distinct_mmsi_count"] == 10
        assert "promotion" in manifest["transform_version"]
        assert manifest["write_status"] == "committed"
        assert manifest["bbox"]["south"] < manifest["bbox"]["north"]
        assert manifest["bbox"]["west"] < manifest["bbox"]["east"]


@skip_no_duckdb
class TestEndToEndDuckDB:
    """Full pipeline: stream → DuckDBSink → query."""

    @soak
    def test_stream_to_duckdb_query(self):
        """Messages flow from stream into DuckDB and are queryable."""
        async def _test():
            sink = DuckDBSink(db_path=":memory:", table_name="positions")
            async with NeptuneStream() as stream:
                for i in range(50):
                    await stream.ingest(
                        _msg(mmsi=i % 10, ts=_ts(i, date="2024-06-15"))
                    )
                # Flush manually (not run_sink, which closes the connection).
                await stream.run_sink(sink, max_messages=50)

            # run_sink calls close(), which flushes + closes connection.
            # Verify stats before connection closed.
            assert sink.rows_written == 50

        _run(_test())

    @soak
    def test_duckdb_queryable_before_close(self):
        """Data is queryable via SQL after flush but before close."""
        async def _test():
            sink = DuckDBSink(db_path=":memory:", table_name="positions")
            msgs = [
                _msg(mmsi=i % 10, ts=f"2024-06-15T00:00:{i:02d}")
                for i in range(50)
            ]
            await sink.write(msgs)
            await sink.flush()

            # Query while connection is still open.
            result = sink.connection.execute(
                "SELECT count(*) FROM positions"
            ).fetchone()
            assert result[0] == 50

            result = sink.connection.execute(
                "SELECT mmsi, count(*) as cnt FROM positions "
                "GROUP BY mmsi ORDER BY mmsi"
            ).fetchall()
            assert len(result) == 10
            assert all(r[1] == 5 for r in result)

            await sink.close()

        _run(_test())


# ---------------------------------------------------------------------------
# Compaction reliability
# ---------------------------------------------------------------------------


class TestCompactionReliability:
    """Verify compaction integrity under sustained operation."""

    @soak
    def test_compaction_across_many_batches(self):
        """StreamCompactor maintains correct stats across 20 compact cycles."""
        from neptune_ais.stream import CompactionConfig, StreamCompactor

        compactor = StreamCompactor(CompactionConfig(trigger_count=30))
        total_added = 0
        total_compacted = 0

        for batch in range(20):
            msgs = [
                _msg(mmsi=i, ts=f"2024-01-01T{batch:02d}:{i:02d}:00")
                for i in range(15)  # 15 per batch, all unique across batches
            ]
            # Add within-batch duplicates (same mmsi+ts as first msg in batch).
            msgs.append(_msg(mmsi=0, ts=f"2024-01-01T{batch:02d}:00:00"))

            compactor.add(msgs)
            total_added += len(msgs)

            if compactor.should_compact():
                result = compactor.compact()
                total_compacted += len(result)

        # Final compact for remaining.
        if compactor.pending_count > 0:
            result = compactor.compact()
            total_compacted += len(result)

        # Each batch has 1 intra-batch dupe → 20 removed total.
        assert compactor.stats.messages_before == total_added
        assert compactor.stats.messages_removed == 20  # 1 dup per batch × 20 batches
        assert total_compacted < total_added


# ---------------------------------------------------------------------------
# Stats invariants
# ---------------------------------------------------------------------------


class TestStatsInvariants:
    """Verify statistical invariants hold under various conditions."""

    @soak
    def test_stats_invariant_no_backpressure(self):
        """received = delivered + deduplicated (no drops)."""
        async def _test():
            async with NeptuneStream() as stream:
                for i in range(50):
                    await stream.ingest(
                        _msg(mmsi=i % 10, ts=f"2024-01-01T00:{i:02d}:00")
                    )
                # Send some dupes.
                for i in range(10):
                    await stream.ingest(
                        _msg(mmsi=i, ts=f"2024-01-01T00:{i:02d}:00")
                    )

                s = stream.stats
                assert s.messages_received == 60
                assert s.messages_deduplicated == 10
                assert s.messages_delivered == 50
                assert s.messages_dropped == 0
                assert s.backpressure_events == 0
                # Invariant: received = delivered + deduplicated + dropped.
                assert s.messages_received == (
                    s.messages_delivered + s.messages_deduplicated + s.messages_dropped
                )

        _run(_test())

    @soak
    def test_stats_invariant_with_drops(self):
        """received = delivered + deduplicated + dropped (with backpressure)."""
        async def _test():
            config = StreamConfig(
                max_queue_size=5,
                backpressure=BackpressurePolicy.DROP_OLDEST,
            )
            async with NeptuneStream(config=config) as stream:
                for i in range(30):
                    await stream.ingest(
                        _msg(mmsi=i, ts=f"2024-01-01T00:{i:02d}:00")
                    )

                s = stream.stats
                # Invariant must hold regardless of drops.
                assert s.messages_received == (
                    s.messages_delivered + s.messages_deduplicated + s.messages_dropped
                )
                # 30 messages into queue of 5 → at least 25 drops.
                assert s.messages_dropped >= 25
                assert s.backpressure_events > 0

        _run(_test())

    @soak
    def test_dedup_rate_accuracy(self):
        """dedup_rate reflects actual duplicate ratio."""
        async def _test():
            async with NeptuneStream() as stream:
                # 20 unique + 10 dupes = 30 received, 10/30 dedup rate.
                for i in range(20):
                    await stream.ingest(
                        _msg(mmsi=i, ts=f"2024-01-01T00:{i:02d}:00")
                    )
                for i in range(10):
                    await stream.ingest(
                        _msg(mmsi=i, ts=f"2024-01-01T00:{i:02d}:00")
                    )

                assert stream.stats.dedup_rate == pytest.approx(10 / 30)

        _run(_test())
