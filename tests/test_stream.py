"""Tests for NeptuneStream async interface and lifecycle.

Covers: stream lifecycle (context manager), async iteration,
message ingestion, deduplication, sink runner, and statistics.

Uses asyncio.run() directly instead of pytest-asyncio to avoid
requiring the optional pytest-asyncio dependency.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from neptune_ais.stream import (
    BackpressurePolicy,
    Checkpoint,
    NeptuneStream,
    StreamConfig,
    StreamSink,
    StreamStats,
    _message_hash,
    load_checkpoint,
    run_with_reconnect,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_message(mmsi: int = 111, ts: str = "2024-06-15T00:00:00") -> dict:
    return {
        "mmsi": mmsi,
        "timestamp": ts,
        "lat": 40.0,
        "lon": -74.0,
        "sog": 5.0,
        "source": "aisstream",
    }


class MockSink:
    """In-memory sink for testing."""

    def __init__(self) -> None:
        self.batches: list[list[dict[str, Any]]] = []
        self.flushed = False
        self.closed = False

    async def write(self, messages: list[dict[str, Any]]) -> None:
        self.batches.append(list(messages))

    async def flush(self) -> None:
        self.flushed = True

    async def close(self) -> None:
        self.closed = True

    @property
    def total_messages(self) -> int:
        return sum(len(b) for b in self.batches)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# StreamConfig
# ---------------------------------------------------------------------------


class TestStreamConfig:
    def test_defaults(self):
        config = StreamConfig()
        assert config.source == "aisstream"
        assert config.reconnect_delay_s == 5.0
        assert config.dedup_window_size == 10_000
        assert config.flush_interval_s == 60

    def test_custom_config(self):
        config = StreamConfig(
            source="finland",
            api_key="test-key",
            bbox=(10.0, 55.0, 30.0, 70.0),
            mmsi=[111, 222],
        )
        assert config.source == "finland"
        assert config.api_key == "test-key"
        assert config.bbox == (10.0, 55.0, 30.0, 70.0)
        assert config.mmsi == [111, 222]


# ---------------------------------------------------------------------------
# StreamStats
# ---------------------------------------------------------------------------


class TestStreamStats:
    def test_initial_stats(self):
        stats = StreamStats()
        assert stats.messages_received == 0
        assert stats.messages_delivered == 0
        assert stats.dedup_rate == 0.0

    def test_dedup_rate(self):
        stats = StreamStats(messages_received=100, messages_deduplicated=20)
        assert stats.dedup_rate == 0.2


# ---------------------------------------------------------------------------
# Message hash
# ---------------------------------------------------------------------------


class TestMessageHash:
    def test_deterministic(self):
        msg = _sample_message()
        assert _message_hash(msg) == _message_hash(msg)

    def test_differs_by_mmsi(self):
        a = _message_hash(_sample_message(mmsi=111))
        b = _message_hash(_sample_message(mmsi=222))
        assert a != b

    def test_differs_by_timestamp(self):
        a = _message_hash(_sample_message(ts="2024-06-15T00:00:00"))
        b = _message_hash(_sample_message(ts="2024-06-15T00:01:00"))
        assert a != b

    def test_format(self):
        h = _message_hash(_sample_message())
        assert isinstance(h, str)
        assert len(h) == 16  # SHA-1 hex prefix


# ---------------------------------------------------------------------------
# NeptuneStream lifecycle
# ---------------------------------------------------------------------------


class TestStreamLifecycle:
    def test_context_manager(self):
        async def _test():
            async with NeptuneStream(source="aisstream") as stream:
                assert stream.is_running
            assert not stream.is_running
        _run(_test())

    def test_config_via_constructor(self):
        async def _test():
            async with NeptuneStream(source="test", api_key="key123") as stream:
                assert stream.config.source == "test"
                assert stream.config.api_key == "key123"
        _run(_test())

    def test_config_object(self):
        async def _test():
            config = StreamConfig(source="custom", dedup_window_size=500)
            async with NeptuneStream(config=config) as stream:
                assert stream.config.source == "custom"
                assert stream.config.dedup_window_size == 500
        _run(_test())

    def test_stats_initial(self):
        async def _test():
            async with NeptuneStream() as stream:
                assert stream.stats.messages_received == 0
        _run(_test())


# ---------------------------------------------------------------------------
# Message ingestion and dedup
# ---------------------------------------------------------------------------


class TestIngestion:
    def test_ingest_accepted(self):
        async def _test():
            async with NeptuneStream() as stream:
                accepted = await stream.ingest(_sample_message())
                assert accepted is True
                assert stream.stats.messages_received == 1
                assert stream.stats.messages_delivered == 1
        _run(_test())

    def test_duplicate_rejected(self):
        async def _test():
            async with NeptuneStream() as stream:
                msg = _sample_message()
                await stream.ingest(msg)
                accepted = await stream.ingest(msg)
                assert accepted is False
                assert stream.stats.messages_received == 2
                assert stream.stats.messages_delivered == 1
                assert stream.stats.messages_deduplicated == 1
        _run(_test())

    def test_different_messages_accepted(self):
        async def _test():
            async with NeptuneStream() as stream:
                await stream.ingest(_sample_message(mmsi=111))
                await stream.ingest(_sample_message(mmsi=222))
                assert stream.stats.messages_delivered == 2
        _run(_test())

    def test_dedup_window_size(self):
        """Old hashes are evicted from the dedup window."""
        async def _test():
            config = StreamConfig(dedup_window_size=3)
            async with NeptuneStream(config=config) as stream:
                for i in range(4):
                    await stream.ingest(_sample_message(mmsi=i))
                accepted = await stream.ingest(_sample_message(mmsi=0))
                assert accepted is True
        _run(_test())


# ---------------------------------------------------------------------------
# Async iteration
# ---------------------------------------------------------------------------


class TestAsyncIteration:
    def test_iterate_messages(self):
        """Ingest messages, then use run_sink to consume them."""
        async def _test():
            sink = MockSink()
            async with NeptuneStream() as stream:
                await stream.ingest(_sample_message(mmsi=111))
                await stream.ingest(_sample_message(mmsi=222))
                await stream.run_sink(sink, max_messages=2)

            assert sink.total_messages == 2
            msgs = [m for batch in sink.batches for m in batch]
            assert msgs[0]["mmsi"] == 111
            assert msgs[1]["mmsi"] == 222
        _run(_test())


# ---------------------------------------------------------------------------
# Sink runner
# ---------------------------------------------------------------------------


class TestSinkRunner:
    def test_sink_receives_messages(self):
        async def _test():
            sink = MockSink()
            async with NeptuneStream() as stream:
                for i in range(5):
                    await stream.ingest(
                        _sample_message(mmsi=i, ts=f"2024-06-15T00:0{i}:00")
                    )
                await stream.run_sink(sink, max_messages=5, batch_size=2)

            assert sink.total_messages == 5
            assert sink.flushed
            assert sink.closed
        _run(_test())

    def test_sink_batch_size(self):
        async def _test():
            sink = MockSink()
            async with NeptuneStream() as stream:
                for i in range(4):
                    await stream.ingest(
                        _sample_message(mmsi=i, ts=f"2024-06-15T00:0{i}:00")
                    )
                await stream.run_sink(sink, max_messages=4, batch_size=3)

            assert len(sink.batches) == 2
            assert len(sink.batches[0]) == 3
            assert len(sink.batches[1]) == 1
        _run(_test())

    def test_sink_protocol_conformance(self):
        """MockSink satisfies StreamSink protocol."""
        assert isinstance(MockSink(), StreamSink)

    def test_sink_cleanup_on_max_messages(self):
        async def _test():
            sink = MockSink()
            async with NeptuneStream() as stream:
                for i in range(10):
                    await stream.ingest(
                        _sample_message(mmsi=i, ts=f"2024-06-15T00:{i:02d}:00")
                    )
                await stream.run_sink(sink, max_messages=3)

            assert sink.total_messages == 3
            assert sink.closed
        _run(_test())


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_roundtrip(self):
        cp = Checkpoint(
            source="aisstream",
            last_timestamp="2024-06-15T00:00:00+00:00",
            messages_total=1000,
            session_count=3,
            last_saved="2024-06-15T01:00:00+00:00",
        )
        restored = Checkpoint.from_json(cp.to_json())
        assert restored.source == "aisstream"
        assert restored.messages_total == 1000
        assert restored.session_count == 3

    def test_save_and_load(self, tmp_path):
        cp = Checkpoint(
            source="test",
            last_timestamp="2024-06-15T00:00:00+00:00",
            messages_total=500,
            session_count=1,
        )
        save_checkpoint(cp, str(tmp_path))
        loaded = load_checkpoint("test", str(tmp_path))
        assert loaded is not None
        assert loaded.source == "test"
        assert loaded.messages_total == 500
        assert loaded.last_saved != ""  # populated by save_checkpoint

    def test_load_missing_returns_none(self, tmp_path):
        assert load_checkpoint("nonexistent", str(tmp_path)) is None

    def test_save_creates_directory(self, tmp_path):
        nested = tmp_path / "deep" / "nested"
        cp = Checkpoint(source="test", messages_total=1)
        save_checkpoint(cp, str(nested))
        assert (nested / "test.checkpoint.json").exists()

    def test_corrupt_checkpoint_returns_none(self, tmp_path):
        filepath = tmp_path / "bad.checkpoint.json"
        filepath.write_text("not json")
        assert load_checkpoint("bad", str(tmp_path)) is None


# ---------------------------------------------------------------------------
# Reconnect loop
# ---------------------------------------------------------------------------


class TestReconnect:
    def test_successful_connection_no_retry(self):
        """A successful connection exits immediately."""
        call_count = 0

        async def _test():
            nonlocal call_count

            async def connect():
                nonlocal call_count
                call_count += 1

            async with NeptuneStream() as stream:
                await run_with_reconnect(stream, connect, max_retries=3)

            assert call_count == 1

        _run(_test())

    def test_retry_on_failure(self):
        """Connection retries after failure."""
        call_count = 0

        async def _test():
            nonlocal call_count

            async def connect():
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("test failure")

            config = StreamConfig(reconnect_delay_s=0.01, max_reconnect_delay_s=0.02)
            async with NeptuneStream(config=config) as stream:
                await run_with_reconnect(stream, connect, max_retries=5)

            assert call_count == 3

        _run(_test())

    def test_max_retries_exceeded(self):
        """Stops after max_retries."""
        call_count = 0

        async def _test():
            nonlocal call_count

            async def connect():
                nonlocal call_count
                call_count += 1
                raise ConnectionError("always fails")

            config = StreamConfig(reconnect_delay_s=0.01)
            async with NeptuneStream(config=config) as stream:
                await run_with_reconnect(stream, connect, max_retries=2)

            # 1 initial + 2 retries = 3 calls total.
            assert call_count == 3
            # reconnections counts only actual retries (not the terminal failure).
            assert stream.stats.reconnections == 2
            # errors counts all failures including the terminal one.
            assert stream.stats.errors == 3

        _run(_test())

    def test_checkpoint_saved_on_reconnect(self, tmp_path):
        """Checkpoint is saved between reconnection attempts."""
        async def _test():
            async def connect():
                raise ConnectionError("fail")

            config = StreamConfig(
                source="test",
                reconnect_delay_s=0.01,
                checkpoint_dir=str(tmp_path),
            )
            async with NeptuneStream(config=config) as stream:
                await run_with_reconnect(stream, connect, max_retries=1)

        _run(_test())
        loaded = load_checkpoint("test", str(tmp_path))
        assert loaded is not None
        assert loaded.session_count >= 1

    def test_stats_track_reconnections(self):
        """StreamStats.reconnections increments on each failure."""
        async def _test():
            call_count = 0

            async def connect():
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise ConnectionError("fail")

            config = StreamConfig(reconnect_delay_s=0.01)
            async with NeptuneStream(config=config) as stream:
                await run_with_reconnect(stream, connect, max_retries=5)
                assert stream.stats.reconnections == 2

        _run(_test())


# ---------------------------------------------------------------------------
# Bounded buffers and backpressure
# ---------------------------------------------------------------------------


class TestBackpressure:
    def test_default_queue_is_bounded(self):
        """Queue has a finite maxsize by default."""
        stream = NeptuneStream()
        assert stream._message_queue.maxsize == 10_000

    def test_custom_queue_size(self):
        config = StreamConfig(max_queue_size=50)
        stream = NeptuneStream(config=config)
        assert stream._message_queue.maxsize == 50

    def test_invalid_backpressure_policy(self):
        with pytest.raises(ValueError):
            StreamConfig(backpressure="ignore")

    def test_block_policy_awaits_consumer(self):
        """With 'block' policy, ingest awaits until queue has space."""
        async def _test():
            config = StreamConfig(max_queue_size=2, backpressure="block")
            async with NeptuneStream(config=config) as stream:
                # Fill the queue.
                await stream.ingest(_sample_message(mmsi=1, ts="2024-01-01T00:00:00"))
                await stream.ingest(_sample_message(mmsi=2, ts="2024-01-01T00:01:00"))

                # Third ingest should block. Use a task + short timeout to verify.
                blocked = True

                async def _try_ingest():
                    nonlocal blocked
                    await stream.ingest(_sample_message(mmsi=3, ts="2024-01-01T00:02:00"))
                    blocked = False

                task = asyncio.create_task(_try_ingest())
                await asyncio.sleep(0.05)
                assert blocked  # still waiting

                # Consume one message to unblock.
                msg = await stream._message_queue.get()
                assert msg is not None
                await asyncio.sleep(0.05)
                assert not blocked  # ingest completed

                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        _run(_test())

    def test_drop_oldest_policy(self):
        """With 'drop_oldest' policy, oldest message is discarded."""
        async def _test():
            config = StreamConfig(max_queue_size=2, backpressure="drop_oldest")
            async with NeptuneStream(config=config) as stream:
                await stream.ingest(_sample_message(mmsi=1, ts="2024-01-01T00:00:00"))
                await stream.ingest(_sample_message(mmsi=2, ts="2024-01-01T00:01:00"))
                # Queue is full. This should drop the oldest (mmsi=1).
                await stream.ingest(_sample_message(mmsi=3, ts="2024-01-01T00:02:00"))

                assert stream.stats.messages_dropped == 1
                assert stream.stats.backpressure_events == 1
                assert stream._message_queue.qsize() == 2

                # Verify the oldest was dropped — remaining are mmsi=2 and mmsi=3.
                msg1 = stream._message_queue.get_nowait()
                msg2 = stream._message_queue.get_nowait()
                assert msg1["mmsi"] == 2
                assert msg2["mmsi"] == 3

        _run(_test())

    def test_backpressure_stats_accumulate(self):
        """Backpressure events are counted across multiple occurrences."""
        async def _test():
            config = StreamConfig(max_queue_size=1, backpressure="drop_oldest")
            async with NeptuneStream(config=config) as stream:
                for i in range(5):
                    await stream.ingest(
                        _sample_message(mmsi=i, ts=f"2024-01-01T00:0{i}:00")
                    )
                # Queue size 1 means 4 drops (first fills, next 4 each drop oldest).
                assert stream.stats.backpressure_events == 4
                assert stream.stats.messages_dropped == 4
                assert stream.stats.messages_delivered == 5

        _run(_test())

    def test_block_policy_records_backpressure_event(self):
        """Block policy also records backpressure_events for observability."""
        async def _test():
            config = StreamConfig(max_queue_size=1, backpressure="block")
            async with NeptuneStream(config=config) as stream:
                await stream.ingest(_sample_message(mmsi=1, ts="2024-01-01T00:00:00"))

                # Second ingest will trigger backpressure. Consume to unblock.
                async def _delayed_consume():
                    await asyncio.sleep(0.02)
                    await stream._message_queue.get()

                consumer = asyncio.create_task(_delayed_consume())
                await stream.ingest(_sample_message(mmsi=2, ts="2024-01-01T00:01:00"))
                await consumer

                assert stream.stats.backpressure_events == 1
                assert stream.stats.messages_dropped == 0  # block never drops

        _run(_test())

    def test_drop_oldest_with_sink(self):
        """Drop-oldest integrates correctly with sink runner."""
        async def _test():
            config = StreamConfig(max_queue_size=3, backpressure="drop_oldest")
            sink = MockSink()
            async with NeptuneStream(config=config) as stream:
                # Overfill the queue.
                for i in range(5):
                    await stream.ingest(
                        _sample_message(mmsi=i, ts=f"2024-01-01T00:0{i}:00")
                    )
                # Queue keeps the 3 newest: mmsi 2, 3, 4.
                await stream.run_sink(sink, max_messages=3)

            assert sink.total_messages == 3
            msgs = [m for batch in sink.batches for m in batch]
            mmsis = [m["mmsi"] for m in msgs]
            assert mmsis == [2, 3, 4]

        _run(_test())
