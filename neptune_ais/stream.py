"""NeptuneStream — async streaming AIS interface.

Connects to real-time AIS feeds (e.g. aisstream.io, Digitraffic MQTT) and
provides checkpointed, back-pressure-aware ingestion with pluggable sinks.

Module role — separate lifecycle from Neptune
---------------------------------------------
**Owns:**
- The ``NeptuneStream`` async iterator and sink framework.
- Checkpoint management, reconnect/retry policy, backpressure.
- Rolling dedup within a streaming window.
- Pluggable sinks: ``ParquetSink``, ``DuckDBSink``, future Kafka/Arrow
  Flight.
- Promotion of stream landing zone data into canonical partitions.

**Does not own:**
- Source connection details — delegates to streaming-capable ``adapters``.
- Schema definitions — uses ``datasets``.
- Catalog registration — delegates to ``catalog`` when promoting to
  canonical.
- Storage layout — delegates to ``storage``.

**Import rule:** Stream may import from ``adapters`` (streaming-capable
adapters), ``datasets``, ``storage``, ``catalog``, and ``qc``. It must
not import from ``derive``, ``geometry``, ``viz``, ``helpers``, or ``cli``.

**Install extra:** ``pip install neptune-ais[stream]`` (websockets).

Lifecycle
---------
``NeptuneStream`` is an async context manager::

    async with NeptuneStream(source="aisstream", api_key="...") as stream:
        async for message in stream:
            print(message["mmsi"], message["lat"], message["lon"])

Or with a sink::

    async with NeptuneStream(source="aisstream", api_key="...") as stream:
        await stream.run_sink(ParquetSink("/data/live"))

The stream handles reconnection on failure, deduplicates within a
rolling window, and supports checkpointing for restart-safe operation.

Non-goals for the current implementation:
- Full orchestration framework (use Airflow/Prefect for that).
- Kafka/Arrow Flight sinks (future work).
- Stream-to-canonical promotion (requires catalog integration).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stream configuration
# ---------------------------------------------------------------------------


@dataclass
class StreamConfig:
    """Configuration for a NeptuneStream instance.

    Args:
        source: Source adapter ID (e.g. ``"aisstream"``).
        api_key: API key for authenticated sources.
        bbox: Optional spatial filter ``(west, south, east, north)``.
        mmsi: Optional MMSI filter list.
        reconnect_delay_s: Seconds to wait before reconnecting
            after a connection failure. Default 5.
        max_reconnect_delay_s: Maximum backoff delay. Default 60.
        dedup_window_size: Number of recent message hashes to keep
            for rolling deduplication. Default 10000.
        max_queue_size: Maximum number of messages that can be buffered
            in the internal queue before backpressure is applied.
            Default 10000.
        backpressure: Policy when the queue is full. ``"block"``
            (default) awaits until space is available, naturally
            throttling the producer. ``"drop_oldest"`` discards the
            oldest queued message to make room, prioritizing freshness.
        checkpoint_dir: Directory for checkpoint files. None disables
            checkpointing.
        flush_interval_s: How often sinks should flush. Default 60.
    """

    source: str = "aisstream"
    api_key: str = ""
    bbox: tuple[float, float, float, float] | None = None
    mmsi: list[int] | None = None
    reconnect_delay_s: float = 5.0
    max_reconnect_delay_s: float = 60.0
    dedup_window_size: int = 10_000
    max_queue_size: int = 10_000
    backpressure: str = "block"  # "block" or "drop_oldest"
    checkpoint_dir: str | None = None
    flush_interval_s: int = 60


# ---------------------------------------------------------------------------
# Sink protocol — pluggable output targets
# ---------------------------------------------------------------------------


@runtime_checkable
class StreamSink(Protocol):
    """Protocol for stream output sinks.

    Sinks receive batches of normalized position dicts and write
    them to their target (Parquet files, DuckDB, etc.).
    """

    async def write(self, messages: list[dict[str, Any]]) -> None:
        """Write a batch of messages to the sink."""
        ...

    async def flush(self) -> None:
        """Flush any buffered data to storage."""
        ...

    async def close(self) -> None:
        """Close the sink and release resources."""
        ...


# ---------------------------------------------------------------------------
# Stream state
# ---------------------------------------------------------------------------


@dataclass
class StreamStats:
    """Runtime statistics for a stream session."""

    messages_received: int = 0
    messages_deduplicated: int = 0
    messages_delivered: int = 0
    messages_dropped: int = 0
    backpressure_events: int = 0
    reconnections: int = 0
    errors: int = 0
    last_message_time: float | None = None

    @property
    def dedup_rate(self) -> float:
        """Fraction of messages that were duplicates."""
        if self.messages_received == 0:
            return 0.0
        return self.messages_deduplicated / self.messages_received


# ---------------------------------------------------------------------------
# NeptuneStream
# ---------------------------------------------------------------------------


class NeptuneStream:
    """Async streaming AIS interface.

    Connects to a live AIS source and yields normalized position
    messages as dicts. Supports pluggable sinks, rolling dedup,
    reconnection with exponential backoff, and checkpointing.

    Usage as async iterator::

        async with NeptuneStream(config=StreamConfig(source="aisstream")) as s:
            async for msg in s:
                print(msg["mmsi"])

    Usage with a sink::

        async with NeptuneStream(config=StreamConfig(source="aisstream")) as s:
            await s.run_sink(my_sink, max_messages=1000)

    Args:
        config: Stream configuration. If None, uses defaults.
        source: Shorthand for ``StreamConfig(source=source)``.
        api_key: Shorthand for ``StreamConfig(api_key=api_key)``.
    """

    def __init__(
        self,
        *,
        config: StreamConfig | None = None,
        source: str = "aisstream",
        api_key: str = "",
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = StreamConfig(source=source, api_key=api_key)

        if self._config.backpressure not in ("block", "drop_oldest"):
            raise ValueError(
                f"backpressure must be 'block' or 'drop_oldest', "
                f"got {self._config.backpressure!r}"
            )

        self._stats = StreamStats()
        self._dedup_queue: deque[str] = deque(
            maxlen=self._config.dedup_window_size
        )
        self._dedup_set: set[str] = set()
        self._running = False
        self._message_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
            maxsize=self._config.max_queue_size
        )

    @property
    def config(self) -> StreamConfig:
        """Return the stream configuration."""
        return self._config

    @property
    def stats(self) -> StreamStats:
        """Return current stream statistics."""
        return self._stats

    @property
    def is_running(self) -> bool:
        """Whether the stream is currently connected and running."""
        return self._running

    # --- Async context manager ---

    async def __aenter__(self) -> NeptuneStream:
        self._running = True
        logger.info(
            "NeptuneStream started: source=%s", self._config.source
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._running = False
        # Drain any remaining messages and put a single sentinel.
        # This ensures no orphaned sentinels accumulate.
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        await self._message_queue.put(None)
        logger.info(
            "NeptuneStream stopped: %d messages received, %d delivered, %d deduped",
            self._stats.messages_received,
            self._stats.messages_delivered,
            self._stats.messages_deduplicated,
        )

    # --- Async iterator ---

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self._running and self._message_queue.empty():
            raise StopAsyncIteration
        msg = await self._message_queue.get()
        if msg is None:
            raise StopAsyncIteration
        return msg

    # --- Message ingestion (called by adapter) ---

    async def ingest(self, message: dict[str, Any]) -> bool:
        """Ingest a single normalized message from an adapter.

        Applies deduplication. Returns True if the message was
        accepted (not a duplicate), False if it was deduplicated.

        This is the adapter→stream interface. Streaming adapters
        call this method for each normalized position they produce.
        """
        self._stats.messages_received += 1

        # Rolling dedup by message hash (O(1) lookup via parallel set).
        msg_hash = _message_hash(message)
        if msg_hash in self._dedup_set:
            self._stats.messages_deduplicated += 1
            return False

        # Evict oldest hash if window is full.
        if len(self._dedup_queue) == self._dedup_queue.maxlen:
            self._dedup_set.discard(self._dedup_queue[0])
        self._dedup_queue.append(msg_hash)
        self._dedup_set.add(msg_hash)
        self._stats.messages_delivered += 1

        # Bounded queue with backpressure policy.
        if self._message_queue.full():
            self._stats.backpressure_events += 1
            if self._config.backpressure == "drop_oldest":
                try:
                    self._message_queue.get_nowait()
                    self._stats.messages_dropped += 1
                except asyncio.QueueEmpty:
                    pass  # pragma: no cover — race condition guard
                self._message_queue.put_nowait(message)
                return True
            # "block" policy: fall through to await put().

        await self._message_queue.put(message)
        return True

    # --- Sink runner ---

    async def run_sink(
        self,
        sink: StreamSink,
        *,
        max_messages: int | None = None,
        batch_size: int = 100,
    ) -> None:
        """Run a sink against the stream until stopped.

        Collects messages into batches of ``batch_size`` and writes
        them to the sink. Flushes periodically based on
        ``config.flush_interval_s``.

        Args:
            sink: The output sink to write to.
            max_messages: Stop after this many messages. None = run
                until the stream is closed.
            batch_size: Number of messages per write batch.
        """
        batch: list[dict[str, Any]] = []
        count = 0

        try:
            async for msg in self:
                batch.append(msg)
                count += 1

                if len(batch) >= batch_size:
                    await sink.write(batch)
                    batch = []

                if max_messages is not None and count >= max_messages:
                    break

            # Write any remaining messages.
            if batch:
                await sink.write(batch)
        finally:
            await sink.flush()
            await sink.close()


# ---------------------------------------------------------------------------
# Checkpoint — restart-safe state persistence
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    """Persistent state for restart-safe streaming.

    Stored as a JSON file in ``StreamConfig.checkpoint_dir``. On restart,
    the stream loads the checkpoint to resume from the last known state.

    Since most AIS WebSocket sources (including AISStream) do not support
    replay from a cursor, the checkpoint serves primarily for:
    - **Gap detection** — how long was the stream down?
    - **Dedup on restart** — skip messages older than ``last_timestamp``.
    - **Operational monitoring** — total counts across restarts.

    Attributes:
        source: Source adapter ID.
        last_timestamp: ISO-8601 timestamp of the last processed message.
        messages_total: Cumulative message count across all sessions.
        session_count: Number of stream sessions (restarts).
        last_saved: When this checkpoint was written.
    """

    source: str = ""
    last_timestamp: str = ""
    messages_total: int = 0
    session_count: int = 0
    last_saved: str = ""

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({
            "source": self.source,
            "last_timestamp": self.last_timestamp,
            "messages_total": self.messages_total,
            "session_count": self.session_count,
            "last_saved": self.last_saved,
        }, indent=2)

    @classmethod
    def from_json(cls, data: str) -> Checkpoint:
        """Deserialize from JSON string."""
        d = json.loads(data)
        return cls(
            source=d.get("source", ""),
            last_timestamp=d.get("last_timestamp", ""),
            messages_total=d.get("messages_total", 0),
            session_count=d.get("session_count", 0),
            last_saved=d.get("last_saved", ""),
        )


def save_checkpoint(checkpoint: Checkpoint, checkpoint_dir: str) -> None:
    """Write a checkpoint to disk.

    Creates the directory if it doesn't exist. The checkpoint file
    is named ``<source>.checkpoint.json``.
    """
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)

    checkpoint.last_saved = datetime.now(timezone.utc).isoformat()
    filepath = path / f"{checkpoint.source}.checkpoint.json"
    filepath.write_text(checkpoint.to_json())
    logger.debug("Checkpoint saved: %s", filepath)


def load_checkpoint(source: str, checkpoint_dir: str) -> Checkpoint | None:
    """Load a checkpoint from disk, or return None if not found."""
    filepath = Path(checkpoint_dir) / f"{source}.checkpoint.json"
    if not filepath.exists():
        return None

    try:
        return Checkpoint.from_json(filepath.read_text())
    except (ValueError, KeyError):
        logger.warning("Corrupt checkpoint file: %s", filepath)
        return None


# ---------------------------------------------------------------------------
# Reconnect loop — exponential backoff with jitter
# ---------------------------------------------------------------------------


async def run_with_reconnect(
    stream: NeptuneStream,
    connect_fn,
    *,
    max_retries: int | None = None,
) -> None:
    """Run a streaming connection with automatic reconnection.

    Wraps a connect function (e.g. ``connect_and_stream``) in a
    retry loop with exponential backoff. Saves checkpoints between
    reconnection attempts if checkpointing is enabled.

    Args:
        stream: A running NeptuneStream instance.
        connect_fn: An async callable that connects to the source and
            feeds messages into ``stream.ingest()``. Should return when
            the connection drops.
        max_retries: Maximum number of reconnection attempts. None
            means retry indefinitely.

    The backoff schedule uses ``stream.config.reconnect_delay_s`` as
    the initial delay and doubles on each attempt up to
    ``stream.config.max_reconnect_delay_s``, with ±25% jitter.
    """
    config = stream.config
    delay = config.reconnect_delay_s
    attempts = 0

    # Load checkpoint if available.
    checkpoint = None
    if config.checkpoint_dir:
        checkpoint = load_checkpoint(config.source, config.checkpoint_dir)
        if checkpoint:
            logger.info(
                "Resuming from checkpoint: %d messages, last at %s",
                checkpoint.messages_total, checkpoint.last_timestamp,
            )
            checkpoint = Checkpoint(
                source=checkpoint.source,
                last_timestamp=checkpoint.last_timestamp,
                messages_total=checkpoint.messages_total,
                session_count=checkpoint.session_count + 1,
            )
        else:
            checkpoint = Checkpoint(source=config.source, session_count=1)

    # Track delivered count at the start so we compute deltas, not
    # cumulative totals (which would double-count across reconnections).
    delivered_baseline = stream.stats.messages_delivered

    while stream.is_running:
        try:
            await connect_fn()
            # Normal exit — save checkpoint and stop.
            delta = stream.stats.messages_delivered - delivered_baseline
            if config.checkpoint_dir and checkpoint:
                checkpoint.last_timestamp = _now_iso()
                checkpoint.messages_total += delta
                save_checkpoint(checkpoint, config.checkpoint_dir)
            break

        except Exception as e:
            attempts += 1
            stream._stats.errors += 1

            if max_retries is not None and attempts > max_retries:
                logger.error(
                    "Max retries (%d) exceeded, stopping", max_retries
                )
                break

            # Only count as a reconnection if we're actually going to retry.
            stream._stats.reconnections += 1

            # Save checkpoint before reconnecting.
            delta = stream.stats.messages_delivered - delivered_baseline
            if config.checkpoint_dir and checkpoint:
                checkpoint.last_timestamp = _now_iso()
                checkpoint.messages_total += delta
                save_checkpoint(checkpoint, config.checkpoint_dir)
            # Reset baseline so next iteration doesn't re-count.
            delivered_baseline = stream.stats.messages_delivered

            # Exponential backoff with jitter.
            jitter = delay * random.uniform(-0.25, 0.25)
            wait = min(delay + jitter, config.max_reconnect_delay_s)
            logger.warning(
                "Connection failed (%s), reconnecting in %.1fs (attempt %d)",
                e, wait, attempts,
            )
            await asyncio.sleep(wait)
            if not stream.is_running:
                break
            delay = min(delay * 2, config.max_reconnect_delay_s)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _message_hash(msg: dict[str, Any]) -> str:
    """Compute a dedup hash for a position message.

    Uses mmsi + timestamp + lat + lon as the dedup key. Two messages
    with the same vessel, time, and location are considered duplicates.
    """
    key = f"{msg.get('mmsi')}:{msg.get('timestamp')}:{msg.get('lat')}:{msg.get('lon')}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]
