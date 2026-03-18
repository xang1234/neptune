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
"""

from __future__ import annotations
