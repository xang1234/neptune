"""Example 5: Streaming pipeline — live AIS data with sinks and promotion.

Shows the full streaming workflow: connect to a live source, land data
durably via sinks, monitor health, and promote to canonical storage.

Requires: pip install neptune-ais[stream]
"""

import asyncio

# from neptune_ais.stream import (
#     NeptuneStream,
#     StreamConfig,
#     BackpressurePolicy,
#     StreamHealth,
# )
# from neptune_ais.sinks import ParquetSink, DuckDBSink, promote_landing

# --- Configure the stream ---
# config = StreamConfig(
#     source="aisstream",
#     api_key="YOUR_AISSTREAM_API_KEY",
#     bbox=(-74.5, 40.0, -73.5, 41.0),  # New York harbor
#     max_queue_size=10_000,
#     backpressure=BackpressurePolicy.BLOCK,
#     lag_threshold_s=30.0,
#     stale_threshold_s=120.0,
# )

# --- Stream to a Parquet sink ---
# async def run_stream():
#     sink = ParquetSink("/tmp/neptune_landing", source="aisstream")
#     async with NeptuneStream(config=config) as stream:
#         # Monitor health during streaming.
#         print(f"Health: {stream.health.value}")
#
#         # Run the sink for 1000 messages.
#         await stream.run_sink(sink, max_messages=1000)
#
#         # Check final health.
#         snap = stream.health_snapshot()
#         print(f"Health: {snap['health']}")
#         print(f"Received: {snap['messages_received']}")
#         print(f"Delivered: {snap['messages_delivered']}")
#         print(f"Dedup rate: {snap['dedup_rate']:.1%}")

# asyncio.run(run_stream())

# --- Promote landed data to canonical storage ---
# results = promote_landing(
#     landing_dir="/tmp/neptune_landing",
#     store_root="/tmp/neptune_demo",
#     source="aisstream",
#     cleanup=True,
# )
# for r in results:
#     print(f"  {r.date}: {r.record_count:,} rows promoted")

# --- Or use DuckDB for direct SQL access ---
# async def run_duckdb_stream():
#     sink = DuckDBSink(db_path="/tmp/ais_live.duckdb", table_name="positions")
#     async with NeptuneStream(config=config) as stream:
#         await stream.run_sink(sink, max_messages=500)
#     # Query immediately after landing:
#     # result = sink.connection.execute(
#     #     "SELECT mmsi, count(*) FROM positions GROUP BY mmsi"
#     # ).fetchall()

print("Example 5: Streaming pipeline.")
print("Uncomment after setting an AISStream API key.")
print()
print("Pipeline: NeptuneStream → ParquetSink → promote_landing → canonical")
print("Health states: HEALTHY → LAGGING → STALE → DISCONNECTED")
print("Backpressure: BLOCK (throttle producer) or DROP_OLDEST (freshness)")
