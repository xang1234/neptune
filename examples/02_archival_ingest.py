"""Example 2: Archival ingest — download, query, and explore AIS data.

Shows the core archival workflow: download data from a source,
query it with Polars or SQL, and use helpers for common operations.

Note: This example requires network access to download from NOAA.
For offline exploration, skip the download step and point to local data.
"""

from neptune_ais.api import Neptune

# --- Download a day of NOAA data ---
# Neptune handles: fetch → normalize → QC → partition → catalog
n = Neptune("2024-06-15", sources=["noaa"], cache_dir="/tmp/neptune_demo")
# Uncomment to actually download (requires network):
# n.download()

# --- Query positions as a Polars LazyFrame ---
# positions = n.positions()
# print(f"Positions: {positions.collect().shape}")

# --- Latest known position per vessel ---
# from neptune_ais.helpers import latest_positions
# latest = latest_positions(positions)
# print(f"Unique vessels: {latest.collect().shape[0]}")

# --- Point-in-time snapshot ---
# from neptune_ais.helpers import snapshot
# snap = snapshot(positions, when="2024-06-15T12:00:00")
# print(f"Vessels at noon: {snap.collect().shape[0]}")

# --- SQL queries via DuckDB ---
# result = n.sql("SELECT mmsi, count(*) as n FROM positions GROUP BY mmsi ORDER BY n DESC LIMIT 10")
# print(result)

# --- Vessel history ---
# from neptune_ais.helpers import vessel_history
# history = vessel_history(367000001, positions=positions)
# # Returns dict[str, LazyFrame] with keys like "positions", "tracks", "events".
# print(f"Positions for vessel: {history['positions'].collect().shape[0]}")

print("Example 2: See comments for archival workflow steps.")
print("Uncomment lines after setting up a cache directory with real data.")
