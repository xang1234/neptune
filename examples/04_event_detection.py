"""Example 4: Event detection — derive maritime events from position data.

Neptune detects four types of maritime events from raw AIS positions:
port calls, EEZ crossings, vessel encounters, and loitering.

Each event includes confidence scores, timestamps, and provenance
tracing back to the source data.
"""

# from neptune_ais.api import Neptune
# from neptune_ais.derive.events import (
#     detect_port_calls,
#     detect_eez_crossings,
#     detect_encounters,
#     detect_loitering,
#     PortCallConfig,
# )

# --- Load positions ---
# n = Neptune("2024-06-15", sources=["noaa"], cache_dir="/tmp/neptune_demo")
# positions = n.positions().collect()

# --- Detect port calls ---
# Requires positions DataFrame and a port_regions Series (from BoundaryRegistry).
# port_calls = detect_port_calls(positions, port_regions)
# print(f"Port calls detected: {len(port_calls)}")

# --- Query events via the API ---
# events_lf = n.events(kind="port_call", min_confidence=0.7)
# events = events_lf.collect()
# print(f"High-confidence port calls: {len(events)}")

# --- Event schema ---
# Each event has:
#   event_id     — deterministic SHA-1 hash for deduplication
#   event_type   — port_call, eez_crossing, encounter, loitering
#   mmsi         — vessel identifier
#   start_time   — event start time
#   end_time     — event end time (if applicable)
#   lat, lon     — event location
#   confidence_score — 0.0-1.0 score
#   provenance   — source:detector/version[upstream] token

print("Example 4: Event detection.")
print("Uncomment after downloading position data.")
print()
print("Event types: port_call, eez_crossing, encounter, loitering")
print("Confidence bands: HIGH (>=0.7), MEDIUM (0.3-0.7), LOW (<0.3)")
