# Event Heuristics: Assumptions, Confidence Limits, and Non-Goals

This document records the assumptions and limits of Neptune's event
detection heuristics. All detectors live in `neptune_ais/derive/events.py`.

## Port Calls

**What it detects:** A vessel with sustained low-speed positions inside
a port boundary.

**Key assumptions:**
- Port boundaries are bbox-only (no polygon containment unless shapely is installed).
- SOG <= 3.0 knots is "stopped" (configurable via `PortCallConfig.max_speed_knots`).
- Minimum 1-hour duration by default.

**Confidence tiers:**
| Duration | Score |
|----------|-------|
| >= 4 hours | 0.9 |
| >= 2 hours | 0.7 |
| < 2 hours | 0.5 |

**Known limits:**
- Anchorage vs. berth is not distinguished.
- Vessels drifting at low speed near (but outside) a port boundary are missed.
- Boundary quality directly affects detection quality.

## EEZ Crossings

**What it detects:** A vessel whose EEZ region label changes between
consecutive positions.

**Key assumptions:**
- Both the "before" and "after" positions must have a non-null EEZ label.
  Transitions from/to unknown waters (null EEZ) are excluded.
- The crossing location is the midpoint of the two positions.
- Equirectangular distance approximation.

**Confidence tiers:**
| Gap + Distance | Score |
|----------------|-------|
| <= 30 min, <= 20 km | 0.9 |
| <= 2 hours, <= 100 km | 0.7 |
| Otherwise | 0.5 |

**Known limits:**
- Sparse AIS data (large time gaps) will miss crossings entirely.
- The midpoint location may be far from the actual boundary line.
- Antimeridian-crossing boundaries are not supported.

## Encounters

**What it detects:** Two vessels within `max_distance_m` (default 500m)
of each other for a sustained period.

**Key assumptions:**
- Time-bucketed self-join with `time_bucket_s` (default 5 min).
- Only pairs where `mmsi_a < mmsi_b` are kept (no duplicates).
- Equirectangular distance at the mean latitude of the pair.

**Confidence tiers:**
| Duration + Distance | Score |
|---------------------|-------|
| >= 1 hour, <= 200m | 0.9 |
| >= 10 min, <= 500m | 0.7 |
| Otherwise | 0.5 |

**Known limits:**
- Self-join is O(n^2/bucket) — large datasets with many vessels in one
  area will produce many pairs. Use bbox filtering upstream.
- Time-bucket alignment means very brief encounters (< bucket width)
  may be missed.
- `other_mmsi` records the secondary vessel but encounter events are
  only stored once per pair (mmsi < other_mmsi).

## Loitering

**What it detects:** A vessel with sustained low-speed positions
clustered in a small area (not inside a port).

**Key assumptions:**
- SOG <= 2.0 knots (configurable).
- Spatial spread measured as max distance from centroid (not convex hull).
- Default radius threshold: 1000m.

**Confidence tiers:**
| Duration + Radius | Score |
|-------------------|-------|
| >= 2 hours, <= 500m | 0.9 |
| >= 30 min, <= 1000m | 0.7 |
| Otherwise | 0.5 |

**Known limits:**
- Does not distinguish loitering from anchoring (no port-exclusion filter).
- Figure-8 or zigzag patterns may exceed the radius threshold despite
  staying in a small operational area.
- The spatial radius is equirectangular, not geodesic.

## Confidence Model

All confidence scores are in [0.0, 1.0]:

| Band | Range | Meaning |
|------|-------|---------|
| High | [0.7, 1.0] | Strong heuristic support |
| Medium | [0.3, 0.7) | Reasonable but not definitive |
| Low | [0.0, 0.3) | Speculative, sparse evidence |

Use `classify_confidence(score)` or filter with
`events.filter(pl.col("confidence_score") >= CONFIDENCE_HIGH)`.

## Provenance

Every event carries a `record_provenance` token:
```
source:detector_name/version[upstream_datasets]
```

Examples:
- `noaa:port_call_detector/0.1.0[boundaries+positions]`
- `noaa:encounter_detector/0.1.0[positions]`

Parse with `parse_provenance(token)` to get structured components.

## Explicit Non-Goals

The following are **intentionally not implemented** in the current version:

- **Berth-level resolution** — port calls identify the port, not the berth.
- **Geodesic distances** — equirectangular is used throughout. Error is
  < 1% at typical AIS scales (< 100 km) but grows at high latitudes.
- **Probabilistic confidence** — scores are tier-based, not Bayesian.
- **Antimeridian support** — bbox containment uses `west <= lon <= east`,
  which fails for regions straddling the 180th meridian.
- **Streaming detection** — all detectors operate on batch DataFrames.
- **Fishing effort** — listed in the schema vocabulary but no detector exists.
- **Complete maritime ontology** — Neptune detects 4 event families, not all
  possible vessel behaviors.
