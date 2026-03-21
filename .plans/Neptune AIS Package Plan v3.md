# Neptune: Open AIS Data Platform for Python

## What is Neptune

Neptune downloads, normalizes, catalogs, fuses, and analyzes open-source AIS
vessel tracking data. It is to maritime AIS what Herbie is to weather model
data: one interface, many archives, clean output — but with a stronger focus on
reproducible ingestion, scalable local analytics, and higher-level maritime
primitives.

Neptune is not just a file downloader. It is a local maritime data platform
built around four linked canonical datasets:

- `positions` — normalized AIS point observations
- `vessels` — vessel identity and slowly changing reference attributes
- `tracks` — derived trip/trajectory segments
- `events` — inferred or source-native maritime events

```python
from neptune_ais import Neptune

# Open one day from one source
n = Neptune("2024-06-15", sources=["noaa"])
positions = n.positions()            # polars.LazyFrame
positions.collect().shape

# Open a date range across multiple sources with explicit fusion policy
n = Neptune(
    ("2024-01-01", "2024-01-31"),
    sources=["noaa", "dma"],
    merge="best",
    bbox=(8.0, 54.5, 15.5, 58.5),
)

# Query derived datasets
tracks = n.tracks(gap="30m", generalize="1m")
events = n.events(kind="port_call")

# Interactive mapping
m = n.map(layer="positions")
m = n.map(layer="tracks", color_by="mean_speed")

# SQL over the same local catalog
con = n.duckdb()
con.sql("""
    SELECT ship_type, count(*)
    FROM positions
    GROUP BY 1
    ORDER BY 2 DESC
""").df()
```

---

## Product goals

Neptune should be:

- **Simple** enough for a notebook user to get a day of AIS data in one line
- **Robust** enough to support long-lived local caches and repeatable analyses
- **Fast** enough to scan large multi-day or multi-source stores lazily
- **Extensible** enough to add new archival and streaming sources cleanly
- **Useful** enough to answer maritime questions, not just expose raw rows

The package should feel compelling even before a user writes custom group-bys.
That means first-class support for common maritime questions like:

- Where were these vessels during a time window?
- What tracks/trips were taken?
- Which vessels crossed an EEZ boundary?
- Which vessels likely made a port call?
- Where are high-density or loitering areas?
- What source(s) contributed to this result?

---

## Core design decisions

### Polars first, DuckDB first-class

Polars remains the primary dataframe engine because AIS workloads are large,
columnar, and filter-heavy. Neptune returns `polars.LazyFrame` for default
programmatic access and keeps as much work lazy as possible.

DuckDB is also a first-class execution engine rather than just an export
target. Neptune should expose a local DuckDB connection or relation layer so
users can run SQL directly over the same cataloged Parquet datasets.

Why this architecture is better:

- Polars is excellent for lazy scans, expression pipelines, and Python-native workflows
- DuckDB is excellent for ad hoc SQL, joins, aggregations, and multi-file analytics
- Supporting both prevents users from exporting out of Neptune just to do normal work
- The package can choose the best engine internally for different workloads over time

### Canonical dataset family, not one giant canonical table

A single canonical row model is too restrictive once Neptune includes archive,
streaming, identity, derived tracks, and event-style sources. Neptune should
standardize a small family of linked datasets instead of forcing every source
into one sparse positions table.

Canonical datasets:

1. **Positions** — timestamped vessel observations
2. **Vessels** — vessel identity and dimensions
3. **Tracks** — trip/trajectory segments derived from positions
4. **Events** — inferred or source-native events such as port calls,
   EEZ crossings, loitering, encounters, or fishing effort markers

This makes the package more expressive, reduces schema bloat, and creates a
clean foundation for future features.

### Cataloged local lakehouse instead of anonymous cached Parquet files

Neptune should still use Parquet as the primary local storage format, but not
as a loose collection of files with only a sidecar metadata JSON. The local
store should behave like a small cataloged lakehouse with:

- deterministic partitioning
- manifest files
- schema versioning
- transform/adapter versioning
- provenance and checksum tracking
- atomic writes and commit markers
- quality-control summaries

This is what makes a local cache reliable across months of use instead of just
convenient during a single notebook session.

### Native track derivation pipeline; GeoPandas only at optional boundaries

Geometry-heavy libraries are still useful, but GeoDataFrame should not be the
default execution engine for core track building. Neptune should derive tracks
natively using Polars and/or DuckDB, then expose optional bridges to
MovingPandas/GeoPandas where they add value.

This change matters because the old “collect everything, convert to pandas,
then build trajectories” path becomes the main bottleneck at scale.

### Explicit multi-source fusion

If a user requests multiple sources, Neptune should not silently “try a
priority order.” It should expose explicit fusion behavior:

- `merge="best"` — choose best record per entity/time from configured precedence
- `merge="union"` — union all matched records with provenance retained
- `merge="prefer:noaa"` — deterministic source preference order
- configurable dedup keys and field precedence rules

This is what turns Neptune into a unified maritime data product rather than a
bag of independent adapters.

---

## Local storage architecture

### Storage layers

Neptune should maintain three logical storage layers:

1. **Raw store** — optional retained source payloads or source-native extracts
2. **Canonical store** — normalized `positions`, `vessels`, `tracks`, `events`
3. **Derived store** — cached map-ready, event-ready, or aggregate products

### Raw retention policy

Raw data should be configurable instead of always discarded after
normalization:

- `raw_policy="none"` — do not retain raw artifacts
- `raw_policy="metadata"` — retain only references, checksums, and key headers
- `raw_policy="full"` — retain original source files for replay and reprocessing

This makes Neptune much more reproducible. If a normalizer changes, users can
rebuild canonical outputs without depending on the upstream provider still
serving the exact same payload.

### Partitioning and sort order

Date-only partitioning is not enough for AIS. Neptune should store canonical
Parquet with partitioning and layout optimized for the actual query surface:

- partition by dataset / source / date
- optionally shard within a day
- maintain sort order by `mmsi, timestamp`
- maintain coarse spatial partition or spatial key such as H3/geohash/quadbin
- write row-group statistics that help bounding-box pruning

Example layout:

```text
~/.neptune/
  catalog/
    catalog.json
    versions.json
  raw/
    noaa/
      2024/06/15/source-0000.parquet
    dma/
      2024/03/18/source-0000.csv.gz
  canonical/
    positions/
      source=noaa/date=2024-06-15/part-0000.parquet
      source=noaa/date=2024-06-15/part-0001.parquet
      source=dma/date=2024-03-18/part-0000.parquet
    vessels/
      source=noaa/date=2024-06-15/part-0000.parquet
    tracks/
      source=noaa/date=2024-06-15/part-0000.parquet
    events/
      kind=port_call/date=2024-06-15/part-0000.parquet
  manifests/
    positions/
      source=noaa/date=2024-06-15.json
    tracks/
      source=noaa/date=2024-06-15.json
```

### Manifest schema

Every write should create a manifest with fields such as:

- dataset name
- source id
- schema version
- transform version
- adapter version
- file list
- raw checksum(s)
- record count
- distinct MMSI count
- min/max timestamp
- bbox
- QC counters
- write timestamp
- write status / commit marker

This lets Neptune detect stale partitions, mixed schema versions, and partial
failures cleanly.

---

## Data model

## Canonical dataset: `positions`

This is the most frequently queried dataset and contains normalized AIS point
observations.

| Field | Polars type | Description | Required |
|-------|-------------|-------------|----------|
| `mmsi` | Int64 | Maritime Mobile Service Identity | yes |
| `timestamp` | Datetime(μs, UTC) | Observation time | yes |
| `lat` | Float64 | Latitude WGS-84 | yes |
| `lon` | Float64 | Longitude WGS-84 | yes |
| `sog` | Float64 | Speed over ground (knots) | no |
| `cog` | Float64 | Course over ground (degrees) | no |
| `heading` | Float64 | True heading (degrees; null if unavailable) | no |
| `nav_status` | Utf8 | Navigational status text | no |
| `imo` | Utf8 | IMO number when available | no |
| `callsign` | Utf8 | Radio call sign | no |
| `ship_type` | Utf8 | Standardized vessel type | no |
| `vessel_name` | Utf8 | Vessel name reported at observation time | no |
| `length` | Float64 | Length overall (meters) | no |
| `beam` | Float64 | Beam (meters) | no |
| `draught` | Float64 | Current draught (meters) | no |
| `destination` | Utf8 | Reported destination | no |
| `eta` | Datetime(μs, UTC) | Reported ETA | no |
| `flag` | Utf8 | ISO-3 or standardized flag code | no |
| `source` | Utf8 | Source identifier | yes |
| `source_record_id` | Utf8 | Stable id for dedup/provenance if available | no |
| `source_file` | Utf8 | Original filename/URL/batch id | yes |
| `ingest_id` | Utf8 | Neptune ingest batch id | yes |
| `spatial_key` | Utf8 | Optional coarse spatial key | no |
| `qc_flags` | List[Utf8] | QC warnings attached to the row | no |
| `qc_severity` | Utf8 | `ok`, `warning`, `error` | yes |
| `confidence_score` | Float64 | Row-level trust score [0,1] | no |
| `record_provenance` | Utf8 | Merge/fusion provenance token | yes |

## Canonical dataset: `vessels`

This dataset stores identity and slowly changing vessel attributes.

| Field | Polars type | Description |
|-------|-------------|-------------|
| `mmsi` | Int64 | Vessel MMSI |
| `imo` | Utf8 | IMO number |
| `callsign` | Utf8 | Callsign |
| `vessel_name` | Utf8 | Standardized preferred vessel name |
| `ship_type` | Utf8 | Standardized vessel type |
| `length` | Float64 | Vessel length |
| `beam` | Float64 | Vessel beam |
| `flag` | Utf8 | Flag state |
| `first_seen` | Datetime(μs, UTC) | First seen in Neptune |
| `last_seen` | Datetime(μs, UTC) | Last seen in Neptune |
| `source` | Utf8 | Contributing source |
| `record_provenance` | Utf8 | Source/fusion lineage |

## Canonical dataset: `tracks`

Tracks are derived trip segments grouped by vessel and split by time gaps and
other heuristics.

| Field | Polars type | Description |
|-------|-------------|-------------|
| `track_id` | Utf8 | Stable trajectory/trip identifier |
| `mmsi` | Int64 | Vessel MMSI |
| `start_time` | Datetime(μs, UTC) | Start of segment |
| `end_time` | Datetime(μs, UTC) | End of segment |
| `point_count` | Int64 | Number of contributing points |
| `distance_m` | Float64 | Estimated track distance |
| `duration_s` | Float64 | Segment duration |
| `mean_speed` | Float64 | Average speed |
| `max_speed` | Float64 | Maximum speed |
| `bbox_west` | Float64 | Segment bbox west |
| `bbox_south` | Float64 | Segment bbox south |
| `bbox_east` | Float64 | Segment bbox east |
| `bbox_north` | Float64 | Segment bbox north |
| `geometry_wkb` | Binary | Optional serialized LineString |
| `timestamp_offsets_ms` | List[Int64] | Per-vertex relative timestamps for animation |
| `source` | Utf8 | Source or fusion source |
| `record_provenance` | Utf8 | Source/fusion lineage |

## Canonical dataset: `events`

Events are either inferred by Neptune or ingested from source-native event
feeds.

| Field | Polars type | Description |
|-------|-------------|-------------|
| `event_id` | Utf8 | Stable event id |
| `event_type` | Utf8 | `port_call`, `eez_crossing`, `encounter`, `loitering`, etc. |
| `mmsi` | Int64 | Primary vessel MMSI |
| `other_mmsi` | Int64 | Secondary vessel MMSI when relevant |
| `start_time` | Datetime(μs, UTC) | Event start |
| `end_time` | Datetime(μs, UTC) | Event end |
| `lat` | Float64 | Representative latitude |
| `lon` | Float64 | Representative longitude |
| `geometry_wkb` | Binary | Optional event geometry |
| `confidence_score` | Float64 | Event confidence |
| `source` | Utf8 | Source or inferred marker |
| `record_provenance` | Utf8 | Lineage |

---

## Multi-source fusion model

Neptune should let the user specify one or more sources explicitly.

```python
n = Neptune(
    ("2024-06-01", "2024-06-07"),
    sources=["noaa", "dma", "aishub"],
    merge="best",
)
```

Fusion policy should include:

- source precedence order
- dedup key selection
- timestamp tolerance for near-duplicate points
- field-level precedence for conflicts
- provenance tagging
- optional per-source confidence weights

Example dedup key for positions:

- `mmsi`
- rounded/normalized timestamp window
- approximate coordinate match or source record id

Fusion outputs should preserve enough provenance that a user can always ask:

- which source won?
- which sources contributed?
- was this deduplicated or unioned?

---

## Query engine architecture

### Unified dataset object

Neptune should replace the split between `Neptune` and `FastNeptune` with one
core dataset object that accepts:

- a single date
- a date range tuple
- a list of dates

This simplifies the API and the implementation while keeping convenience.

```python
class Neptune:
    """Open Neptune datasets over one or more dates and sources."""

    def __init__(
        self,
        dates,
        *,
        sources: list[str] | None = None,
        merge: str = "best",
        bbox: tuple[float, float, float, float] | None = None,
        mmsi: list[int] | None = None,
        cache_dir: str | None = None,
        raw_policy: str = "metadata",
        overwrite: bool = False,
    ) -> None: ...
```

### Core accessors

```python
n.positions()         # polars.LazyFrame
n.vessels()           # polars.LazyFrame
n.tracks()            # polars.LazyFrame by default
n.events()            # polars.LazyFrame
n.inventory()         # summary dataframe/dict
n.provenance()        # manifest-level provenance
n.quality_report()    # QC summary
n.duckdb()            # duckdb connection bound to catalog
n.sql("SELECT ...")   # convenience SQL method
```

### Higher-level maritime helpers

These should be first-class because they make the package valuable beyond raw
access.

```python
n.latest_positions()
n.port_calls()
n.encounters(max_distance_m=500)
n.loitering(min_duration="30m")
n.eez_crossings(boundary="idn_eez")
n.density(resolution=6)
n.vessel_history(mmsi=123456789)
n.snapshot("2024-06-15T12:00:00Z")
```

### Why this API is better

- One object is easier to learn and document than separate single-date and range classes
- `positions / vessels / tracks / events` mirrors the canonical model directly
- Users get high-level maritime questions answered without immediately dropping to custom code
- SQL becomes a normal part of the workflow instead of an export escape hatch

---

## Track derivation architecture

### Tracks are a first-class derived dataset

Raw AIS is point data. Many real analyses want trip segments. Neptune should
therefore build tracks as a native derived dataset, cache them, and let users
query them directly.

### Native segmentation pipeline

Default track derivation should be implemented in Polars and/or DuckDB rather
than by materializing full GeoDataFrames.

Pipeline:

1. Read `positions` lazily
2. Sort by `mmsi, timestamp`
3. Detect segment boundaries using:
   - observation gaps
   - non-monotonic timestamp resets
   - implausible jumps
   - optional status changes
4. Assign `track_id`
5. Aggregate per-track statistics
6. Optionally emit WKB geometry and per-vertex timestamp offsets
7. Cache results to the `tracks` dataset

Config:

```python
tracks = n.tracks(
    gap="30m",
    min_points=3,
    min_duration="5m",
    min_distance_m=100,
    generalize="1m",
    include_geometry=True,
    refresh=False,
)
```

### Optional geometry bridges

Neptune can still offer optional conversion helpers for downstream geometry
workflows:

```python
n.tracks(as_geo=True)                 # GeoDataFrame
n.tracks(as_movingpandas=True)        # TrajectoryCollection
```

That keeps interoperability without making pandas/GeoPandas the scaling limit.

---

## Visualization architecture

lonboard remains a strong fit for rendering large local datasets in notebooks
or exported HTML, but the visualization layer should be fed by map-ready
products rather than arbitrary eager GeoDataFrame conversions.

### Design principles

- Prefer Arrow/GeoArrow-friendly representations when possible
- Clip to viewport before building heavy layers
- Support sampling and aggregation for dense point clouds
- Treat animated trip playback as an advanced mode, not the only trajectory story
- Cache map-ready derived layers for repeated views

### Supported layers

| Layer | Use case | Backing dataset |
|-------|----------|-----------------|
| `positions` | Snapshot positions | `positions` |
| `tracks` | Static vessel tracks | `tracks` |
| `trips` | Animated track replay | `tracks` + per-vertex timestamps |
| `density` | Heat/density map | derived aggregates |
| `events` | Port calls, crossings, encounters | `events` |

### Example

```python
m = n.map(layer="positions", color_by="ship_type")
m = n.map(layer="tracks", color_by="mean_speed", opacity=0.4)
m = n.map(layer="density", resolution=6)
m = n.map(layer="events", event_type="port_call")
m = n.map(layer="trips", animate=True, trail_length=120)
```

### Key improvement

For animated trips, Neptune should explicitly store or derive per-coordinate
timestamp offsets needed by the visualization layer. This avoids the previous
gap where animated tracks were conceptually supported but not fully modeled in
the underlying track output.

---

## Streaming architecture

`NeptuneStream` should remain a separate async interface, but it needs real
operational semantics rather than only “append to Parquet forever.”

### Streaming design

```python
class NeptuneStream:
    async def __aiter__(self) -> AsyncIterator[dict]: ...
    async def sink(self, sink, *, checkpoint_dir: str | None = None): ...
    async def to_parquet(self, output_dir: str, *, flush_seconds: int = 60): ...
    async def to_duckdb(self, db_path: str, *, table: str = "positions"): ...
    async def tracks(self, *, window: str = "1h") -> AsyncIterator[pl.DataFrame]: ...
```

### Streaming requirements

- checkpointing
- reconnect and retry policy
- bounded buffers / backpressure
- dedup within a rolling window
- rolling file compaction
- heartbeat and source lag monitoring
- eventual promotion from stream landing zone into canonical partitions

### Pluggable sinks

- `ParquetSink`
- `DuckDBSink`
- future `KafkaSink` or `ArrowFlightSink`

This makes the streaming path durable and production-like instead of just a
long-running demo.

---

## Data quality model

Neptune should adopt a layered quality model instead of only dropping bad rows
at ingest.

### QC classes

1. **Hard invalid** — structurally broken or impossible; drop
2. **Suspicious** — keep, but flag and lower confidence
3. **Source-specific quirk** — normalize and annotate provenance/QC note

### Example checks

- latitude outside [-90, 90]
- longitude outside [-180, 180]
- MMSI malformed
- duplicate raw messages
- non-monotonic timestamps within vessel stream
- impossible implied speed between points
- repeated stale positions over long windows
- heading sentinels / unavailable values
- source-specific field mapping anomalies

### Why this is better

Many analysts want clean defaults, but they also want the ability to inspect
anomalies. Keeping suspicious rows with flags is far more useful than silently
throwing away everything that looks odd.

### QC outputs

- row-level `qc_flags`
- row-level `qc_severity`
- row-level `confidence_score`
- dataset-level `quality_report()`
- manifest-level QC counters and error summaries

---

## Source adapters

Neptune should keep the adapter model, but make the contract richer and more
self-describing.

### Adapter interface

```python
class SourceAdapter(Protocol):
    source_id: str
    supports_backfill: bool
    supports_streaming: bool
    supports_server_side_bbox: bool
    supports_incremental: bool
    auth_scheme: str | None
    rate_limit: str | None
    expected_latency: str | None
    license_requirements: str | None

    def available_dates(self) -> list[date] | tuple[date, date] | None: ...
    def fetch_raw(self, spec: "FetchSpec") -> list["RawArtifact"]: ...
    def normalize_positions(self, raw: list["RawArtifact"]) -> pl.DataFrame: ...
    def normalize_vessels(self, raw: list["RawArtifact"]) -> pl.DataFrame | None: ...
    def normalize_events(self, raw: list["RawArtifact"]) -> pl.DataFrame | None: ...
    def qc_rules(self) -> list["QCRule"]: ...
```

### Extensibility

Adapters should be discoverable through plugin registration so external teams
can add sources without modifying Neptune core.

### Testing

Every adapter should ship with:

- golden raw fixtures
- normalization tests
- schema conformance tests
- QC tests
- manifest/provenance tests

That is what makes the adapter layer sustainable as the source list grows.

---

## Data sources

Initial source catalog remains valuable, but Neptune should document each
source in terms of both coverage and capabilities.

| Source ID | Provider | Delivery | Coverage | History | Backfill | Streaming | Notes |
|-----------|----------|----------|----------|---------|----------|-----------|-------|
| `noaa` | NOAA Marine Cadastre | Daily GeoParquet / CSV+ZIP | U.S. coastal | 2009– | Yes | No | Strong archival source |
| `aisstream` | aisstream.io | WebSocket JSON | Global | — | No | Yes | Real-time only |
| `dma` | Danish Maritime Authority | Daily CSV | Danish waters | 2006– | Yes | No | Regional archive |
| `norway` | Norwegian Coastal Admin | TCP NMEA + bulk CSV | Norwegian EEZ | Archive | Yes | Yes/partial | Mixed delivery |
| `finland` | Fintraffic Digitraffic | REST JSON + MQTT | Baltic Sea | limited | Partial | Yes | Regional live source |
| `gfw` | Global Fishing Watch | REST API JSON | Global by region | 2012– | Yes | No | Identity/events enrichments |
| `aishub` | AISHub community | REST CSV/JSON | Global terrestrial | limited | No | Partial | Variable quality |

Neptune should also expose source capabilities programmatically:

```python
from neptune_ais import sources

sources.catalog()
sources.available("2024-06-15")
sources.info("dma")
sources.capabilities("noaa")
```

---

## Python API summary

```python
class Neptune:
    def download(self) -> list[str]: ...
    def refresh(self) -> list[str]: ...

    def positions(self, *, raw: bool = False) -> pl.LazyFrame: ...
    def vessels(self) -> pl.LazyFrame: ...
    def tracks(self, **kwargs) -> pl.LazyFrame | "gpd.GeoDataFrame": ...
    def events(self, *, kind: str | None = None) -> pl.LazyFrame: ...

    def latest_positions(self) -> pl.LazyFrame: ...
    def port_calls(self, **kwargs) -> pl.LazyFrame: ...
    def encounters(self, **kwargs) -> pl.LazyFrame: ...
    def loitering(self, **kwargs) -> pl.LazyFrame: ...
    def eez_crossings(self, boundary: str, **kwargs) -> pl.LazyFrame: ...
    def density(self, resolution: int = 6, **kwargs) -> pl.LazyFrame: ...
    def vessel_history(self, mmsi: int) -> pl.LazyFrame: ...
    def snapshot(self, when) -> pl.LazyFrame: ...

    def map(self, *, layer: str = "positions", **kwargs): ...
    def inventory(self): ...
    def provenance(self): ...
    def quality_report(self): ...

    def duckdb(self): ...
    def sql(self, query: str): ...

    def to_parquet(self, path: str, *, dataset: str = "positions") -> list[str]: ...
    def to_duckdb(self, db_path: str, *, dataset: str = "positions") -> int: ...
```

---

## CLI

```bash
# Download / refresh
neptune download --source noaa --date 2024-06-15
neptune download --sources noaa,dma --start 2024-01-01 --end 2024-01-31 --merge best
neptune refresh --dataset positions --source noaa --date 2024-06-15

# Inspect catalog
neptune inventory --dataset positions --source noaa --date 2024-06-15
neptune manifests --dataset positions --source noaa --date 2024-06-15
neptune qc --dataset positions --source noaa --date 2024-06-15

# Query / export
neptune sql "SELECT ship_type, count(*) FROM positions GROUP BY 1"
neptune export parquet --dataset tracks --start 2024-06-01 --end 2024-06-07 --out ./tracks/
neptune export duckdb --dataset positions --source noaa --date 2024-06-15 --db ais.duckdb

# Derived analytics
neptune tracks --source noaa --date 2024-06-15 --gap 30m
neptune events port-calls --source noaa --start 2024-06-01 --end 2024-06-07
neptune density --source noaa --date 2024-06-15 --resolution 6

# Visualization
neptune map positions --source noaa --date 2024-06-15 --html map.html
neptune map tracks --source noaa --date 2024-06-15 --color-by mean_speed --html tracks.html
```

---

## Package structure

```text
neptune_ais/
  __init__.py
  api.py                 # Neptune and user-facing API
  stream.py              # NeptuneStream
  catalog.py             # manifests, versions, dataset registry
  storage.py             # partition layout, atomic writes, retention
  fusion.py              # multi-source merge and dedup rules
  qc.py                  # quality checks and confidence scoring
  sql.py                 # DuckDB integration
  viz.py                 # map layer helpers
  helpers.py             # high-level maritime primitives

  datasets/
    positions.py
    vessels.py
    tracks.py
    events.py

  adapters/
    base.py
    registry.py
    noaa.py
    aisstream.py
    dma.py
    norway.py
    finland.py
    gfw.py
    aishub.py

  derive/
    tracks.py            # native track segmentation
    events.py            # port calls, crossings, encounters, loitering
    density.py           # spatial density products

  geometry/
    boundaries.py        # EEZ/port/region support
    bridges.py           # optional GeoPandas/MovingPandas conversions

  cli/
    main.py
    commands_download.py
    commands_inventory.py
    commands_qc.py
    commands_sql.py
    commands_export.py
    commands_map.py
```

---

## Dependencies

### Core

- `polars`
- `pyarrow`
- `duckdb`
- `fsspec`
- `orjson`
- `pydantic` or `msgspec`
- `httpx`
- `tenacity`

### Optional analytics / geometry

- `shapely`
- `geopandas`
- `movingpandas`
- `lonboard`
- `h3` or alternative spatial indexing library

### Optional streaming

- `websockets`
- `aiokafka` (future)

### Why this dependency shape is better

The core remains performant and mostly columnar. Heavier geospatial packages
stay optional so Neptune remains usable in lean environments while still
supporting advanced geometry workflows when installed.

---

## Key implementation notes

### Atomic writes and commit protocol

Every dataset write should be staged to a temporary location and committed only
after validation succeeds. This prevents half-written partitions from being
mistaken for valid cache entries.

### Schema evolution

Neptune should maintain explicit schema versions per dataset and migration rules
for manifest compatibility. Mixed-version partitions should be detectable.

### Derived dataset caching

Tracks, events, and map-ready aggregates should be cached with their own
manifests. Derived outputs should be invalidated if the upstream schema,
upstream source partition, or derivation config changes.

### Provenance model

Provenance should exist at both the manifest and row level. A user should be
able to trace a result back to:

- source adapter
- raw artifact or source batch
- transform version
- fusion policy used
- QC state

---

## Implementation phases

### Phase 1: Core storage + catalog + NOAA

Deliver:

- unified `Neptune` dataset object
- canonical `positions` and `vessels`
- manifest/catalog layer
- atomic writes and schema versioning
- NOAA adapter
- Polars accessors
- first-class DuckDB integration
- base QC framework

### Phase 2: Second archival source + fusion

Deliver:

- DMA or Norway adapter
- explicit multi-source fusion policies
- dedup and precedence rules
- manifest provenance for fused outputs
- inventory, provenance, and QC reports

### Phase 3: Native tracks + map foundations

Deliver:

- native track segmentation pipeline
- cached `tracks` dataset
- optional GeoPandas/MovingPandas bridges
- viewport-aware mapping
- static track and position layers

### Phase 4: Events and higher-level helpers

Deliver:

- `port_calls()`
- `eez_crossings()`
- `encounters()`
- `loitering()`
- `density()`
- cached `events` dataset

### Phase 5: Streaming hardening

Deliver:

- `NeptuneStream`
- checkpointed streaming sinks
- rolling dedup/compaction
- stream-to-canonical landing pipeline

### Phase 6: Additional sources + ecosystem polish

Deliver:

- GFW, Finland, AISHub adapters
- richer source capability metadata
- CLI expansion
- documentation and examples
- plugin registration for external adapters

### Phase 7: Release readiness

Deliver:

- performance benchmarks
- adapter test harness
- schema migration tests
- end-to-end reproducibility tests
- packaging, docs, and examples

---

## Why this revised plan is better

This revision improves Neptune in the ways that matter most for a serious data
package:

- **More robust** because storage is cataloged, versioned, and atomically written
- **More performant** because query layout and track derivation are aligned with AIS workloads
- **More scalable** because pandas/GeoDataFrame are no longer the core trajectory bottleneck
- **More reliable** because raw retention, manifests, QC layers, and provenance are explicit
- **More useful** because Neptune exposes maritime primitives, not just rows and plots
- **More compelling** because multi-source fusion is a real feature, not just adapter selection
- **More extensible** because the adapter contract and plugin model are stronger

The result is a package that still feels simple for a notebook user but has a
much stronger architecture for long-lived real-world use.
