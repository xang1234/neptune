<p align="center">
  <b>🔱 Neptune AIS</b>
</p>

<h3 align="center">Open AIS data platform for Python</h3>

<p align="center">
  Download, normalize, fuse, and analyze vessel tracking data from multiple open-source AIS archives.<br>
  One interface. Many sources. Clean output.
</p>

<p align="center">
  <a href="#installation">Installation</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#data-sources">Data Sources</a> &bull;
  <a href="#features">Features</a> &bull;
  <a href="#cli">CLI</a> &bull;
  <a href="#documentation">Docs</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/neptune-ais/"><img src="https://img.shields.io/pypi/v/neptune-ais?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/neptune-ais/"><img src="https://img.shields.io/pypi/pyversions/neptune-ais" alt="Python"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://github.com/yourorg/neptune-ais/actions"><img src="https://img.shields.io/badge/tests-768%20passing-brightgreen" alt="Tests"></a>
</p>

---

## What is Neptune?

**Neptune** is a Python library that gives you a single, unified interface to download, normalize, and analyze [AIS](https://en.wikipedia.org/wiki/Automatic_identification_system) (Automatic Identification System) vessel tracking data from multiple open-source archives.

AIS data powers maritime domain awareness — vessel tracking, trade analytics, environmental monitoring, fishing surveillance, and port operations. But working with it is painful: every provider uses a different format, schema, delivery mechanism, and quality profile. Neptune handles all of that so you can focus on analysis.

```python
from neptune_ais import Neptune

n = Neptune("2024-06-15", sources=["noaa"])
n.download()

positions = n.positions()          # Polars LazyFrame — normalized, QC'd
result = n.sql("SELECT mmsi, count(*) as n FROM positions GROUP BY mmsi ORDER BY n DESC LIMIT 5")
```

**Think of Neptune as [Herbie](https://github.com/blaylockbk/Herbie) for maritime data** — a clean data-access layer that handles the messy plumbing of fetching, normalizing, and cataloging data from heterogeneous archives, so you get reproducible, analysis-ready output every time.

## Key Features

- **Multi-source ingestion** — Download from NOAA, DMA, AISHub through one API, ingest GFW fishing events and effort grids, and stream from AISStream and Finland (Digitraffic)
- **Automatic normalization** — Every source is normalized to a canonical schema with QC scoring and provenance tracking
- **Multi-source fusion** — Merge overlapping sources with configurable dedup strategies (`best`, `union`, `prefer:<source>`)
- **Polars-native** — Query positions, vessels, tracks, and events as lazy DataFrames with full predicate pushdown
- **SQL via DuckDB** — Run SQL queries directly over your cataloged data
- **Event detection** — Derive port calls, EEZ crossings, vessel encounters, and loitering from raw positions
- **Real-time streaming** — Connect to live AIS feeds with backpressure, checkpointing, and durable sinks
- **Interactive maps** — Visualize positions, tracks, and events with lonboard
- **Plugin system** — Add custom source adapters via Python entry points
- **CLI included** — `neptune download`, `neptune inventory`, `neptune sql`, and more

## Installation

Neptune's core is lightweight — only [Polars](https://pola.rs/), [Pydantic](https://docs.pydantic.dev/), and [httpx](https://www.python-httpx.org/) are required. Everything else is opt-in.

```bash
# Core (Polars + Pydantic + httpx)
pip install neptune-ais

# With SQL support (DuckDB)
pip install neptune-ais[sql]

# With spatial & visualization (GeoDataFrames, lonboard, H3)
pip install neptune-ais[geo]

# With real-time streaming (WebSocket feeds)
pip install neptune-ais[stream]

# With the CLI (Click + Rich)
pip install neptune-ais[cli]

# Everything
pip install neptune-ais[all]
```

**Requirements:** Python 3.10+

<details>
<summary><b>Optional dependency groups explained</b></summary>

| Extra | Adds | Used by |
|---|---|---|
| `sql` | duckdb | `Neptune.sql()`, `Neptune.duckdb()`, `DuckDBSink` |
| `parquet` | pyarrow | Full Parquet write options (compression, statistics) |
| `geo` | shapely, geopandas, movingpandas, lonboard, h3 | Boundary lookups, GeoDataFrame bridges, maps |
| `stream` | websockets, aiomqtt | `NeptuneStream`, live AIS feeds |
| `cli` | click, rich | `neptune` console commands |
| `notebooks` | jupyter, ipykernel | Interactive notebook examples |
| `dev` | pytest, mypy, ruff, coverage, nbstripout | Development and testing |
| `all` | All of the above (except dev) | Full-featured install |

</details>

## Quick Start

### Download and query AIS data

```python
from neptune_ais import Neptune

# Download a day of NOAA AIS data
# Neptune handles: fetch → normalize → QC → partition → catalog
n = Neptune("2024-06-15", sources=["noaa"])
n.download()

# Query as a Polars LazyFrame
positions = n.positions()
df = positions.collect()
print(f"{len(df):,} position reports from {df['mmsi'].n_unique():,} vessels")

# SQL queries via DuckDB
top_vessels = n.sql("""
    SELECT mmsi, count(*) as n
    FROM positions
    GROUP BY mmsi
    ORDER BY n DESC
    LIMIT 10
""")
```

### Common operations with helpers

```python
from neptune_ais.helpers import latest_positions, snapshot, vessel_history

# Most recent position per vessel
latest = latest_positions(positions)

# Point-in-time snapshot — where was every vessel at noon?
noon = snapshot(positions, when="2024-06-15T12:00:00")

# Full history for a single vessel
history = vessel_history(367000001, positions=positions)
```

### Multi-source fusion

```python
# Combine NOAA and DMA with automatic deduplication
n = Neptune(
    ("2024-06-15", "2024-06-16"),
    sources=["noaa", "dma"],
    merge="best",       # "best" | "union" | "prefer:noaa"
)
n.download()
fused = n.positions()   # Deduplicated across sources
```

### Event detection

```python
# Derive maritime events from position data
events = n.events(kind="port_call", min_confidence=0.7)

# Event types: port_call, eez_crossing, encounter, loitering
# Each event includes confidence scores and full provenance
```

### Real-time streaming

```python
import asyncio
from neptune_ais.stream import NeptuneStream, StreamConfig
from neptune_ais.sinks import ParquetSink, promote_landing

config = StreamConfig(
    source="aisstream",
    api_key="YOUR_KEY",
    bbox=(-74.5, 40.0, -73.5, 41.0),  # New York harbor
)

async def ingest():
    sink = ParquetSink("/tmp/neptune_landing", source="aisstream")
    async with NeptuneStream(config=config) as stream:
        await stream.run(sink, max_messages=10_000)
    # Promote to canonical storage
    promote_landing("/tmp/neptune_landing", store_root="~/.neptune", source="aisstream")

asyncio.run(ingest())
```

## Data Sources

Neptune includes adapters for six open AIS data providers, with a plugin system for adding more.

| Source | Provider | Coverage | Delivery | Auth | Backfill |
|---|---|---|---|---|---|
| `noaa` | NOAA AIS Archive | US waters, global ATON | Daily files | None | Yes |
| `dma` | Danish Maritime Authority | European waters | Daily files | None | Yes |
| `gfw` | Global Fishing Watch | Global | Events API + 4Wings | API token | Yes (2020+) |
| `finland` | Digitraffic Finland | Finnish waters | MQTT streaming | None | No (live only) |
| `aishub` | AISHub | Global (variable quality) | Multiple feeds | API key | Yes |
| `aisstream` | AISStream | Global (real-time) | WebSocket | API key | No (live only) |

### Discover sources programmatically

```python
from neptune_ais import sources

sources.load_all_adapters()

# List all sources
for s in sources.catalog():
    print(f"{s.source_id:<12} {s.provider:<30} auth={s.auth_scheme or 'none'}")

# Find open-data sources with backfill
for s in sources.find_sources(backfill=True, auth=False):
    print(s.source_id)
```

### Add a custom source via plugin

External packages register adapters through Python entry points:

```toml
# In your plugin's pyproject.toml
[project.entry-points."neptune_ais.adapters"]
my_source = "my_package.adapter:MyAdapter"
```

## Features

### Architecture

Neptune is organized around a **canonical dataset family** and a **three-layer local store**:

```
                                ┌────────────────┐
                                │   Your Code    │
                                │  Polars / SQL  │
                                └───────┬────────┘
                                        │
                              ┌─────────▼─────────┐
                              │     Neptune API    │
                              │ .positions()       │
                              │ .tracks()          │
                              │ .events()          │
                              │ .sql()             │
                              └──┬──────────────┬──┘
                   ┌─────────────▼──┐     ┌─────▼──────────────┐
                   │  Archival Path │     │  Streaming Path    │
                   │  fetch → norm  │     │  NeptuneStream     │
                   │  → QC → store  │     │  → sink → promote  │
                   └──┬─────────────┘     └─────┬──────────────┘
       ┌──────────────▼─────────────────────────▼────────────────┐
       │                  Three-Layer Store                       │
       │  raw/ (source payloads)  →  canonical/ (normalized)     │
       │                          →  derived/ (cached products)  │
       └─────────────────────────────────────────────────────────┘
       ┌─────────────────────────────────────────────────────────┐
       │              Catalog & Manifests                         │
       │  partition tracking · schema versions · QC summaries    │
       │  staleness detection · atomic writes (stage → commit)   │
       └─────────────────────────────────────────────────────────┘
```

### Canonical Datasets

| Dataset | Description | Schema |
|---|---|---|
| **positions** | Timestamped AIS point observations (mmsi, lat, lon, sog, cog, ...) | `positions/v1` |
| **vessels** | Vessel identity and reference data (slowly changing dimensions) | `vessels/v1` |
| **tracks** | Derived trip/trajectory segments | `tracks/v1` |
| **events** | Maritime events (port calls, EEZ crossings, encounters, loitering) | `events/v1` |

### Quality Control

Every ingested record passes through row-level and partition-level QC checks:

- Data type validation, range checks, sentinel detection
- Confidence scoring in three tiers: **HIGH** (>= 0.7), **MEDIUM** (0.3–0.7), **LOW** (< 0.3)
- Per-adapter QC rule injection for source-specific quirks
- Full provenance tracking from source through fusion

### Fusion Modes

When querying across multiple sources, Neptune supports three merge strategies:

| Mode | Behavior |
|---|---|
| `best` | Deduplicate with configurable field-level precedence |
| `union` | Keep all records from all sources, tag provenance |
| `prefer:<source>` | Deterministic source preference (e.g., `prefer:noaa`) |

### Event Detection

Neptune derives four maritime event families from position data using heuristic detectors:

| Event | Description |
|---|---|
| **Port calls** | Sustained low-speed presence within a port boundary |
| **EEZ crossings** | Transitions between exclusive economic zones |
| **Encounters** | Two vessels within 500m for a sustained duration |
| **Loitering** | Sustained low-speed movement in a small area |

Each event includes a deterministic `event_id`, confidence score, timestamps, and full provenance linking back to source positions. See [HEURISTICS.md](HEURISTICS.md) for detection assumptions and known limitations.

## CLI

Neptune includes a full command-line interface (requires `pip install neptune-ais[cli]`):

```bash
# Download data
neptune download --source noaa --date 2024-06-15
neptune download --source noaa --source dma --start 2024-06-01 --end 2024-06-07

# Inspect what you have
neptune inventory
neptune inventory --dataset positions

# Quality reports
neptune qc --source noaa --date 2024-06-15

# SQL queries from the terminal
neptune sql "SELECT count(*) FROM positions WHERE source = 'noaa'"

# Source catalog
neptune sources
neptune sources --compare noaa dma gfw

# Event queries
neptune events --kind port_call --date 2024-06-15

# Health and provenance
neptune health
neptune provenance --date 2024-06-15

# Promote streaming data to canonical store
neptune promote --landing-dir /tmp/neptune_landing --source aisstream
```

## Documentation

Full Sphinx documentation is planned. In the meantime:

| Resource | Description |
|---|---|
| [`examples/`](examples/) | Seven narrative examples covering the full workflow |
| [HEURISTICS.md](HEURISTICS.md) | Event detection assumptions, confidence limits, non-goals |
| [RELEASING.md](RELEASING.md) | Release procedures and checklist |
| [RC_CHECKLIST.md](RC_CHECKLIST.md) | Release-candidate validation results |

### Examples

| # | Example | Topics |
|---|---|---|
| 1 | [Source Discovery](examples/01_source_discovery.ipynb) ([.py](examples/01_source_discovery.py)) | Inspect sources, capabilities, filters |
| 2 | [Archival Ingest](examples/02_archival_ingest.ipynb) ([.py](examples/02_archival_ingest.py)) | Download, Polars queries, SQL, helpers |
| 3 | [Multi-Source Fusion](examples/03_multi_source_fusion.ipynb) ([.py](examples/03_multi_source_fusion.py)) | Merge strategies, fusion config |
| 4 | [Event Detection](examples/04_event_detection.ipynb) ([.py](examples/04_event_detection.py)) | Port calls, EEZ crossings, encounters |
| 5 | [Streaming Pipeline](examples/05_streaming_pipeline.ipynb) ([.py](examples/05_streaming_pipeline.py)) | Live feeds, sinks, promotion |
| 6 | [External Plugin](examples/06_external_plugin.py) | Custom adapter via entry point |
| 7 | [Fishing Intelligence](examples/07_fishing_intelligence.ipynb) | GFW events, vessel identity, fishing effort grids |

> **Tip:** Install notebook support with `pip install neptune-ais[notebooks]` to run the interactive examples.

## Contributing

Contributions are welcome. To get started:

```bash
git clone https://github.com/yourorg/neptune-ais.git
cd neptune-ais
pip install -e ".[all,dev]"
pytest
```

The test suite includes 768 tests covering adapter certification, schema reproducibility, streaming soak tests, and packaging validation.

## License

[MIT](LICENSE)
