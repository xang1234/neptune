# Neptune AIS Examples

Interactive notebooks covering the full Neptune workflow — from data discovery to animated vessel replay.

> **Tip:** Install notebook support with `pip install neptune-ais[geo,sql,notebooks]` to run all examples.

## Notebooks

| # | Example | Topics | nbviewer |
|---|---------|--------|----------|
| 1 | [Source Discovery](01_source_discovery.ipynb) | Inspect sources, capabilities, filters | [view](https://nbviewer.org/github/xang1234/neptune/blob/main/examples/01_source_discovery.ipynb) |
| 2 | [Archival Ingest](02_archival_ingest.ipynb) | Download, Polars queries, SQL, helpers | [view](https://nbviewer.org/github/xang1234/neptune/blob/main/examples/02_archival_ingest.ipynb) |
| 3 | [Multi-Source Fusion](03_multi_source_fusion.ipynb) | Merge strategies, fusion config | [view](https://nbviewer.org/github/xang1234/neptune/blob/main/examples/03_multi_source_fusion.ipynb) |
| 4 | [Event Detection](04_event_detection.ipynb) | Port calls, EEZ crossings, encounters | [view](https://nbviewer.org/github/xang1234/neptune/blob/main/examples/04_event_detection.ipynb) |
| 5 | [Streaming Pipeline](05_streaming_pipeline.ipynb) | Live feeds, sinks, promotion | [view](https://nbviewer.org/github/xang1234/neptune/blob/main/examples/05_streaming_pipeline.ipynb) |
| 6 | [External Plugin](06_external_plugin.py) | Custom adapter via entry point | — |
| 7 | [Fishing Intelligence](07_fishing_intelligence.ipynb) | GFW events, vessels, fishing effort grids | [view](https://nbviewer.org/github/xang1234/neptune/blob/main/examples/07_fishing_intelligence.ipynb) |
| 8 | [Spatial Visualization](08_spatial_visualization.ipynb) | Interactive maps, GeoDataFrames, MovingPandas, animated replay | [view](https://nbviewer.org/github/xang1234/neptune/blob/main/examples/08_spatial_visualization.ipynb) |

## Suggested order

**Start here:** Examples 1-2 are self-contained and require no API keys.

```
1 Source Discovery     → What data sources are available?
2 Archival Ingest      → Download and query NOAA data
3 Multi-Source Fusion  → Combine NOAA + DMA
4 Event Detection      → Derive port calls, encounters from positions
5 Streaming Pipeline   → Live AIS feeds (requires AISStream API key)
6 External Plugin      → Build your own adapter
7 Fishing Intelligence → GFW events & effort (requires GFW API token)
8 Spatial Visualization → Maps, trajectories, animated replay
```

## Requirements by example

| Example | Install extra | API key needed |
|---------|--------------|----------------|
| 1 | `pip install neptune-ais` | No |
| 2 | `pip install neptune-ais[sql]` | No |
| 3 | `pip install neptune-ais[sql]` | No |
| 4 | `pip install neptune-ais[geo]` | No |
| 5 | `pip install neptune-ais[stream]` | AISStream |
| 6 | — | No |
| 7 | `pip install neptune-ais[notebooks,gfw]` | GFW |
| 8 | `pip install neptune-ais[geo,sql]` | No |
