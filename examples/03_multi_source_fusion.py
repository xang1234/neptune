"""Example 3: Multi-source fusion — combine data from multiple providers.

When multiple sources cover the same area, Neptune fuses them using
configurable merge strategies. This example shows how fusion works
and how to inspect the fusion configuration.
"""

from neptune_ais.api import Neptune

# --- Configure multi-source fusion ---
# "best" mode: deduplicate across sources, prefer higher-quality data.
# "union" mode: keep all records from all sources.
# "prefer:noaa" mode: prefer NOAA when sources overlap.

# n = Neptune(
#     ("2024-06-15", "2024-06-16"),
#     sources=["noaa", "dma"],
#     merge="best",
#     cache_dir="/tmp/neptune_demo",
# )

# --- Inspect fusion configuration ---
# info = n.fusion_info()
# print(f"Mode:        {info['fusion']['mode']}")
# print(f"Sources:     {info['sources']}")
# print(f"Precedence:  {info['fusion']['source_precedence']}")
# print(f"Multi-source: {info['multi_source']}")

# --- Per-source breakdown ---
# for sd in info["per_source"]:
#     print(f"  {sd['source']}: {sd['rows']:,} rows across {sd['partitions']} partitions")

# --- Query fused data (deduplication happens automatically) ---
# positions = n.positions()
# df = positions.collect()
# print(f"Total fused positions: {len(df):,}")
# print(f"Sources in result: {df['source'].unique().to_list()}")

print("Example 3: Multi-source fusion.")
print("Uncomment after downloading data from multiple sources.")
print()
print("Merge modes:")
print("  'best'   — deduplicate, prefer highest-quality source")
print("  'union'  — keep all records from all sources")
print("  'prefer:<source>' — prefer a specific source when overlapping")
