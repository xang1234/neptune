"""Example 1: Source discovery — find the right data source for your workflow.

Neptune supports multiple AIS data sources with different coverage,
delivery modes, and access requirements. This example shows how to
inspect what is available before downloading anything.
"""

from neptune_ais.adapters.registry import (
    load_all_adapters,
    catalog,
    info,
    compare,
    find_sources,
)

# Load all built-in adapters and any installed plugins.
load_all_adapters()

# --- List all registered sources ---
print("=== All registered sources ===")
for caps in catalog():
    print(f"  {caps.source_id:<12} {caps.provider:<30} {caps.coverage}")

# --- Inspect a specific source ---
print("\n=== NOAA details ===")
noaa = info("noaa")
print(f"  Provider:  {noaa.provider}")
print(f"  Coverage:  {noaa.coverage}")
print(f"  History:   {noaa.history_start}")
print(f"  Backfill:  {noaa.supports_backfill}")
print(f"  Auth:      {noaa.auth_scheme or 'none'}")
print(f"  License:   {noaa.license_requirements}")

# --- Compare sources side-by-side ---
print("\n=== Compare NOAA vs GFW ===")
for summary in compare("noaa", "gfw"):
    print(f"  {summary['source']:<8} backfill={summary['backfill']}, "
          f"auth={summary['auth']}, coverage={summary['coverage']}")

# --- Filter sources by capability ---
print("\n=== Open-data sources (no authentication required) ===")
for caps in find_sources(auth=False):
    print(f"  {caps.source_id}: {caps.provider}")

print("\n=== Sources supporting streaming ===")
for caps in find_sources(streaming=True):
    print(f"  {caps.source_id}: {caps.provider}")

print("\n=== Sources with backfill + no auth ===")
for caps in find_sources(backfill=True, auth=False):
    print(f"  {caps.source_id}: {caps.provider}")
