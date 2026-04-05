# Data Attribution

This directory contains reference data files derived from the following
open-data sources. Neptune AIS gratefully acknowledges their publishers.

## NGA World Port Index (WPI)

- **Source:** National Geospatial-Intelligence Agency (NGA), Pub. 150
- **URL:** https://msi.nga.mil/Publications/WPI
- **License:** Public domain (US Government work)
- **File:** `wpi_ports.parquet`
- **Description:** Port identity, metadata, facilities, and services
  for ~3,800 ports worldwide.

## UN/LOCODE

- **Source:** United Nations Economic Commission for Europe (UNECE)
- **URL:** https://unece.org/trade/cefact/UNLOCODE-Download
- **License:** Public domain
- **File:** `unlocode_ports.parquet`
- **Description:** Standardized 5-character location codes for ~12,000
  port-function locations with coordinates. Filtered from ~116,000
  entries to port-function codes with valid coordinates.

## MarineRegions.org EEZ Boundaries v12

- **Source:** Flanders Marine Institute (VLIZ)
- **URL:** https://www.marineregions.org/eez.php
- **License:** Creative Commons Attribution 4.0 International (CC-BY 4.0)
- **Files:** `eez_meta.parquet`, `eez_polygons.parquet`
- **Description:** Exclusive Economic Zone boundaries for 285 maritime
  jurisdiction zones worldwide. Polygons simplified for file size.
- **Citation:** Flanders Marine Institute (2023). Maritime Boundaries
  Geodatabase: Maritime Boundaries and Exclusive Economic Zones (200NM),
  version 12. Available online at https://www.marineregions.org/.
  https://doi.org/10.14284/628
