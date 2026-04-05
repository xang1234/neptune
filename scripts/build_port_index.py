#!/usr/bin/env python3
"""Build the World Port Index Parquet data files.

Downloads, parses, and compresses open maritime reference data into
Parquet files that ship with the neptune-ais package.

Data sources:
    - NGA World Port Index (WPI): ~3,800 ports, US Gov public domain
      https://msi.nga.mil/Publications/WPI
    - UN/LOCODE: ~116K locations filtered to ~8K port entries, public domain
      https://unece.org/trade/cefact/UNLOCODE-Download

Output (default: WPI + UNLOCODE):
    neptune_ais/ports/data/wpi_ports.parquet
    neptune_ais/ports/data/unlocode_ports.parquet

Output (--only eez, separate step):
    neptune_ais/ports/data/eez_meta.parquet
    neptune_ais/ports/data/eez_polygons.parquet

Usage:
    python scripts/build_port_index.py                # WPI + UNLOCODE
    python scripts/build_port_index.py --only eez     # EEZ (requires geopandas)

Requirements (build-time only):
    polars, httpx  (both are neptune-ais core deps)
    geopandas, shapely  (only for --only eez)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import sys
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WPI_CSV_URL = (
    "https://msi.nga.mil/api/publications/download"
    "?type=view&key=16920959/SFH00000/UpdatedPub150.csv"
)

UNLOCODE_CSV_URL = (
    "https://datahub.io/core/un-locode/_r/-/data/code-list.csv"
)

# VLIZ WFS endpoint — no authentication required
EEZ_WFS_URL = (
    "https://geo.vliz.be/geoserver/MarineRegions/wfs"
    "?service=WFS&version=2.0.0&request=GetFeature"
    "&typeNames=eez&outputFormat=application/json"
)

# Simplification tolerance in degrees (~0.05° ≈ 5.5 km at equator).
# EEZ boundaries are 200 nautical mile (370 km) zones from shore —
# 5 km precision is more than adequate for containment checks.
# The polygon file is lazy-downloaded at runtime, not shipped in the wheel.
EEZ_SIMPLIFY_TOLERANCE = 0.05

# Fallback: GitHub mirror (set up by maintainers via `neptune ports download`)
WPI_MIRROR_URL = ""  # TODO: set after first GitHub Release

# GitHub Release URL for EEZ polygon data (lazy-downloaded at runtime).
# Pattern: https://github.com/{owner}/{repo}/releases/download/{tag}/eez_polygons.parquet
# Set this after creating the data release (e.g., v0.1.0-data).
EEZ_POLYGONS_RELEASE_URL = ""  # TODO: set after uploading to GitHub Releases
# Used by the runtime loader (_loader.py) for integrity verification on download.
# Regenerate after rebuilding eez_polygons.parquet.
EEZ_POLYGONS_SHA256 = "e8d1b376a677a0ff5e7e9799107130bd7145079421196af5d0b6dd410eae4424"

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "neptune_ais" / "ports" / "data"

DEFAULT_RADIUS_KM = 2.0  # For ports with missing harbor size

# NGA WPI country names → ISO 3166-1 alpha-2 codes.
# The WPI "Country Code" column contains full country names, not ISO codes.
# This mapping covers all ~195 countries that appear in the WPI.
COUNTRY_NAME_TO_ISO: dict[str, str] = {
    "Afghanistan": "AF", "Albania": "AL", "Algeria": "DZ", "American Samoa": "AS",
    "Angola": "AO", "Anguilla": "AI", "Antarctica": "AQ",
    "Antigua and Barbuda": "AG", "Argentina": "AR", "Aruba": "AW",
    "Australia": "AU", "Azerbaijan": "AZ", "Bahamas  The": "BS",
    "Bahrain": "BH", "Bangladesh": "BD", "Barbados": "BB", "Belgium": "BE",
    "Belize": "BZ", "Benin": "BJ", "Bermuda": "BM", "Brazil": "BR",
    "British Virgin Islands": "VG", "Brunei": "BN", "Bulgaria": "BG",
    "Burma": "MM", "Cabo Verde": "CV", "Cambodia": "KH", "Cameroon": "CM",
    "Canada": "CA", "Cayman Islands": "KY", "Chile": "CL", "China": "CN",
    "Christmas Island": "CX", "Cocos (Keeling) Islands": "CC",
    "Colombia": "CO", "Comoros": "KM",
    "Congo  Democratic Republic of the": "CD",
    "Congo  Republic of the": "CG", "Cook Islands": "CK",
    "Costa Rica": "CR", "Cote d'Ivoire": "CI", "Croatia": "HR",
    "Cuba": "CU", "Curacao": "CW", "Cyprus": "CY", "Denmark": "DK",
    "Djibouti": "DJ", "Dominica": "DM", "Dominican Republic": "DO",
    "East Timor": "TL", "Ecuador": "EC", "Egypt": "EG",
    "El Salvador": "SV", "Equatorial Guinea": "GQ", "Eritrea": "ER",
    "Estonia": "EE", "Eswatini": "SZ", "Ethiopia": "ET",
    "Falkland Islands (Islas Malvinas)": "FK", "Faroe Islands": "FO",
    "Fiji": "FJ", "Finland": "FI", "France": "FR",
    "French Guiana": "GF", "French Polynesia": "PF",
    "French Southern and Antarctic Lands": "TF",
    "Gabon": "GA", "Gambia  The": "GM", "Georgia": "GE", "Germany": "DE",
    "Ghana": "GH", "Gibraltar": "GI", "Greece": "GR", "Greenland": "GL",
    "Grenada": "GD", "Guadeloupe": "GP", "Guam": "GU", "Guatemala": "GT",
    "Guernsey": "GG", "Guinea": "GN", "Guinea-Bissau": "GW",
    "Guyana": "GY", "Haiti": "HT", "Honduras": "HN", "Hong Kong": "HK",
    "Iceland": "IS", "India": "IN", "Indonesia": "ID", "Iran": "IR",
    "Iraq": "IQ", "Ireland": "IE", "Isle of Man": "IM", "Israel": "IL",
    "Italy": "IT", "Jamaica": "JM", "Japan": "JP", "Jersey": "JE",
    "Jordan": "JO", "Kazakhstan": "KZ", "Kenya": "KE", "Kiribati": "KI",
    "Korea  North": "KP", "North Korea": "KP",
    "Korea  South": "KR", "Kosovo": "XK",
    "Kuwait": "KW", "Latvia": "LV", "Lebanon": "LB", "Liberia": "LR",
    "Libya": "LY", "Lithuania": "LT", "Luxembourg": "LU", "Macau": "MO",
    "Madagascar": "MG", "Malaysia": "MY", "Maldives": "MV", "Malta": "MT",
    "Marshall Islands": "MH", "Martinique": "MQ", "Mauritania": "MR",
    "Mauritius": "MU", "Mayotte": "YT", "Mexico": "MX",
    "Micronesia  Federated States of": "FM",
    "Federated States of Micronesia": "FM", "Moldova": "MD",
    "Monaco": "MC", "Montenegro": "ME", "Montserrat": "MS",
    "Morocco": "MA", "Mozambique": "MZ", "Namibia": "NA", "Nauru": "NR",
    "Netherlands": "NL", "New Caledonia": "NC", "New Zealand": "NZ",
    "Nicaragua": "NI", "Nigeria": "NG", "Niue": "NU",
    "Norfolk Island": "NF", "Northern Mariana Islands": "MP",
    "Norway": "NO", "Oman": "OM", "Pakistan": "PK", "Palau": "PW",
    "Panama": "PA", "Papua New Guinea": "PG", "Paraguay": "PY",
    "Peru": "PE", "Philippines": "PH", "Pitcairn Islands": "PN",
    "Poland": "PL", "Portugal": "PT", "Puerto Rico": "PR", "Qatar": "QA",
    "Reunion": "RE", "Romania": "RO", "Russia": "RU", "Rwanda": "RW",
    "Saint Helena  Ascension  and Tristan da Cunha": "SH",
    "Saint Helena, Ascension, and Tristan da Cunha": "SH",
    "South Georgia and South Sandwich Islands": "GS",
    "Johnson Atoll": "UM", "Johnston Atoll": "UM",
    "Midway Islands": "UM", "Wake Island": "UM",
    "Bosnia and Herzegovina": "BA",
    "British Indian Ocean Territory": "IO",
    "Saint Kitts and Nevis": "KN", "Saint Lucia": "LC",
    "Saint Pierre and Miquelon": "PM",
    "Saint Vincent and the Grenadines": "VC", "Samoa": "WS",
    "Sao Tome and Principe": "ST", "Saudi Arabia": "SA", "Senegal": "SN",
    "Serbia": "RS", "Seychelles": "SC", "Sierra Leone": "SL",
    "Singapore": "SG", "Sint Maarten": "SX", "Slovenia": "SI",
    "Solomon Islands": "SB", "Somalia": "SO", "South Africa": "ZA",
    "Spain": "ES", "Spratly Islands": "XS", "Sri Lanka": "LK",
    "Sudan": "SD", "Suriname": "SR", "Svalbard": "SJ", "Sweden": "SE",
    "Switzerland": "CH", "Syria": "SY", "Taiwan": "TW", "Tanzania": "TZ",
    "Thailand": "TH", "Togo": "TG", "Tokelau": "TK", "Tonga": "TO",
    "Trinidad and Tobago": "TT", "Tunisia": "TN", "Turkey": "TR",
    "Turkmenistan": "TM", "Turks and Caicos Islands": "TC",
    "Tuvalu": "TV", "Ukraine": "UA",
    "United Arab Emirates": "AE", "United Kingdom": "GB",
    "United States": "US", "Uruguay": "UY", "Vanuatu": "VU",
    "Venezuela": "VE", "Vietnam": "VN",
    "Virgin Islands": "VI", "Wallis and Futuna": "WF",
    "Western Sahara": "EH", "Yemen": "YE", "Zambia": "ZM",
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _download_cached(
    urls: list[str],
    cache_path: Path | None = None,
    *,
    timeout: float = 120.0,
) -> bytes:
    """Download from a list of URLs with local caching and fallback.

    Tries each URL in order; returns cached data if available.
    """
    import httpx

    if cache_path and cache_path.exists():
        logger.info("Using cached file: %s", cache_path)
        return cache_path.read_bytes()

    last_err: Exception | None = None
    for url in urls:
        try:
            logger.info("Downloading from %s ...", url[:80])
            resp = httpx.get(url, follow_redirects=True, timeout=timeout)
            resp.raise_for_status()
            data = resp.content
            logger.info("Downloaded %d bytes", len(data))

            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(data)

            return data
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("Download failed from %s: %s", url[:80], exc)
            last_err = exc

    raise RuntimeError(f"Failed to download from all URLs: {last_err}")


def download_wpi_csv(cache_dir: Path | None = None) -> bytes:
    """Download the WPI CSV from NGA, with optional mirror fallback."""
    urls = [WPI_CSV_URL]
    if WPI_MIRROR_URL:
        urls.append(WPI_MIRROR_URL)
    cache_path = (cache_dir / "UpdatedPub150.csv") if cache_dir else None
    return _download_cached(urls, cache_path, timeout=60.0)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Columns we care about from the WPI CSV → our internal names.
WPI_COLUMN_MAP: dict[str, str] = {
    "World Port Index Number": "wpi_number",
    "Main Port Name": "name",
    "Alternate Port Name": "alternate_name",
    "UN/LOCODE": "unlocode_raw",
    "Country Code": "country_name",
    "Latitude": "lat",
    "Longitude": "lon",
    "Harbor Size": "harbor_size",
    "Harbor Type": "harbor_type",
    "Shelter Afforded": "shelter_quality",
    "Channel Depth (m)": "channel_depth_m",
    "Anchorage Depth (m)": "anchorage_depth_m",
    "Cargo Pier Depth (m)": "cargo_pier_depth_m",
    "Maximum Vessel Length (m)": "max_vessel_length_m",
    "Tidal Range (m)": "tide_range_m",
    "Pilotage - Compulsory": "has_pilotage",
    "Tugs - Assistance": "has_tugs",
    "Supplies - Fuel Oil": "has_fuel",
    "Dry Dock": "has_drydock_raw",
    "Cranes - Fixed": "has_cranes_fixed",
    "Cranes - Mobile": "has_cranes_mobile",
    "Cranes - Floating": "has_cranes_floating",
    "Medical Facilities": "has_medical",
}


def _ternary_to_bool(col: pl.Expr) -> pl.Expr:
    """Convert 'Yes'/'No'/'Unknown'/null → True/False/False."""
    return col.str.to_lowercase().eq(pl.lit("yes"))


def _wpi_zero_to_null(col: pl.Expr) -> pl.Expr:
    """Convert 0.0 → null (WPI uses 0.0 as sentinel for unknown numeric values)."""
    return pl.when(col == 0.0).then(None).otherwise(col)


def _normalize_unlocode(raw: pl.Expr) -> pl.Expr:
    """Convert 'NL RTM' → 'NLRTM' (remove internal space)."""
    return raw.str.replace_all(" ", "").str.to_uppercase()


def _extract_country_code_from_unlocode(raw: pl.Expr) -> pl.Expr:
    """Extract ISO alpha-2 from UNLOCODE ('NL RTM' → 'NL')."""
    return raw.str.replace_all(" ", "").str.slice(0, 2).str.to_uppercase()


def _harbor_size_code(full: pl.Expr) -> pl.Expr:
    """Convert 'Large' → 'L', 'Medium' → 'M', etc."""
    return (
        pl.when(full == "Large").then(pl.lit("L"))
        .when(full == "Medium").then(pl.lit("M"))
        .when(full == "Small").then(pl.lit("S"))
        .when(full == "Very Small").then(pl.lit("V"))
        .otherwise(pl.lit(None))
    )


def _shelter_code(full: pl.Expr) -> pl.Expr:
    """Convert 'Excellent' → 'E', 'Good' → 'G', etc."""
    return (
        pl.when(full == "Excellent").then(pl.lit("E"))
        .when(full == "Good").then(pl.lit("G"))
        .when(full == "Fair").then(pl.lit("F"))
        .when(full == "Poor").then(pl.lit("P"))
        .when(full == "None").then(pl.lit("N"))
        .otherwise(pl.lit(None))
    )


def parse_wpi_csv(csv_bytes: bytes) -> pl.DataFrame:
    """Parse the NGA WPI CSV into a clean Polars DataFrame.

    Handles:
    - UTF-8 BOM stripping
    - Column renaming to internal names
    - Type coercion (float → int for WPI number)
    - Ternary → bool conversion for facility flags
    - Depth 0.0 → null normalization
    - UNLOCODE space removal
    - Country name → ISO alpha-2 mapping
    - Harbor size/shelter quality code extraction
    - Bbox computation from harbor size
    """
    if csv_bytes[:3] == b"\xef\xbb\xbf":
        csv_bytes = csv_bytes[3:]

    # Read CSV with Polars — only the columns we need
    all_columns = list(WPI_COLUMN_MAP.keys())
    df = pl.read_csv(
        io.BytesIO(csv_bytes),
        columns=all_columns,
        infer_schema_length=5000,
        null_values=["", "null", "NULL"],
    )

    # Rename to internal names
    df = df.rename(WPI_COLUMN_MAP)

    # --- Type coercion ---

    # WPI number: float → int
    df = df.with_columns(
        pl.col("wpi_number").cast(pl.Float64).cast(pl.Int64),
    )

    # --- UNLOCODE normalization ---
    # Normalize and convert empty strings to null
    df = df.with_columns(
        pl.when(
            pl.col("unlocode_raw").is_not_null()
            & (pl.col("unlocode_raw").str.strip_chars().str.len_chars() > 2)
        )
        .then(_normalize_unlocode(pl.col("unlocode_raw")))
        .otherwise(pl.lit(None))
        .alias("unlocode"),
    )

    # --- Country code ---
    # Strategy: extract from UNLOCODE where available, else map from name
    country_map = pl.DataFrame({
        "country_name": list(COUNTRY_NAME_TO_ISO.keys()),
        "_iso_from_name": list(COUNTRY_NAME_TO_ISO.values()),
    })

    # Prefer the name-mapped ISO code (reliable).
    # Fall back to UNLOCODE prefix only when name mapping is missing,
    # because NGA uses non-standard UNLOCODE prefixes for some countries
    # (e.g., GC for Equatorial Guinea instead of ISO GQ).
    df = df.with_columns(
        _extract_country_code_from_unlocode(
            pl.col("unlocode_raw")
        ).alias("_iso_from_locode"),
    )
    df = df.join(country_map, on="country_name", how="left")
    df = df.with_columns(
        pl.when(pl.col("_iso_from_name").is_not_null())
        .then(pl.col("_iso_from_name"))
        .when(
            pl.col("_iso_from_locode").is_not_null()
            & (pl.col("_iso_from_locode").str.len_chars() == 2)
        )
        .then(pl.col("_iso_from_locode"))
        .otherwise(pl.lit(None))
        .alias("country_code"),
    )
    df = df.drop(["unlocode_raw", "_iso_from_locode", "_iso_from_name"])

    # Normalize double-spaces in country names (older NGA formats used "Bahamas  The")
    df = df.with_columns(
        pl.col("country_name").str.replace_all("  ", " ").alias("country_name"),
    )

    # --- Empty strings → null for string fields ---
    df = df.with_columns(
        pl.when(pl.col("alternate_name").str.strip_chars().str.len_chars() == 0)
        .then(pl.lit(None))
        .otherwise(pl.col("alternate_name"))
        .alias("alternate_name"),
    )

    # --- Harbor size / shelter codes ---
    df = df.with_columns(
        _harbor_size_code(pl.col("harbor_size")).alias("harbor_size"),
        _shelter_code(pl.col("shelter_quality")).alias("shelter_quality"),
    )

    # --- Harbor type: abbreviate ---
    df = df.with_columns(
        pl.when(pl.col("harbor_type").str.starts_with("Coastal (Natural)"))
        .then(pl.lit("CN"))
        .when(pl.col("harbor_type").str.starts_with("Coastal (Breakwater)"))
        .then(pl.lit("CB"))
        .when(pl.col("harbor_type").str.starts_with("Coastal (Tide"))
        .then(pl.lit("CT"))
        .when(pl.col("harbor_type").str.starts_with("River (Natural)"))
        .then(pl.lit("RN"))
        .when(pl.col("harbor_type").str.starts_with("River (Basin"))
        .then(pl.lit("RB"))
        .when(pl.col("harbor_type").str.starts_with("River (Tide"))
        .then(pl.lit("RT"))
        .when(pl.col("harbor_type").str.starts_with("Open Roadstead"))
        .then(pl.lit("OR"))
        .when(pl.col("harbor_type").str.starts_with("Canal or Lake"))
        .then(pl.lit("CL"))
        .when(pl.col("harbor_type").str.starts_with("Typhoon"))
        .then(pl.lit("TH"))
        .otherwise(pl.lit(None))
        .alias("harbor_type"),
    )

    depth_cols = [
        "channel_depth_m", "anchorage_depth_m", "cargo_pier_depth_m",
        "max_vessel_length_m", "tide_range_m",
    ]
    df = df.with_columns(
        [_wpi_zero_to_null(pl.col(c)).alias(c) for c in depth_cols]
    )

    bool_cols = ["has_pilotage", "has_tugs", "has_fuel", "has_medical"]
    df = df.with_columns(
        [_ternary_to_bool(pl.col(c)).alias(c) for c in bool_cols]
    )

    # Drydock: has a drydock if value is not None/Unknown/null
    df = df.with_columns(
        (
            pl.col("has_drydock_raw").is_not_null()
            & ~pl.col("has_drydock_raw").is_in(["None", "Unknown"])
        ).alias("has_drydock"),
    )
    df = df.drop("has_drydock_raw")

    # Cranes: any of fixed/mobile/floating = Yes → has_cranes = True
    df = df.with_columns(
        (
            _ternary_to_bool(pl.col("has_cranes_fixed"))
            | _ternary_to_bool(pl.col("has_cranes_mobile"))
            | _ternary_to_bool(pl.col("has_cranes_floating"))
        ).alias("has_cranes"),
    )
    df = df.drop(["has_cranes_fixed", "has_cranes_mobile", "has_cranes_floating"])

    # --- Compute bboxes (vectorized) ---
    radius_km = (
        pl.when(pl.col("harbor_size") == "L").then(10.0)
        .when(pl.col("harbor_size") == "M").then(5.0)
        .when(pl.col("harbor_size") == "S").then(2.0)
        .when(pl.col("harbor_size") == "V").then(1.0)
        .otherwise(DEFAULT_RADIUS_KM)
    )
    deg_per_km = 1.0 / 111.32
    lat_delta = radius_km * deg_per_km
    cos_lat = pl.col("lat").radians().cos().clip(lower_bound=0.01)
    lon_delta = radius_km * deg_per_km / cos_lat

    df = df.with_columns(
        (pl.col("lon") - lon_delta).clip(-180.0, 180.0).alias("bbox_west"),
        (pl.col("lat") - lat_delta).clip(-90.0, 90.0).alias("bbox_south"),
        (pl.col("lon") + lon_delta).clip(-180.0, 180.0).alias("bbox_east"),
        (pl.col("lat") + lat_delta).clip(-90.0, 90.0).alias("bbox_north"),
    )

    # --- has_derived_polygon placeholder ---
    df = df.with_columns(
        pl.lit(False).alias("has_derived_polygon"),
    )

    # --- Select final columns in canonical order ---
    final_columns = [
        "wpi_number", "name", "alternate_name", "unlocode",
        "country_code", "country_name",
        "lat", "lon",
        "bbox_west", "bbox_south", "bbox_east", "bbox_north",
        "harbor_size", "harbor_type", "shelter_quality",
        "channel_depth_m", "anchorage_depth_m", "cargo_pier_depth_m",
        "max_vessel_length_m", "tide_range_m",
        "has_pilotage", "has_tugs", "has_fuel", "has_drydock", "has_cranes",
        "has_medical",
        "has_derived_polygon",
    ]
    df = df.select(final_columns)

    # --- Deduplicate by WPI number (keep first occurrence) ---
    # NGA data has rare duplicates (e.g., WPI 400: two spellings of same port)
    n_before = len(df)
    df = df.unique(subset=["wpi_number"], keep="first", maintain_order=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("Dropped %d duplicate WPI numbers", n_dropped)

    # --- Sort by WPI number ---
    df = df.sort("wpi_number")

    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_wpi(df: pl.DataFrame) -> None:
    """Run basic sanity checks on the parsed WPI data."""
    n = len(df)
    logger.info("Parsed %d ports", n)

    if n < 3500:
        raise ValueError(f"Expected ~3800 ports, got {n}")

    # Check required columns have no nulls
    for col in ["wpi_number", "name", "lat", "lon"]:
        null_count = df[col].null_count()
        if null_count > 0:
            raise ValueError(f"Column {col} has {null_count} nulls")

    # Check WPI numbers are unique
    n_unique = df["wpi_number"].n_unique()
    if n_unique != n:
        raise ValueError(f"WPI numbers not unique: {n_unique} unique vs {n} rows")

    # Check coordinate ranges
    lat_range = df["lat"].min(), df["lat"].max()
    lon_range = df["lon"].min(), df["lon"].max()
    logger.info("Lat range: [%.2f, %.2f]", *lat_range)
    logger.info("Lon range: [%.2f, %.2f]", *lon_range)

    if lat_range[0] < -90 or lat_range[1] > 90:
        raise ValueError(f"Latitude out of range: {lat_range}")
    if lon_range[0] < -180 or lon_range[1] > 180:
        raise ValueError(f"Longitude out of range: {lon_range}")

    # Check country codes
    no_cc = df.filter(pl.col("country_code").is_null()).height
    if no_cc > 50:
        logger.warning("%d ports have no country code", no_cc)

    # Harbor size distribution
    hs_counts = df.group_by("harbor_size").len().sort("len", descending=True)
    logger.info("Harbor size distribution:\n%s", hs_counts)


# ---------------------------------------------------------------------------
# UN/LOCODE — download, parse, filter
# ---------------------------------------------------------------------------


def download_unlocode_csv(cache_dir: Path | None = None) -> bytes:
    """Download the UN/LOCODE CSV from datahub.io (pre-cleaned UTF-8)."""
    cache_path = (cache_dir / "un-locode-code-list.csv") if cache_dir else None
    return _download_cached([UNLOCODE_CSV_URL], cache_path)


def _parse_unlocode_coord(coord_str: str | None) -> tuple[float | None, float | None]:
    """Parse UN/LOCODE coordinate string to (lat, lon).

    Format: '4042N 07400W' → (40.7, -74.0)
    Latitude: DDMM[N|S], Longitude: DDDMM[E|W]
    """
    if not coord_str or len(coord_str) < 11:
        return (None, None)

    coord_str = coord_str.strip()
    parts = coord_str.split()
    if len(parts) != 2:
        return (None, None)

    lat_str, lon_str = parts

    try:
        # Latitude: DDMMN (5 chars)
        lat_dir = lat_str[-1]
        lat_deg = int(lat_str[:2])
        lat_min = int(lat_str[2:4])
        lat = lat_deg + lat_min / 60.0
        if lat_dir == "S":
            lat = -lat

        # Longitude: DDDMME (6 chars)
        lon_dir = lon_str[-1]
        lon_deg = int(lon_str[:3])
        lon_min = int(lon_str[3:5])
        lon = lon_deg + lon_min / 60.0
        if lon_dir == "W":
            lon = -lon

        # Sanity check
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            return (None, None)

        return (lat, lon)
    except (ValueError, IndexError):
        return (None, None)


def parse_unlocode_csv(csv_bytes: bytes) -> pl.DataFrame:
    """Parse and filter UN/LOCODE CSV to port-function entries with coordinates.

    Filters to entries where:
    1. Function field contains '1' (seaport)
    2. Coordinates are present and parseable

    Returns a DataFrame with columns:
        code, country_code, location_code, name, subdivision,
        function_codes, lat, lon
    """
    df = pl.read_csv(
        io.BytesIO(csv_bytes),
        infer_schema_length=10000,
        null_values=[""],
    )

    # The datahub.io CSV has these columns:
    # Change, Country, Location, Name, NameWoDiacritics, Subdivision,
    # Status, Function, Date, IATA, Coordinates, Remarks
    required = {"Country", "Location", "Name", "NameWoDiacritics",
                "Function", "Coordinates", "Subdivision"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"UNLOCODE CSV missing columns: {missing}")

    # Filter to entries with a Function value containing '1' (seaport)
    df = df.filter(
        pl.col("Function").is_not_null()
        & pl.col("Function").str.contains("1")
    )
    logger.info("Entries with port function: %d", len(df))

    # Filter to entries with coordinates
    df = df.filter(pl.col("Coordinates").is_not_null())
    logger.info("Port entries with coordinates: %d", len(df))

    # Parse coordinates
    coords = [
        _parse_unlocode_coord(c)
        for c in df["Coordinates"].to_list()
    ]
    df = df.with_columns(
        pl.Series("lat", [c[0] for c in coords], dtype=pl.Float64),
        pl.Series("lon", [c[1] for c in coords], dtype=pl.Float64),
    )

    # Drop entries where coordinate parsing failed
    df = df.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
    logger.info("Port entries with valid parsed coordinates: %d", len(df))

    # Build the 5-char LOCODE: country + location (e.g., "NL" + "RTM" = "NLRTM")
    df = df.with_columns(
        (pl.col("Country") + pl.col("Location")).alias("code"),
    )

    # Use NameWoDiacritics for ASCII-safe matching, fall back to Name
    df = df.with_columns(
        pl.when(pl.col("NameWoDiacritics").is_not_null())
        .then(pl.col("NameWoDiacritics"))
        .otherwise(pl.col("Name"))
        .alias("name"),
    )

    # Select and rename final columns
    result = df.select(
        pl.col("code"),
        pl.col("Country").alias("country_code"),
        pl.col("Location").alias("location_code"),
        pl.col("name"),
        pl.col("Subdivision").alias("subdivision"),
        pl.col("Function").alias("function_codes"),
        pl.col("lat"),
        pl.col("lon"),
    )

    # Deduplicate by code (keep first)
    n_before = len(result)
    result = result.unique(subset=["code"], keep="first", maintain_order=True)
    n_dropped = n_before - len(result)
    if n_dropped:
        logger.info("Dropped %d duplicate LOCODE entries", n_dropped)

    return result.sort("code")


def validate_unlocode(df: pl.DataFrame) -> None:
    """Run sanity checks on parsed UNLOCODE data."""
    n = len(df)
    logger.info("Parsed %d UNLOCODE port entries", n)

    if n < 5000:
        raise ValueError(f"Expected ~8K+ port entries, got {n}")

    for col in ["code", "country_code", "name", "lat", "lon"]:
        null_count = df[col].null_count()
        if null_count > 0:
            raise ValueError(f"UNLOCODE column {col} has {null_count} nulls")

    # Validate code format: 5 chars, all uppercase alpha
    bad_codes = df.filter(pl.col("code").str.len_chars() != 5)
    if len(bad_codes) > 0:
        logger.warning("%d codes with unexpected length", len(bad_codes))

    # Coordinate ranges
    lat_range = df["lat"].min(), df["lat"].max()
    lon_range = df["lon"].min(), df["lon"].max()
    logger.info("Lat range: [%.2f, %.2f]", *lat_range)
    logger.info("Lon range: [%.2f, %.2f]", *lon_range)


def build_unlocode(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Build unlocode_ports.parquet from the UN/LOCODE CSV."""
    csv_bytes = download_unlocode_csv(cache_dir=cache_dir)

    df = parse_unlocode_csv(csv_bytes)
    validate_unlocode(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "unlocode_ports.parquet"
    df.write_parquet(output_path, compression="zstd", compression_level=9)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%.2f MB, %d rows)", output_path, size_mb, len(df))

    if size_mb > 1.0:
        logger.warning("Output exceeds 1 MB target: %.2f MB", size_mb)

    return output_path


# ---------------------------------------------------------------------------
# EEZ — download from WFS, extract metadata, simplify polygons
# ---------------------------------------------------------------------------


def download_eez_geojson(cache_dir: Path | None = None) -> bytes:
    """Download all EEZ polygons as GeoJSON from the VLIZ WFS endpoint."""
    cache_path = (cache_dir / "eez_v12.geojson") if cache_dir else None
    return _download_cached([EEZ_WFS_URL], cache_path, timeout=300.0)


def parse_eez_geojson(
    geojson_bytes: bytes,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parse EEZ GeoJSON into metadata and polygon DataFrames.

    Requires geopandas and shapely (build-time deps).

    Returns:
        (eez_meta, eez_polygons) — metadata-only and WKB polygon DataFrames.
    """
    try:
        import geopandas as gpd
        import shapely
    except ImportError:
        raise ImportError(
            "EEZ processing requires geopandas and shapely. "
            "Install with: pip install geopandas shapely"
        ) from None

    gdf = gpd.read_file(io.BytesIO(geojson_bytes))
    del geojson_bytes  # free ~270 MB
    logger.info("Loaded %d EEZ features", len(gdf))

    meta_rows = []
    poly_rows = []

    for _, row in gdf.iterrows():
        mrgid = int(row.get("MRGID", row.get("mrgid", 0)))
        name = row.get("GEONAME", row.get("geoname")) or ""
        sovereign = row.get("SOVEREIGN1", row.get("sovereign1")) or ""
        iso_3 = row.get("ISO_SOV1", row.get("iso_sov1")) or ""
        area_km2 = float(row.get("AREA_KM2", row.get("area_km2", 0.0)))

        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Compute bbox from original (unsimplified) geometry
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        bbox_west, bbox_south, bbox_east, bbox_north = bounds

        meta_rows.append({
            "mrgid": mrgid,
            "name": name,
            "sovereign": sovereign,
            "iso_3": iso_3,
            "bbox_west": bbox_west,
            "bbox_south": bbox_south,
            "bbox_east": bbox_east,
            "bbox_north": bbox_north,
            "area_km2": area_km2,
        })

        simplified = geom.simplify(EEZ_SIMPLIFY_TOLERANCE)
        poly_rows.append({
            "mrgid": mrgid,
            "geometry_wkb": shapely.to_wkb(simplified),
        })

    # Build Polars DataFrames
    meta_df = pl.DataFrame(meta_rows).sort("mrgid")
    poly_df = pl.DataFrame(poly_rows).sort("mrgid")

    logger.info("Extracted %d EEZ regions with geometry", len(meta_df))
    return meta_df, poly_df


def validate_eez(meta_df: pl.DataFrame, poly_df: pl.DataFrame) -> None:
    """Run sanity checks on parsed EEZ data."""
    n = len(meta_df)
    logger.info("EEZ metadata: %d regions", n)

    if n < 200:
        raise ValueError(f"Expected ~285 EEZ regions, got {n}")

    if len(poly_df) != n:
        raise ValueError(
            f"Metadata ({n}) and polygon ({len(poly_df)}) row counts differ"
        )

    for col in ["mrgid", "name", "iso_3"]:
        null_count = meta_df[col].null_count()
        if null_count > 0:
            logger.warning("EEZ column %s has %d nulls", col, null_count)


def build_eez(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    cache_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Build eez_meta.parquet and eez_polygons.parquet from VLIZ WFS."""
    import os
    # EEZ GeoJSON is ~270 MB — GDAL rejects it without this override
    os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

    geojson_bytes = download_eez_geojson(cache_dir=cache_dir)

    meta_df, poly_df = parse_eez_geojson(geojson_bytes)
    validate_eez(meta_df, poly_df)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write metadata (ships in-package)
    meta_path = output_dir / "eez_meta.parquet"
    meta_df.write_parquet(meta_path, compression="zstd", compression_level=9)
    meta_mb = meta_path.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%.2f MB, %d rows)", meta_path, meta_mb, len(meta_df))
    if meta_mb > 0.5:
        logger.warning("eez_meta exceeds 500 KB target: %.2f MB", meta_mb)

    # Write polygons (hosted externally, lazy-downloaded at runtime)
    poly_path = output_dir / "eez_polygons.parquet"
    poly_df.write_parquet(poly_path, compression="zstd", compression_level=9)
    poly_mb = poly_path.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%.2f MB, %d rows)", poly_path, poly_mb, len(poly_df))
    if poly_mb > 25.0:
        logger.warning("eez_polygons exceeds 25 MB target: %.2f MB", poly_mb)

    return meta_path, poly_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_wpi(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    skip_download: bool = False,
    cache_dir: Path | None = None,
    csv_path: Path | None = None,
) -> Path:
    """Build wpi_ports.parquet from the NGA WPI CSV.

    Args:
        output_dir: Where to write the output Parquet file.
        skip_download: If True, expect csv_path to be provided.
        cache_dir: Directory to cache the downloaded CSV.
        csv_path: Path to a pre-downloaded CSV file.

    Returns:
        Path to the output Parquet file.
    """
    if csv_path and csv_path.exists():
        logger.info("Reading WPI CSV from %s", csv_path)
        csv_bytes = csv_path.read_bytes()
    elif skip_download:
        raise FileNotFoundError("--skip-download requires --csv-path")
    else:
        csv_bytes = download_wpi_csv(cache_dir=cache_dir)

    logger.info("Source CSV SHA-256: %s", hashlib.sha256(csv_bytes).hexdigest())

    df = parse_wpi_csv(csv_bytes)
    validate_wpi(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "wpi_ports.parquet"
    df.write_parquet(output_path, compression="zstd", compression_level=9)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Wrote %s (%.2f MB, %d rows)", output_path, size_mb, len(df))

    if size_mb > 3.0:
        logger.warning("Output file exceeds 3 MB target: %.2f MB", size_mb)

    # Write SHA-256 sidecar for the output Parquet (for integrity verification)
    parquet_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
    sha_path = output_path.with_suffix(".parquet.sha256")
    sha_path.write_text(f"{parquet_sha}  wpi_ports.parquet\n")
    logger.info("Wrote SHA-256 sidecar: %s", sha_path)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build port index Parquet data files"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: neptune_ais/ports/data/)",
    )
    parser.add_argument(
        "--csv-path", type=Path, default=None,
        help="Path to pre-downloaded WPI CSV file",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Directory to cache downloaded CSV files",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download, use --csv-path instead",
    )
    parser.add_argument(
        "--only", choices=["wpi", "unlocode", "eez"], default=None,
        help="Build one dataset (default: wpi + unlocode; eez requires geopandas)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    outputs: list[Path] = []
    try:
        if args.only in (None, "wpi"):
            outputs.append(build_wpi(
                output_dir=args.output_dir,
                skip_download=args.skip_download,
                cache_dir=args.cache_dir,
                csv_path=args.csv_path,
            ))
        if args.only in (None, "unlocode"):
            outputs.append(build_unlocode(
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
            ))
        if args.only == "eez":
            # EEZ requires geopandas/shapely — not in the default build path
            meta_path, poly_path = build_eez(
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
            )
            outputs.extend([meta_path, poly_path])
        for p in outputs:
            print(f"Success: {p}")
    except Exception as exc:
        logger.error("Build failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
