"""Data loading layer for the port dictionary.

Reads built-in Parquet data files via ``importlib.resources``, handles
user overlay merging, and provides lazy download of EEZ polygon data.

All functions return Polars DataFrames. The in-package data files
(wpi_ports, unlocode_ports, eez_meta) are always available. The EEZ
polygon file is large (~16 MB) and downloaded on first use.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data file paths (via importlib.resources)
# ---------------------------------------------------------------------------

_DATA_DIR: Path | None = None


def _get_data_dir() -> Path:
    """Resolve the path to the built-in data directory."""
    global _DATA_DIR
    if _DATA_DIR is None:
        from importlib.resources import files

        _DATA_DIR = Path(str(files("neptune_ais.ports") / "data"))
    return _DATA_DIR


# ---------------------------------------------------------------------------
# EEZ polygon download constants
# ---------------------------------------------------------------------------

# GitHub Release URL for lazy download. Empty until release is published.
_EEZ_POLYGONS_URL = ""  # TODO: set after first GitHub Release

# Expected SHA-256 of the EEZ polygon Parquet file.
_EEZ_POLYGONS_SHA256 = (
    "e8d1b376a677a0ff5e7e9799107130bd7145079421196af5d0b6dd410eae4424"
)

# Local cache directory for downloaded data.
_CACHE_DIR = Path.home() / ".neptune" / "ports"


# ---------------------------------------------------------------------------
# In-package loaders
# ---------------------------------------------------------------------------


def load_ports(
    *,
    user_overlays: list[Path] | None = None,
) -> pl.DataFrame:
    """Load the built-in WPI port dataset, optionally merging user overlays.

    User overlays are Parquet files with the same schema. Rows in
    overlays replace built-in rows by ``wpi_number`` (anti-join +
    concat). Rows with ``wpi_number < 0`` or null are appended as
    user-defined ports.

    Overlay lookup order:
        1. Explicit ``user_overlays`` parameter.
        2. ``$NEPTUNE_PORTS_OVERLAY`` environment variable (path).
        3. ``~/.neptune/ports/custom_ports.parquet`` (persistent).

    Args:
        user_overlays: Optional list of Parquet file paths to merge.

    Returns:
        A Polars DataFrame with the WPI port schema.
    """
    path = _get_data_dir() / "wpi_ports.parquet"
    df = pl.read_parquet(path)

    # Collect overlay paths in priority order (lowest first, highest last).
    # Last-applied wins, so: persistent < env < explicit.
    overlay_paths: list[Path] = []

    default_overlay = _CACHE_DIR / "custom_ports.parquet"
    if default_overlay.exists():
        overlay_paths.append(default_overlay)

    env_overlay = os.environ.get("NEPTUNE_PORTS_OVERLAY")
    if env_overlay:
        p = Path(env_overlay)
        if p.exists():
            overlay_paths.append(p)

    if user_overlays:
        overlay_paths.extend(user_overlays)

    # Merge overlays
    for overlay_path in overlay_paths:
        overlay = pl.read_parquet(overlay_path)
        logger.info("Merging port overlay: %s (%d rows)", overlay_path, len(overlay))
        df = _merge_overlay(df, overlay, key="wpi_number")

    return df


def load_unlocodes() -> pl.DataFrame:
    """Load the built-in UNLOCODE port dataset.

    Returns:
        A Polars DataFrame with columns: code, country_code,
        location_code, name, subdivision, function_codes, lat, lon.
    """
    path = _get_data_dir() / "unlocode_ports.parquet"
    return pl.read_parquet(path)


def load_eez_meta() -> pl.DataFrame:
    """Load EEZ metadata (names, bboxes, areas). Always in-package.

    Returns:
        A Polars DataFrame with columns: mrgid, name, sovereign,
        iso_3, bbox_west/south/east/north, area_km2.
    """
    path = _get_data_dir() / "eez_meta.parquet"
    return pl.read_parquet(path)


# ---------------------------------------------------------------------------
# Lazy-downloaded EEZ polygons
# ---------------------------------------------------------------------------


def load_eez_polygons(
    *,
    cache_dir: Path | None = None,
) -> pl.DataFrame:
    """Load simplified EEZ polygon geometries (WKB).

    The polygon file is not shipped in the wheel (~16 MB). On first
    call, it is downloaded from a GitHub Release and cached locally.

    Cache lookup order:
        1. ``cache_dir`` parameter (if provided).
        2. ``~/.neptune/ports/eez_polygons.parquet``.
        3. In-package data directory (if file exists, e.g. dev checkout).

    Args:
        cache_dir: Override cache directory.

    Returns:
        A Polars DataFrame with columns: mrgid, geometry_wkb.

    Raises:
        RuntimeError: If the file cannot be found or downloaded.
    """
    target_dir = cache_dir or _CACHE_DIR
    cache_path = target_dir / "eez_polygons.parquet"

    # Check cache first
    if cache_path.exists():
        logger.debug("Loading cached EEZ polygons: %s", cache_path)
        return pl.read_parquet(cache_path)

    # Check in-package (dev checkout has the file)
    in_package = _get_data_dir() / "eez_polygons.parquet"
    if in_package.exists():
        logger.debug("Loading in-package EEZ polygons: %s", in_package)
        return pl.read_parquet(in_package)

    # Download
    if not _EEZ_POLYGONS_URL:
        raise RuntimeError(
            "EEZ polygon data is not available. The download URL has not "
            "been configured yet. Run `neptune ports download` or place "
            "eez_polygons.parquet in ~/.neptune/ports/"
        )

    _download_eez_polygons(cache_path)
    return pl.read_parquet(cache_path)


def _download_eez_polygons(dest: Path) -> None:
    """Download EEZ polygon Parquet and verify integrity."""
    import httpx

    logger.info("Downloading EEZ polygons from %s ...", _EEZ_POLYGONS_URL[:80])
    dest.parent.mkdir(parents=True, exist_ok=True)

    resp = httpx.get(_EEZ_POLYGONS_URL, follow_redirects=True, timeout=300.0)
    resp.raise_for_status()
    data = resp.content
    logger.info("Downloaded %d bytes", len(data))

    # Verify SHA-256
    actual_sha = hashlib.sha256(data).hexdigest()
    if actual_sha != _EEZ_POLYGONS_SHA256:
        raise RuntimeError(
            f"EEZ polygon integrity check failed. "
            f"Expected SHA-256: {_EEZ_POLYGONS_SHA256}, "
            f"got: {actual_sha}"
        )

    dest.write_bytes(data)
    logger.info("Cached EEZ polygons to %s", dest)


# ---------------------------------------------------------------------------
# Overlay merging
# ---------------------------------------------------------------------------


def _merge_overlay(
    base: pl.DataFrame,
    overlay: pl.DataFrame,
    *,
    key: str,
) -> pl.DataFrame:
    """Merge an overlay DataFrame into a base DataFrame.

    Rows in the overlay with a matching ``key`` value replace the
    corresponding base rows. Rows with null or negative key values
    are appended as new entries.
    """
    # Separate overlay rows: replacements vs additions
    replacements = overlay.filter(
        pl.col(key).is_not_null() & (pl.col(key) >= 0)
    )
    additions = overlay.filter(
        pl.col(key).is_null() | (pl.col(key) < 0)
    )

    if len(replacements) > 0:
        # Remove matching rows from base, then concat replacements
        replace_keys = replacements[key]
        base = base.filter(~pl.col(key).is_in(replace_keys))
        # Select only columns present in base to prevent schema drift
        common_cols = [c for c in base.columns if c in replacements.columns]
        base = pl.concat([base, replacements.select(common_cols)], how="vertical_relaxed")

    if len(additions) > 0:
        common_cols = [c for c in base.columns if c in additions.columns]
        base = pl.concat([base, additions.select(common_cols)], how="vertical_relaxed")

    return base.sort(key, nulls_last=True)
