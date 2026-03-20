"""Base — SourceAdapter protocol and shared adapter types.

Defines the contract that all source adapters must satisfy, including
fetch, normalize, and QC rule hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

import polars as pl


# ---------------------------------------------------------------------------
# AIS protocol constants — shared across adapters
# ---------------------------------------------------------------------------

AIS_NAV_STATUS: dict[int, str] = {
    0: "Under way using engine",
    1: "At anchor",
    2: "Not under command",
    3: "Restricted maneuverability",
    4: "Constrained by draught",
    5: "Moored",
    6: "Aground",
    7: "Engaged in fishing",
    8: "Under way sailing",
    9: "Reserved for HSC",
    10: "Reserved for WIG",
    14: "AIS-SART",
    15: "Not defined",
}


# ---------------------------------------------------------------------------
# Fetch specification — what to download
# ---------------------------------------------------------------------------


@dataclass
class FetchSpec:
    """Specification for a single fetch operation.

    Passed to ``SourceAdapter.fetch_raw()`` to describe what data to
    retrieve from the source.
    """

    date: date
    """The date to fetch data for."""

    bbox: tuple[float, float, float, float] | None = None
    """Optional bounding box filter (west, south, east, north).
    Only used by sources that support server-side spatial filtering."""

    overwrite: bool = False
    """If True, re-download even if raw artifacts already exist locally."""


# ---------------------------------------------------------------------------
# Raw artifact — returned by fetch_raw()
# ---------------------------------------------------------------------------


@dataclass
class RawArtifact:
    """A raw source artifact returned by an adapter's fetch operation.

    This is the adapter-side representation. The orchestration layer
    converts these to ``catalog.RawArtifact`` Pydantic models for
    manifest storage.
    """

    source_url: str
    """Where the data was fetched from."""

    filename: str
    """Original filename."""

    local_path: str
    """Absolute path to the downloaded file on disk."""

    content_hash: str
    """SHA-256 hex digest of the file contents."""

    size_bytes: int
    """File size in bytes."""

    fetch_timestamp: datetime
    """When the file was fetched."""

    content_type: str | None = None
    """MIME type or format hint."""

    headers: dict[str, str] = field(default_factory=dict)
    """Captured HTTP response headers."""


# ---------------------------------------------------------------------------
# Shared fetch helper
# ---------------------------------------------------------------------------


def download_and_hash(
    url: str,
    dest: Path,
    *,
    overwrite: bool = False,
    content_type: str | None = None,
    headers: dict[str, str] | None = None,
    retries: int = 3,
) -> RawArtifact:
    """Download a file and return a RawArtifact with its SHA-256 hash.

    Shared by all HTTP-based adapters. Handles caching (skip if exists
    and not overwrite), streaming download with retries, and hash
    computation.

    Args:
        headers: Optional HTTP headers (e.g. for API key authentication).
        retries: Number of retry attempts for transient network errors.
    """
    import hashlib
    import logging
    import time

    import httpx

    logger = logging.getLogger(__name__)

    content_hash: str | None = None

    if dest.exists() and not overwrite:
        logger.info("Using cached raw file: %s", dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_suffix(dest.suffix + ".partial")

        timeout = httpx.Timeout(connect=30, read=120, write=30, pool=30)
        last_exc: Exception | None = None
        max_wait = 60

        for attempt in range(1, retries + 1):
            logger.info(
                "Downloading %s → %s (attempt %d/%d)", url, dest, attempt, retries,
            )
            try:
                sha256 = hashlib.sha256()
                size_bytes = 0
                with httpx.stream(
                    "GET", url,
                    follow_redirects=True,
                    headers=headers or {},
                    timeout=timeout,
                ) as resp:
                    resp.raise_for_status()
                    with open(partial, "wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=65536):
                            f.write(chunk)
                            sha256.update(chunk)
                            size_bytes += len(chunk)
                partial.rename(dest)
                content_hash = sha256.hexdigest()
                logger.info("Downloaded %s (%d bytes)", dest.name, size_bytes)
                break
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
                partial.unlink(missing_ok=True)
                if (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code < 500
                ):
                    raise  # 4xx errors are not transient
                logger.warning(
                    "Download failed (attempt %d/%d): %s", attempt, retries, exc,
                )
                if attempt < retries:
                    wait = min(2 ** attempt, max_wait)
                    logger.warning("Retrying in %ds", wait)
                    time.sleep(wait)
        else:
            raise RuntimeError(
                f"Download failed after {retries} attempts ({last_exc}): {url}"
            ) from last_exc

    # For cache hits, hash must be computed from the existing file.
    if content_hash is None:
        sha256 = hashlib.sha256()
        with open(dest, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        content_hash = sha256.hexdigest()

    return RawArtifact(
        source_url=url,
        filename=dest.name,
        local_path=str(dest),
        content_hash=content_hash,
        size_bytes=dest.stat().st_size,
        fetch_timestamp=datetime.now(timezone.utc),
        content_type=content_type,
    )


# ---------------------------------------------------------------------------
# Shared vessel extraction helper
# ---------------------------------------------------------------------------


def extract_vessels(
    positions_df: pl.DataFrame,
    source_id: str,
    vessel_cols: list[str] | None = None,
) -> pl.DataFrame | None:
    """Extract vessel identity records from a positions DataFrame.

    Shared by all adapters that embed vessel identity in position reports.
    Groups by MMSI, takes the last non-null value per identity field, and
    computes first/last seen timestamps.
    """
    if vessel_cols is None:
        vessel_cols = [
            "mmsi", "imo", "callsign", "vessel_name", "ship_type",
            "length", "beam",
        ]

    available = [c for c in vessel_cols if c in positions_df.columns]
    if "mmsi" not in available:
        return None

    vessels = (
        positions_df
        .select(available + ["timestamp"])
        .group_by("mmsi")
        .agg([
            pl.col(c).drop_nulls().last().alias(c)
            for c in available if c != "mmsi"
        ] + [
            pl.col("timestamp").min().alias("first_seen"),
            pl.col("timestamp").max().alias("last_seen"),
        ])
    )

    vessels = vessels.with_columns(
        pl.lit(source_id).alias("source"),
        pl.lit(f"{source_id}:direct").alias("record_provenance"),
    )

    return vessels


# ---------------------------------------------------------------------------
# Source capabilities — self-describing adapter metadata
# ---------------------------------------------------------------------------


@dataclass
class SourceCapabilities:
    """Metadata describing what a source adapter can do.

    Exposed via ``neptune_ais.sources.capabilities("noaa")``.
    Used for source discovery, fusion planning, and CLI reporting.
    """

    source_id: str
    """Unique identifier for this source, e.g. 'noaa'."""

    provider: str
    """Human-readable provider name."""

    description: str
    """Short description of the source."""

    # --- capability flags ---

    supports_backfill: bool = False
    """Whether historical date-range downloads are supported."""

    supports_streaming: bool = False
    """Whether real-time streaming is supported."""

    supports_server_side_bbox: bool = False
    """Whether the source API supports spatial filtering."""

    supports_incremental: bool = False
    """Whether incremental/delta downloads are supported."""

    # --- operational metadata ---

    auth_scheme: str | None = None
    """Authentication requirement, e.g. 'api_key', 'oauth2', None."""

    rate_limit: str | None = None
    """Rate limit description, e.g. '100 req/min'."""

    expected_latency: str | None = None
    """Typical data availability delay, e.g. '1 day', 'real-time'."""

    license_requirements: str | None = None
    """License or attribution requirements."""

    # --- coverage ---

    coverage: str = ""
    """Geographic coverage description."""

    history_start: str | None = None
    """Earliest available date, e.g. '2009-01-01'."""

    # --- data characteristics ---

    datasets_provided: list[str] = field(default_factory=lambda: ["positions"])
    """Which canonical datasets this source can produce.
    Most sources produce 'positions'; some also produce 'vessels' or 'events'."""

    delivery_format: str = ""
    """Raw data format, e.g. 'GeoParquet', 'CSV-in-ZIP', 'WebSocket JSON'."""

    typical_daily_rows: str | None = None
    """Approximate row count per day, e.g. '500K-2M', '10M+'."""

    known_quirks: list[str] = field(default_factory=list)
    """Source-specific data quirks that normalization handles.
    E.g. ['heading=511 means unavailable', 'IMO field uses IMO0000000 sentinel']."""

    def summary(self) -> dict[str, str]:
        """Return a flat dict summary suitable for table display."""
        return {
            "source": self.source_id,
            "provider": self.provider,
            "coverage": self.coverage,
            "history": self.history_start or "unknown",
            "backfill": "yes" if self.supports_backfill else "no",
            "streaming": "yes" if self.supports_streaming else "no",
            "bbox": "yes" if self.supports_server_side_bbox else "no",
            "auth": self.auth_scheme or "none",
            "latency": self.expected_latency or "unknown",
            "format": self.delivery_format,
            "datasets": ", ".join(self.datasets_provided),
            "license": self.license_requirements or "unknown",
        }


# ---------------------------------------------------------------------------
# Source adapter protocol — the contract
# ---------------------------------------------------------------------------


@runtime_checkable
class SourceAdapter(Protocol):
    """Protocol that all source adapters must implement.

    Adapters are responsible for:
    1. Declaring their capabilities (``capabilities``).
    2. Fetching raw data (``fetch_raw``).
    3. Normalizing to canonical schemas (``normalize_positions``,
       ``normalize_vessels``).
    4. Supplying source-specific QC rules (``qc_rules``).

    Adapters must NOT call storage or catalog directly. They return
    DataFrames and RawArtifacts to the orchestration layer in ``api``.
    """

    @property
    def source_id(self) -> str:
        """Unique identifier for this source."""
        ...

    @property
    def capabilities(self) -> SourceCapabilities:
        """Self-describing metadata about this adapter."""
        ...

    def available_dates(self) -> list[date] | tuple[date, date] | None:
        """Return available date range or list of dates.

        Returns:
            - A list of specific available dates, or
            - A (start, end) tuple for a continuous range, or
            - None if the source doesn't support date enumeration.
        """
        ...

    def fetch_raw(self, spec: FetchSpec) -> list[RawArtifact]:
        """Fetch raw source data for the given specification.

        Returns a list of RawArtifact records pointing to downloaded
        files on disk. The orchestration layer handles retention policy.
        """
        ...

    def normalize_positions(
        self, artifacts: list[RawArtifact]
    ) -> pl.DataFrame:
        """Normalize raw artifacts into the canonical positions schema.

        Returns a DataFrame conforming to
        ``datasets.positions.SCHEMA``. The orchestration layer adds
        pipeline-generated columns (source_file, ingest_id, qc_severity,
        record_provenance) after normalization.
        """
        ...

    def normalize_vessels(
        self, artifacts: list[RawArtifact]
    ) -> pl.DataFrame | None:
        """Normalize raw artifacts into the canonical vessels schema.

        Returns a DataFrame conforming to ``datasets.vessels.SCHEMA``,
        or None if this source does not provide vessel identity data
        separately from position reports.
        """
        ...

    def qc_rules(self) -> list:
        """Return source-specific QC checks.

        Returns a list of objects satisfying the ``qc.QCCheck`` protocol.
        These are appended to the built-in checks during the ingest
        pipeline. Return an empty list if no source-specific rules.
        """
        ...
