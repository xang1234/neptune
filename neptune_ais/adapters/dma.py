"""DMA — Danish Maritime Authority archival adapter.

CSV downloads covering Danish waters, available from 2006 onward.

DMA publishes AIS data as semicolon-delimited CSV files in ZIP
archives via an S3-hosted portal at ``aisdata.ais.dk``.

Data granularity changed over time:
- 2006 – 2024 Feb: **Monthly** files (~17 GB each)
  ``/{year}/aisdk-{year}-{month:02d}.zip``
- 2024 Mar – present: **Daily** files (~600 MB each)
  ``/{year}/aisdk-{year}-{month:02d}-{day:02d}.zip``

When a user requests a specific date that falls within a monthly
archive, the adapter downloads the full month and filters to the
requested date during normalization.

Coverage: Danish waters / Western Baltic Sea, 2006–present.
Delivery: Monthly or daily CSV-in-ZIP (see above).

Raw column mapping (DMA → canonical):
    Timestamp → timestamp (Datetime μs UTC, format "DD/MM/YYYY HH:MM:SS")
    Type of mobile → (not mapped to canonical)
    MMSI → mmsi (Int64)
    Latitude → lat (Float64)
    Longitude → lon (Float64)
    Navigational status → nav_status (String)
    ROT → (not mapped to canonical)
    SOG → sog (Float64)
    COG → cog (Float64)
    Heading → heading (Float64, 511 → null)
    IMO → imo (String, "0" or "" → null)
    Callsign → callsign (String)
    Name → vessel_name (String)
    Ship type → ship_type (String)
    Cargo type → (not mapped)
    Width → beam (Float64)
    Length → length (Float64)
    Type of position fixing device → (not mapped)
    Draught → draught (Float64)
    Destination → destination (String)
    ETA → (not mapped)
    Data source type → (not mapped)
    Size A/B/C/D → (not mapped, GPS-to-hull offsets)
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import polars as pl

from neptune_ais.adapters.base import (
    FetchSpec,
    RawArtifact,
    SourceAdapter,
    SourceCapabilities,
    download_and_hash,
    extract_vessels,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_ID = "dma"
ADAPTER_VERSION = "dma/0.1.0"

# DMA download URL pattern.
# Daily:   http://aisdata.ais.dk/{year}/aisdk-YYYY-MM-DD.zip
# Monthly: http://aisdata.ais.dk/{year}/aisdk-YYYY-MM.zip
BASE_URL = "http://aisdata.ais.dk"

# Daily files are available from 2024-03-01 onward.
# Before that, data is published as monthly archives.
DAILY_START = date(2024, 3, 1)

# DMA raw column names → canonical column names.
COLUMN_MAP: dict[str, str] = {
    "# Timestamp": "timestamp",
    "Timestamp": "timestamp",  # alternate header without #
    "MMSI": "mmsi",
    "Latitude": "lat",
    "Longitude": "lon",
    "SOG": "sog",
    "COG": "cog",
    "Heading": "heading",
    "IMO": "imo",
    "Callsign": "callsign",
    "Name": "vessel_name",
    "Ship type": "ship_type",
    "Navigational status": "nav_status",
    "Width": "beam",
    "Length": "length",
    "Draught": "draught",
    "Destination": "destination",
}

# DMA sentinel values.
HEADING_UNAVAILABLE = 511.0
IMO_UNAVAILABLE_VALUES = {"0", "", "Unknown"}

# DMA files used semicolons historically (web.ais.dk) but switched to
# commas (aisdata.ais.dk, 2024+). Auto-detect from the header line.
_DMA_DELIMITERS = {";", ","}

# DMA timestamp formats (DD/MM/YYYY HH:MM:SS is the primary format).
DMA_TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def _detect_separator(data: bytes) -> str:
    """Detect CSV separator from the first line of data.

    Historical DMA files (web.ais.dk) used semicolons. Newer files
    (aisdata.ais.dk, 2024+) use commas. Detect from the header line.
    """
    first_line = data.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    semicolons = first_line.count(";")
    commas = first_line.count(",")
    return ";" if semicolons > commas else ","


def _is_daily(target_date: date) -> bool:
    """Return True if daily files exist for this date."""
    return target_date >= DAILY_START


def _build_url(target_date: date) -> str:
    """Build the DMA download URL for a given date.

    Daily dates (>= 2024-03-01) use per-day files.
    Earlier dates use monthly archives.
    """
    return f"{BASE_URL}/{target_date.year}/{_build_filename(target_date)}"


def _build_filename(target_date: date) -> str:
    """Build the expected local filename for a given date."""
    if _is_daily(target_date):
        return f"aisdk-{target_date.isoformat()}.zip"
    return f"aisdk-{target_date.year}-{target_date.month:02d}.zip"


# ---------------------------------------------------------------------------
# DMA adapter
# ---------------------------------------------------------------------------


class DMAAdapter:
    """Danish Maritime Authority AIS source adapter.

    Implements the ``SourceAdapter`` protocol for fetching and
    normalizing DMA AIS data.
    """

    SOURCE_ID = SOURCE_ID

    def __init__(self, download_dir: Path | None = None) -> None:
        self._download_dir = download_dir

    @property
    def source_id(self) -> str:
        return SOURCE_ID

    @property
    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            source_id=SOURCE_ID,
            provider="Danish Maritime Authority",
            description="Danish waters AIS from DMA, daily/monthly CSV archives.",
            supports_backfill=True,
            supports_streaming=False,
            supports_server_side_bbox=False,
            supports_incremental=False,
            auth_scheme=None,
            rate_limit=None,
            expected_latency="2-3 days",
            license_requirements="Open data (Danish open data license)",
            coverage="Danish waters / Western Baltic Sea",
            history_start="2006-03-01",
            datasets_provided=["positions", "vessels"],
            delivery_format="Semicolon-delimited CSV in ZIP (monthly pre-2024-03, daily after)",
            typical_daily_rows="1M-5M",
            known_quirks=[
                "Heading=511 means unavailable (normalized to null)",
                "IMO='0' or '' means unavailable (normalized to null)",
                "Timestamp format is DD/MM/YYYY HH:MM:SS",
                "CSV delimiter varies: semicolon (historical) or comma (2024+)",
                "Pre-2024-03 data is monthly (~17 GB per file)",
            ],
        )

    def available_dates(self) -> tuple[date, date]:
        """DMA has data from 2006-03 to ~yesterday (monthly before 2024-03, daily after)."""
        return (date(2006, 3, 1), date.today())

    def fetch_raw(self, spec: FetchSpec) -> list[RawArtifact]:
        """Download the DMA ZIP file for the specified date.

        For dates before 2024-03-01, downloads the monthly archive
        (can be ~17 GB). The file is cached, so subsequent requests
        for other dates in the same month reuse the download.
        """
        if self._download_dir is None:
            raise ValueError(
                "download_dir must be set before fetching. "
                "Pass it via DMAAdapter(download_dir=...)"
            )

        url = _build_url(spec.date)
        dest = self._download_dir / _build_filename(spec.date)

        if not _is_daily(spec.date) and not (dest.exists() and not spec.overwrite):
            logger.warning(
                "DMA data before 2024-03-01 is published as monthly archives "
                "(~17 GB). Downloading %s-%02d month file; will filter to %s "
                "during normalization.",
                spec.date.year, spec.date.month, spec.date.isoformat(),
            )

        artifact = download_and_hash(
            url, dest, overwrite=spec.overwrite, content_type="application/zip",
        )
        # Tag monthly artifacts with the requested date so
        # normalize_positions can filter to just the requested day.
        if not _is_daily(spec.date):
            artifact.headers["x-neptune-request-date"] = spec.date.isoformat()
        return [artifact]

    def normalize_positions(
        self, artifacts: list[RawArtifact]
    ) -> pl.DataFrame:
        """Normalize DMA raw CSV data into canonical positions schema.

        DMA CSV files are semicolon-delimited with a specific timestamp
        format (DD/MM/YYYY HH:MM:SS). For monthly archives, filters to
        the requested date after parsing.
        """
        import io
        import zipfile

        frames: list[pl.DataFrame] = []

        for art in artifacts:
            path = Path(art.local_path)

            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as zf:
                    csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                    if not csv_names:
                        raise ValueError(f"No CSV found in {path}")
                    with zf.open(csv_names[0]) as csv_file:
                        csv_bytes = csv_file.read()
                        sep = _detect_separator(csv_bytes)
                        raw = pl.read_csv(
                            io.BytesIO(csv_bytes),
                            separator=sep,
                            try_parse_dates=False,
                        )
            elif path.suffix == ".csv":
                with open(path, "rb") as f:
                    first_line = f.readline()
                sep = _detect_separator(first_line)
                raw = pl.read_csv(
                    path,
                    separator=sep,
                    try_parse_dates=False,
                )
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

            # Rename columns to canonical names.
            rename = {
                src: dst
                for src, dst in COLUMN_MAP.items()
                if src in raw.columns
            }
            df = raw.rename(rename)

            # Select only mapped columns that exist.
            canonical_cols = [c for c in COLUMN_MAP.values() if c in df.columns]
            # Deduplicate (COLUMN_MAP has two entries for timestamp).
            canonical_cols = list(dict.fromkeys(canonical_cols))
            df = df.select(canonical_cols)

            # Type casting and sentinel normalization.
            df = self._cast_and_normalize(df)

            # For monthly archives, filter to the requested date.
            request_date_str = art.headers.get("x-neptune-request-date")
            if request_date_str:
                request_date = date.fromisoformat(request_date_str)
                before_count = len(df)
                df = df.filter(
                    pl.col("timestamp").dt.date() == request_date
                )
                logger.info(
                    "Filtered monthly archive to %s: %d → %d rows",
                    request_date_str, before_count, len(df),
                )

            frames.append(df)

        if not frames:
            raise ValueError("No data frames produced from artifacts")

        return pl.concat(frames) if len(frames) > 1 else frames[0]

    def normalize_vessels(
        self, artifacts: list[RawArtifact]
    ) -> pl.DataFrame | None:
        """Extract vessel identity records from DMA position data."""
        positions = self.normalize_positions(artifacts)
        return extract_vessels(positions, SOURCE_ID)

    def qc_rules(self) -> list:
        """Return DMA-specific QC rules.

        DMA data shares the same AIS sentinel conventions as NOAA
        (heading=511 → null, IMO="0" → null). These are handled during
        normalization. No additional QC rules beyond built-ins.
        """
        return []

    # --- Internal helpers ---

    @staticmethod
    def _cast_and_normalize(df: pl.DataFrame) -> pl.DataFrame:
        """Apply type casts and sentinel normalization to a raw DMA frame."""
        exprs: list[pl.Expr] = []

        for col_name in df.columns:
            col = pl.col(col_name)

            if col_name == "mmsi":
                exprs.append(col.cast(pl.Int64, strict=False))
            elif col_name == "timestamp":
                # DMA uses DD/MM/YYYY HH:MM:SS format.
                exprs.append(
                    col.str.to_datetime(DMA_TIMESTAMP_FORMAT, strict=False)
                    .cast(pl.Datetime("us", "UTC"))
                )
            elif col_name in ("lat", "lon", "sog", "cog", "length", "beam", "draught"):
                exprs.append(col.cast(pl.Float64, strict=False))
            elif col_name == "heading":
                exprs.append(
                    pl.when(col.cast(pl.Float64, strict=False) == HEADING_UNAVAILABLE)
                    .then(None)
                    .otherwise(col.cast(pl.Float64, strict=False))
                    .alias("heading")
                )
            elif col_name == "imo":
                # DMA uses "0", "", or "Unknown" for unavailable IMO.
                exprs.append(
                    pl.when(col.cast(pl.String).is_in(IMO_UNAVAILABLE_VALUES))
                    .then(None)
                    .otherwise(col.cast(pl.String))
                    .alias("imo")
                )
            elif col_name == "ship_type":
                exprs.append(col.cast(pl.String))
            else:
                exprs.append(col.cast(pl.String, strict=False))

        df = df.with_columns(exprs)

        # Add source provenance.
        df = df.with_columns(
            pl.lit(SOURCE_ID).alias("source"),
        )

        return df


# ---------------------------------------------------------------------------
# Auto-register on import
# ---------------------------------------------------------------------------

from neptune_ais.adapters.registry import register  # noqa: E402

register(DMAAdapter)
