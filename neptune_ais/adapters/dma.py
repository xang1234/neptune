"""DMA — Danish Maritime Authority archival adapter.

Daily CSV downloads covering Danish waters, available from 2006 onward.

DMA publishes AIS data as daily CSV files (semicolon-delimited) via
their web portal. Files are typically named ``aisdk-YYYY-MM-DD.csv``
and compressed as ``.zip``.

Coverage: Danish waters / Western Baltic Sea, 2006–present.
Delivery: Daily CSV files, typically available within a few days.

Raw column mapping (DMA → canonical):
    # Timestamp → timestamp (Datetime μs UTC, format "DD/MM/YYYY HH:MM:SS")
    Type of mobile → message_type (not mapped to canonical)
    MMSI → mmsi (Int64)
    Latitude → lat (Float64)
    Longitude → lon (Float64)
    Navigational status → nav_status (String)
    ROT → rot (not mapped to canonical)
    SOG → sog (Float64, 1/10 knot precision)
    COG → cog (Float64, 1/10 degree precision)
    Heading → heading (Float64, 511 → null)
    IMO → imo (String, "0" or "" → null)
    Callsign → callsign (String)
    Name → vessel_name (String)
    Ship type → ship_type (String, numeric code → string)
    Cargo type → cargo_type (not mapped)
    Width → beam (Float64)
    Length → length (Float64)
    Type of position fixing device → pos_fix_device (not mapped)
    Draught → draught (Float64, 1/10 meter)
    Destination → destination (String)
    ETA → eta (Datetime, various formats → null if unparseable)
    Data source type → data_source_type (not mapped)
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
# https://web.ais.dk/aisdata/aisdk-YYYY-MM-DD.zip
BASE_URL = "https://web.ais.dk/aisdata"

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

# DMA timestamp formats (DD/MM/YYYY HH:MM:SS is the primary format).
DMA_TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def _build_url(target_date: date) -> str:
    """Build the DMA download URL for a given date."""
    return f"{BASE_URL}/aisdk-{target_date.isoformat()}.zip"


def _build_filename(target_date: date) -> str:
    """Build the expected filename for a given date."""
    return f"aisdk-{target_date.isoformat()}.zip"


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
            description="Danish waters AIS from DMA, daily CSV archives.",
            supports_backfill=True,
            supports_streaming=False,
            supports_server_side_bbox=False,
            supports_incremental=False,
            auth_scheme=None,
            rate_limit=None,
            expected_latency="2-3 days",
            license_requirements="Open data (Danish open data license)",
            coverage="Danish waters / Western Baltic Sea",
            history_start="2006-01-01",
            datasets_provided=["positions", "vessels"],
            delivery_format="Semicolon-delimited CSV in ZIP",
            typical_daily_rows="1M-5M",
            known_quirks=[
                "Heading=511 means unavailable (normalized to null)",
                "IMO='0' or '' means unavailable (normalized to null)",
                "Timestamp format is DD/MM/YYYY HH:MM:SS",
                "CSV uses semicolon delimiter",
            ],
        )

    def available_dates(self) -> tuple[date, date]:
        """DMA has continuous daily data from 2006-01-01 to ~recently."""
        return (date(2006, 1, 1), date.today())

    def fetch_raw(self, spec: FetchSpec) -> list[RawArtifact]:
        """Download the DMA daily ZIP file for the specified date."""
        if self._download_dir is None:
            raise ValueError(
                "download_dir must be set before fetching. "
                "Pass it via DMAAdapter(download_dir=...)"
            )

        url = _build_url(spec.date)
        dest = self._download_dir / _build_filename(spec.date)

        return [download_and_hash(
            url, dest, overwrite=spec.overwrite, content_type="application/zip",
        )]

    def normalize_positions(
        self, artifacts: list[RawArtifact]
    ) -> pl.DataFrame:
        """Normalize DMA raw CSV data into canonical positions schema.

        DMA CSV files are semicolon-delimited with a specific timestamp
        format (DD/MM/YYYY HH:MM:SS).
        """
        frames: list[pl.DataFrame] = []

        for art in artifacts:
            path = Path(art.local_path)

            if path.suffix == ".zip":
                import io
                import zipfile

                with zipfile.ZipFile(path) as zf:
                    csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                    if not csv_names:
                        raise ValueError(f"No CSV found in {path}")
                    with zf.open(csv_names[0]) as csv_file:
                        raw = pl.read_csv(
                            io.BytesIO(csv_file.read()),
                            separator=";",
                            try_parse_dates=False,
                        )
            elif path.suffix == ".csv":
                raw = pl.read_csv(
                    path,
                    separator=";",
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
