"""NOAA — NOAA Marine Cadastre AIS adapter.

Downloads daily CSV-in-ZIP archives from the Marine Cadastre,
normalizes to the canonical positions and vessels schemas.

NOAA Marine Cadastre serves AIS data as daily CSV-in-ZIP files
for all years (2009–present).

Coverage: U.S. coastal waters, 2009–present.
Delivery: Daily files, typically available next day.

Raw column mapping (NOAA → canonical):
    MMSI → mmsi (Int64)
    BaseDateTime → timestamp (Datetime μs UTC)
    LAT → lat (Float64)
    LON → lon (Float64)
    SOG → sog (Float64)
    COG → cog (Float64)
    Heading → heading (Float64, 511 → null)
    VesselName → vessel_name (String)
    IMO → imo (String, "IMO0000000" → null)
    CallSign → callsign (String)
    VesselType → ship_type (String, numeric code → standardized)
    Status → nav_status (String)
    Length → length (Float64)
    Width → beam (Float64)
    Draft → draught (Float64)
    Cargo → cargo (not mapped to canonical; NOAA-specific)
    TransceiverClass → transceiver_class (not mapped)
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

SOURCE_ID = "noaa"
ADAPTER_VERSION = "noaa/0.1.0"

# NOAA Marine Cadastre base URL pattern.
# https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip
BASE_URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler"

# NOAA raw column names → canonical column names.
COLUMN_MAP: dict[str, str] = {
    "MMSI": "mmsi",
    "BaseDateTime": "timestamp",
    "LAT": "lat",
    "LON": "lon",
    "SOG": "sog",
    "COG": "cog",
    "Heading": "heading",
    "VesselName": "vessel_name",
    "IMO": "imo",
    "CallSign": "callsign",
    "VesselType": "ship_type",
    "Status": "nav_status",
    "Length": "length",
    "Width": "beam",
    "Draft": "draught",
}

# NOAA sentinel values that should be normalized to null.
HEADING_UNAVAILABLE = 511.0
IMO_UNAVAILABLE = "IMO0000000"


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def _build_url(target_date: date) -> str:
    """Build the download URL for a given date."""
    return f"{BASE_URL}/{target_date.year}/{_build_filename(target_date)}"


def _build_filename(target_date: date) -> str:
    """Build the expected filename for a given date."""
    return f"AIS_{target_date.year}_{target_date.month:02d}_{target_date.day:02d}.zip"


# ---------------------------------------------------------------------------
# NOAA adapter
# ---------------------------------------------------------------------------


class NOAAAdapter:
    """NOAA Marine Cadastre AIS source adapter.

    Implements the ``SourceAdapter`` protocol for fetching and
    normalizing NOAA AIS data.
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
            provider="NOAA Marine Cadastre",
            description="U.S. coastal AIS from NOAA Marine Cadastre, daily archives.",
            supports_backfill=True,
            supports_streaming=False,
            supports_server_side_bbox=False,
            supports_incremental=False,
            auth_scheme=None,
            rate_limit=None,
            expected_latency="1 day",
            license_requirements="Public domain (U.S. Government work)",
            coverage="U.S. coastal waters",
            history_start="2009-01-01",
            datasets_provided=["positions", "vessels"],
            delivery_format="CSV-in-ZIP (all years)",
            typical_daily_rows="500K-2M",
            known_quirks=[
                "Heading=511 means unavailable (normalized to null)",
                "IMO='IMO0000000' means unavailable (normalized to null)",
                "VesselType is numeric code (cast to string)",
            ],
        )

    def available_dates(self) -> tuple[date, date]:
        """NOAA has continuous daily data from 2009-01-01 to ~yesterday."""
        return (date(2009, 1, 1), date.today())

    def fetch_raw(self, spec: FetchSpec) -> list[RawArtifact]:
        """Download the NOAA daily file for the specified date."""
        if self._download_dir is None:
            raise ValueError(
                "download_dir must be set before fetching. "
                "Pass it via NOAAAdapter(download_dir=...)"
            )

        url = _build_url(spec.date)
        dest = self._download_dir / _build_filename(spec.date)
        return [download_and_hash(
            url, dest, overwrite=spec.overwrite, content_type="application/zip",
        )]

    def normalize_positions(
        self, artifacts: list[RawArtifact]
    ) -> pl.DataFrame:
        """Normalize NOAA raw data into canonical positions schema.

        Reads daily CSV-in-ZIP archives and applies column renaming,
        type casting, and sentinel normalization.
        """
        import io
        import zipfile

        frames: list[pl.DataFrame] = []

        for art in artifacts:
            path = Path(art.local_path)

            if path.suffix != ".zip":
                raise ValueError(f"Expected .zip file, got: {path.suffix}")

            with zipfile.ZipFile(path) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    raise ValueError(f"No CSV found in {path}")
                with zf.open(csv_names[0]) as csv_file:
                    raw = pl.read_csv(io.BytesIO(csv_file.read()))

            # Rename columns to canonical names.
            rename = {
                src: dst
                for src, dst in COLUMN_MAP.items()
                if src in raw.columns
            }
            df = raw.rename(rename)

            # Select only mapped columns that exist.
            canonical_cols = [c for c in COLUMN_MAP.values() if c in df.columns]
            df = df.select(canonical_cols)

            # Type casting and sentinel normalization.
            df = self._cast_and_normalize(df, art)

            frames.append(df)

        if not frames:
            raise ValueError("No data frames produced from artifacts")

        return pl.concat(frames) if len(frames) > 1 else frames[0]

    def normalize_vessels(
        self, artifacts: list[RawArtifact]
    ) -> pl.DataFrame | None:
        """Extract vessel identity records from NOAA position data."""
        positions = self.normalize_positions(artifacts)
        return extract_vessels(positions, SOURCE_ID)

    def qc_rules(self) -> list:
        """Return NOAA-specific QC rules.

        NOAA data has known quirks:
        - Heading=511 means "not available" (normalized to null)
        - IMO="IMO0000000" means "not available" (normalized to null)
        - VesselType is a numeric code (mapped to string)

        These are handled during normalization (SOURCE_QUIRK class),
        so no additional QC rules are needed beyond the built-ins.
        """
        return []

    # --- Internal helpers ---

    @staticmethod
    def _cast_and_normalize(df: pl.DataFrame, artifact: RawArtifact) -> pl.DataFrame:
        """Apply type casts and sentinel normalization to a raw NOAA frame."""
        exprs: list[pl.Expr] = []

        for col_name in df.columns:
            col = pl.col(col_name)

            if col_name == "mmsi":
                exprs.append(col.cast(pl.Int64, strict=False))
            elif col_name == "timestamp":
                exprs.append(
                    col.str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False)
                    .cast(pl.Datetime("us", "UTC"))
                )
            elif col_name in ("lat", "lon", "sog", "cog", "length", "beam", "draught"):
                exprs.append(col.cast(pl.Float64, strict=False))
            elif col_name == "heading":
                # 511 = not available → null
                heading = col.cast(pl.Float64, strict=False)
                exprs.append(
                    pl.when(heading == HEADING_UNAVAILABLE)
                    .then(None)
                    .otherwise(heading)
                    .alias("heading")
                )
            elif col_name == "imo":
                # "IMO0000000" = not available → null
                imo = col.cast(pl.String)
                exprs.append(
                    pl.when(imo == IMO_UNAVAILABLE)
                    .then(None)
                    .otherwise(imo)
                    .alias("imo")
                )
            elif col_name == "ship_type":
                # NOAA uses numeric vessel type codes; cast to string.
                exprs.append(col.cast(pl.String))
            else:
                exprs.append(col.cast(pl.String, strict=False))

        df = df.with_columns(exprs)

        # Add source provenance (adapter-provided, not pipeline-generated).
        df = df.with_columns(
            pl.lit(SOURCE_ID).alias("source"),
        )

        return df


# ---------------------------------------------------------------------------
# Auto-register on import
# ---------------------------------------------------------------------------

from neptune_ais.adapters.registry import register  # noqa: E402

register(NOAAAdapter)
