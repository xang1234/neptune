"""AISHub — AISHub community-sourced AIS adapter.

Downloads vessel position data from the AISHub API, normalizes to the
canonical positions and vessels schemas.

AISHub provides:
- Community-aggregated AIS positions from terrestrial receivers worldwide
- Variable quality: receiver coverage gaps, clock drift, GPS anomalies
- Near-real-time snapshots and limited historical data

Coverage: Global terrestrial (dependent on community receiver network).
Delivery: REST API returning CSV or XML, API key required.
Latency: Near real-time for latest positions.

Raw column mapping (AISHub → canonical):
    MMSI → mmsi (Int64)
    TIME → timestamp (Datetime, ISO-8601)
    LATITUDE → lat (Float64)
    LONGITUDE → lon (Float64)
    SPEED → sog (Float64, knots × 10 → knots)
    COURSE → cog (Float64, degrees × 10 → degrees)
    HEADING → heading (Float64, 511 → null)
    NAME → vessel_name (String)
    IMO → imo (String, "0" → null)
    CALLSIGN → callsign (String)
    TYPE → ship_type (String, numeric → string)
    A + B → length (Float64, meters, derived from A+B)
    C + D → beam (Float64, meters, derived from C+D)
    DRAUGHT → draught (Float64, meters × 10 → meters)
    DEST → destination (String)
    FLAG → flag (String, ISO country code)
    NAVSTAT → nav_status (String, numeric → string)
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
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

SOURCE_ID = "aishub"
ADAPTER_VERSION = "aishub/0.1.0"

# AISHub API base URL.
BASE_URL = "https://data.aishub.net/ws.php"

# AISHub raw field names → canonical column names.
COLUMN_MAP: dict[str, str] = {
    "MMSI": "mmsi",
    "TIME": "timestamp",
    "LATITUDE": "lat",
    "LONGITUDE": "lon",
    "SPEED": "sog",
    "COURSE": "cog",
    "HEADING": "heading",
    "NAME": "vessel_name",
    "IMO": "imo",
    "CALLSIGN": "callsign",
    "TYPE": "ship_type",
    "DRAUGHT": "draught",
    "DEST": "destination",
    "FLAG": "flag",
    "NAVSTAT": "nav_status",
    "A": "_dim_a",
    "B": "_dim_b",
    "C": "_dim_c",
    "D": "_dim_d",
}

# Sentinel values.
HEADING_UNAVAILABLE = 511.0
IMO_UNAVAILABLE_VALUES = {"0", "", "Unknown"}

# AISHub scales SOG/COG/draught by 10 (same as raw AIS encoding).
AISHUB_SCALE_FACTOR = 10.0


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def _build_url(api_key: str) -> str:
    """Build the AISHub API URL for latest positions."""
    return f"{BASE_URL}?username={api_key}&format=1&output=json&compress=0"


def _build_filename(target_date: date) -> str:
    """Build the expected filename for a given date."""
    return f"aishub_{target_date.isoformat()}.json"


# ---------------------------------------------------------------------------
# AISHub adapter
# ---------------------------------------------------------------------------


class AISHubAdapter:
    """AISHub community-sourced AIS adapter.

    Implements the ``SourceAdapter`` protocol for fetching and
    normalizing AISHub data. AISHub is a community aggregator of
    terrestrial AIS receivers with variable quality characteristics.

    Quality caveats are made explicit in capabilities and known_quirks
    rather than hidden behind generic normalization.
    """

    SOURCE_ID = SOURCE_ID

    def __init__(
        self,
        download_dir: Path | None = None,
        *,
        api_key: str = "",
    ) -> None:
        self._download_dir = download_dir
        self._api_key = api_key

    @property
    def source_id(self) -> str:
        return SOURCE_ID

    @property
    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            source_id=SOURCE_ID,
            provider="AISHub",
            description=(
                "Community-sourced global AIS from terrestrial receivers. "
                "Variable quality depending on receiver network coverage."
            ),
            supports_backfill=False,
            supports_streaming=False,
            supports_server_side_bbox=True,
            supports_incremental=True,
            auth_scheme="api_key",
            rate_limit="1 req/min",
            expected_latency="< 1 minute",
            license_requirements=(
                "AISHub membership required. Data sharing reciprocity. "
                "See https://www.aishub.net/ais-dispatcher"
            ),
            coverage="Global terrestrial (community receivers)",
            history_start=None,  # No guaranteed history
            datasets_provided=["positions", "vessels"],
            delivery_format="REST API JSON",
            typical_daily_rows="50K-500K",
            known_quirks=[
                "Community-sourced: variable receiver quality and coverage",
                "SPEED/COURSE are × 10 (raw AIS encoding, divide by 10)",
                "DRAUGHT is × 10 (divide by 10)",
                "Heading=511 means unavailable (normalized to null)",
                "IMO='0' or empty means unavailable (normalized to null)",
                "Dimensions A+B=length, C+D=beam (derived fields)",
                "Timestamps may have clock drift from community receivers",
                "No guaranteed historical backfill — snapshot-oriented API",
                "GPS spoofing and position anomalies more common than institutional sources",
            ],
        )

    def available_dates(self) -> None:
        """AISHub is snapshot-based with no date enumeration."""
        return None

    def fetch_raw(self, spec: FetchSpec) -> list[RawArtifact]:
        """Download AISHub snapshot data."""
        if self._download_dir is None:
            raise ValueError(
                "download_dir must be set before fetching. "
                "Pass it via AISHubAdapter(download_dir=...)"
            )
        if not self._api_key:
            raise ValueError(
                "api_key must be set before fetching. "
                "Pass it via AISHubAdapter(api_key=...)"
            )

        url = _build_url(self._api_key)
        filename = _build_filename(spec.date)
        dest = self._download_dir / filename

        return [download_and_hash(
            url, dest, overwrite=spec.overwrite, content_type="application/json",
        )]

    def normalize_positions(
        self, artifacts: list[RawArtifact]
    ) -> pl.DataFrame:
        """Normalize AISHub raw JSON data into canonical positions schema."""
        frames: list[pl.DataFrame] = []

        for art in artifacts:
            fpath = Path(art.local_path)
            raw_data = json.loads(fpath.read_text())

            # AISHub returns [{...}, ...] or [metadata, [{...}, ...]].
            if isinstance(raw_data, list) and len(raw_data) == 2 and isinstance(raw_data[1], list):
                records = raw_data[1]
            elif isinstance(raw_data, list):
                records = raw_data
            elif isinstance(raw_data, dict):
                records = raw_data.get("data", raw_data.get("entries", []))
                if not records and "data" not in raw_data and "entries" not in raw_data:
                    logger.warning(
                        "AISHub response has no 'data' or 'entries' key: %s",
                        sorted(raw_data.keys()),
                    )
            else:
                raise ValueError(
                    f"Unexpected AISHub response format: {type(raw_data)}"
                )

            if not records:
                continue

            raw = pl.DataFrame(records)

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

            # Derive length and beam from dimension fields if present.
            df = self._derive_dimensions(df)

            # Type casting, scaling, and sentinel normalization.
            df = self._cast_and_normalize(df)

            frames.append(df)

        if not frames:
            raise ValueError("No data frames produced from AISHub artifacts")

        return pl.concat(frames) if len(frames) > 1 else frames[0]

    def normalize_vessels(
        self, artifacts: list[RawArtifact]
    ) -> pl.DataFrame | None:
        """Extract vessel identity records from AISHub position data."""
        positions = self.normalize_positions(artifacts)
        return extract_vessels(positions, SOURCE_ID)

    def qc_rules(self) -> list:
        """AISHub-specific QC rules.

        AISHub data has higher quality variance than institutional
        sources. Scaling and sentinels are handled during normalization.
        Known quirks (clock drift, GPS anomalies) are documented in
        capabilities but not automatically correctable.
        """
        return []

    # --- Internal helpers ---

    @staticmethod
    def _derive_dimensions(df: pl.DataFrame) -> pl.DataFrame:
        """Derive length (A+B) and beam (C+D) from AIS dimension fields."""
        if "_dim_a" in df.columns and "_dim_b" in df.columns:
            df = df.with_columns(
                (pl.col("_dim_a").cast(pl.Float64, strict=False)
                 + pl.col("_dim_b").cast(pl.Float64, strict=False))
                .alias("length")
            ).drop(["_dim_a", "_dim_b"])
        else:
            for col in ("_dim_a", "_dim_b"):
                if col in df.columns:
                    df = df.drop(col)

        if "_dim_c" in df.columns and "_dim_d" in df.columns:
            df = df.with_columns(
                (pl.col("_dim_c").cast(pl.Float64, strict=False)
                 + pl.col("_dim_d").cast(pl.Float64, strict=False))
                .alias("beam")
            ).drop(["_dim_c", "_dim_d"])
        else:
            for col in ("_dim_c", "_dim_d"):
                if col in df.columns:
                    df = df.drop(col)

        return df

    @staticmethod
    def _cast_and_normalize(df: pl.DataFrame) -> pl.DataFrame:
        """Apply type casts, scaling, and sentinel normalization."""
        exprs: list[pl.Expr] = []

        for col_name in df.columns:
            col = pl.col(col_name)

            if col_name == "mmsi":
                exprs.append(col.cast(pl.Int64, strict=False))
            elif col_name == "timestamp":
                exprs.append(
                    col.str.to_datetime(
                        "%Y-%m-%dT%H:%M:%S%#z", strict=False
                    ).cast(pl.Datetime("us", "UTC"))
                )
            elif col_name in ("lat", "lon"):
                exprs.append(col.cast(pl.Float64, strict=False))
            elif col_name in ("sog", "cog", "draught"):
                # AISHub sends SOG/COG/draught × 10 (raw AIS encoding).
                exprs.append(
                    (col.cast(pl.Float64, strict=False) / AISHUB_SCALE_FACTOR)
                    .alias(col_name)
                )
            elif col_name == "heading":
                # 511 = not available → null.
                float_col = col.cast(pl.Float64, strict=False)
                exprs.append(
                    pl.when(float_col == HEADING_UNAVAILABLE)
                    .then(None)
                    .otherwise(float_col)
                    .alias("heading")
                )
            elif col_name in ("length", "beam"):
                exprs.append(col.cast(pl.Float64, strict=False))
            elif col_name == "imo":
                # "0", "", "Unknown" mean unavailable → null.
                str_col = col.cast(pl.String, strict=False)
                exprs.append(
                    pl.when(str_col.is_null() | str_col.is_in(IMO_UNAVAILABLE_VALUES))
                    .then(pl.lit(None, dtype=pl.String))
                    .otherwise(str_col)
                    .alias("imo")
                )
            elif col_name == "ship_type":
                exprs.append(col.cast(pl.String))
            elif col_name == "nav_status":
                exprs.append(col.cast(pl.String))
            else:
                exprs.append(col.cast(pl.String, strict=False))

        exprs.append(pl.lit(SOURCE_ID).alias("source"))
        return df.with_columns(exprs)


# ---------------------------------------------------------------------------
# Auto-register on import
# ---------------------------------------------------------------------------

from neptune_ais.adapters.registry import register  # noqa: E402

register(AISHubAdapter)
