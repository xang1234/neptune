"""Finland — Fintraffic Digitraffic marine AIS adapter.

Downloads vessel position data from the Digitraffic REST API,
normalizes to the canonical positions and vessels schemas.

Digitraffic provides:
- Real-time vessel locations via REST API (no auth required)
- Historical vessel locations via date-based queries
- Vessel metadata (name, type, dimensions)

Coverage: Finnish waters and Baltic Sea, 2019–present.
Delivery: REST API JSON (open data, no authentication).
Latency: Near real-time (<1 minute for latest positions).

This is a mixed-delivery source: it supports both archival backfill
via date-based REST queries and near-real-time position snapshots.

Raw column mapping (Digitraffic → canonical):
    mmsi → mmsi (Int64)
    timestampExternal → timestamp (Datetime μs UTC, epoch ms)
    x → lon (Float64)
    y → lat (Float64)
    sog → sog (Float64, knots × 10 → knots)
    cog → cog (Float64, degrees × 10 → degrees)
    heading → heading (Float64, 511 → null)
    name → vessel_name (String)
    imo → imo (Int64, 0 → null)
    callSign → callsign (String)
    destination → destination (String)
    shipType → ship_type (String, numeric → string)
    draught → draught (Float64, meters × 10 → meters)
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

SOURCE_ID = "finland"
ADAPTER_VERSION = "finland/0.1.0"

# Digitraffic marine API base URL.
BASE_URL = "https://meri.digitraffic.fi/api/ais/v1"

# Digitraffic raw field names → canonical column names.
COLUMN_MAP: dict[str, str] = {
    "mmsi": "mmsi",
    "timestampExternal": "timestamp",
    "y": "lat",
    "x": "lon",
    "sog": "sog",
    "cog": "cog",
    "heading": "heading",
    "name": "vessel_name",
    "imo": "imo",
    "callSign": "callsign",
    "destination": "destination",
    "shipType": "ship_type",
    "draught": "draught",
}

# Sentinel values.
HEADING_UNAVAILABLE = 511.0
IMO_UNAVAILABLE = 0

# Digitraffic scales SOG/COG/draught by 10.
DIGITRAFFIC_SCALE_FACTOR = 10.0


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def _build_url(target_date: date) -> str:
    """Build the download URL for a given date's position data."""
    return (
        f"{BASE_URL}/locations/history"
        f"?date={target_date.isoformat()}&format=json"
    )


def _build_filename(target_date: date) -> str:
    """Build the expected filename for a given date."""
    return f"finland_{target_date.isoformat()}.json"


# ---------------------------------------------------------------------------
# Finland adapter
# ---------------------------------------------------------------------------


class FinlandAdapter:
    """Fintraffic Digitraffic marine AIS source adapter.

    Implements the ``SourceAdapter`` protocol for fetching and
    normalizing Finnish maritime AIS data. Digitraffic is open data
    with no authentication required.

    This is a mixed-delivery source supporting both archival backfill
    (date-based REST queries) and near-real-time snapshots.
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
            provider="Fintraffic Digitraffic",
            description=(
                "Finnish waters and Baltic Sea AIS from Digitraffic. "
                "Open data, near-real-time and historical."
            ),
            supports_backfill=True,
            supports_streaming=True,
            supports_server_side_bbox=False,
            supports_incremental=False,
            auth_scheme=None,
            rate_limit=None,
            expected_latency="< 1 minute",
            license_requirements=(
                "CC BY 4.0 — Fintraffic / Digitraffic. "
                "See https://www.digitraffic.fi/en/terms/"
            ),
            coverage="Finnish waters and Baltic Sea",
            history_start="2019-01-01",
            datasets_provided=["positions", "vessels"],
            delivery_format="REST API JSON",
            typical_daily_rows="50K-200K",
            known_quirks=[
                "SOG is knots × 10 (divide by 10)",
                "COG is degrees × 10 (divide by 10)",
                "Draught is meters × 10 (divide by 10)",
                "Heading=511 means unavailable (normalized to null)",
                "IMO=0 means unavailable (normalized to null)",
                "Coordinates: x=longitude, y=latitude (reversed naming)",
                "Timestamps are Unix epoch milliseconds",
                "shipType is numeric (cast to string)",
            ],
        )

    def available_dates(self) -> tuple[date, date]:
        """Digitraffic has data from 2019-01-01 to today."""
        return (date(2019, 1, 1), date.today())

    def fetch_raw(self, spec: FetchSpec) -> list[RawArtifact]:
        """Download Digitraffic data for the specified date."""
        if self._download_dir is None:
            raise ValueError(
                "download_dir must be set before fetching. "
                "Pass it via FinlandAdapter(download_dir=...)"
            )

        url = _build_url(spec.date)
        filename = _build_filename(spec.date)
        dest = self._download_dir / filename

        return [download_and_hash(
            url, dest, overwrite=spec.overwrite, content_type="application/json",
        )]

    def normalize_positions(
        self, artifacts: list[RawArtifact]
    ) -> pl.DataFrame:
        """Normalize Digitraffic raw JSON data into canonical positions schema."""
        frames: list[pl.DataFrame] = []

        for art in artifacts:
            fpath = Path(art.local_path)
            raw_data = json.loads(fpath.read_text())

            # Digitraffic returns a list of location records or a
            # wrapper with a "features" key (GeoJSON style).
            if isinstance(raw_data, dict):
                records = raw_data.get("features", raw_data.get("data", []))
                if not records and "features" not in raw_data and "data" not in raw_data:
                    logger.warning(
                        "Digitraffic response has no 'features' or 'data' key: %s",
                        sorted(raw_data.keys()),
                    )
            elif isinstance(raw_data, list):
                records = raw_data
            else:
                raise ValueError(
                    f"Unexpected Digitraffic response format: {type(raw_data)}"
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

            # Type casting, scaling, and sentinel normalization.
            df = self._cast_and_normalize(df)

            frames.append(df)

        if not frames:
            raise ValueError("No data frames produced from Digitraffic artifacts")

        return pl.concat(frames) if len(frames) > 1 else frames[0]

    def normalize_vessels(
        self, artifacts: list[RawArtifact]
    ) -> pl.DataFrame | None:
        """Extract vessel identity records from Digitraffic position data."""
        positions = self.normalize_positions(artifacts)
        return extract_vessels(positions, SOURCE_ID)

    def qc_rules(self) -> list:
        """Digitraffic-specific QC rules.

        Scaling (SOG/COG/draught ÷ 10) and sentinels (heading=511,
        IMO=0) are handled during normalization. No additional QC
        rules beyond built-ins.
        """
        return []

    # --- Internal helpers ---

    @staticmethod
    def _cast_and_normalize(df: pl.DataFrame) -> pl.DataFrame:
        """Apply type casts, scaling, and sentinel normalization."""
        exprs: list[pl.Expr] = []

        for col_name in df.columns:
            col = pl.col(col_name)

            if col_name == "mmsi":
                exprs.append(col.cast(pl.Int64, strict=False))
            elif col_name == "timestamp":
                # Digitraffic uses Unix epoch milliseconds.
                exprs.append(
                    (col.cast(pl.Int64, strict=False) * 1000)
                    .cast(pl.Datetime("us", "UTC"))
                )
            elif col_name in ("lat", "lon"):
                exprs.append(col.cast(pl.Float64, strict=False))
            elif col_name == "sog":
                # Digitraffic SOG is knots × 10.
                exprs.append(
                    (col.cast(pl.Float64, strict=False) / DIGITRAFFIC_SCALE_FACTOR)
                    .alias("sog")
                )
            elif col_name == "cog":
                # Digitraffic COG is degrees × 10.
                exprs.append(
                    (col.cast(pl.Float64, strict=False) / DIGITRAFFIC_SCALE_FACTOR)
                    .alias("cog")
                )
            elif col_name == "heading":
                # 511 = not available → null.
                exprs.append(
                    pl.when(col.cast(pl.Float64, strict=False) == HEADING_UNAVAILABLE)
                    .then(None)
                    .otherwise(col.cast(pl.Float64, strict=False))
                    .alias("heading")
                )
            elif col_name == "draught":
                # Digitraffic draught is meters × 10.
                exprs.append(
                    (col.cast(pl.Float64, strict=False) / DIGITRAFFIC_SCALE_FACTOR)
                    .alias("draught")
                )
            elif col_name == "imo":
                # IMO=0 or null means unavailable → null.
                casted = col.cast(pl.Int64, strict=False)
                exprs.append(
                    pl.when(casted.is_null() | (casted == IMO_UNAVAILABLE))
                    .then(pl.lit(None, dtype=pl.String))
                    .otherwise(col.cast(pl.String))
                    .alias("imo")
                )
            elif col_name == "ship_type":
                # Digitraffic uses numeric vessel type codes.
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

register(FinlandAdapter)
