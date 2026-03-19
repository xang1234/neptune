"""GFW — Global Fishing Watch adapter.

Downloads vessel position and identity data from the GFW API, normalizes
to the canonical positions and vessels schemas.

GFW provides:
- Vessel identity (MMSI, IMO, name, flag, type, length)
- AIS-derived positions (lat, lon, timestamp, speed, course)
- Fishing activity classifications (not mapped to positions; future events)

Coverage: Global, 2012–present.
Delivery: REST API with API key authentication, JSON responses.
Latency: 3+ days (processing pipeline delay).

Raw column mapping (GFW → canonical):
    mmsi → mmsi (Int64)
    timestamp → timestamp (Datetime μs UTC)
    lat → lat (Float64)
    lon → lon (Float64)
    speed → sog (Float64, knots)
    course → cog (Float64, degrees)
    heading → heading (Float64, null if absent)
    shipname → vessel_name (String)
    imo → imo (String, null if 0 or absent)
    callsign → callsign (String)
    flag → flag (String, ISO 3166-1 alpha-3)
    vessel_type → ship_type (String)
    length → length (Float64, meters)
"""

from __future__ import annotations

import json
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

SOURCE_ID = "gfw"
ADAPTER_VERSION = "gfw/0.1.0"

# GFW API base URL.
BASE_URL = "https://gateway.api.globalfishingwatch.org/v3"

# GFW raw field names → canonical column names.
COLUMN_MAP: dict[str, str] = {
    "mmsi": "mmsi",
    "timestamp": "timestamp",
    "lat": "lat",
    "lon": "lon",
    "speed": "sog",
    "course": "cog",
    "heading": "heading",
    "shipname": "vessel_name",
    "imo": "imo",
    "callsign": "callsign",
    "flag": "flag",
    "vessel_type": "ship_type",
    "length": "length",
}

# GFW sentinel values.
IMO_UNAVAILABLE = 0


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def _build_url(target_date: date) -> str:
    """Build the download URL for a given date's position data."""
    return (
        f"{BASE_URL}/datasets/public-global-fishing-effort:latest/"
        f"download?date={target_date.isoformat()}&format=json"
    )


def _build_filename(target_date: date) -> str:
    """Build the expected filename for a given date."""
    return f"gfw_{target_date.isoformat()}.json"


# ---------------------------------------------------------------------------
# GFW adapter
# ---------------------------------------------------------------------------


class GFWAdapter:
    """Global Fishing Watch source adapter.

    Implements the ``SourceAdapter`` protocol for fetching and
    normalizing GFW AIS data via their public API.
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
            provider="Global Fishing Watch",
            description="Global AIS positions and vessel identity from GFW API.",
            supports_backfill=True,
            supports_streaming=False,
            supports_server_side_bbox=True,
            supports_incremental=False,
            auth_scheme="api_key",
            rate_limit="100 req/min",
            expected_latency="3 days",
            license_requirements=(
                "CC BY-SA 4.0 — attribution required. "
                "See https://globalfishingwatch.org/terms-of-use"
            ),
            coverage="Global",
            history_start="2012-01-01",
            datasets_provided=["positions", "vessels"],
            delivery_format="REST API JSON",
            typical_daily_rows="100K-500K",
            known_quirks=[
                "IMO=0 means unavailable (normalized to null)",
                "speed is in knots (mapped to sog)",
                "course is in degrees (mapped to cog)",
                "Processing pipeline delay: positions available ~3 days after collection",
                "flag uses ISO 3166-1 alpha-3 codes",
            ],
        )

    def available_dates(self) -> tuple[date, date]:
        """GFW has data from 2012-01-01 to ~3 days ago."""
        from datetime import timedelta
        return (date(2012, 1, 1), date.today() - timedelta(days=3))

    def fetch_raw(self, spec: FetchSpec) -> list[RawArtifact]:
        """Download GFW data for the specified date."""
        if self._download_dir is None:
            raise ValueError(
                "download_dir must be set before fetching. "
                "Pass it via GFWAdapter(download_dir=...)"
            )

        url = _build_url(spec.date)
        filename = _build_filename(spec.date)
        dest = self._download_dir / filename

        auth_headers = (
            {"Authorization": f"Bearer {self._api_key}"}
            if self._api_key
            else None
        )

        return [download_and_hash(
            url, dest,
            overwrite=spec.overwrite,
            content_type="application/json",
            headers=auth_headers,
        )]

    def normalize_positions(
        self, artifacts: list[RawArtifact]
    ) -> pl.DataFrame:
        """Normalize GFW raw JSON data into canonical positions schema."""
        frames: list[pl.DataFrame] = []

        for art in artifacts:
            fpath = Path(art.local_path)
            raw_data = json.loads(fpath.read_text())

            # GFW API returns a list of position records or a
            # wrapper object with an "entries" key.
            if isinstance(raw_data, dict):
                records = raw_data.get("entries", raw_data.get("data", []))
                if not records and "entries" not in raw_data and "data" not in raw_data:
                    logger.warning(
                        "GFW response has no 'entries' or 'data' key: %s",
                        sorted(raw_data.keys()),
                    )
            elif isinstance(raw_data, list):
                records = raw_data
            else:
                raise ValueError(f"Unexpected GFW response format: {type(raw_data)}")

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

            # Type casting and sentinel normalization.
            df = self._cast_and_normalize(df)

            frames.append(df)

        if not frames:
            raise ValueError("No data frames produced from GFW artifacts")

        return pl.concat(frames) if len(frames) > 1 else frames[0]

    def normalize_vessels(
        self, artifacts: list[RawArtifact]
    ) -> pl.DataFrame | None:
        """Extract vessel identity records from GFW position data."""
        positions = self.normalize_positions(artifacts)
        return extract_vessels(positions, SOURCE_ID)

    def qc_rules(self) -> list:
        """GFW-specific QC rules.

        GFW data is pre-processed by their pipeline, so quality is
        generally high. Sentinel values (IMO=0) are handled during
        normalization. No additional QC rules beyond built-ins.
        """
        return []

    # --- Internal helpers ---

    @staticmethod
    def _cast_and_normalize(df: pl.DataFrame) -> pl.DataFrame:
        """Apply type casts and sentinel normalization to GFW data."""
        exprs: list[pl.Expr] = []

        for col_name in df.columns:
            col = pl.col(col_name)

            if col_name == "mmsi":
                exprs.append(col.cast(pl.Int64, strict=False))
            elif col_name == "timestamp":
                # GFW timestamps use "Z" or "+00:00" suffix.
                # %#z handles both Zulu literal and offset notation.
                exprs.append(
                    col.str.to_datetime(
                        "%Y-%m-%dT%H:%M:%S%#z", strict=False
                    ).cast(pl.Datetime("us", "UTC"))
                )
            elif col_name in ("lat", "lon", "sog", "cog", "length"):
                exprs.append(col.cast(pl.Float64, strict=False))
            elif col_name == "heading":
                exprs.append(col.cast(pl.Float64, strict=False))
            elif col_name == "imo":
                # IMO=0 or null means unavailable → null.
                casted = col.cast(pl.Int64, strict=False)
                exprs.append(
                    pl.when(casted.is_null() | (casted == IMO_UNAVAILABLE))
                    .then(pl.lit(None, dtype=pl.String))
                    .otherwise(col.cast(pl.String))
                    .alias("imo")
                )
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

register(GFWAdapter)
