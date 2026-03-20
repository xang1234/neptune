"""GFW — Global Fishing Watch adapter.

Uses the official ``gfw-api-python-client`` (``gfwapiclient``) to access:

- **Events API** — fishing events, vessel encounters, port visits, loitering
- **Vessels API** — vessel identity search by MMSI
- **4Wings API** — aggregated fishing effort grids

GFW does **not** provide individual AIS position reports (like NOAA/DMA).
The ``normalize_positions()`` method raises ``NotImplementedError`` to
signal this clearly.

Coverage: Global, 2020–present (Events API).
Delivery: REST API with API token authentication.
Latency: 3+ days (processing pipeline delay).

Requires: ``pip install gfw-api-python-client>=1.0``
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from neptune_ais.adapters.base import (
    FetchSpec,
    RawArtifact,
    SourceAdapter,
    SourceCapabilities,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_ID = "gfw"
ADAPTER_VERSION = "gfw/0.2.0"

# GFW sentinel values.
IMO_UNAVAILABLE = 0

# GFW event type → Neptune event type mapping.
# Values use constants from datasets.events to avoid stringly-typed drift.
from neptune_ais.datasets.events import (
    EVENT_TYPE_ENCOUNTER,
    EVENT_TYPE_FISHING,
    EVENT_TYPE_LOITERING,
    EVENT_TYPE_PORT_CALL,
)

GFW_EVENT_TYPE_MAP: dict[str, str] = {
    "FISHING": EVENT_TYPE_FISHING,
    "ENCOUNTER": EVENT_TYPE_ENCOUNTER,
    "PORT_VISIT": EVENT_TYPE_PORT_CALL,
    "LOITERING": EVENT_TYPE_LOITERING,
}


# ---------------------------------------------------------------------------
# GFW adapter
# ---------------------------------------------------------------------------


class GFWAdapter:
    """Global Fishing Watch source adapter.

    Implements the ``SourceAdapter`` protocol for fetching and
    normalizing GFW data via the official Python client.

    GFW provides events, vessels, and fishing effort — **not** individual
    AIS positions. Use ``normalize_events()`` and
    ``normalize_fishing_effort()`` instead of ``normalize_positions()``.
    """

    SOURCE_ID = SOURCE_ID

    def __init__(
        self,
        download_dir: Path | None = None,
        *,
        api_key: str = "",
        access_token: str = "",
    ) -> None:
        self._download_dir = download_dir
        # _api_key is the single source of truth for the token.
        # api.py sets adapter._api_key after construction.
        self._api_key = (
            access_token
            or api_key
            or os.environ.get("GFW_API_TOKEN", "")
            or os.environ.get("GFW_API_KEY", "")
            or ""
        )

    @property
    def source_id(self) -> str:
        return SOURCE_ID

    @property
    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            source_id=SOURCE_ID,
            provider="Global Fishing Watch",
            description=(
                "Global fishing events, vessel identity, and aggregated "
                "fishing effort from GFW APIs."
            ),
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
            history_start="2020-01-01",
            datasets_provided=["events", "vessels", "fishing_effort"],
            delivery_format="REST API JSON (gfwapiclient)",
            typical_daily_rows="10K-100K events",
            known_quirks=[
                "GFW does not provide individual AIS positions",
                "Events API available from 2020-01-01 onwards",
                "vessel.ssvid is the MMSI as a string",
                "Event types are uppercase (FISHING, ENCOUNTER, etc.)",
                "4Wings effort is gridded — each row is a spatial cell, not a vessel",
                "Requires gfw-api-python-client package",
            ],
        )

    def available_dates(self) -> tuple[date, date]:
        """GFW Events API has data from 2020-01-01 to ~3 days ago."""
        from datetime import timedelta

        return (date(2020, 1, 1), date.today() - timedelta(days=3))

    # --- Fetch ---

    def fetch_raw(self, spec: FetchSpec) -> list[RawArtifact]:
        """Fetch events and effort data for the given date.

        Uses the official ``gfwapiclient`` async client, bridged to sync
        via ``asyncio.run()``.

        Returns a list of RawArtifact records pointing to JSON files
        on disk (one for events, one for effort).
        """
        if self._download_dir is None:
            raise ValueError(
                "download_dir must be set before fetching. "
                "Pass it via GFWAdapter(download_dir=...)"
            )

        token = self._api_key
        if not token:
            raise ValueError(
                "GFW API token required. Set GFW_API_TOKEN environment "
                "variable or pass access_token/api_key to GFWAdapter()."
            )

        self._download_dir.mkdir(parents=True, exist_ok=True)
        date_str = spec.date.isoformat()

        events_data, effort_data = self._run_async(
            self._fetch_all(spec, token)
        )

        artifacts: list[RawArtifact] = []

        # Write events JSON.
        events_path = self._download_dir / f"gfw_events_{date_str}.json"
        events_path.write_text(json.dumps(events_data, default=str))
        artifacts.append(self._make_artifact(
            events_path, f"gfwapiclient:events:{date_str}"
        ))

        # Write effort JSON.
        effort_path = self._download_dir / f"gfw_effort_{date_str}.json"
        effort_path.write_text(json.dumps(effort_data, default=str))
        artifacts.append(self._make_artifact(
            effort_path, f"gfwapiclient:effort:{date_str}"
        ))

        return artifacts

    @staticmethod
    def _run_async(coro):
        """Bridge async gfwapiclient calls into sync adapter protocol."""
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop running — safe to use asyncio.run().
            return asyncio.run(coro)
        raise RuntimeError(
            "GFWAdapter.fetch_raw() cannot be called from within a running "
            "async event loop. If in Jupyter, run: import nest_asyncio; "
            "nest_asyncio.apply() before calling Neptune.download()."
        )

    @staticmethod
    async def _fetch_all(spec: FetchSpec, token: str):
        """Async: fetch events + effort for one date concurrently."""
        import asyncio

        try:
            import gfwapiclient
        except ImportError:
            raise ImportError(
                "gfwapiclient is required for the GFW adapter. "
                "Install with: pip install gfw-api-python-client>=1.0"
            ) from None

        client = gfwapiclient.Client(access_token=token)
        date_str = spec.date.isoformat()

        # Fetch events and effort concurrently.
        events_result, effort_result = await asyncio.gather(
            client.events.get_all_events(
                datasets=[
                    "public-global-fishing-events:latest",
                    "public-global-encounters-events:latest",
                    "public-global-loitering-events:latest",
                    "public-global-port-visits-events:latest",
                ],
                start_date=date_str,
                end_date=date_str,
            ),
            client.fourwings.create_fishing_effort_report(
                spatial_resolution="LOW",
                temporal_resolution="DAILY",
                group_by="FLAGANDGEARTYPE",
                start_date=date_str,
                end_date=date_str,
            ),
        )

        events_data = [
            item.model_dump(mode="json", by_alias=True)
            for item in events_result.data()
        ]
        effort_data = [
            item.model_dump(mode="json", by_alias=True)
            for item in effort_result.data()
        ]

        return events_data, effort_data

    @staticmethod
    def _make_artifact(path: Path, source_url: str) -> RawArtifact:
        """Build a RawArtifact from a file written to disk."""
        import hashlib

        content = path.read_bytes()
        return RawArtifact(
            source_url=source_url,
            filename=path.name,
            local_path=str(path),
            content_hash=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            fetch_timestamp=datetime.now(timezone.utc),
            content_type="application/json",
        )

    # --- Normalize positions (not supported) ---

    def normalize_positions(
        self, artifacts: list[RawArtifact]
    ) -> pl.DataFrame:
        """GFW does not provide individual AIS positions.

        Raises NotImplementedError with guidance to use events() or
        fishing_effort() instead.
        """
        raise NotImplementedError(
            "GFW does not provide individual AIS positions. "
            "Use normalize_events() for fishing events/encounters/port visits, "
            "or normalize_fishing_effort() for aggregated spatial fishing effort."
        )

    # --- Shared artifact reader ---

    @staticmethod
    def _read_artifacts(
        artifacts: list[RawArtifact], filename_filter: str,
    ) -> list[list[dict]] | None:
        """Filter artifacts by filename and parse JSON.

        Returns None if no matching artifacts, otherwise a list of
        parsed record lists (one per matching artifact file).
        """
        matched = [a for a in artifacts if filename_filter in a.filename]
        if not matched:
            return None
        result = []
        for art in matched:
            raw_data = json.loads(Path(art.local_path).read_text())
            if not isinstance(raw_data, list):
                raw_data = [raw_data]
            result.append(raw_data)
        return result

    # --- Normalize events ---

    def normalize_events(
        self, artifacts: list[RawArtifact]
    ) -> pl.DataFrame | None:
        """Parse GFW events JSON into Neptune events/v1 schema."""
        from neptune_ais.datasets.events import SCHEMA

        parsed = self._read_artifacts(artifacts, "events")
        if parsed is None:
            return None

        rows: list[dict] = []
        for raw_data in parsed:
            for item in raw_data:
                gfw_type = (item.get("type") or "").upper()
                event_type = GFW_EVENT_TYPE_MAP.get(gfw_type)
                if event_type is None:
                    continue

                vessel = item.get("vessel") or {}
                position = item.get("position") or {}

                # Parse MMSI from vessel.ssvid.
                ssvid = vessel.get("ssvid")
                try:
                    mmsi = int(ssvid) if ssvid else None
                except (ValueError, TypeError):
                    mmsi = None
                if mmsi is None:
                    continue

                # For encounters, extract the other vessel's MMSI.
                other_mmsi = None
                encounter = item.get("encounter")
                if encounter and isinstance(encounter, dict):
                    other_vessel = encounter.get("vessel") or {}
                    other_ssvid = other_vessel.get("ssvid")
                    if other_ssvid:
                        try:
                            other_mmsi = int(other_ssvid)
                        except (ValueError, TypeError):
                            pass

                # Parse timestamps.
                start_time = _parse_dt(item.get("start"))
                end_time = _parse_dt(item.get("end"))
                if start_time is None or end_time is None:
                    continue

                rows.append({
                    "event_id": item.get("id", ""),
                    "event_type": event_type,
                    "mmsi": mmsi,
                    "other_mmsi": other_mmsi,
                    "start_time": start_time,
                    "end_time": end_time,
                    "lat": position.get("lat"),
                    "lon": position.get("lon"),
                    "geometry_wkb": None,
                    "confidence_score": 1.0,
                    "source": SOURCE_ID,
                    "record_provenance": f"{SOURCE_ID}:events_api",
                })

        if not rows:
            return pl.DataFrame(schema=SCHEMA)

        return pl.DataFrame(rows, schema=SCHEMA)

    # --- Normalize vessels ---

    def normalize_vessels(
        self, artifacts: list[RawArtifact]
    ) -> pl.DataFrame | None:
        """Extract vessel identity from GFW event data.

        Builds a vessel record for each unique MMSI seen in the events,
        using the vessel metadata embedded in event responses.
        """
        parsed = self._read_artifacts(artifacts, "events")
        if parsed is None:
            return None

        seen: dict[int, dict] = {}
        for raw_data in parsed:
            for item in raw_data:
                vessel = item.get("vessel") or {}
                ssvid = vessel.get("ssvid")
                if not ssvid:
                    continue
                try:
                    mmsi = int(ssvid)
                except (ValueError, TypeError):
                    continue

                if mmsi not in seen:
                    seen[mmsi] = {
                        "mmsi": mmsi,
                        "vessel_name": vessel.get("name"),
                        "flag": vessel.get("flag"),
                        "ship_type": vessel.get("type"),
                        "source": SOURCE_ID,
                        "record_provenance": f"{SOURCE_ID}:events_api",
                    }

        if not seen:
            return None

        from neptune_ais.datasets.vessels import SCHEMA

        rows = list(seen.values())
        # Add required temporal columns as None (not available from events).
        for row in rows:
            row.setdefault("first_seen", None)
            row.setdefault("last_seen", None)
            row.setdefault("imo", None)
            row.setdefault("callsign", None)
            row.setdefault("length", None)
            row.setdefault("beam", None)

        return pl.DataFrame(rows, schema=SCHEMA)

    # --- Normalize fishing effort ---

    def normalize_fishing_effort(
        self, artifacts: list[RawArtifact]
    ) -> pl.DataFrame | None:
        """Parse 4Wings effort JSON into Neptune fishing_effort/v1 schema."""
        from neptune_ais.datasets.fishing_effort import SCHEMA

        parsed = self._read_artifacts(artifacts, "effort")
        if parsed is None:
            return None

        rows: list[dict] = []
        for raw_data in parsed:
            for item in raw_data:
                date_str = item.get("date")
                if date_str is None:
                    continue

                try:
                    effort_date = date.fromisoformat(date_str)
                except ValueError:
                    continue

                hours = item.get("hours")
                if hours is None:
                    continue

                rows.append({
                    "date": effort_date,
                    "lat": item.get("lat"),
                    "lon": item.get("lon"),
                    "flag": item.get("flag"),
                    "geartype": item.get("geartype"),
                    "vessel_hours": float(hours),
                    "source": SOURCE_ID,
                    "record_provenance": f"{SOURCE_ID}:fourwings_api",
                })

        if not rows:
            return pl.DataFrame(schema=SCHEMA)

        return pl.DataFrame(rows, schema=SCHEMA)

    # --- QC ---

    def qc_rules(self) -> list:
        """GFW data is pre-processed — no additional QC rules."""
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_dt(val) -> datetime | None:
    """Parse a datetime string or object to UTC datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val
    if isinstance(val, str):
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Auto-register on import
# ---------------------------------------------------------------------------

from neptune_ais.adapters.registry import register  # noqa: E402

register(GFWAdapter)
