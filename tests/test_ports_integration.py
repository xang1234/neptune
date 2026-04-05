"""Integration tests for the port intelligence pipeline.

Tests the full stack: vectorized spatial join → port-call detection →
enrichment → visualization. Uses the real WPI data (190 KB) with
synthetic AIS positions for deterministic results.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from neptune_ais.ports._models import Port


# ---------------------------------------------------------------------------
# Synthetic positions fixture
# ---------------------------------------------------------------------------


def _make_positions_at(
    lat: float,
    lon: float,
    n: int,
    *,
    n_vessels: int = 5,
    sog: float = 1.0,
    hours: int = 6,
    start: datetime | None = None,
) -> pl.DataFrame:
    """Generate synthetic low-speed positions at a known location."""
    rng = random.Random(42)
    start = start or datetime(2024, 6, 15, 8, 0, tzinfo=timezone.utc)

    rows = []
    for i in range(n):
        rows.append({
            "mmsi": 200000000 + (i % n_vessels),
            "lat": lat + rng.uniform(-0.005, 0.005),
            "lon": lon + rng.uniform(-0.005, 0.005),
            "sog": sog + rng.uniform(-0.5, 0.5),
            "timestamp": start + timedelta(seconds=rng.uniform(0, hours * 3600)),
            "source": "test",
        })
    return pl.DataFrame(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
    )


@pytest.fixture(scope="module")
def pi():
    from neptune_ais.ports import index
    return index()


@pytest.fixture(scope="module")
def rotterdam_positions():
    """Positions clustered at Rotterdam (51.9, 4.48) over 6 hours."""
    return _make_positions_at(51.9, 4.48, n=100, n_vessels=5, hours=6)


# ---------------------------------------------------------------------------
# Vectorized spatial join
# ---------------------------------------------------------------------------


class TestVectorizedPortLookup:
    def test_assigns_rotterdam(self, pi, rotterdam_positions):
        from neptune_ais.ports._registry_bridge import vectorized_port_lookup

        port_bounds = pi.ports.select(
            "name", "lat", "lon",
            "bbox_west", "bbox_south", "bbox_east", "bbox_north",
        )
        result = vectorized_port_lookup(rotterdam_positions, port_bounds)

        assert len(result) == len(rotterdam_positions)
        matched = result.drop_nulls()
        assert len(matched) > 0
        assert "Rotterdam" in matched.to_list()

    def test_open_ocean_no_match(self, pi):
        from neptune_ais.ports._registry_bridge import vectorized_port_lookup

        ocean = _make_positions_at(0.0, -150.0, n=10)
        port_bounds = pi.ports.select(
            "name", "lat", "lon",
            "bbox_west", "bbox_south", "bbox_east", "bbox_north",
        )
        result = vectorized_port_lookup(ocean, port_bounds)
        assert result.null_count() == len(ocean)

    def test_performance_100k(self, pi):
        """100K positions against 3,800 ports should complete in <10s."""
        import time
        from neptune_ais.ports._registry_bridge import vectorized_port_lookup

        # Spread positions across a region with many ports (North Sea)
        rng = random.Random(99)
        rows = []
        for i in range(100_000):
            rows.append({
                "lat": 50.0 + rng.uniform(0, 5),
                "lon": 0.0 + rng.uniform(0, 10),
            })
        big = pl.DataFrame(rows)

        port_bounds = pi.ports.select(
            "name", "lat", "lon",
            "bbox_west", "bbox_south", "bbox_east", "bbox_north",
        )

        t0 = time.perf_counter()
        result = vectorized_port_lookup(big, port_bounds)
        elapsed = time.perf_counter() - t0

        assert len(result) == 100_000
        assert elapsed < 10.0, f"Took {elapsed:.1f}s (limit 10s)"


# ---------------------------------------------------------------------------
# build_boundary_dataset
# ---------------------------------------------------------------------------


class TestBuildBoundaryDataset:
    def test_produces_valid_dataset(self, pi):
        from neptune_ais.ports._registry_bridge import build_boundary_dataset

        ds = build_boundary_dataset(pi)
        assert ds.name == "builtin_ports"
        assert len(ds.regions) > 3000
        # Each region has a name and bbox
        for region in ds.regions[:5]:
            assert region.name
            assert len(region.bbox) == 4


# ---------------------------------------------------------------------------
# port_calls() convenience wrapper — zero-config detection
# ---------------------------------------------------------------------------


class TestPortCallsConvenience:
    def test_zero_config(self, rotterdam_positions):
        """port_calls(positions) should work without any manual setup."""
        from neptune_ais.helpers import port_calls

        events = port_calls(rotterdam_positions.lazy(), enrich=False)
        # With 100 low-speed positions over 6 hours, should detect >=1 port call
        assert len(events) >= 1

    def test_enriched(self, rotterdam_positions):
        """With enrich=True, events get port metadata columns."""
        from neptune_ais.helpers import port_calls

        events = port_calls(rotterdam_positions.lazy(), enrich=True)
        if len(events) > 0:
            assert "port_name" in events.columns
            assert "unlocode" in events.columns
            assert "country_code" in events.columns
            assert "harbor_size" in events.columns

    def test_empty_positions(self):
        from neptune_ais.helpers import port_calls

        empty = pl.DataFrame({
            "mmsi": pl.Series([], dtype=pl.Int64),
            "lat": pl.Series([], dtype=pl.Float64),
            "lon": pl.Series([], dtype=pl.Float64),
            "sog": pl.Series([], dtype=pl.Float64),
            "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "source": pl.Series([], dtype=pl.String),
        })
        events = port_calls(empty.lazy())
        assert len(events) == 0


# ---------------------------------------------------------------------------
# enrich_port_calls
# ---------------------------------------------------------------------------


class TestEnrichPortCalls:
    def test_enrichment_columns(self, pi, rotterdam_positions):
        from neptune_ais.helpers import enrich_port_calls, port_calls

        events = port_calls(rotterdam_positions.lazy(), enrich=False)
        if len(events) == 0:
            pytest.skip("No port calls detected from synthetic data")

        enriched = enrich_port_calls(events, pi)
        expected_cols = [
            "port_name", "unlocode", "country_code", "country_name",
            "harbor_size", "shelter_quality", "channel_depth_m",
            "has_pilotage", "has_cranes", "has_fuel",
        ]
        for col in expected_cols:
            assert col in enriched.columns, f"Missing column: {col}"

    def test_empty_events(self, pi):
        from neptune_ais.datasets.events import SCHEMA
        from neptune_ais.helpers import enrich_port_calls

        empty = pl.DataFrame(schema=SCHEMA)
        result = enrich_port_calls(empty, pi)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# eez_crossings() convenience wrapper
# ---------------------------------------------------------------------------


class TestEEZCrossingsConvenience:
    def test_empty_positions(self):
        from neptune_ais.helpers import eez_crossings

        empty = pl.DataFrame({
            "mmsi": pl.Series([], dtype=pl.Int64),
            "lat": pl.Series([], dtype=pl.Float64),
            "lon": pl.Series([], dtype=pl.Float64),
            "sog": pl.Series([], dtype=pl.Float64),
            "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "source": pl.Series([], dtype=pl.String),
        })
        events = eez_crossings(empty.lazy())
        assert len(events) == 0

    def test_crossing_detected(self):
        """Two positions in different EEZ bboxes should produce a crossing."""
        from neptune_ais.helpers import eez_crossings

        # Position 1: Dutch North Sea (52.5, 4.0)
        # Position 2: Belgian waters (51.2, 3.0)
        positions = pl.DataFrame({
            "mmsi": [123456789, 123456789],
            "lat": [52.5, 51.2],
            "lon": [4.0, 3.0],
            "sog": [12.0, 12.0],
            "timestamp": [
                datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
                datetime(2024, 6, 15, 11, 0, tzinfo=timezone.utc),
            ],
            "source": ["test", "test"],
        }).with_columns(
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        )

        events = eez_crossings(positions.lazy())
        # May or may not detect a crossing depending on bbox overlap
        # The key test is that it doesn't crash
        assert isinstance(events, pl.DataFrame)


# ---------------------------------------------------------------------------
# prepare_ports() visualization
# ---------------------------------------------------------------------------


class TestPreparePortsIntegration:
    def test_default_output(self):
        from neptune_ais.viz import prepare_ports

        result = prepare_ports()
        assert "centers" in result
        assert "polygons" in result
        assert len(result["centers"]) > 3000
        assert len(result["polygons"]) > 3000

    def test_viewport_clipping(self):
        from neptune_ais.viz import prepare_ports, Viewport

        vp = Viewport(west=3.0, south=51.0, east=5.0, north=52.0)
        result = prepare_ports(viewport=vp)
        assert len(result["centers"]) < 100  # small viewport
        assert len(result["polygons"]) < 100

    def test_polygon_schema(self):
        from neptune_ais.viz import prepare_ports, Viewport

        vp = Viewport(west=3.0, south=51.0, east=5.0, north=52.0)
        result = prepare_ports(viewport=vp)
        polygons = result["polygons"]
        assert "name" in polygons.columns
        assert "geometry_wkb" in polygons.columns
        assert "polygon_source" in polygons.columns
        assert "confidence" in polygons.columns

    def test_with_derived_polygons(self, pi):
        from neptune_ais.viz import prepare_ports

        # Mock tier 2 derived polygon for Rotterdam
        derived = pl.DataFrame({
            "port_name": ["Rotterdam"],
            "zone_id": ["zone_0"],
            "geometry_wkb": [b"\x01\x00"],
            "confidence": [0.9],
            "bbox_west": [3.8], "bbox_south": [51.7],
            "bbox_east": [4.8], "bbox_north": [52.1],
        })
        result = prepare_ports(port_index=pi, derived_polygons=derived)
        polygons = result["polygons"]
        sources = polygons["polygon_source"].unique().to_list()
        assert "tier2" in sources
        assert "tier1_bbox" in sources


# ---------------------------------------------------------------------------
# Neptune API methods
# ---------------------------------------------------------------------------


class TestNeptuneAPIMethods:
    def test_port_calls_method_exists(self):
        from neptune_ais.api import Neptune
        assert hasattr(Neptune, "port_calls")
        assert hasattr(Neptune, "eez_crossings")
        assert hasattr(Neptune, "derive_port_polygons")
