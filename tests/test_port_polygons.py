"""Unit tests for the port polygon derivation pipeline (Tier 2).

Uses synthetic AIS positions clustered around known port centers.
Requires shapely (the [geo] extra) for hull computation.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

# Skip entire module if shapely is not installed
shapely = pytest.importorskip("shapely")


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _make_positions(
    center_lat: float,
    center_lon: float,
    n: int = 200,
    *,
    n_vessels: int = 20,
    sog_range: tuple[float, float] = (0.0, 2.0),
    spread_deg: float = 0.02,
    start: datetime | None = None,
    hours: int = 24 * 30,
) -> pl.DataFrame:
    """Generate synthetic low-speed positions around a point."""
    rng = random.Random(42)
    start = start or datetime(2024, 6, 1, tzinfo=timezone.utc)

    rows: list[dict] = []
    for i in range(n):
        rows.append({
            "mmsi": 200000000 + (i % n_vessels),
            "lat": center_lat + rng.uniform(-spread_deg, spread_deg),
            "lon": center_lon + rng.uniform(-spread_deg, spread_deg),
            "sog": rng.uniform(*sog_range),
            "timestamp": start + timedelta(hours=rng.uniform(0, hours)),
            "source": "synthetic",
        })
    return pl.DataFrame(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
    )


@pytest.fixture(scope="module")
def port_index():
    """Real PortIndex (190 KB Parquet, fast)."""
    from neptune_ais.ports import index
    return index()


@pytest.fixture(scope="module")
def rotterdam_positions():
    """200 low-speed positions near Rotterdam."""
    return _make_positions(51.9, 4.48, n=200, n_vessels=25, hours=24 * 60)


@pytest.fixture(scope="module")
def multi_port_positions():
    """Positions at Rotterdam + Hamburg (two clusters)."""
    rdam = _make_positions(51.9, 4.48, n=150, n_vessels=15, hours=24 * 45)
    hamb = _make_positions(53.55, 9.93, n=100, n_vessels=10, hours=24 * 30)
    return pl.concat([rdam, hamb])


# ---------------------------------------------------------------------------
# PortPolygonConfig
# ---------------------------------------------------------------------------


class TestPortPolygonConfig:
    def test_defaults(self):
        from neptune_ais.derive.port_polygons import PortPolygonConfig

        cfg = PortPolygonConfig()
        assert cfg.max_speed_knots == 3.0
        assert cfg.min_positions == 50
        assert cfg.concave_ratio == 0.3

    def test_config_hash_deterministic(self):
        from neptune_ais.derive.port_polygons import PortPolygonConfig

        a = PortPolygonConfig()
        b = PortPolygonConfig()
        assert a.config_hash() == b.config_hash()

    def test_config_hash_changes(self):
        from neptune_ais.derive.port_polygons import PortPolygonConfig

        a = PortPolygonConfig(max_speed_knots=3.0)
        b = PortPolygonConfig(max_speed_knots=5.0)
        assert a.config_hash() != b.config_hash()


# ---------------------------------------------------------------------------
# assign_positions_to_ports
# ---------------------------------------------------------------------------


class TestAssignPositions:
    def test_basic(self, port_index, rotterdam_positions):
        from neptune_ais.derive.port_polygons import assign_positions_to_ports

        result = assign_positions_to_ports(rotterdam_positions, port_index)
        assert "assigned_port" in result.columns
        assert len(result) > 0
        # Most should be assigned to Rotterdam
        ports = result["assigned_port"].value_counts()
        top_port = ports.sort("count", descending=True)["assigned_port"][0]
        assert "Rotterdam" in top_port

    def test_empty_positions(self, port_index):
        from neptune_ais.derive.port_polygons import assign_positions_to_ports

        empty = pl.DataFrame({
            "lat": pl.Series([], dtype=pl.Float64),
            "lon": pl.Series([], dtype=pl.Float64),
            "sog": pl.Series([], dtype=pl.Float64),
            "mmsi": pl.Series([], dtype=pl.Int64),
            "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "source": pl.Series([], dtype=pl.String),
        })
        result = assign_positions_to_ports(empty, port_index)
        assert "assigned_port" in result.columns
        assert len(result) == 0

    def test_high_speed_filtered(self, port_index):
        """Positions above max_speed_knots should be filtered out."""
        from neptune_ais.derive.port_polygons import assign_positions_to_ports

        fast = _make_positions(51.9, 4.48, n=50, sog_range=(10.0, 15.0))
        result = assign_positions_to_ports(fast, port_index)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# compute_confidence
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    def test_all_maxed(self):
        from neptune_ais.derive.port_polygons import compute_confidence

        c = compute_confidence(
            position_count=1000, vessel_count=50,
            temporal_span_days=120, area_km2=2.0,
        )
        assert c == 1.0

    def test_zero_span(self):
        from neptune_ais.derive.port_polygons import compute_confidence

        c = compute_confidence(
            position_count=1000, vessel_count=50,
            temporal_span_days=0, area_km2=2.0,
        )
        assert c == 0.0  # temporal factor = 0 → min is 0

    def test_one_vessel(self):
        from neptune_ais.derive.port_polygons import compute_confidence

        c = compute_confidence(
            position_count=1000, vessel_count=1,
            temporal_span_days=120, area_km2=2.0,
        )
        assert c == pytest.approx(1.0 / 30.0, abs=0.01)

    def test_destination_boost(self):
        from neptune_ais.derive.port_polygons import compute_confidence

        base = compute_confidence(
            position_count=50, vessel_count=5,
            temporal_span_days=10, area_km2=2.0,
        )
        boosted = compute_confidence(
            position_count=50, vessel_count=5,
            temporal_span_days=10, area_km2=2.0,
            destination_match_rate=1.0,
        )
        assert boosted > base
        assert boosted - base == pytest.approx(0.1, abs=0.001)

    def test_destination_zero_rate_no_change(self):
        from neptune_ais.derive.port_polygons import compute_confidence

        a = compute_confidence(
            position_count=100, vessel_count=10,
            temporal_span_days=30, area_km2=1.0,
        )
        b = compute_confidence(
            position_count=100, vessel_count=10,
            temporal_span_days=30, area_km2=1.0,
            destination_match_rate=0.0,
        )
        assert a == b

    def test_capped_at_one(self):
        from neptune_ais.derive.port_polygons import compute_confidence

        c = compute_confidence(
            position_count=10000, vessel_count=100,
            temporal_span_days=365, area_km2=1.0,
            destination_match_rate=1.0,
        )
        assert c == 1.0


# ---------------------------------------------------------------------------
# suggest_zone_type
# ---------------------------------------------------------------------------


class TestSuggestZoneType:
    def test_terminal(self):
        from neptune_ais.derive.port_polygons import suggest_zone_type

        label, conf = suggest_zone_type({
            "mean_sog": 0.5, "vessel_count": 50, "position_count": 500,
        })
        assert label == "terminal"
        assert 0.0 < conf <= 1.0

    def test_approach(self):
        from neptune_ais.derive.port_polygons import suggest_zone_type

        label, conf = suggest_zone_type({
            "mean_sog": 3.0, "vessel_count": 20, "position_count": 100,
        })
        assert label == "approach"

    def test_anchorage(self):
        from neptune_ais.derive.port_polygons import suggest_zone_type

        label, conf = suggest_zone_type({
            "mean_sog": 0.3, "vessel_count": 3, "position_count": 30,
        })
        assert label == "anchorage"

    def test_unknown(self):
        from neptune_ais.derive.port_polygons import suggest_zone_type

        label, conf = suggest_zone_type({
            "mean_sog": 1.5, "vessel_count": 5, "position_count": 40,
        })
        assert label == "unknown"


# ---------------------------------------------------------------------------
# Leader clustering
# ---------------------------------------------------------------------------


class TestLeaderCluster:
    def test_single_cluster(self):
        from neptune_ais.derive.port_polygons import _leader_cluster

        lats = [51.9 + i * 0.001 for i in range(20)]
        lons = [4.5 + i * 0.001 for i in range(20)]
        clusters = _leader_cluster(lats, lons, separation_m=5000)
        assert len(clusters) == 1
        assert len(clusters[0]) == 20

    def test_two_clusters(self):
        from neptune_ais.derive.port_polygons import _leader_cluster

        # Two groups ~100 km apart
        lats = [51.9] * 10 + [52.8] * 10
        lons = [4.5] * 10 + [4.5] * 10
        clusters = _leader_cluster(lats, lons, separation_m=5000)
        assert len(clusters) == 2

    def test_sorted_by_size(self):
        from neptune_ais.derive.port_polygons import _leader_cluster

        lats = [0.0] * 5 + [10.0] * 15
        lons = [0.0] * 5 + [10.0] * 15
        clusters = _leader_cluster(lats, lons, separation_m=5000)
        assert len(clusters[0]) >= len(clusters[1])


# ---------------------------------------------------------------------------
# split_port_zones (end-to-end with synthetic data)
# ---------------------------------------------------------------------------


class TestSplitPortZones:
    def test_produces_zones(self, port_index, rotterdam_positions):
        from neptune_ais.derive.port_polygons import (
            assign_positions_to_ports,
            split_port_zones,
            PortPolygonConfig,
        )

        cfg = PortPolygonConfig(min_positions=20, min_cluster_points=10)
        assigned = assign_positions_to_ports(rotterdam_positions, port_index, config=cfg)
        zones = split_port_zones(assigned, port_index, config=cfg)

        assert len(zones) > 0
        assert "port_name" in zones.columns
        assert "zone_id" in zones.columns
        assert "geometry_wkb" in zones.columns
        assert "confidence" in zones.columns
        assert "destination_match_rate" in zones.columns

    def test_zone_geometry_is_valid(self, port_index, rotterdam_positions):
        from neptune_ais.derive.port_polygons import (
            assign_positions_to_ports,
            split_port_zones,
            PortPolygonConfig,
        )

        cfg = PortPolygonConfig(min_positions=20, min_cluster_points=10)
        assigned = assign_positions_to_ports(rotterdam_positions, port_index, config=cfg)
        zones = split_port_zones(assigned, port_index, config=cfg)

        for wkb in zones["geometry_wkb"].to_list():
            geom = shapely.from_wkb(wkb)
            assert geom.is_valid
            assert geom.geom_type in ("Polygon", "MultiPolygon")

    def test_empty_assigned(self, port_index):
        from neptune_ais.derive.port_polygons import split_port_zones, _empty_zone_df

        empty = _empty_zone_df().with_columns(
            pl.lit(None).cast(pl.String).alias("assigned_port"),
        ).head(0)
        # Won't crash on empty but need proper assigned schema
        # Use an actually empty DataFrame with correct columns
        empty_assigned = pl.DataFrame({
            "lat": pl.Series([], dtype=pl.Float64),
            "lon": pl.Series([], dtype=pl.Float64),
            "sog": pl.Series([], dtype=pl.Float64),
            "mmsi": pl.Series([], dtype=pl.Int64),
            "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "assigned_port": pl.Series([], dtype=pl.String),
        })
        result = split_port_zones(empty_assigned, port_index)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# derive_port_polygons (full pipeline)
# ---------------------------------------------------------------------------


class TestDerivePortPolygons:
    def test_full_pipeline(self, port_index, rotterdam_positions):
        from neptune_ais.derive.port_polygons import (
            derive_port_polygons,
            PortPolygonConfig,
        )

        cfg = PortPolygonConfig(min_positions=20, min_cluster_points=10)
        result = derive_port_polygons(rotterdam_positions, port_index, config=cfg)

        assert len(result) > 0
        assert "port_name" in result.columns
        assert "config_hash" in result.columns
        assert "derived_at" in result.columns

    def test_multi_port(self, port_index, multi_port_positions):
        from neptune_ais.derive.port_polygons import (
            derive_port_polygons,
            PortPolygonConfig,
        )

        cfg = PortPolygonConfig(min_positions=20, min_cluster_points=10)
        result = derive_port_polygons(multi_port_positions, port_index, config=cfg)

        ports = result["port_name"].unique().to_list()
        assert len(ports) >= 2

    def test_empty_positions(self, port_index):
        from neptune_ais.derive.port_polygons import derive_port_polygons

        empty = pl.DataFrame({
            "lat": pl.Series([], dtype=pl.Float64),
            "lon": pl.Series([], dtype=pl.Float64),
            "sog": pl.Series([], dtype=pl.Float64),
            "mmsi": pl.Series([], dtype=pl.Int64),
            "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "source": pl.Series([], dtype=pl.String),
        })
        result = derive_port_polygons(empty, port_index)
        assert len(result) == 0

    def test_persistence(self, port_index, rotterdam_positions, tmp_path):
        from neptune_ais.derive.port_polygons import (
            derive_port_polygons,
            PortPolygonConfig,
        )

        cfg = PortPolygonConfig(min_positions=20, min_cluster_points=10)
        result = derive_port_polygons(
            rotterdam_positions, port_index,
            config=cfg, output_dir=str(tmp_path),
        )

        # Check file was written
        parquet_files = list(tmp_path.rglob("*.parquet"))
        assert len(parquet_files) == 1
        assert cfg.config_hash() in parquet_files[0].name

        # Check file is readable and matches result
        loaded = pl.read_parquet(parquet_files[0])
        assert len(loaded) == len(result)

    def test_resolved_destinations_validation(self, port_index, rotterdam_positions):
        from neptune_ais.derive.port_polygons import derive_port_polygons

        # Wrong length
        bad = pl.DataFrame({"resolved_port_name": ["Rotterdam"]})
        with pytest.raises(ValueError, match="rows"):
            derive_port_polygons(
                rotterdam_positions, port_index, resolved_destinations=bad,
            )

        # Missing column
        bad2 = pl.DataFrame({
            "wrong": ["x"] * len(rotterdam_positions),
        })
        with pytest.raises(ValueError, match="resolved_port_name"):
            derive_port_polygons(
                rotterdam_positions, port_index, resolved_destinations=bad2,
            )
