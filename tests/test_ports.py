"""Unit tests for the ports subsystem.

Tests: models, index queries, spatial math, destination resolver,
overlay merging, and edge cases.
"""

from __future__ import annotations

import polars as pl
import pytest

from neptune_ais.ports._models import EEZRegion, Port


# ---------------------------------------------------------------------------
# Fixtures — synthetic port data
# ---------------------------------------------------------------------------

def _make_port(**overrides) -> dict:
    """Build a port row dict with sensible defaults."""
    base = dict(
        wpi_number=99999,
        name="Test Port",
        alternate_name=None,
        unlocode="XXTES",
        country_code="XX",
        country_name="Testland",
        lat=51.9,
        lon=4.5,
        bbox_west=4.3,
        bbox_south=51.7,
        bbox_east=4.7,
        bbox_north=52.1,
        harbor_size="M",
        harbor_type="CN",
        shelter_quality="G",
        channel_depth_m=10.0,
        anchorage_depth_m=8.0,
        cargo_pier_depth_m=9.0,
        max_vessel_length_m=200.0,
        tide_range_m=2.0,
        has_pilotage=True,
        has_tugs=True,
        has_fuel=True,
        has_drydock=False,
        has_cranes=True,
        has_medical=False,
        has_derived_polygon=False,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestPort:
    def test_construction(self):
        p = Port(**_make_port())
        assert p.name == "Test Port"
        assert p.wpi_number == 99999

    def test_frozen(self):
        p = Port(**_make_port())
        with pytest.raises(AttributeError):
            p.name = "Changed"  # type: ignore[misc]

    def test_bbox_property(self):
        p = Port(**_make_port(bbox_west=1.0, bbox_south=2.0, bbox_east=3.0, bbox_north=4.0))
        assert p.bbox == (1.0, 2.0, 3.0, 4.0)

    def test_none_depths(self):
        p = Port(**_make_port(channel_depth_m=None, anchorage_depth_m=None))
        assert p.channel_depth_m is None
        assert p.anchorage_depth_m is None


class TestEEZRegion:
    def test_construction(self):
        r = EEZRegion(
            mrgid=5668, name="Dutch EEZ", sovereign="Netherlands",
            iso_3="NLD", bbox_west=2.5, bbox_south=51.3,
            bbox_east=7.2, bbox_north=55.8, area_km2=64292.0,
        )
        assert r.name == "Dutch EEZ"
        assert r.area_km2 == 64292.0

    def test_frozen(self):
        r = EEZRegion(
            mrgid=1, name="Test", sovereign="Test", iso_3="TST",
            bbox_west=0, bbox_south=0, bbox_east=1, bbox_north=1, area_km2=100,
        )
        with pytest.raises(AttributeError):
            r.name = "Changed"  # type: ignore[misc]

    def test_bbox_property(self):
        r = EEZRegion(
            mrgid=1, name="Test", sovereign="Test", iso_3="TST",
            bbox_west=10, bbox_south=20, bbox_east=30, bbox_north=40, area_km2=100,
        )
        assert r.bbox == (10, 20, 30, 40)


# ---------------------------------------------------------------------------
# Spatial math
# ---------------------------------------------------------------------------


class TestHaversine:
    def test_known_distance(self):
        """Rotterdam to Hamburg: ~410 km."""
        from neptune_ais.ports._spatial import haversine_distance

        d = haversine_distance(51.9, 4.48, 53.55, 9.93)
        assert 405_000 < d < 415_000  # meters

    def test_zero_distance(self):
        from neptune_ais.ports._spatial import haversine_distance

        assert haversine_distance(0.0, 0.0, 0.0, 0.0) == 0.0

    def test_antipodal(self):
        """North pole to south pole: ~20,000 km."""
        from neptune_ais.ports._spatial import haversine_distance

        d = haversine_distance(90.0, 0.0, -90.0, 0.0)
        assert 19_900_000 < d < 20_100_000

    def test_equator_one_degree(self):
        """One degree along the equator: ~111 km."""
        from neptune_ais.ports._spatial import haversine_distance

        d = haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert 111_000 < d < 111_400


class TestBboxFromCenter:
    def test_basic(self):
        from neptune_ais.ports._spatial import bbox_from_center

        west, south, east, north = bbox_from_center(0.0, 0.0, 100.0)
        assert west < 0 < east
        assert south < 0 < north
        # ~100 km radius ≈ ~0.9 degrees at equator
        assert -1.1 < west < -0.7
        assert 0.7 < east < 1.1

    def test_symmetry(self):
        from neptune_ais.ports._spatial import bbox_from_center

        w, s, e, n = bbox_from_center(45.0, 10.0, 50.0)
        assert abs((n - s) / 2 - (n - 45.0)) < 0.001  # symmetric in lat
        assert abs((e - w) / 2 - (e - 10.0)) < 0.01   # symmetric in lon


# ---------------------------------------------------------------------------
# PortIndex queries (uses real data — 190 KB Parquet, fast to load)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pi():
    """Shared PortIndex for the test module."""
    from neptune_ais.ports import index
    return index()


class TestPortIndexSearch:
    def test_exact_unlocode(self, pi):
        results = pi.search("NLRTM")
        assert len(results) >= 1
        assert "Rotterdam" in results["name"].to_list()

    def test_name_search(self, pi):
        results = pi.search("Rotterdam")
        assert len(results) >= 1
        assert results["name"][0] == "Rotterdam"

    def test_no_results(self, pi):
        results = pi.search("ZZZZNONEXISTENT")
        assert len(results) == 0

    def test_empty_query(self, pi):
        results = pi.search("")
        assert len(results) == 0

    def test_limit(self, pi):
        results = pi.search("port", limit=5)
        assert len(results) <= 5


class TestPortIndexNear:
    def test_rotterdam_area(self, pi):
        results = pi.near(51.9, 4.5, radius_km=50)
        names = results["name"].to_list()
        assert "Rotterdam" in names
        assert "distance_km" in results.columns
        # Results should be sorted by distance
        dists = results["distance_km"].to_list()
        assert dists == sorted(dists)

    def test_no_ports_nearby(self, pi):
        # Middle of the Pacific
        results = pi.near(0.0, -150.0, radius_km=10)
        assert len(results) == 0

    def test_limit(self, pi):
        results = pi.near(51.9, 4.5, radius_km=500, limit=3)
        assert len(results) <= 3


class TestPortIndexLookups:
    def test_by_unlocode(self, pi):
        port = pi.by_unlocode("NLRTM")
        assert port is not None
        assert port.name == "Rotterdam"

    def test_by_unlocode_miss(self, pi):
        assert pi.by_unlocode("XXXXX") is None

    def test_by_wpi(self, pi):
        port = pi.by_wpi(31140)  # Rotterdam
        assert port is not None
        assert "Rotterdam" in port.name or "NLRTM" in (port.unlocode or "")

    def test_by_wpi_miss(self, pi):
        assert pi.by_wpi(999999) is None

    def test_get_unlocode(self, pi):
        port = pi.get("NLRTM")
        assert port is not None
        assert port.name == "Rotterdam"

    def test_get_wpi_string(self, pi):
        port = pi.get("31140")
        assert port is not None

    def test_get_name(self, pi):
        port = pi.get("Hamburg")
        assert port is not None
        assert port.name == "Hamburg"

    def test_get_miss(self, pi):
        assert pi.get("ZZZZZ") is None

    def test_by_country(self, pi):
        results = pi.by_country("NL")
        assert len(results) > 0
        assert all(c == "NL" for c in results["country_code"].to_list())

    def test_by_country_miss(self, pi):
        results = pi.by_country("XX")
        assert len(results) == 0


class TestPortIndexFilters:
    def test_in_bbox(self, pi):
        # North Sea bbox
        results = pi.in_bbox(west=3.0, south=51.0, east=5.0, north=52.0)
        assert len(results) > 0
        for row in results.iter_rows(named=True):
            assert 51.0 <= row["lat"] <= 52.0
            assert 3.0 <= row["lon"] <= 5.0

    def test_large_ports(self, pi):
        results = pi.large_ports()
        assert len(results) > 0
        assert all(s == "L" for s in results["harbor_size"].to_list())

    def test_with_facilities(self, pi):
        results = pi.with_facilities(has_drydock=True)
        assert len(results) > 0
        assert all(results["has_drydock"].to_list())


# ---------------------------------------------------------------------------
# EEZ queries
# ---------------------------------------------------------------------------


class TestEEZ:
    def test_eez_for_point_north_sea(self, pi):
        # Point in the Dutch North Sea
        eez = pi.eez_for_point(52.5, 4.0)
        assert eez is not None
        assert "Dutch" in eez.name or "Netherlands" in eez.sovereign

    def test_eez_for_point_open_ocean(self, pi):
        # Middle of the Pacific — might not be in any EEZ
        eez = pi.eez_for_point(0.0, -150.0)
        # Could be None or could match a distant EEZ bbox

    def test_eez_by_country(self, pi):
        eezs = pi.eez_by_country("NLD")
        assert len(eezs) >= 1
        assert all(isinstance(e, EEZRegion) for e in eezs)


# ---------------------------------------------------------------------------
# Destination resolver
# ---------------------------------------------------------------------------


class TestDestinationResolver:
    def test_exact_unlocode(self, pi):
        from neptune_ais.ports._destination import resolve_destination

        port = resolve_destination("NLRTM", pi)
        assert port is not None
        assert port.name == "Rotterdam"

    def test_unlocode_with_space(self, pi):
        from neptune_ais.ports._destination import resolve_destination

        port = resolve_destination("NL RTM", pi)
        assert port is not None
        assert port.name == "Rotterdam"

    def test_exact_name(self, pi):
        from neptune_ais.ports._destination import resolve_destination

        port = resolve_destination("ROTTERDAM", pi)
        assert port is not None
        assert port.name == "Rotterdam"

    def test_substring_match(self, pi):
        from neptune_ais.ports._destination import resolve_destination

        port = resolve_destination("PORT OF ROTTERDAM", pi)
        assert port is not None
        assert "Rotterdam" in port.name

    def test_no_match(self, pi):
        from neptune_ais.ports._destination import resolve_destination

        assert resolve_destination("XYZNONEXISTENT", pi) is None

    def test_empty_string(self, pi):
        from neptune_ais.ports._destination import resolve_destination

        assert resolve_destination("", pi) is None

    def test_garbage(self, pi):
        from neptune_ais.ports._destination import resolve_destination

        assert resolve_destination(">>><<<", pi) is None


class TestNormalize:
    def test_basic(self):
        from neptune_ais.ports._destination import _normalize

        assert _normalize("  Rotterdam  ") == "ROTTERDAM"

    def test_strips_punctuation(self):
        from neptune_ais.ports._destination import _normalize

        assert _normalize("ROTT/DAM>>") == "ROTTDAM"

    def test_collapses_whitespace(self):
        from neptune_ais.ports._destination import _normalize

        assert _normalize("PORT  OF   ROTT") == "PORT OF ROTT"

    def test_empty_input(self):
        from neptune_ais.ports._destination import _normalize

        assert _normalize("") == ""

    def test_all_symbols(self):
        from neptune_ais.ports._destination import _normalize

        assert _normalize(">>><<<///") == ""


class TestDestinationColumn:
    def test_vectorized(self, pi):
        from neptune_ais.ports._destination import resolve_destination_column

        dests = pl.Series(["NLRTM", "HAMBURG", None, "XYZXYZ", ""])
        result = resolve_destination_column(dests, pi)
        assert len(result) == 5
        assert result["resolved_port_name"][0] == "Rotterdam"
        assert result["resolved_port_name"][1] == "Hamburg"
        assert result["resolved_port_name"][2] is None
        assert result["resolved_port_name"][3] is None
        assert result["resolved_port_name"][4] is None
        assert result["match_confidence"][0] == 1.0  # exact UNLOCODE

    def test_empty_series(self, pi):
        from neptune_ais.ports._destination import resolve_destination_column

        result = resolve_destination_column(
            pl.Series([], dtype=pl.String), pi,
        )
        assert len(result) == 0
        assert "resolved_port_name" in result.columns


# ---------------------------------------------------------------------------
# Overlay merging
# ---------------------------------------------------------------------------


class TestOverlayMerge:
    def test_replacement(self):
        from neptune_ais.ports._loader import _merge_overlay

        base = pl.DataFrame({"wpi_number": [1, 2, 3], "name": ["A", "B", "C"]})
        overlay = pl.DataFrame({"wpi_number": [2], "name": ["B_UPDATED"]})
        result = _merge_overlay(base, overlay, key="wpi_number")
        assert len(result) == 3
        names = dict(zip(
            result["wpi_number"].to_list(),
            result["name"].to_list(),
        ))
        assert names[2] == "B_UPDATED"
        assert names[1] == "A"

    def test_addition(self):
        from neptune_ais.ports._loader import _merge_overlay

        base = pl.DataFrame({"wpi_number": [1], "name": ["A"]})
        overlay = pl.DataFrame({"wpi_number": [-1], "name": ["NEW"]})
        result = _merge_overlay(base, overlay, key="wpi_number")
        assert len(result) == 2

    def test_null_key_addition(self):
        from neptune_ais.ports._loader import _merge_overlay

        base = pl.DataFrame({"wpi_number": [1], "name": ["A"]})
        overlay = pl.DataFrame({
            "wpi_number": pl.Series([None], dtype=pl.Int64),
            "name": ["NEW"],
        })
        result = _merge_overlay(base, overlay, key="wpi_number")
        assert len(result) == 2

    def test_empty_overlay(self):
        from neptune_ais.ports._loader import _merge_overlay

        base = pl.DataFrame({"wpi_number": [1, 2], "name": ["A", "B"]})
        overlay = pl.DataFrame({"wpi_number": pl.Series([], dtype=pl.Int64), "name": pl.Series([], dtype=pl.String)})
        result = _merge_overlay(base, overlay, key="wpi_number")
        assert len(result) == 2
