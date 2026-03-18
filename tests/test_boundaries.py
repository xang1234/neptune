"""Tests for boundary datasets, registry, and spatial lookups.

Validates that BoundaryDataset/BoundaryRegion capture provenance,
BoundaryRegistry manages datasets, and point lookups work for
both bbox-only and geometry-enhanced regions.
"""

from __future__ import annotations

import polars as pl
import pytest

from neptune_ais.geometry.boundaries import (
    BoundaryDataset,
    BoundaryRegion,
    BoundaryRegistry,
    _bbox_contains,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _port_dataset() -> BoundaryDataset:
    """Synthetic port boundary dataset with two ports."""
    return BoundaryDataset(
        name="world_ports",
        version="2024.1",
        source_url="https://example.com/ports.geojson",
        description="World port boundaries",
        regions=(
            BoundaryRegion(
                name="Rotterdam",
                bbox=(3.8, 51.8, 4.5, 52.1),
            ),
            BoundaryRegion(
                name="Singapore",
                bbox=(103.6, 1.1, 104.1, 1.5),
            ),
        ),
    )


def _eez_dataset() -> BoundaryDataset:
    """Synthetic EEZ dataset with two zones."""
    return BoundaryDataset(
        name="eez",
        version="11.0",
        source_url="https://marineregions.org",
        description="Exclusive Economic Zones",
        regions=(
            BoundaryRegion(
                name="NLD_EEZ",
                bbox=(2.5, 51.0, 7.2, 55.8),
            ),
            BoundaryRegion(
                name="SGP_EEZ",
                bbox=(103.4, 1.0, 104.4, 1.6),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# BoundaryRegion
# ---------------------------------------------------------------------------


class TestBoundaryRegion:
    def test_creation(self):
        r = BoundaryRegion(name="test", bbox=(0.0, 0.0, 1.0, 1.0))
        assert r.name == "test"
        assert r.bbox == (0.0, 0.0, 1.0, 1.0)
        assert r.geometry is None

    def test_frozen(self):
        r = BoundaryRegion(name="test", bbox=(0.0, 0.0, 1.0, 1.0))
        with pytest.raises(AttributeError):
            r.name = "changed"


# ---------------------------------------------------------------------------
# BoundaryDataset
# ---------------------------------------------------------------------------


class TestBoundaryDataset:
    def test_creation(self):
        ds = _port_dataset()
        assert ds.name == "world_ports"
        assert ds.version == "2024.1"
        assert len(ds.regions) == 2

    def test_provenance_tag(self):
        ds = _port_dataset()
        assert ds.provenance_tag() == "world_ports/2024.1"

    def test_provenance_tag_format(self):
        ds = _eez_dataset()
        tag = ds.provenance_tag()
        assert "/" in tag
        assert tag == "eez/11.0"

    def test_frozen(self):
        ds = _port_dataset()
        with pytest.raises(AttributeError):
            ds.name = "changed"


# ---------------------------------------------------------------------------
# BoundaryRegistry
# ---------------------------------------------------------------------------


class TestBoundaryRegistry:
    def test_register_and_get(self):
        reg = BoundaryRegistry()
        ds = _port_dataset()
        reg.register(ds)
        assert reg.get("world_ports") is ds

    def test_get_missing_returns_none(self):
        reg = BoundaryRegistry()
        assert reg.get("nonexistent") is None

    def test_register_replaces(self):
        reg = BoundaryRegistry()
        ds1 = _port_dataset()
        ds2 = BoundaryDataset(name="world_ports", version="2025.1")
        reg.register(ds1)
        reg.register(ds2)
        assert reg.get("world_ports").version == "2025.1"

    def test_datasets_property(self):
        reg = BoundaryRegistry()
        reg.register(_port_dataset())
        reg.register(_eez_dataset())
        assert len(reg.datasets) == 2

    def test_provenance_tags(self):
        reg = BoundaryRegistry()
        reg.register(_port_dataset())
        reg.register(_eez_dataset())
        tags = reg.provenance_tags()
        assert tags == ["eez/11.0", "world_ports/2024.1"]

    def test_provenance_tags_empty(self):
        reg = BoundaryRegistry()
        assert reg.provenance_tags() == []


# ---------------------------------------------------------------------------
# Bounding-box containment
# ---------------------------------------------------------------------------


class TestBboxContains:
    def test_inside(self):
        assert _bbox_contains((0.0, 0.0, 10.0, 10.0), 5.0, 5.0)

    def test_outside(self):
        assert not _bbox_contains((0.0, 0.0, 10.0, 10.0), 15.0, 5.0)

    def test_on_boundary_inclusive(self):
        assert _bbox_contains((0.0, 0.0, 10.0, 10.0), 0.0, 0.0)
        assert _bbox_contains((0.0, 0.0, 10.0, 10.0), 10.0, 10.0)

    def test_outside_longitude(self):
        assert not _bbox_contains((0.0, 0.0, 10.0, 10.0), 5.0, 15.0)


# ---------------------------------------------------------------------------
# Point lookup — bbox only
# ---------------------------------------------------------------------------


class TestLookupBboxOnly:
    """Lookup tests using bbox-only regions (no shapely needed)."""

    def test_point_in_rotterdam(self):
        reg = BoundaryRegistry()
        reg.register(_port_dataset())
        matches = reg.lookup(lat=51.9, lon=4.0)
        assert ("world_ports", "Rotterdam") in matches

    def test_point_in_singapore(self):
        reg = BoundaryRegistry()
        reg.register(_port_dataset())
        matches = reg.lookup(lat=1.3, lon=103.8)
        assert ("world_ports", "Singapore") in matches

    def test_point_in_ocean(self):
        reg = BoundaryRegistry()
        reg.register(_port_dataset())
        matches = reg.lookup(lat=20.0, lon=-30.0)
        assert matches == []

    def test_point_in_multiple_datasets(self):
        """A point can match regions from multiple datasets."""
        reg = BoundaryRegistry()
        reg.register(_port_dataset())
        reg.register(_eez_dataset())
        # Rotterdam is inside both the port bbox and the NLD_EEZ bbox.
        matches = reg.lookup(lat=51.9, lon=4.0)
        dataset_names = {m[0] for m in matches}
        assert "world_ports" in dataset_names
        assert "eez" in dataset_names

    def test_point_in_overlapping_regions(self):
        """When bboxes overlap, both regions are returned."""
        ds = BoundaryDataset(
            name="overlap_test",
            version="1.0",
            regions=(
                BoundaryRegion(name="A", bbox=(0.0, 0.0, 10.0, 10.0)),
                BoundaryRegion(name="B", bbox=(5.0, 5.0, 15.0, 15.0)),
            ),
        )
        reg = BoundaryRegistry()
        reg.register(ds)
        matches = reg.lookup(lat=7.0, lon=7.0)
        region_names = {m[1] for m in matches}
        assert region_names == {"A", "B"}


# ---------------------------------------------------------------------------
# Column lookup
# ---------------------------------------------------------------------------


class TestLookupColumn:
    def test_basic_lookup(self):
        reg = BoundaryRegistry()
        reg.register(_port_dataset())

        df = pl.DataFrame({
            "lat": [51.9, 1.3, 20.0],
            "lon": [4.0, 103.8, -30.0],
        })

        result = reg.lookup_column(df, "world_ports")
        assert result.to_list() == ["Rotterdam", "Singapore", None]

    def test_missing_dataset_returns_nulls(self):
        reg = BoundaryRegistry()
        df = pl.DataFrame({"lat": [51.9], "lon": [4.0]})
        result = reg.lookup_column(df, "nonexistent")
        assert result.to_list() == [None]

    def test_custom_column_names(self):
        reg = BoundaryRegistry()
        reg.register(_port_dataset())

        df = pl.DataFrame({
            "latitude": [51.9],
            "longitude": [4.0],
        })

        result = reg.lookup_column(
            df, "world_ports", lat_col="latitude", lon_col="longitude"
        )
        assert result.to_list() == ["Rotterdam"]

    def test_result_is_string_series(self):
        reg = BoundaryRegistry()
        reg.register(_port_dataset())
        df = pl.DataFrame({"lat": [51.9], "lon": [4.0]})
        result = reg.lookup_column(df, "world_ports")
        assert result.dtype == pl.String
        assert result.name == "region"

    def test_empty_dataframe(self):
        reg = BoundaryRegistry()
        reg.register(_port_dataset())
        df = pl.DataFrame({"lat": pl.Series([], dtype=pl.Float64),
                           "lon": pl.Series([], dtype=pl.Float64)})
        result = reg.lookup_column(df, "world_ports")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Provenance integration
# ---------------------------------------------------------------------------


class TestProvenanceIntegration:
    """Verify boundary provenance flows into EventProvenance."""

    def test_provenance_tag_in_event_upstream(self):
        from neptune_ais.derive.events import EventProvenance

        reg = BoundaryRegistry()
        reg.register(_eez_dataset())
        tags = reg.provenance_tags()

        prov = EventProvenance(
            source="noaa",
            detector="eez_detector",
            detector_version="0.1.0",
            upstream_datasets=["tracks", "boundaries"],
        )
        token = prov.to_token()
        assert "boundaries" in token
        assert "tracks" in token

    def test_boundary_version_traceable(self):
        """Boundary version can be recorded alongside event provenance."""
        ds = _eez_dataset()
        assert ds.provenance_tag() == "eez/11.0"
        # A detector would include this in its config or notes.
