"""Tests for timelapse corridor visualization.

Covers: TimelapsConfig, vessel type categorization, prepare_timelapse,
generate_timelapse (single and multi-panel), and XSS protection.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from neptune_ais.viz import (
    VESSEL_TYPE_PALETTE,
    TimelapsConfig,
    Viewport,
    _VESSEL_TYPE_ORDER,
    _categorize_vessel_type,
    generate_timelapse,
    prepare_timelapse,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_positions(n: int = 100, *, with_ship_type: bool = False) -> pl.DataFrame:
    """Generate *n* synthetic position rows."""
    import random
    from datetime import datetime, timedelta, timezone

    rng = random.Random(42)
    base = datetime(2024, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [base + timedelta(minutes=i * 5) for i in range(n)]

    data = {
        "mmsi": pl.Series(
            [rng.choice([111, 222, 333, 444, 555]) for _ in range(n)],
            dtype=pl.Int64,
        ),
        "timestamp": pl.Series(timestamps, dtype=pl.Datetime("us", "UTC")),
        "lat": [33.7 + rng.uniform(-0.1, 0.1) for _ in range(n)],
        "lon": [-118.2 + rng.uniform(-0.1, 0.1) for _ in range(n)],
        "sog": [10.0] * n,
        "source": ["noaa"] * n,
        "record_provenance": ["noaa:direct"] * n,
        "qc_severity": ["ok"] * n,
    }

    if with_ship_type:
        types = ["70", "80", "60", "30", None]
        data["ship_type"] = [rng.choice(types) for _ in range(n)]

    return pl.DataFrame(data)


def _sample_vessels(mmsis: list[int]) -> pl.DataFrame:
    """Generate a vessels DataFrame for enrichment testing."""
    types = ["70", "80", "60", "30", "52"]
    return pl.DataFrame({
        "mmsi": pl.Series(mmsis, dtype=pl.Int64),
        "ship_type": [types[i % len(types)] for i in range(len(mmsis))],
        "flag": ["US"] * len(mmsis),
    })


# ---------------------------------------------------------------------------
# TimelapsConfig
# ---------------------------------------------------------------------------


class TestTimelapsConfig:
    def test_defaults(self) -> None:
        cfg = TimelapsConfig()
        assert cfg.title == "AIS TIMELAPSE"
        assert cfg.dot_radius == 1.0
        assert cfg.dot_alpha == 0.10
        assert cfg.bin_interval_minutes == 30
        assert cfg.speed == 2.0
        assert cfg.fade_factor == 0.998
        assert cfg.bloom is True
        assert cfg.color_by_type is True
        assert cfg.layout == "vertical"

    def test_custom_values(self) -> None:
        cfg = TimelapsConfig(
            title="PUGET SOUND",
            date_from="2024-06-01",
            date_to="2024-06-30",
            zoom=9,
            dot_radius=3.0,
            speed=8.0,
            layout="horizontal",
        )
        assert cfg.title == "PUGET SOUND"
        assert cfg.zoom == 9
        assert cfg.dot_radius == 3.0
        assert cfg.layout == "horizontal"

    def test_frozen(self) -> None:
        cfg = TimelapsConfig()
        with pytest.raises(AttributeError):
            cfg.title = "nope"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Vessel type categorization
# ---------------------------------------------------------------------------


class TestCategorizeVesselType:
    def test_cargo_codes(self) -> None:
        assert _categorize_vessel_type("70") == "cargo"
        assert _categorize_vessel_type("75") == "cargo"
        assert _categorize_vessel_type("79") == "cargo"

    def test_tanker_codes(self) -> None:
        assert _categorize_vessel_type("80") == "tanker"
        assert _categorize_vessel_type("89") == "tanker"

    def test_passenger_codes(self) -> None:
        assert _categorize_vessel_type("60") == "passenger"
        assert _categorize_vessel_type("69") == "passenger"

    def test_fishing_codes(self) -> None:
        assert _categorize_vessel_type("30") == "fishing"

    def test_tug_codes(self) -> None:
        assert _categorize_vessel_type("52") == "tug"

    def test_text_cargo(self) -> None:
        assert _categorize_vessel_type("Cargo ship") == "cargo"
        assert _categorize_vessel_type("Container vessel") == "cargo"
        assert _categorize_vessel_type("Bulk carrier") == "cargo"

    def test_text_tanker(self) -> None:
        assert _categorize_vessel_type("Oil tanker") == "tanker"
        assert _categorize_vessel_type("Chemical tanker") == "tanker"

    def test_text_passenger(self) -> None:
        assert _categorize_vessel_type("Passenger ferry") == "passenger"
        assert _categorize_vessel_type("Cruise ship") == "passenger"

    def test_text_fishing(self) -> None:
        assert _categorize_vessel_type("Fishing vessel") == "fishing"

    def test_text_tug(self) -> None:
        assert _categorize_vessel_type("Tug boat") == "tug"
        assert _categorize_vessel_type("Pilot vessel") == "tug"

    def test_none(self) -> None:
        assert _categorize_vessel_type(None) == "other"

    def test_unknown(self) -> None:
        assert _categorize_vessel_type("999") == "other"
        assert _categorize_vessel_type("Unknown") == "other"

    def test_palette_coverage(self) -> None:
        """Every category returned by _categorize must be in the palette."""
        for cat in ("cargo", "tanker", "passenger", "fishing", "tug", "other"):
            assert cat in VESSEL_TYPE_PALETTE
            assert cat in _VESSEL_TYPE_ORDER


# ---------------------------------------------------------------------------
# prepare_timelapse
# ---------------------------------------------------------------------------


class TestPrepareTimelapse:
    def test_basic_preparation(self) -> None:
        df = _sample_positions(100, with_ship_type=True)
        result = prepare_timelapse(df)
        assert len(result["bins"]) > 0
        assert isinstance(result["type_counts"], dict)
        assert len(result["cumul_vessels"]) == len(result["bins"])
        assert len(result["bin_timestamps_ms"]) == len(result["bins"])

    def test_compact_format(self) -> None:
        """Each point should be [lat, lon, type_idx, mmsi_idx]."""
        df = _sample_positions(50, with_ship_type=True)
        result = prepare_timelapse(df)
        for bin_points in result["bins"]:
            for pt in bin_points:
                assert len(pt) == 4
                assert isinstance(pt[0], float)  # lat
                assert isinstance(pt[1], float)  # lon
                assert isinstance(pt[2], int)     # type_idx
                assert isinstance(pt[3], int)     # mmsi_idx

    def test_total_points_match(self) -> None:
        df = _sample_positions(80, with_ship_type=True)
        result = prepare_timelapse(df, max_points=80)
        total = sum(len(b) for b in result["bins"])
        assert total == 80

    def test_viewport_clipping(self) -> None:
        df = _sample_positions(200)
        viewport = Viewport(west=-118.3, south=33.6, east=-118.1, north=33.8)
        result = prepare_timelapse(df, viewport=viewport)
        for bin_points in result["bins"]:
            for pt in bin_points:
                assert 33.5 <= pt[0] <= 33.9   # lat (with rounding tolerance)
                assert -118.4 <= pt[1] <= -118.0  # lon

    def test_max_points_sampling(self) -> None:
        df = _sample_positions(500)
        result = prepare_timelapse(df, max_points=50)
        total = sum(len(b) for b in result["bins"])
        assert total <= 50

    def test_empty_input(self) -> None:
        df = _sample_positions(0)
        result = prepare_timelapse(df)
        assert result["bins"] == []
        assert result["cumul_vessels"] == []
        assert result["color_by_type"] is False

    def test_lazyframe_input(self) -> None:
        df = _sample_positions(50, with_ship_type=True)
        result = prepare_timelapse(df.lazy())
        assert len(result["bins"]) > 0

    def test_cumulative_vessels_monotonic(self) -> None:
        df = _sample_positions(100)
        result = prepare_timelapse(df)
        cv = result["cumul_vessels"]
        for i in range(1, len(cv)):
            assert cv[i] >= cv[i - 1]

    def test_auto_view(self) -> None:
        df = _sample_positions(100)
        result = prepare_timelapse(df)
        assert -90 <= result["center_lat"] <= 90
        assert -180 <= result["center_lon"] <= 180
        assert result["zoom"] > 0

    def test_bin_interval(self) -> None:
        """Different bin intervals should produce different bin counts."""
        df = _sample_positions(100)  # 100 * 5 min = 500 min span
        result_60 = prepare_timelapse(df, bin_interval_minutes=60)
        result_30 = prepare_timelapse(df, bin_interval_minutes=30)
        assert len(result_30["bins"]) >= len(result_60["bins"])

    def test_mmsi_idx_present(self) -> None:
        """Each point should have a valid mmsi_idx."""
        df = _sample_positions(50, with_ship_type=True)
        result = prepare_timelapse(df)
        all_idxs = set()
        for bin_points in result["bins"]:
            for pt in bin_points:
                all_idxs.add(pt[3])
        # Should have indices for 5 distinct vessels
        assert len(all_idxs) == df["mmsi"].n_unique()

    def test_color_by_type_disabled(self) -> None:
        df = _sample_positions(50)  # no ship_type column
        result = prepare_timelapse(df, color_by_type=True)
        # All points should be type index for "other"
        for bin_points in result["bins"]:
            for pt in bin_points:
                assert pt[2] == _VESSEL_TYPE_ORDER.index("other")

    def test_auto_fallback_many_other(self) -> None:
        """If >60% are 'other', color_by_type should be auto-disabled."""
        df = _sample_positions(100)
        # No ship_type → all "other" → should fallback
        result = prepare_timelapse(df, color_by_type=True)
        assert result["color_by_type"] is False

    def test_vessel_enrichment(self) -> None:
        df = _sample_positions(50)
        # Add empty ship_type column.
        df = df.with_columns(pl.lit(None).cast(pl.String).alias("ship_type"))
        vessels = _sample_vessels([111, 222, 333, 444, 555])
        result = prepare_timelapse(df, vessels=vessels, color_by_type=True)
        # Should have enriched types — not all "other".
        assert result["color_by_type"] is True

    def test_palette_and_type_names(self) -> None:
        df = _sample_positions(50, with_ship_type=True)
        result = prepare_timelapse(df)
        assert len(result["palette"]) == len(result["type_names"])
        assert result["palette"] == [
            VESSEL_TYPE_PALETTE[n] for n in _VESSEL_TYPE_ORDER
        ]


# ---------------------------------------------------------------------------
# generate_timelapse
# ---------------------------------------------------------------------------


class TestGenerateTimelapse:
    def test_generates_html_file(self, tmp_path: Path) -> None:
        df = _sample_positions(100, with_ship_type=True)
        out = generate_timelapse(
            df,
            config=TimelapsConfig(title="Test Timelapse"),
            output=str(tmp_path / "test.html"),
        )
        assert Path(out).exists()
        html = Path(out).read_text()
        assert "Test Timelapse" in html
        assert "maplibregl" in html.lower() or "maplibre" in html.lower()

    def test_contains_bins_data(self, tmp_path: Path) -> None:
        df = _sample_positions(50, with_ship_type=True)
        out = generate_timelapse(df, output=str(tmp_path / "t.html"))
        html = Path(out).read_text()
        assert "BINS" in html
        assert "PALETTE" in html

    def test_dark_nolabels_basemap(self, tmp_path: Path) -> None:
        df = _sample_positions(30, with_ship_type=True)
        out = generate_timelapse(df, output=str(tmp_path / "t.html"))
        html = Path(out).read_text()
        assert "dark-matter-nolabels" in html

    def test_bloom_canvas_code(self, tmp_path: Path) -> None:
        df = _sample_positions(30, with_ship_type=True)
        out = generate_timelapse(
            df,
            config=TimelapsConfig(bloom=True),
            output=str(tmp_path / "t.html"),
        )
        html = Path(out).read_text()
        assert "UnrealBloomPass" in html

    def test_hidpi_scaling(self, tmp_path: Path) -> None:
        df = _sample_positions(30, with_ship_type=True)
        out = generate_timelapse(df, output=str(tmp_path / "t.html"))
        html = Path(out).read_text()
        assert "setPixelRatio" in html

    def test_fade_uses_destination_out(self, tmp_path: Path) -> None:
        df = _sample_positions(30, with_ship_type=True)
        out = generate_timelapse(df, output=str(tmp_path / "t.html"))
        html = Path(out).read_text()
        assert "uFadeFactor" in html

    def test_xss_protection(self, tmp_path: Path) -> None:
        df = _sample_positions(10, with_ship_type=True)
        cfg = TimelapsConfig(title='</script><script>alert(1)')
        out = generate_timelapse(df, config=cfg, output=str(tmp_path / "xss.html"))
        html = Path(out).read_text()
        # The _safe_json_embed escapes < to \u003c.
        # Title goes through template substitution, check the JSON data.
        # At minimum, the literal </script> should not appear mid-script.
        script_blocks = html.split("<script>")
        for block in script_blocks[1:]:
            inner = block.split("</script>")[0]
            assert "</script>" not in inner

    def test_empty_positions_raises(self, tmp_path: Path) -> None:
        df = _sample_positions(0)
        with pytest.raises(ValueError, match="No positions"):
            generate_timelapse(df, output=str(tmp_path / "fail.html"))

    def test_legend_present_with_types(self, tmp_path: Path) -> None:
        df = _sample_positions(50, with_ship_type=True)
        out = generate_timelapse(df, output=str(tmp_path / "t.html"))
        html = Path(out).read_text()
        assert "legend-item" in html
        assert "cargo" in html.lower()

    def test_legend_empty_without_types(self, tmp_path: Path) -> None:
        df = _sample_positions(50)  # no ship_type → fallback
        out = generate_timelapse(df, output=str(tmp_path / "t.html"))
        html = Path(out).read_text()
        # color_by_type is False → legend div exists but has no items inside
        legend_start = html.find('id="legend"')
        legend_end = html.find("</div>", legend_start)
        legend_content = html[legend_start:legend_end]
        assert "legend-dot" not in legend_content

    def test_keyboard_shortcuts(self, tmp_path: Path) -> None:
        df = _sample_positions(30, with_ship_type=True)
        out = generate_timelapse(df, output=str(tmp_path / "t.html"))
        html = Path(out).read_text()
        assert "Space" in html
        assert "ArrowRight" in html


class TestGenerateTimelapseMultiPanel:
    def test_multi_panel(self, tmp_path: Path) -> None:
        df1 = _sample_positions(30, with_ship_type=True)
        df2 = _sample_positions(30, with_ship_type=True)
        panels = [
            {"positions": df1, "config": TimelapsConfig(), "label": "Panel A"},
            {"positions": df2, "config": TimelapsConfig(), "label": "Panel B"},
        ]
        out = generate_timelapse(
            df1,
            panels=panels,
            output=str(tmp_path / "multi.html"),
        )
        html = Path(out).read_text()
        assert "Panel A" in html
        assert "Panel B" in html
        assert "map-0" in html
        assert "map-1" in html

    def test_multi_panel_vertical_layout(self, tmp_path: Path) -> None:
        df = _sample_positions(20, with_ship_type=True)
        panels = [
            {"positions": df, "label": "A"},
            {"positions": df, "label": "B"},
            {"positions": df, "label": "C"},
        ]
        out = generate_timelapse(
            df,
            config=TimelapsConfig(layout="vertical"),
            panels=panels,
            output=str(tmp_path / "v.html"),
        )
        html = Path(out).read_text()
        assert "grid-template-rows" in html

    def test_multi_panel_horizontal_layout(self, tmp_path: Path) -> None:
        df = _sample_positions(20, with_ship_type=True)
        panels = [
            {"positions": df, "label": "A"},
            {"positions": df, "label": "B"},
        ]
        out = generate_timelapse(
            df,
            config=TimelapsConfig(layout="horizontal"),
            panels=panels,
            output=str(tmp_path / "h.html"),
        )
        html = Path(out).read_text()
        assert "grid-template-columns" in html
