"""Regression and golden tests for fusion merge modes, dedup, and provenance.

Covers: union, best, prefer:<source>, field precedence, confidence weights,
coordinate dedup, provenance token format, and edge cases.
"""

from __future__ import annotations

import polars as pl
import pytest

from neptune_ais.fusion import (
    FusionConfig,
    MergeMode,
    compute_dedup_buckets,
    dedup_subset_columns,
    merge,
    parse_merge_arg,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable multi-source test data
# ---------------------------------------------------------------------------

def _noaa_positions() -> pl.DataFrame:
    """NOAA positions: vessel 111 at T+0, T+5min; vessel 222 (noaa-only)."""
    return pl.DataFrame({
        "mmsi": pl.Series([111, 111, 222], dtype=pl.Int64),
        "timestamp": pl.Series([
            "2024-06-15T00:00:00",
            "2024-06-15T00:05:00",
            "2024-06-15T00:00:00",
        ], dtype=pl.Datetime("us", "UTC")),
        "lat": [40.0, 40.01, 55.0],
        "lon": [-74.0, -74.01, 12.0],
        "sog": [10.0, 11.0, 0.0],
        "source": ["noaa"] * 3,
        "record_provenance": ["noaa:direct"] * 3,
        "vessel_name": ["GOOD NAME", "GOOD NAME", "US VESSEL"],
        "ship_type": ["70", "70", "80"],
    })


def _dma_positions() -> pl.DataFrame:
    """DMA positions: vessel 111 near-duplicates (+10s, +5s); vessel 333 (dma-only)."""
    return pl.DataFrame({
        "mmsi": pl.Series([111, 111, 333], dtype=pl.Int64),
        "timestamp": pl.Series([
            "2024-06-15T00:00:10",
            "2024-06-15T00:05:05",
            "2024-06-15T00:00:00",
        ], dtype=pl.Datetime("us", "UTC")),
        "lat": [40.001, 40.011, 56.0],
        "lon": [-73.999, -74.009, 10.0],
        "sog": [10.1, 10.9, 0.0],
        "source": ["dma"] * 3,
        "record_provenance": ["dma:direct"] * 3,
        "vessel_name": ["DMA NAME", "DMA NAME", "DK VESSEL"],
        "ship_type": ["Cargo", "Cargo", "Fishing"],
    })


def _frames() -> dict[str, pl.DataFrame]:
    return {"noaa": _noaa_positions(), "dma": _dma_positions()}


# ---------------------------------------------------------------------------
# parse_merge_arg
# ---------------------------------------------------------------------------


def test_parse_union():
    c = parse_merge_arg("union")
    assert c.mode == MergeMode.UNION


def test_parse_best():
    c = parse_merge_arg("best", ["noaa", "dma"])
    assert c.mode == MergeMode.BEST
    assert c.source_precedence == ["noaa", "dma"]


def test_parse_prefer():
    c = parse_merge_arg("prefer:noaa", ["noaa", "dma"])
    assert c.mode == MergeMode.PREFER
    assert c.prefer_source == "noaa"


def test_parse_case_insensitive():
    c = parse_merge_arg("BEST")
    assert c.mode == MergeMode.BEST


def test_parse_unknown_raises():
    try:
        parse_merge_arg("bogus")
        assert False, "Should have raised"
    except ValueError:
        pass


def test_parse_prefer_empty_raises():
    try:
        parse_merge_arg("prefer:")
        assert False, "Should have raised"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# FusionConfig validation
# ---------------------------------------------------------------------------


def test_config_prefer_requires_source():
    try:
        FusionConfig(mode=MergeMode.PREFER)
        assert False, "Should have raised"
    except ValueError as e:
        assert "prefer_source" in str(e)


def test_config_prefer_source_wrong_mode():
    try:
        FusionConfig(mode=MergeMode.UNION, prefer_source="noaa")
        assert False, "Should have raised"
    except ValueError as e:
        assert "PREFER mode" in str(e)


def test_config_negative_tolerance():
    try:
        FusionConfig(timestamp_tolerance_seconds=-1)
        assert False, "Should have raised"
    except ValueError:
        pass


def test_config_bad_weight():
    try:
        FusionConfig(source_confidence_weights={"noaa": 1.5})
        assert False, "Should have raised"
    except ValueError:
        pass


def test_config_protected_field_precedence():
    try:
        FusionConfig(field_precedence={"source": ["dma", "noaa"]})
        assert False, "Should have raised"
    except ValueError as e:
        assert "protected column" in str(e)


def test_config_valid_field_precedence():
    c = FusionConfig(field_precedence={"vessel_name": ["noaa", "dma"]})
    assert c.field_precedence == {"vessel_name": ["noaa", "dma"]}


# ---------------------------------------------------------------------------
# Union merge — golden tests
# ---------------------------------------------------------------------------


def test_union_preserves_all_rows():
    config = FusionConfig(mode=MergeMode.UNION)
    result = merge(_frames(), config)
    assert len(result) == 6


def test_union_tags_provenance():
    config = FusionConfig(mode=MergeMode.UNION)
    result = merge(_frames(), config)
    provs = set(result["record_provenance"].to_list())
    assert "noaa:union" in provs
    assert "dma:union" in provs


def test_union_no_provenance():
    config = FusionConfig(mode=MergeMode.UNION, tag_provenance=False)
    result = merge(_frames(), config)
    # Should retain original provenance.
    provs = set(result["record_provenance"].to_list())
    assert "noaa:direct" in provs


def test_union_aligns_columns():
    """Union with mismatched columns should fill missing with null."""
    noaa = _noaa_positions()
    dma = _dma_positions().with_columns(pl.lit("CPH").alias("destination"))
    config = FusionConfig(mode=MergeMode.UNION)
    result = merge({"noaa": noaa, "dma": dma}, config)
    assert "destination" in result.columns
    assert len(result) == 6


# ---------------------------------------------------------------------------
# Best merge — golden tests
# ---------------------------------------------------------------------------


def test_best_deduplicates():
    config = FusionConfig(
        mode=MergeMode.BEST,
        source_precedence=["noaa", "dma"],
        timestamp_tolerance_seconds=30,
    )
    result = merge(_frames(), config)
    # 111: 2 near-dup pairs → 2, 222: 1, 333: 1 = 4
    assert len(result) == 4


def test_best_respects_precedence_noaa_first():
    config = FusionConfig(
        mode=MergeMode.BEST,
        source_precedence=["noaa", "dma"],
        timestamp_tolerance_seconds=30,
    )
    result = merge(_frames(), config)
    mmsi111 = result.filter(pl.col("mmsi") == 111)
    assert all(s == "noaa" for s in mmsi111["source"].to_list())


def test_best_respects_precedence_dma_first():
    config = FusionConfig(
        mode=MergeMode.BEST,
        source_precedence=["dma", "noaa"],
        timestamp_tolerance_seconds=30,
    )
    result = merge(_frames(), config)
    mmsi111 = result.filter(pl.col("mmsi") == 111)
    assert all(s == "dma" for s in mmsi111["source"].to_list())


def test_best_preserves_single_source_rows():
    config = FusionConfig(
        mode=MergeMode.BEST,
        source_precedence=["noaa", "dma"],
        timestamp_tolerance_seconds=30,
    )
    result = merge(_frames(), config)
    assert len(result.filter(pl.col("mmsi") == 222)) == 1  # noaa-only
    assert len(result.filter(pl.col("mmsi") == 333)) == 1  # dma-only


def test_best_provenance_multi_source():
    config = FusionConfig(
        mode=MergeMode.BEST,
        source_precedence=["noaa", "dma"],
        timestamp_tolerance_seconds=30,
    )
    result = merge(_frames(), config)
    mmsi111 = result.filter(pl.col("mmsi") == 111)
    provs = mmsi111["record_provenance"].to_list()
    # Multi-source observations should have contributor brackets.
    assert all("[+" in p for p in provs)
    assert all("noaa:best" in p for p in provs)


def test_best_provenance_single_source():
    config = FusionConfig(
        mode=MergeMode.BEST,
        source_precedence=["noaa", "dma"],
        timestamp_tolerance_seconds=30,
    )
    result = merge(_frames(), config)
    mmsi222 = result.filter(pl.col("mmsi") == 222)
    prov = mmsi222["record_provenance"][0]
    assert prov == "noaa:best"
    assert "[+" not in prov


# ---------------------------------------------------------------------------
# Prefer merge — golden tests
# ---------------------------------------------------------------------------


def test_prefer_overrides_source_order():
    """prefer:noaa wins even when sources list dma first."""
    config = FusionConfig(
        mode=MergeMode.PREFER,
        prefer_source="noaa",
        source_precedence=["dma", "noaa"],
        timestamp_tolerance_seconds=30,
    )
    result = merge(_frames(), config)
    mmsi111 = result.filter(pl.col("mmsi") == 111)
    assert all(s == "noaa" for s in mmsi111["source"].to_list())


def test_prefer_provenance_tag():
    config = FusionConfig(
        mode=MergeMode.PREFER,
        prefer_source="noaa",
        timestamp_tolerance_seconds=30,
    )
    result = merge(_frames(), config)
    mmsi111 = result.filter(pl.col("mmsi") == 111)
    provs = mmsi111["record_provenance"].to_list()
    assert all(":prefer:noaa" in p for p in provs)


# ---------------------------------------------------------------------------
# Field precedence — golden tests
# ---------------------------------------------------------------------------


def test_field_precedence_per_field_winner():
    config = FusionConfig(
        mode=MergeMode.BEST,
        source_precedence=["noaa", "dma"],
        timestamp_tolerance_seconds=30,
        field_precedence={
            "vessel_name": ["noaa", "dma"],
            "ship_type": ["dma", "noaa"],
        },
    )
    result = merge(_frames(), config)
    mmsi111 = result.filter(pl.col("mmsi") == 111)

    # vessel_name from NOAA (higher precedence for this field).
    assert all(v == "GOOD NAME" for v in mmsi111["vessel_name"].to_list())

    # ship_type from DMA (higher precedence for this field).
    assert all(v == "Cargo" for v in mmsi111["ship_type"].to_list())


def test_field_precedence_null_fill():
    """When preferred source has null, falls through to next source."""
    noaa = _noaa_positions().with_columns(
        pl.lit(None).cast(pl.String).alias("ship_type")
    )
    config = FusionConfig(
        mode=MergeMode.BEST,
        source_precedence=["noaa", "dma"],
        timestamp_tolerance_seconds=30,
        field_precedence={"ship_type": ["noaa", "dma"]},
    )
    result = merge({"noaa": noaa, "dma": _dma_positions()}, config)
    mmsi111 = result.filter(pl.col("mmsi") == 111)
    # NOAA has null ship_type → falls through to DMA.
    assert all(v == "Cargo" for v in mmsi111["ship_type"].to_list())


# ---------------------------------------------------------------------------
# Confidence weights — golden tests
# ---------------------------------------------------------------------------


def test_confidence_weights_applied():
    config = FusionConfig(
        mode=MergeMode.BEST,
        source_precedence=["noaa", "dma"],
        timestamp_tolerance_seconds=30,
        source_confidence_weights={"noaa": 1.0, "dma": 0.8},
    )
    result = merge(_frames(), config)
    noaa_rows = result.filter(pl.col("source") == "noaa")
    dma_rows = result.filter(pl.col("source") == "dma")
    assert all(c == 1.0 for c in noaa_rows["confidence_score"].to_list())
    assert all(c == 0.8 for c in dma_rows["confidence_score"].to_list())


# ---------------------------------------------------------------------------
# Coordinate dedup — golden tests
# ---------------------------------------------------------------------------


def test_coordinate_dedup_same_location():
    """Near-duplicate observations at same location are deduped."""
    config = FusionConfig(
        mode=MergeMode.BEST,
        source_precedence=["noaa", "dma"],
        timestamp_tolerance_seconds=30,
        coordinate_tolerance_degrees=0.01,
    )
    result = merge(_frames(), config)
    assert len(result) == 4  # Same as time-only dedup (coords are close)


def test_coordinate_dedup_different_location():
    """Same MMSI + time but far apart should NOT be deduped."""
    noaa = pl.DataFrame({
        "mmsi": pl.Series([111, 111], dtype=pl.Int64),
        "timestamp": pl.Series(["2024-06-15T00:00:00"] * 2, dtype=pl.Datetime("us", "UTC")),
        "lat": [40.0, 45.0],  # 5° apart
        "lon": [-74.0, -74.0],
        "source": ["noaa"] * 2,
        "record_provenance": ["noaa:direct"] * 2,
    })
    dma = pl.DataFrame({
        "mmsi": pl.Series([111], dtype=pl.Int64),
        "timestamp": pl.Series(["2024-06-15T00:00:05"], dtype=pl.Datetime("us", "UTC")),
        "lat": [40.001],
        "lon": [-73.999],
        "source": ["dma"],
        "record_provenance": ["dma:direct"],
    })
    config = FusionConfig(
        mode=MergeMode.BEST,
        source_precedence=["noaa", "dma"],
        timestamp_tolerance_seconds=30,
        coordinate_tolerance_degrees=0.01,
    )
    result = merge({"noaa": noaa, "dma": dma}, config)
    # noaa@40° + dma@40.001° → dedup to 1. noaa@45° → unique. Total: 2.
    assert len(result) == 2


def test_coordinate_dedup_disabled():
    """With coordinate_tolerance=0, only time+mmsi dedup."""
    config = FusionConfig(
        mode=MergeMode.BEST,
        source_precedence=["noaa", "dma"],
        timestamp_tolerance_seconds=30,
        coordinate_tolerance_degrees=0.0,
    )
    result = merge(_frames(), config)
    assert len(result) == 4


# ---------------------------------------------------------------------------
# Single source — edge cases
# ---------------------------------------------------------------------------


def test_single_source_no_fusion():
    config = FusionConfig(mode=MergeMode.BEST)
    result = merge({"noaa": _noaa_positions()}, config)
    assert len(result) == 3
    assert all(":only" in p for p in result["record_provenance"].to_list())


def test_empty_frames_raises():
    try:
        merge({}, FusionConfig(mode=MergeMode.UNION))
        assert False, "Should have raised"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Dedup bucket internals
# ---------------------------------------------------------------------------


def test_dedup_buckets_timestamp():
    df = pl.DataFrame({
        "mmsi": [111, 111],
        "timestamp": pl.Series(["2024-06-15T00:00:00", "2024-06-15T00:00:10"],
                               dtype=pl.Datetime("us", "UTC")),
    })
    config = FusionConfig(timestamp_tolerance_seconds=30)
    bucketed = compute_dedup_buckets(df, config)
    assert "_dedup_ts_bucket" in bucketed.columns
    # Same 30s window.
    assert bucketed["_dedup_ts_bucket"][0] == bucketed["_dedup_ts_bucket"][1]


def test_dedup_buckets_coordinates():
    df = pl.DataFrame({
        "mmsi": [111, 111],
        "timestamp": pl.Series(["2024-06-15T00:00:00", "2024-06-15T00:00:00"],
                               dtype=pl.Datetime("us", "UTC")),
        "lat": [40.0, 40.001],
        "lon": [-74.0, -73.999],
    })
    config = FusionConfig(coordinate_tolerance_degrees=0.01)
    bucketed = compute_dedup_buckets(df, config)
    assert "_dedup_lat_bucket" in bucketed.columns
    assert "_dedup_lon_bucket" in bucketed.columns
    # Same grid cell.
    assert bucketed["_dedup_lat_bucket"][0] == bucketed["_dedup_lat_bucket"][1]


def test_dedup_subset_with_coords():
    df = pl.DataFrame({
        "mmsi": [111],
        "timestamp": pl.Series(["2024-06-15T00:00:00"], dtype=pl.Datetime("us", "UTC")),
        "lat": [40.0], "lon": [-74.0],
    })
    config = FusionConfig(coordinate_tolerance_degrees=0.01)
    bucketed = compute_dedup_buckets(df, config)
    subset = dedup_subset_columns(config, bucketed)
    assert subset == ["mmsi", "_dedup_ts_bucket", "_dedup_lat_bucket", "_dedup_lon_bucket"]


def test_dedup_subset_without_coords():
    df = pl.DataFrame({
        "mmsi": [111],
        "timestamp": pl.Series(["2024-06-15T00:00:00"], dtype=pl.Datetime("us", "UTC")),
    })
    config = FusionConfig(coordinate_tolerance_degrees=0.0)
    bucketed = compute_dedup_buckets(df, config)
    subset = dedup_subset_columns(config, bucketed)
    assert subset == ["mmsi", "_dedup_ts_bucket"]


if __name__ == "__main__":
    import inspect
    tests = [
        (name, obj) for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    ]
    print(f"Running {len(tests)} fusion regression tests...")
    for name, func in sorted(tests):
        func()
        print(f"  {name} ✓")
    print(f"\nAll {len(tests)} tests passed ✓")
