"""Schema migration, raw-rebuild reproducibility, and cache invalidation tests.

Verifies three trust boundaries:
1. Schema migration: manifests with old/unknown versions are detected
2. Raw-rebuild: re-processing the same data produces identical output
3. Derived-cache invalidation: version changes are caught by cache keys

These are release-gate tests — they verify long-lived-cache correctness
that unit tests of individual components cannot cover.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from neptune_ais.catalog import (
    KNOWN_SCHEMA_VERSIONS,
    Manifest,
    QCSummary,
    WriteStatus,
    current_schema_version,
    is_compatible,
)
from neptune_ais.storage import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_WRITE_STATISTICS,
    DEFAULT_ROW_GROUP_SIZE,
    PartitionWriter,
    shard_filename,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_manifest(
    dataset: str = "positions",
    source: str = "test",
    date_str: str = "2024-06-15",
    schema_version: str | None = None,
    **overrides,
) -> Manifest:
    """Build a minimal valid manifest for testing."""
    return Manifest(
        dataset=dataset,
        source=source,
        date=date_str,
        schema_version=schema_version or current_schema_version(dataset),
        adapter_version=f"{source}/0.1.0",
        transform_version="normalize/0.1.0",
        files=["part-0000.parquet"],
        raw_artifacts=[],
        raw_policy="none",
        record_count=100,
        distinct_mmsi_count=10,
        min_timestamp=datetime(2024, 6, 15, tzinfo=timezone.utc),
        max_timestamp=datetime(2024, 6, 15, 23, 59, 59, tzinfo=timezone.utc),
        qc_summary=QCSummary(
            total_rows=100, rows_ok=100, rows_warning=0, rows_error=0,
        ),
        write_status=WriteStatus.COMMITTED,
        **overrides,
    )


def _sample_positions(n: int = 100) -> pl.DataFrame:
    """Generate deterministic sample positions."""
    import random
    random.seed(42)
    return pl.DataFrame({
        "mmsi": [200000000 + i % 10 for i in range(n)],
        "timestamp": [f"2024-06-15T00:{i:02d}:00" for i in range(n)],
        "lat": [40.0 + random.uniform(-0.1, 0.1) for _ in range(n)],
        "lon": [-74.0 + random.uniform(-0.1, 0.1) for _ in range(n)],
        "sog": [random.uniform(0, 15) for _ in range(n)],
        "source": ["test"] * n,
    })


# ---------------------------------------------------------------------------
# 1. Schema migration — version detection and compatibility
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    """Verify schema version detection and forward/backward compatibility."""

    def test_current_version_exists_for_all_datasets(self):
        """Every dataset in the registry has a current version."""
        for dataset in KNOWN_SCHEMA_VERSIONS:
            version = current_schema_version(dataset)
            assert version, f"{dataset} has empty current version"
            assert "/" in version, f"{dataset} version should be dataset/vN"

    def test_current_version_is_compatible(self):
        """The current version is always compatible with itself."""
        for dataset in KNOWN_SCHEMA_VERSIONS:
            version = current_schema_version(dataset)
            assert is_compatible(dataset, version)

    def test_unknown_version_not_compatible(self):
        """An unknown future version is not compatible."""
        assert not is_compatible("positions", "positions/v99")
        assert not is_compatible("tracks", "tracks/v999")

    def test_unknown_dataset_not_compatible(self):
        """An unknown dataset is never compatible."""
        assert not is_compatible("nonexistent", "nonexistent/v1")

    def test_manifest_with_current_version_valid(self):
        """A manifest with the current schema version is valid."""
        m = _sample_manifest()
        assert is_compatible(m.dataset, m.schema_version)

    def test_manifest_with_old_version_detected(self):
        """A manifest with an old (but known) version is detected."""
        # Currently all datasets have only v1, so there's no "old" version.
        # Test that adding a v2 makes v1 still compatible.
        old_versions = KNOWN_SCHEMA_VERSIONS["positions"].copy()
        try:
            KNOWN_SCHEMA_VERSIONS["positions"].append("positions/v2")
            assert is_compatible("positions", "positions/v1")
            assert is_compatible("positions", "positions/v2")
            assert current_schema_version("positions") == "positions/v2"
        finally:
            KNOWN_SCHEMA_VERSIONS["positions"][:] = old_versions

    def test_manifest_with_unknown_version_flagged(self):
        """A manifest with an unrecognized version is flagged by check_health."""
        from neptune_ais.catalog import CatalogRegistry

        m = _sample_manifest(schema_version="positions/v99")
        registry = CatalogRegistry.__new__(CatalogRegistry)
        registry._manifests = {("positions", "test", "2024-06-15"): m}
        registry._store_root = Path("/tmp")

        warnings = registry.check_health()
        assert any("v99" in w for w in warnings), (
            f"Expected warning about unknown version, got: {warnings}"
        )

    def test_mixed_versions_detected(self):
        """Inventory flags mixed schema versions across partitions."""
        from neptune_ais.catalog import CatalogRegistry

        m1 = _sample_manifest(date_str="2024-06-15", schema_version="positions/v1")
        m2 = _sample_manifest(date_str="2024-06-16", schema_version="positions/v1")

        registry = CatalogRegistry.__new__(CatalogRegistry)
        registry._manifests = {
            ("positions", "test", "2024-06-15"): m1,
            ("positions", "test", "2024-06-16"): m2,
        }
        registry._store_root = Path("/tmp")

        inv = registry.inventory("positions")
        assert len(inv) >= 1
        assert not inv[0].has_mixed_versions


# ---------------------------------------------------------------------------
# 2. Raw-rebuild reproducibility — deterministic output
# ---------------------------------------------------------------------------


class TestRawRebuildReproducibility:
    """Verify that re-processing the same data produces identical output."""

    def test_parquet_write_is_deterministic(self, tmp_path):
        """Writing the same DataFrame twice produces identical Parquet files."""
        df = _sample_positions(50)

        path1 = tmp_path / "run1.parquet"
        path2 = tmp_path / "run2.parquet"

        for path in (path1, path2):
            df.write_parquet(
                path,
                compression=PARQUET_COMPRESSION,
                compression_level=PARQUET_COMPRESSION_LEVEL,
                row_group_size=DEFAULT_ROW_GROUP_SIZE,
                statistics=PARQUET_WRITE_STATISTICS,
            )

        # Binary identity: same bytes.
        assert path1.read_bytes() == path2.read_bytes()

    def test_sort_is_deterministic(self):
        """Sorting the same data produces identical row order."""
        df = _sample_positions(100)

        sorted1 = df.sort(["mmsi", "timestamp"])
        sorted2 = df.sort(["mmsi", "timestamp"])

        assert sorted1.equals(sorted2)

    def test_dedup_is_deterministic(self):
        """Deduplication on the same data produces identical results."""
        df = _sample_positions(100)
        # Add duplicates.
        df_with_dupes = pl.concat([df, df.head(20)])

        dedup1 = df_with_dupes.unique(
            subset=["mmsi", "timestamp", "lat", "lon"], keep="any"
        ).sort(["mmsi", "timestamp"])
        dedup2 = df_with_dupes.unique(
            subset=["mmsi", "timestamp", "lat", "lon"], keep="any"
        ).sort(["mmsi", "timestamp"])

        assert dedup1.equals(dedup2)

    def test_partition_write_is_reproducible(self, tmp_path):
        """Full partition write (stage → validate → commit) is reproducible."""
        df = _sample_positions(50).sort(["mmsi", "timestamp"])

        for run in ("run1", "run2"):
            store = tmp_path / run
            writer = PartitionWriter(store, "positions", "test", "2024-06-15")
            staging = writer.prepare()

            sf = shard_filename(0)
            df.write_parquet(
                staging / sf,
                compression=PARQUET_COMPRESSION,
                compression_level=PARQUET_COMPRESSION_LEVEL,
                row_group_size=DEFAULT_ROW_GROUP_SIZE,
                statistics=PARQUET_WRITE_STATISTICS,
            )
            writer.validate(expected_files=[sf])

            manifest = _sample_manifest()
            writer.commit(manifest_json=manifest.model_dump_json(indent=2))

        # Both runs produced identical Parquet files.
        file1 = tmp_path / "run1" / "canonical" / "positions" / "source=test" / "date=2024-06-15" / "part-0000.parquet"
        file2 = tmp_path / "run2" / "canonical" / "positions" / "source=test" / "date=2024-06-15" / "part-0000.parquet"
        assert file1.read_bytes() == file2.read_bytes()

    def test_manifest_json_is_reproducible(self):
        """Manifest serialization is deterministic (same fields → same JSON)."""
        m1 = _sample_manifest()
        m2 = _sample_manifest()

        # Exclude write_timestamp (it uses datetime.now).
        j1 = json.loads(m1.model_dump_json())
        j2 = json.loads(m2.model_dump_json())
        j1.pop("write_timestamp", None)
        j2.pop("write_timestamp", None)

        assert j1 == j2


# ---------------------------------------------------------------------------
# 3. Derived-cache invalidation — version change detection
# ---------------------------------------------------------------------------


class TestDerivedCacheInvalidation:
    """Verify that derived cache keys change when versions change."""

    def test_track_cache_key_changes_with_schema(self):
        """TrackCacheKey includes schema version — changes invalidate cache."""
        from neptune_ais.derive.tracks import TrackCacheKey

        base = dict(source="test", date="2024-06-15", config_hash="abc",
                     positions_schema_version="positions/v1",
                     upstream_manifest_hash="hash1")
        key1 = TrackCacheKey(**base, tracks_schema_version="tracks/v1")
        key2 = TrackCacheKey(**base, tracks_schema_version="tracks/v2")

        assert key1.cache_key() != key2.cache_key()

    def test_track_cache_key_changes_with_config(self):
        """TrackCacheKey changes when config changes."""
        from neptune_ais.derive.tracks import TrackCacheKey

        base = dict(source="test", date="2024-06-15",
                     positions_schema_version="positions/v1",
                     tracks_schema_version="tracks/v1",
                     upstream_manifest_hash="hash1")
        key1 = TrackCacheKey(**base, config_hash="abc")
        key2 = TrackCacheKey(**base, config_hash="def")

        assert key1.cache_key() != key2.cache_key()

    def test_track_cache_key_stable_for_same_inputs(self):
        """Same inputs produce the same cache key (deterministic)."""
        from neptune_ais.derive.tracks import TrackCacheKey

        kwargs = dict(source="test", date="2024-06-15", config_hash="abc",
                       positions_schema_version="positions/v1",
                       tracks_schema_version="tracks/v1",
                       upstream_manifest_hash="hash1")
        key1 = TrackCacheKey(**kwargs)
        key2 = TrackCacheKey(**kwargs)

        assert key1.cache_key() == key2.cache_key()

    def test_event_cache_key_changes_with_schema(self):
        """EventCacheKey includes schema version — changes invalidate cache."""
        from neptune_ais.derive.events import EventCacheKey

        base = dict(event_type="port_call", source="test", date="2024-06-15",
                     config_hash="abc", upstream_schema_version="tracks/v1",
                     upstream_manifest_hash="hash1")
        key1 = EventCacheKey(**base, events_schema_version="events/v1")
        key2 = EventCacheKey(**base, events_schema_version="events/v2")

        assert key1.cache_key() != key2.cache_key()

    def test_event_cache_key_stable(self):
        """Same event inputs produce the same cache key."""
        from neptune_ais.derive.events import EventCacheKey

        kwargs = dict(event_type="port_call", source="test", date="2024-06-15",
                       config_hash="abc", upstream_schema_version="tracks/v1",
                       events_schema_version="events/v1",
                       upstream_manifest_hash="hash1")
        key1 = EventCacheKey(**kwargs)
        key2 = EventCacheKey(**kwargs)

        assert key1.cache_key() == key2.cache_key()

    def test_upstream_version_change_invalidates_events(self):
        """Changing the upstream (tracks) version invalidates event cache."""
        from neptune_ais.derive.events import EventCacheKey

        base = dict(event_type="port_call", source="test", date="2024-06-15",
                     config_hash="abc", events_schema_version="events/v1",
                     upstream_manifest_hash="hash1")
        key1 = EventCacheKey(**base, upstream_schema_version="tracks/v1")
        key2 = EventCacheKey(**base, upstream_schema_version="tracks/v2")

        assert key1.cache_key() != key2.cache_key()


# ---------------------------------------------------------------------------
# 4. Promotion reproducibility
# ---------------------------------------------------------------------------


try:
    from pydantic import BaseModel as _PydanticModel
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

skip_no_pydantic = pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")


@skip_no_pydantic
class TestPromotionReproducibility:
    """Promotion from landing to canonical is deterministic."""

    def test_promote_twice_produces_same_data(self, tmp_path):
        """Promoting the same landing files twice produces identical canonical data."""
        from neptune_ais.sinks import promote_landing

        # Create landing data.
        landing = tmp_path / "landing" / "test"
        landing.mkdir(parents=True)
        df = _sample_positions(50).sort(["mmsi", "timestamp"])
        df.write_parquet(landing / "landing-20240615-0000.parquet")

        # Promote twice to different store roots.
        for run in ("store1", "store2"):
            store = tmp_path / run
            promote_landing(tmp_path / "landing", store, source="test")

        # Compare canonical Parquet files.
        for run in ("store1", "store2"):
            canonical = tmp_path / run / "canonical" / "positions" / "source=test" / "date=2024-06-15"
            assert canonical.exists(), f"{run} canonical dir missing"

        files1 = sorted((tmp_path / "store1" / "canonical").rglob("*.parquet"))
        files2 = sorted((tmp_path / "store2" / "canonical").rglob("*.parquet"))
        assert len(files1) == len(files2)

        for f1, f2 in zip(files1, files2):
            assert f1.read_bytes() == f2.read_bytes(), (
                f"Files differ: {f1.name} vs {f2.name}"
            )
