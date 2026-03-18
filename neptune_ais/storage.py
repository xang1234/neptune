"""Storage — partition layout, atomic writes, and retention policy.

Manages the three-layer local store (raw / canonical / derived), deterministic
partitioning, staged writes with commit markers, and raw retention policies.

Module role — cross-cutting infrastructure
------------------------------------------
Storage is shared infrastructure used by ``api`` (orchestration), ``catalog``
(manifest paths), and indirectly by ``derive`` (caching derived output).

**Owns:**
- The three-layer directory layout: ``raw/``, ``canonical/``, ``derived/``.
- Partition path computation (dataset / source / date / shard).
- Atomic write protocol: stage to temp → validate → commit.
- Raw retention policy enforcement (none / metadata / full).
- Sort-order guarantees within Parquet files (mmsi, timestamp).

**Does not own:**
- Schema definitions — those live in ``datasets``.
- Manifest metadata — that lives in ``catalog``.
- Fetch or normalization logic — that lives in ``adapters``.
- Which files to read for a query — that is ``catalog``'s job.

**Import rule:** Storage may import from ``datasets`` (to know partition keys
and sort columns). It must not import from ``adapters``, ``derive``,
``geometry``, ``cli``, or ``api``.
"""

from __future__ import annotations

import logging
import shutil
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default store root
# ---------------------------------------------------------------------------

DEFAULT_STORE_ROOT = Path.home() / ".neptune"
"""Default root directory for the Neptune local store.

Can be overridden per-session via ``Neptune(cache_dir=...)``.
"""

# ---------------------------------------------------------------------------
# Storage layers
# ---------------------------------------------------------------------------


class StoreLayer(str, Enum):
    """The three logical storage layers."""

    RAW = "raw"
    """Optional retained source payloads or source-native extracts."""

    CANONICAL = "canonical"
    """Normalized positions, vessels, tracks, events."""

    DERIVED = "derived"
    """Cached map-ready, event-ready, or aggregate products."""


# ---------------------------------------------------------------------------
# Top-level directories
# ---------------------------------------------------------------------------

CATALOG_DIR = "catalog"
"""Catalog metadata (catalog.json, versions.json)."""

MANIFESTS_DIR = "manifests"
"""Per-partition manifest JSON files."""

STAGING_DIR = "_staging"
"""Temporary directory for atomic writes. Files here are in-flight and
should never be read by queries."""

# ---------------------------------------------------------------------------
# Raw retention policies
# ---------------------------------------------------------------------------


class RawPolicy(str, Enum):
    """Controls how much raw source data Neptune retains after normalization."""

    NONE = "none"
    """Do not retain raw artifacts."""

    METADATA = "metadata"
    """Retain only references, checksums, and key headers."""

    FULL = "full"
    """Retain original source files for replay and reprocessing."""


# ---------------------------------------------------------------------------
# Partition layout — path computation
# ---------------------------------------------------------------------------

# Hive-style partition key names used in directory paths.
# Both Polars and DuckDB auto-discover these from path segments.
PARTITION_KEY_SOURCE = "source"
PARTITION_KEY_DATE = "date"

# Shard file naming
SHARD_PREFIX = "part"
SHARD_EXTENSION = ".parquet"
SHARD_DIGITS = 4  # part-0000.parquet, part-0001.parquet, ...

# Maximum rows per shard file. When a partition exceeds this, the writer
# splits into multiple shards to keep individual file sizes manageable
# for memory-mapped reads and row-group statistics.
DEFAULT_MAX_ROWS_PER_SHARD = 2_000_000


def canonical_partition_path(
    dataset: str,
    source: str,
    date: str,
) -> Path:
    """Compute the relative partition directory for canonical data.

    Returns a path like::

        canonical/positions/source=noaa/date=2024-06-15

    This is relative to the store root. The caller appends shard filenames.
    """
    return (
        Path(StoreLayer.CANONICAL.value)
        / dataset
        / f"{PARTITION_KEY_SOURCE}={source}"
        / f"{PARTITION_KEY_DATE}={date}"
    )


def derived_partition_path(
    product: str,
    source: str,
    date: str,
) -> Path:
    """Compute the relative partition directory for derived data.

    Returns a path like::

        derived/tracks/source=noaa/date=2024-06-15
    """
    return (
        Path(StoreLayer.DERIVED.value)
        / product
        / f"{PARTITION_KEY_SOURCE}={source}"
        / f"{PARTITION_KEY_DATE}={date}"
    )


def raw_partition_path(
    source: str,
    date: str,
) -> Path:
    """Compute the relative directory for raw source artifacts.

    Returns a path like::

        raw/noaa/2024/06/15

    Raw layout uses slash-separated date components (not Hive-style)
    because raw files retain their original names and formats.
    """
    # date is "YYYY-MM-DD" → split into year/month/day
    parts = date.split("-")
    return Path(StoreLayer.RAW.value) / source / Path(*parts)


def manifest_path(
    dataset: str,
    source: str,
    date: str,
) -> Path:
    """Compute the relative path for a partition's manifest JSON.

    Returns a path like::

        manifests/positions/source=noaa/date=2024-06-15.json

    One JSON file per partition (dataset + source + date).
    """
    return (
        Path(MANIFESTS_DIR)
        / dataset
        / f"{PARTITION_KEY_SOURCE}={source}"
        / f"{PARTITION_KEY_DATE}={date}.json"
    )


def staging_path(
    dataset: str,
    source: str,
    date: str,
) -> Path:
    """Compute the relative staging directory for an in-flight write.

    Returns a path like::

        _staging/positions/source=noaa/date=2024-06-15

    The atomic write protocol writes here first, then moves to the
    canonical location after validation succeeds.
    """
    return (
        Path(STAGING_DIR)
        / dataset
        / f"{PARTITION_KEY_SOURCE}={source}"
        / f"{PARTITION_KEY_DATE}={date}"
    )


def shard_filename(shard_index: int) -> str:
    """Generate a shard filename like ``part-0000.parquet``.

    >>> shard_filename(0)
    'part-0000.parquet'
    >>> shard_filename(12)
    'part-0012.parquet'
    """
    return f"{SHARD_PREFIX}-{shard_index:0{SHARD_DIGITS}d}{SHARD_EXTENSION}"


# ---------------------------------------------------------------------------
# Sort order — within-file row ordering
# ---------------------------------------------------------------------------

# These are the column names that Parquet files must be sorted by.
# Imported from datasets at call time to avoid top-level polars import,
# but the contract is: [mmsi, timestamp] for positions/tracks,
# [mmsi, last_seen] for vessels.
#
# Sort order serves two purposes:
# 1. Row-group statistics enable predicate pushdown for MMSI and time range.
# 2. Consecutive rows for the same vessel enable efficient streaming scans.

SORT_ORDER_POSITIONS = ["mmsi", "timestamp"]
SORT_ORDER_VESSELS = ["mmsi", "last_seen"]

# Row-group size target. Parquet row groups should be large enough for
# efficient columnar reads but small enough that row-group statistics
# provide meaningful pruning.
DEFAULT_ROW_GROUP_SIZE = 500_000

# ---------------------------------------------------------------------------
# Parquet write options
# ---------------------------------------------------------------------------

PARQUET_COMPRESSION = "zstd"
"""Compression codec for Parquet files. ZSTD offers a good balance of
compression ratio and decode speed for AIS columnar data."""

PARQUET_COMPRESSION_LEVEL = 3
"""ZSTD compression level. Level 3 is the default and provides fast
compression with good ratios."""

PARQUET_WRITE_STATISTICS = True
"""Whether to write column statistics in Parquet row groups. Required
for predicate pushdown and bounding-box pruning."""


# ---------------------------------------------------------------------------
# Atomic write protocol — PartitionWriter
# ---------------------------------------------------------------------------


class PartitionWriteError(Exception):
    """Raised when a partition write fails validation or commit."""


class PartitionWriter:
    """Orchestrates the stage → validate → commit protocol for a partition.

    Usage::

        writer = PartitionWriter(store_root, dataset, source, date)

        # Stage phase: write shards into the staging directory.
        writer.prepare()
        for i, shard_df in enumerate(shards):
            shard_df.write_parquet(writer.shard_path(i), ...)

        # Validate phase: check files exist and are readable.
        writer.validate(expected_files=["part-0000.parquet"])

        # Commit phase: move to canonical location and write manifest.
        writer.commit(manifest_json=manifest.model_dump_json(indent=2))

    If any phase fails, call ``writer.abort()`` to clean up staging.

    The writer does NOT construct Manifest objects — that is the caller's
    job (usually ``api`` orchestration). This avoids a circular import
    between storage and catalog.
    """

    def __init__(
        self,
        store_root: Path,
        dataset: str,
        source: str,
        date: str,
    ) -> None:
        self.store_root = store_root
        self.dataset = dataset
        self.source = source
        self.date = date

        self._staging_dir = store_root / staging_path(dataset, source, date)
        self._canonical_dir = store_root / canonical_partition_path(
            dataset, source, date
        )
        self._manifest_file = store_root / manifest_path(
            dataset, source, date
        )
        self._committed = False

    # --- Stage phase ---

    def prepare(self) -> Path:
        """Create the staging directory for this partition.

        Returns the absolute path to the staging directory where the
        caller should write shard files.

        Raises ``PartitionWriteError`` if the staging directory already
        exists (indicating an incomplete prior write).
        """
        if self._staging_dir.exists():
            raise PartitionWriteError(
                f"Staging directory already exists (incomplete prior write?): "
                f"{self._staging_dir}"
            )
        self._staging_dir.mkdir(parents=True, exist_ok=False)
        logger.debug("Created staging directory: %s", self._staging_dir)
        return self._staging_dir

    def shard_path(self, shard_index: int) -> Path:
        """Return the absolute path for a shard file in the staging directory.

        >>> # writer.shard_path(0) → .../staging/.../part-0000.parquet
        """
        return self._staging_dir / shard_filename(shard_index)

    # --- Validate phase ---

    def validate(self, expected_files: list[str]) -> None:
        """Check that all expected shard files exist in staging.

        Raises ``PartitionWriteError`` if any file is missing or if
        unexpected files are present.
        """
        if not self._staging_dir.exists():
            raise PartitionWriteError(
                f"Staging directory does not exist: {self._staging_dir}"
            )

        staged = {f.name for f in self._staging_dir.iterdir() if f.is_file()}
        expected = set(expected_files)

        missing = expected - staged
        if missing:
            raise PartitionWriteError(
                f"Missing shard files in staging: {sorted(missing)}"
            )

        unexpected = staged - expected
        if unexpected:
            raise PartitionWriteError(
                f"Unexpected files in staging: {sorted(unexpected)}"
            )

        # Verify files are non-empty.
        for fname in expected:
            fpath = self._staging_dir / fname
            if fpath.stat().st_size == 0:
                raise PartitionWriteError(f"Empty shard file: {fname}")

        logger.debug(
            "Validation passed: %d shard(s) in %s",
            len(expected),
            self._staging_dir,
        )

    # --- Commit phase ---

    def commit(self, manifest_json: str) -> Path:
        """Move staged files to canonical location and write manifest.

        This is the atomic commit point. After this method returns
        successfully, the partition is query-visible.

        Returns the path to the written manifest file.

        Raises ``PartitionWriteError`` if the commit fails.
        """
        if self._committed:
            raise PartitionWriteError("Partition already committed")

        if not self._staging_dir.exists():
            raise PartitionWriteError(
                f"Nothing to commit — staging directory missing: "
                f"{self._staging_dir}"
            )

        # If the canonical directory already exists, remove it first
        # (overwrite semantics for re-ingest).
        if self._canonical_dir.exists():
            logger.info(
                "Overwriting existing partition: %s", self._canonical_dir
            )
            shutil.rmtree(self._canonical_dir)

        # Move staging → canonical. Using shutil.move instead of
        # Path.rename to handle cross-device moves.
        self._canonical_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self._staging_dir), str(self._canonical_dir))

        # Write the manifest.
        self._manifest_file.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_file.write_text(manifest_json, encoding="utf-8")

        self._committed = True
        logger.info(
            "Committed partition: %s/%s/%s → %s",
            self.dataset,
            self.source,
            self.date,
            self._canonical_dir,
        )
        return self._manifest_file

    # --- Abort / cleanup ---

    def abort(self) -> None:
        """Clean up the staging directory after a failed write.

        Safe to call multiple times or if staging was never created.
        """
        if self._staging_dir.exists():
            shutil.rmtree(self._staging_dir)
            logger.info("Aborted write, cleaned staging: %s", self._staging_dir)

    # --- Cleanup stale staging ---

    @staticmethod
    def cleanup_stale_staging(store_root: Path) -> list[Path]:
        """Remove all staging directories (orphaned from failed writes).

        Returns a list of directories that were removed.
        """
        staging_root = store_root / STAGING_DIR
        if not staging_root.exists():
            return []

        removed: list[Path] = []
        # Walk dataset directories under _staging/ to report what's there.
        for dataset_dir in staging_root.iterdir():
            if not dataset_dir.is_dir():
                continue
            for source_dir in dataset_dir.iterdir():
                if not source_dir.is_dir():
                    continue
                for date_dir in source_dir.iterdir():
                    if date_dir.is_dir():
                        removed.append(date_dir)

        # Remove the entire staging tree in one operation.
        shutil.rmtree(staging_root)

        if removed:
            logger.info("Cleaned %d stale staging directories", len(removed))
        return removed
