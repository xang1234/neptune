"""Catalog — manifests, schema versions, and dataset registry.

Tracks what has been ingested, which schema version was used, and provides
staleness and mixed-version detection across the local lakehouse.

Module role — cross-cutting infrastructure
------------------------------------------
Catalog is the metadata backbone. Every successful write (canonical or
derived) creates a manifest entry here. Every query begins with a catalog
lookup to find relevant partitions.

**Owns:**
- Manifest schema: dataset, source, schema version, transform version,
  adapter version, file list, checksums, record counts, bbox, QC counters,
  write timestamp, commit status.
- Schema version registry and migration compatibility rules.
- Partition discovery: given a dataset + source + date range, return the
  file list and manifest metadata.
- Staleness detection: mixed schema versions, partial failures, missing
  commit markers.

**Does not own:**
- File I/O or directory layout — that is ``storage``.
- Schema column definitions — those are in ``datasets``.
- The write pipeline itself — that is ``api`` orchestration.

**Import rule:** Catalog may import from ``datasets`` (schema version IDs)
and ``storage`` (path helpers). It must not import from ``adapters``,
``derive``, ``geometry``, ``cli``, or ``api``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from neptune_ais.storage import MANIFESTS_DIR, RawPolicy, canonical_partition_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Write status — commit marker for atomic writes
# ---------------------------------------------------------------------------


class WriteStatus(str, Enum):
    """Commit status for a partition write.

    The atomic write protocol is: stage → validate → commit.
    Only ``COMMITTED`` partitions are visible to queries.
    """

    STAGED = "staged"
    """Write is in progress or has not been validated yet."""

    COMMITTED = "committed"
    """Write completed successfully and is query-visible."""

    FAILED = "failed"
    """Write failed validation; partition should be ignored or cleaned up."""


# ---------------------------------------------------------------------------
# QC summary counters — manifest-level quality overview
# ---------------------------------------------------------------------------


class QCSummary(BaseModel):
    """Aggregate quality counters for a single partition write."""

    total_rows: int = Field(description="Total rows written (before any drops).")
    rows_ok: int = Field(description="Rows with qc_severity = ok.")
    rows_warning: int = Field(description="Rows with qc_severity = warning.")
    rows_error: int = Field(description="Rows with qc_severity = error.")
    rows_dropped: int = Field(
        default=0,
        description="Hard-invalid rows dropped before writing.",
    )
    checks_applied: list[str] = Field(
        default_factory=list,
        description="Names of QC checks that were executed.",
    )


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------


class BBox(BaseModel):
    """Spatial bounding box for a partition (WGS-84)."""

    west: float = Field(ge=-180.0, le=180.0)
    south: float = Field(ge=-90.0, le=90.0)
    east: float = Field(ge=-180.0, le=180.0)
    north: float = Field(ge=-90.0, le=90.0)


# ---------------------------------------------------------------------------
# Raw artifact tracking — source provenance for reproducibility
# ---------------------------------------------------------------------------


class RawArtifact(BaseModel):
    """Metadata for a single raw source artifact that contributed to a partition.

    Adapters produce one or more ``RawArtifact`` records per fetch operation.
    These are stored in the manifest so that canonical outputs can always be
    traced back to the exact raw inputs — even when the raw files themselves
    are not retained locally (``raw_policy="none"`` or ``"metadata"``).

    When ``raw_policy="full"``, the ``local_path`` field points to the
    retained copy in the raw store. When ``raw_policy="metadata"``, only
    this metadata record exists. When ``raw_policy="none"``, the adapter
    still produces ``RawArtifact`` records for manifest inclusion, but
    neither the file nor the metadata is persisted separately.
    """

    source_url: str = Field(
        description=(
            "Original URL, API endpoint, or identifier where the raw data "
            "was fetched from. For local files, use a file:// URI."
        ),
    )
    filename: str = Field(
        description=(
            "Original filename of the raw artifact as provided by the source "
            "(e.g. 'AIS_2024_06_15.zip', 'aisdk-2024-06-15.csv')."
        ),
    )
    content_hash: str = Field(
        description=(
            "SHA-256 hex digest of the raw file contents. Computed before "
            "any decompression or normalization. This is the primary key "
            "for rebuild detection — if the hash matches, the raw input "
            "is identical."
        ),
    )
    size_bytes: int = Field(
        ge=0,
        description="Size of the raw artifact in bytes.",
    )
    fetch_timestamp: datetime = Field(
        description="When the artifact was fetched (UTC).",
    )
    content_type: str | None = Field(
        default=None,
        description=(
            "MIME type or format hint, e.g. 'application/x-geoparquet', "
            "'text/csv', 'application/zip'. None if unknown."
        ),
    )
    local_path: str | None = Field(
        default=None,
        description=(
            "Relative path to the retained raw file in the raw store. "
            "Only set when raw_policy='full'. None otherwise."
        ),
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Key HTTP response headers or source metadata captured at "
            "fetch time (e.g. ETag, Last-Modified, Content-Encoding). "
            "Useful for conditional re-fetch and change detection."
        ),
    )


# ---------------------------------------------------------------------------
# Manifest — the core metadata record for every partition write
# ---------------------------------------------------------------------------


class Manifest(BaseModel):
    """Metadata record created by every successful partition write.

    A manifest ties a set of Parquet files to the exact schema, adapter,
    transform, and quality state that produced them. This is the foundation
    for partition discovery, staleness detection, mixed-version warnings,
    and partial-failure recovery.

    Serialized as JSON alongside each partition in the manifests/ directory.
    """

    # --- identity: what was written ---

    dataset: str = Field(
        description=(
            "Canonical dataset name: 'positions', 'vessels', 'tracks', "
            "'events', or a derived product name."
        ),
    )
    source: str = Field(
        description="Source adapter identifier, e.g. 'noaa', 'dma'.",
    )
    date: str = Field(
        description=(
            "Partition date as ISO-8601 string (YYYY-MM-DD). Derived from "
            "the timestamp range of the written data."
        ),
    )

    # --- versioning: under which assumptions ---

    schema_version: str = Field(
        description=(
            "Dataset schema version identifier, e.g. 'positions/v1'. "
            "Must match the version from the datasets module."
        ),
    )
    adapter_version: str = Field(
        description=(
            "Adapter code version that fetched and normalized the data. "
            "Allows re-ingest detection when the adapter changes."
        ),
    )
    transform_version: str = Field(
        description=(
            "Transform/normalization pipeline version. Distinct from "
            "adapter_version when shared normalization logic changes "
            "independently of the adapter."
        ),
    )

    # --- file inventory ---

    files: list[str] = Field(
        description=(
            "Relative paths to Parquet files in this partition, "
            "ordered by shard number."
        ),
    )
    raw_artifacts: list[RawArtifact] = Field(
        default_factory=list,
        description=(
            "Structured metadata for each raw source artifact that was "
            "normalized into this partition. Always populated regardless "
            "of raw_policy — this is the provenance link from canonical "
            "output back to raw input."
        ),
    )
    raw_policy: RawPolicy = Field(
        default=RawPolicy.METADATA,
        description=(
            "Raw retention policy in effect when this partition was written. "
            "Determines whether raw_artifacts[].local_path is populated."
        ),
    )

    # --- statistics ---

    record_count: int = Field(
        ge=0,
        description="Number of rows written to the partition.",
    )
    distinct_mmsi_count: int = Field(
        ge=0,
        description="Number of distinct MMSI values in the partition.",
    )
    min_timestamp: datetime = Field(
        description="Earliest timestamp in the partition (UTC).",
    )
    max_timestamp: datetime = Field(
        description="Latest timestamp in the partition (UTC).",
    )
    bbox: BBox | None = Field(
        default=None,
        description=(
            "Spatial bounding box of the data. None for datasets without "
            "coordinates (e.g. vessels)."
        ),
    )

    # --- quality ---

    qc_summary: QCSummary = Field(
        description="Aggregate quality counters for this partition.",
    )

    # --- write metadata ---

    write_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this manifest was written (UTC).",
    )
    write_status: WriteStatus = Field(
        default=WriteStatus.STAGED,
        description=(
            "Commit status. Only COMMITTED partitions are query-visible. "
            "Writers set STAGED initially, then promote to COMMITTED after "
            "validation succeeds."
        ),
    )


# ---------------------------------------------------------------------------
# Schema version registry — known versions and compatibility
# ---------------------------------------------------------------------------

KNOWN_SCHEMA_VERSIONS: dict[str, list[str]] = {
    "positions": ["positions/v1"],
    "vessels": ["vessels/v1"],
    "tracks": [],  # to be defined
    "events": [],  # to be defined
}
"""Registry of known schema versions per dataset.

The list is ordered oldest → newest. The last entry is the current version.
This registry enables:
- Mixed-version detection (partitions with different schema versions).
- Forward compatibility checks (reject unknown future versions).
- Migration path discovery (which versions can be upgraded to current).
"""


def current_schema_version(dataset: str) -> str:
    """Return the current (latest) schema version for a dataset.

    Raises ``KeyError`` if the dataset is unknown.
    Raises ``ValueError`` if the dataset has no registered versions.
    """
    versions = KNOWN_SCHEMA_VERSIONS[dataset]
    if not versions:
        raise ValueError(
            f"dataset {dataset!r} has no registered schema versions"
        )
    return versions[-1]


def is_compatible(dataset: str, version: str) -> bool:
    """Check whether *version* is a known, compatible schema version.

    Returns True if *version* appears in the registry for *dataset*.
    Unknown datasets or versions return False.
    """
    return version in KNOWN_SCHEMA_VERSIONS.get(dataset, [])


# ---------------------------------------------------------------------------
# Partition inventory — summary view of a single partition
# ---------------------------------------------------------------------------


class PartitionInfo(BaseModel):
    """Lightweight summary of a committed partition for inventory views.

    Extracted from the full Manifest but carries only the fields needed
    for listing, filtering, and staleness detection.
    """

    dataset: str
    source: str
    date: str
    schema_version: str
    record_count: int
    distinct_mmsi_count: int
    min_timestamp: datetime
    max_timestamp: datetime
    write_timestamp: datetime
    write_status: WriteStatus

    @classmethod
    def from_manifest(cls, m: Manifest) -> PartitionInfo:
        return cls(
            dataset=m.dataset,
            source=m.source,
            date=m.date,
            schema_version=m.schema_version,
            record_count=m.record_count,
            distinct_mmsi_count=m.distinct_mmsi_count,
            min_timestamp=m.min_timestamp,
            max_timestamp=m.max_timestamp,
            write_timestamp=m.write_timestamp,
            write_status=m.write_status,
        )


# ---------------------------------------------------------------------------
# Dataset inventory — aggregate view across partitions
# ---------------------------------------------------------------------------


class DatasetInventory(BaseModel):
    """Aggregate inventory for a single dataset across all sources and dates."""

    dataset: str
    sources: list[str]
    date_range: tuple[str, str] | None = None
    partition_count: int = 0
    total_rows: int = 0
    total_distinct_mmsi: int = 0
    schema_versions: list[str] = Field(default_factory=list)
    has_mixed_versions: bool = False


# ---------------------------------------------------------------------------
# Catalog registry — manifest discovery, loading, and inventory
# ---------------------------------------------------------------------------


class CatalogRegistry:
    """Read-side catalog that discovers and queries partition manifests.

    Scans the ``manifests/`` directory tree, loads committed manifests,
    and provides inventory views for API and CLI consumption.

    Usage::

        registry = CatalogRegistry(store_root)
        registry.scan()

        # What do we have?
        inv = registry.inventory()
        parts = registry.partitions("positions", source="noaa")
        manifest = registry.get_manifest("positions", "noaa", "2024-06-15")
    """

    def __init__(self, store_root: Path) -> None:
        self.store_root = store_root
        self._manifests_dir = store_root / MANIFESTS_DIR
        self._manifests: dict[tuple[str, str, str], Manifest] = {}

    def scan(self) -> int:
        """Scan the manifests directory and load all manifest files.

        Returns the number of manifests loaded. Manifests that fail to
        parse are logged and skipped.
        """
        self._manifests.clear()

        if not self._manifests_dir.exists():
            logger.debug("No manifests directory at %s", self._manifests_dir)
            return 0

        count = 0
        for json_file in self._manifests_dir.rglob("*.json"):
            try:
                text = json_file.read_text(encoding="utf-8")
                m = Manifest.model_validate_json(text)
                key = (m.dataset, m.source, m.date)
                self._manifests[key] = m
                count += 1
            except Exception:
                logger.warning(
                    "Failed to load manifest: %s", json_file, exc_info=True
                )

        logger.debug("Loaded %d manifest(s) from %s", count, self._manifests_dir)
        return count

    # --- Manifest access ---

    def get_manifest(
        self,
        dataset: str,
        source: str,
        date: str,
    ) -> Manifest | None:
        """Return the manifest for a specific partition, or None."""
        return self._manifests.get((dataset, source, date))

    def all_manifests(self) -> list[Manifest]:
        """Return all loaded manifests."""
        return list(self._manifests.values())

    # --- Partition discovery ---

    def partitions(
        self,
        dataset: str | None = None,
        *,
        source: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        committed_only: bool = True,
    ) -> list[PartitionInfo]:
        """List partitions matching the given filters.

        Args:
            dataset: Filter by dataset name. None for all datasets.
            source: Filter by source. None for all sources.
            date_from: Inclusive start date (YYYY-MM-DD). None for no lower bound.
            date_to: Inclusive end date (YYYY-MM-DD). None for no upper bound.
            committed_only: If True, exclude non-committed partitions.

        Returns a list of ``PartitionInfo`` summaries, sorted by
        (dataset, source, date).
        """
        results: list[PartitionInfo] = []

        for m in self._manifests.values():
            if committed_only and m.write_status != WriteStatus.COMMITTED:
                continue
            if dataset is not None and m.dataset != dataset:
                continue
            if source is not None and m.source != source:
                continue
            if date_from is not None and m.date < date_from:
                continue
            if date_to is not None and m.date > date_to:
                continue
            results.append(PartitionInfo.from_manifest(m))

        results.sort(key=lambda p: (p.dataset, p.source, p.date))
        return results

    # --- File discovery for query engines ---

    def parquet_files(
        self,
        dataset: str,
        *,
        source: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[Path]:
        """Return absolute Parquet file paths for committed partitions.

        This is the primary interface for Polars scans and DuckDB
        registration — it returns the actual files to read.
        """
        manifests = self._filter_committed(dataset, source, date_from, date_to)

        files: list[Path] = []
        for m in manifests:
            partition_dir = self.store_root / canonical_partition_path(
                m.dataset, m.source, m.date
            )
            for fname in m.files:
                files.append(partition_dir / fname)

        files.sort()
        return files

    # --- Inventory views ---

    def inventory(
        self,
        dataset: str | None = None,
        *,
        committed_only: bool = True,
    ) -> list[DatasetInventory]:
        """Produce aggregate inventory summaries per dataset.

        Args:
            dataset: Summarize a single dataset. None for all.
            committed_only: If True, only count committed partitions.

        Returns one ``DatasetInventory`` per dataset found.
        """
        # Group manifests by dataset.
        by_dataset: dict[str, list[Manifest]] = {}
        for m in self._manifests.values():
            if committed_only and m.write_status != WriteStatus.COMMITTED:
                continue
            if dataset is not None and m.dataset != dataset:
                continue
            by_dataset.setdefault(m.dataset, []).append(m)

        results: list[DatasetInventory] = []
        for ds_name, manifests in sorted(by_dataset.items()):
            sources = sorted({m.source for m in manifests})
            dates = sorted(m.date for m in manifests)
            versions = sorted({m.schema_version for m in manifests})
            total_rows = sum(m.record_count for m in manifests)
            # distinct_mmsi across partitions is an approximation (sum, not union)
            total_mmsi = sum(m.distinct_mmsi_count for m in manifests)

            results.append(
                DatasetInventory(
                    dataset=ds_name,
                    sources=sources,
                    date_range=(dates[0], dates[-1]) if dates else None,
                    partition_count=len(manifests),
                    total_rows=total_rows,
                    total_distinct_mmsi=total_mmsi,
                    schema_versions=versions,
                    has_mixed_versions=len(versions) > 1,
                )
            )

        return results

    # --- Staleness detection ---

    def check_health(self) -> list[str]:
        """Check for catalog health issues.

        Returns a list of human-readable warning strings. An empty list
        means the catalog is healthy.

        Checks:
        - Non-committed partitions (staged or failed).
        - Mixed schema versions within a dataset.
        - Unknown schema versions.
        """
        warnings: list[str] = []

        for key, m in self._manifests.items():
            ds, src, dt = key

            if m.write_status == WriteStatus.STAGED:
                warnings.append(
                    f"{ds}/{src}/{dt}: partition is STAGED (incomplete write?)"
                )
            elif m.write_status == WriteStatus.FAILED:
                warnings.append(
                    f"{ds}/{src}/{dt}: partition is FAILED (needs cleanup)"
                )

            if not is_compatible(m.dataset, m.schema_version):
                warnings.append(
                    f"{ds}/{src}/{dt}: unknown schema version "
                    f"{m.schema_version!r}"
                )

        # Check for mixed versions per dataset.
        by_dataset: dict[str, set[str]] = {}
        for m in self._manifests.values():
            if m.write_status == WriteStatus.COMMITTED:
                by_dataset.setdefault(m.dataset, set()).add(m.schema_version)

        for ds_name, versions in by_dataset.items():
            if len(versions) > 1:
                warnings.append(
                    f"{ds_name}: mixed schema versions {sorted(versions)}"
                )

        return warnings

    # --- Provenance and quality report surfaces ---

    def quality_report(
        self,
        dataset: str,
        *,
        source: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> QualityReport:
        """Aggregate QC counters across matching committed partitions.

        Returns a ``QualityReport`` summarizing quality across the
        selected scope. This is the backing implementation for
        ``Neptune.quality_report()``.
        """
        from neptune_ais.qc import QualityReport

        manifests = self._filter_committed(dataset, source, date_from, date_to)

        total_rows = 0
        rows_ok = 0
        rows_warning = 0
        rows_error = 0
        rows_dropped = 0
        all_checks: set[str] = set()

        for m in manifests:
            qc = m.qc_summary
            total_rows += qc.total_rows
            rows_ok += qc.rows_ok
            rows_warning += qc.rows_warning
            rows_error += qc.rows_error
            rows_dropped += qc.rows_dropped
            all_checks.update(qc.checks_applied)

        return QualityReport(
            dataset=dataset,
            source=source,
            date_from=date_from,
            date_to=date_to,
            partitions_scanned=len(manifests),
            total_rows=total_rows,
            rows_ok=rows_ok,
            rows_warning=rows_warning,
            rows_error=rows_error,
            rows_dropped=rows_dropped,
            checks_applied=sorted(all_checks),
        )

    def provenance(
        self,
        dataset: str,
        *,
        source: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> ProvenanceSummary:
        """Summarize provenance across matching committed partitions.

        Returns a ``ProvenanceSummary`` tracing canonical data back to
        its origins. This is the backing implementation for
        ``Neptune.provenance()``.
        """
        from neptune_ais.qc import ProvenanceSummary

        manifests = self._filter_committed(dataset, source, date_from, date_to)

        schema_versions: set[str] = set()
        adapter_versions: set[str] = set()
        transform_versions: set[str] = set()
        raw_policies: set[str] = set()
        total_artifacts = 0
        with_local = 0
        without_local = 0

        for m in manifests:
            schema_versions.add(m.schema_version)
            adapter_versions.add(m.adapter_version)
            transform_versions.add(m.transform_version)
            raw_policies.add(m.raw_policy.value)
            for art in m.raw_artifacts:
                total_artifacts += 1
                if art.local_path:
                    with_local += 1
                else:
                    without_local += 1

        return ProvenanceSummary(
            dataset=dataset,
            source=source,
            date_from=date_from,
            date_to=date_to,
            partitions_scanned=len(manifests),
            schema_versions=sorted(schema_versions),
            adapter_versions=sorted(adapter_versions),
            transform_versions=sorted(transform_versions),
            total_raw_artifacts=total_artifacts,
            raw_policies=sorted(raw_policies),
            artifacts_with_local_copy=with_local,
            artifacts_without_local_copy=without_local,
        )

    # --- Internal helpers ---

    def _filter_committed(
        self,
        dataset: str,
        source: str | None,
        date_from: str | None,
        date_to: str | None,
    ) -> list[Manifest]:
        """Return committed manifests matching the given filters."""
        results: list[Manifest] = []
        for m in self._manifests.values():
            if m.write_status != WriteStatus.COMMITTED:
                continue
            if m.dataset != dataset:
                continue
            if source is not None and m.source != source:
                continue
            if date_from is not None and m.date < date_from:
                continue
            if date_to is not None and m.date > date_to:
                continue
            results.append(m)
        return results
