"""QC — data quality checks and confidence scoring.

Layered quality model: hard-invalid rows are dropped, suspicious rows are
flagged with qc_flags/qc_severity, and per-row confidence_score is computed.

Module role — cross-cutting infrastructure
------------------------------------------
QC runs after normalization (on adapter output) and can also run after
fusion. It is invoked by ``api`` as part of the ingest pipeline.

**Owns:**
- The QC check registry and rule execution engine.
- Built-in checks: lat/lon range, MMSI format, impossible speed, stale
  positions, heading sentinels, duplicate detection, timestamp monotonicity.
- Confidence score computation.
- Dataset-level quality report aggregation.
- The ``QCRule`` protocol that adapters can implement to supply
  source-specific checks.

**Does not own:**
- Schema definitions — those are in ``datasets``.
- Source-specific QC rules — those are supplied by ``adapters`` via the
  protocol, but registered and executed here.
- Manifest QC counters — those are written by ``catalog``.

**Import rule:** QC may import from ``datasets`` (column names for checks).
It must not import from ``adapters``, ``derive``, ``geometry``, ``cli``,
or ``api``.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# QC severity levels
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Row-level quality severity.

    Matches the vocabulary in ``datasets.positions.QC_SEVERITY_VALUES``.
    """

    OK = "ok"
    """Row passed all checks."""

    WARNING = "warning"
    """Row is suspicious but retained with flags."""

    ERROR = "error"
    """Row failed a hard check but was retained (e.g. for inspection).
    Depending on pipeline config, error rows may also be dropped."""


# ---------------------------------------------------------------------------
# QC check classes — the three-tier quality taxonomy
# ---------------------------------------------------------------------------


class QCClass(str, Enum):
    """Classification of a QC check by how it handles failures.

    The v3 plan defines three distinct tiers:

    1. **HARD_INVALID** — structurally broken or impossible values.
       Action: drop the row before writing. These rows never reach the
       canonical store. Examples: lat outside [-90, 90], malformed MMSI,
       null required fields.

    2. **SUSPICIOUS** — plausible but anomalous values.
       Action: keep the row, add a flag to ``qc_flags``, set
       ``qc_severity`` to ``warning``, lower ``confidence_score``.
       Examples: implied speed > 50 knots, stale repeated position,
       non-monotonic timestamp within a vessel stream.

    3. **SOURCE_QUIRK** — source-specific encoding or sentinel values
       that need normalization rather than flagging.
       Action: normalize the value (e.g. sentinel → null), annotate
       provenance. No severity penalty. Examples: heading=511 means
       "not available" in AIS, NOAA-specific field encoding.
    """

    HARD_INVALID = "hard_invalid"
    """Structurally broken — drop the row."""

    SUSPICIOUS = "suspicious"
    """Anomalous — flag and lower confidence, but keep."""

    SOURCE_QUIRK = "source_quirk"
    """Source encoding — normalize, don't penalize."""


# ---------------------------------------------------------------------------
# QC check protocol — the contract for all checks
# ---------------------------------------------------------------------------


@runtime_checkable
class QCCheck(Protocol):
    """Protocol that all QC checks must satisfy.

    Built-in checks and adapter-supplied checks both implement this
    interface. The QC engine discovers checks via their metadata
    properties and uses them to:
    - Build the ``qc_flags`` and ``qc_severity`` columns.
    - Compute ``confidence_score``.
    - Decide which rows to drop (for HARD_INVALID checks).
    - Populate ``QCSummary`` in the manifest.

    Note: the ``apply()`` execution method is not part of this protocol
    because different check types need different signatures (e.g. range
    checks operate on a single column, while speed plausibility checks
    need the full DataFrame sorted by MMSI+timestamp). The QC engine
    dispatches to checks by type.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this check, e.g. 'lat_range'."""
        ...

    @property
    def qc_class(self) -> QCClass:
        """Which QC tier this check belongs to."""
        ...

    @property
    def severity(self) -> Severity:
        """Severity assigned to flagged rows."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description of what this check tests."""
        ...


# ---------------------------------------------------------------------------
# Built-in check definitions
# ---------------------------------------------------------------------------


class RangeCheck:
    """Check that a numeric column's values fall within a valid range.

    Values outside the range are flagged. Used for lat, lon, sog, cog,
    heading, and confidence_score bounds.
    """

    def __init__(
        self,
        name: str,
        column: str,
        min_val: float,
        max_val: float,
        *,
        qc_class: QCClass = QCClass.HARD_INVALID,
        severity: Severity = Severity.ERROR,
        description: str = "",
    ) -> None:
        self._name = name
        self.column = column
        self.min_val = min_val
        self.max_val = max_val
        self._qc_class = qc_class
        self._severity = severity
        self._description = description or (
            f"{column} must be in [{min_val}, {max_val}]"
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def qc_class(self) -> QCClass:
        return self._qc_class

    @property
    def severity(self) -> Severity:
        return self._severity

    @property
    def description(self) -> str:
        return self._description


class NotNullCheck:
    """Check that a required column contains no null values.

    Nulls in required columns are hard-invalid.
    """

    def __init__(self, name: str, column: str) -> None:
        self._name = name
        self.column = column

    @property
    def name(self) -> str:
        return self._name

    @property
    def qc_class(self) -> QCClass:
        return QCClass.HARD_INVALID

    @property
    def severity(self) -> Severity:
        return Severity.ERROR

    @property
    def description(self) -> str:
        return f"{self.column} must not be null"


class MMSIFormatCheck:
    """Check that MMSI values are 9-digit positive integers.

    MMSIs outside the valid range [100000000, 999999999] are hard-invalid.
    """

    @property
    def name(self) -> str:
        return "mmsi_format"

    @property
    def qc_class(self) -> QCClass:
        return QCClass.HARD_INVALID

    @property
    def severity(self) -> Severity:
        return Severity.ERROR

    @property
    def description(self) -> str:
        return "MMSI must be a 9-digit integer in [100000000, 999999999]"


class SpeedPlausibilityCheck:
    """Check for implausible implied speed between consecutive positions.

    Vessels reporting positions that imply speeds above the threshold
    (default 50 knots) are flagged as suspicious.
    """

    def __init__(self, max_knots: float = 50.0) -> None:
        self.max_knots = max_knots

    @property
    def name(self) -> str:
        return "speed_plausibility"

    @property
    def qc_class(self) -> QCClass:
        return QCClass.SUSPICIOUS

    @property
    def severity(self) -> Severity:
        return Severity.WARNING

    @property
    def description(self) -> str:
        return f"Implied speed between consecutive points must be <= {self.max_knots} knots"


class StalePositionCheck:
    """Check for repeated identical positions over a long window.

    Vessels reporting the exact same lat/lon for extended periods are
    flagged as suspicious (likely a stuck transponder or anchored vessel
    with stale reports).
    """

    def __init__(self, max_repeats: int = 100) -> None:
        self.max_repeats = max_repeats

    @property
    def name(self) -> str:
        return "stale_position"

    @property
    def qc_class(self) -> QCClass:
        return QCClass.SUSPICIOUS

    @property
    def severity(self) -> Severity:
        return Severity.WARNING

    @property
    def description(self) -> str:
        return f"Flag if same lat/lon repeats > {self.max_repeats} times consecutively"


class TimestampMonotonicityCheck:
    """Check for non-monotonic timestamps within a vessel's stream.

    Within a sorted-by-MMSI partition, timestamps should be non-decreasing
    for each vessel. Non-monotonic timestamps suggest data corruption or
    out-of-order message delivery.
    """

    @property
    def name(self) -> str:
        return "timestamp_monotonicity"

    @property
    def qc_class(self) -> QCClass:
        return QCClass.SUSPICIOUS

    @property
    def severity(self) -> Severity:
        return Severity.WARNING

    @property
    def description(self) -> str:
        return "Timestamps must be non-decreasing within each vessel's stream"


# ---------------------------------------------------------------------------
# Built-in check registry — default checks for positions
# ---------------------------------------------------------------------------

BUILTIN_POSITIONS_CHECKS: list[QCCheck] = [
    # Hard-invalid range checks (from datasets.positions.VALID_RANGES)
    RangeCheck("lat_range", "lat", -90.0, 90.0),
    RangeCheck("lon_range", "lon", -180.0, 180.0),
    RangeCheck("sog_range", "sog", 0.0, 102.3, qc_class=QCClass.SUSPICIOUS, severity=Severity.WARNING),
    RangeCheck("cog_range", "cog", 0.0, 360.0, qc_class=QCClass.SUSPICIOUS, severity=Severity.WARNING),
    RangeCheck("heading_range", "heading", 0.0, 360.0, qc_class=QCClass.SUSPICIOUS, severity=Severity.WARNING),
    # Required field null checks
    NotNullCheck("mmsi_not_null", "mmsi"),
    NotNullCheck("timestamp_not_null", "timestamp"),
    NotNullCheck("lat_not_null", "lat"),
    NotNullCheck("lon_not_null", "lon"),
    NotNullCheck("source_not_null", "source"),
    # Format checks
    MMSIFormatCheck(),
    # Suspicious pattern checks
    SpeedPlausibilityCheck(),
    StalePositionCheck(),
    TimestampMonotonicityCheck(),
]
"""Default QC checks applied to the positions dataset.

Adapters may supply additional source-specific checks via the
``SourceAdapter.qc_rules()`` method. Those are appended to this list
during the ingest pipeline.
"""


# ---------------------------------------------------------------------------
# QC check result — output of a single check on a single partition
# ---------------------------------------------------------------------------


class CheckResult(BaseModel):
    """Result of running one QC check on a batch of data.

    Produced by the QC engine and aggregated into quality reports.
    """

    check_name: str = Field(
        description="Identifier of the QC check, e.g. 'lat_range', 'mmsi_format'.",
    )
    rows_checked: int = Field(
        ge=0,
        description="Number of rows the check was applied to.",
    )
    rows_flagged: int = Field(
        ge=0,
        description="Number of rows that failed this check.",
    )
    severity: Severity = Field(
        description="The severity level assigned to flagged rows by this check.",
    )
    description: str = Field(
        default="",
        description="Human-readable description of what this check tests.",
    )


# ---------------------------------------------------------------------------
# Quality report — dataset- or partition-level quality summary
# ---------------------------------------------------------------------------


class QualityReport(BaseModel):
    """Quality report for one or more partitions.

    Aggregates QC counters and check results into a single inspectable
    object. This is the return type of ``Neptune.quality_report()`` and
    the CLI ``neptune qc`` command.
    """

    # --- scope ---

    dataset: str = Field(description="Dataset name.")
    source: str | None = Field(
        default=None,
        description="Source filter, or None for all sources.",
    )
    date_from: str | None = Field(
        default=None,
        description="Start of date range (inclusive), or None.",
    )
    date_to: str | None = Field(
        default=None,
        description="End of date range (inclusive), or None.",
    )

    # --- aggregate counters ---

    partitions_scanned: int = Field(
        ge=0,
        description="Number of partitions included in this report.",
    )
    total_rows: int = Field(
        ge=0,
        description="Total rows across all scanned partitions (before drops).",
    )
    rows_ok: int = Field(ge=0)
    rows_warning: int = Field(ge=0)
    rows_error: int = Field(ge=0)
    rows_dropped: int = Field(ge=0)

    # --- derived metrics ---

    @property
    def rows_written(self) -> int:
        """Rows actually written (total minus dropped)."""
        return self.total_rows - self.rows_dropped

    @property
    def ok_rate(self) -> float:
        """Fraction of written rows that are ok (0.0–1.0)."""
        written = self.rows_written
        return self.rows_ok / written if written > 0 else 0.0

    @property
    def warning_rate(self) -> float:
        """Fraction of written rows that are warnings (0.0–1.0)."""
        written = self.rows_written
        return self.rows_warning / written if written > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Fraction of written rows that are errors (0.0–1.0)."""
        written = self.rows_written
        return self.rows_error / written if written > 0 else 0.0

    @property
    def drop_rate(self) -> float:
        """Fraction of total rows that were dropped (0.0–1.0)."""
        return self.rows_dropped / self.total_rows if self.total_rows > 0 else 0.0

    # --- per-check detail ---

    checks_applied: list[str] = Field(
        default_factory=list,
        description="Union of all QC check names applied across partitions.",
    )
    check_results: list[CheckResult] = Field(
        default_factory=list,
        description=(
            "Per-check results. Populated when a detailed quality report "
            "is requested; may be empty for summary-only reports."
        ),
    )


# ---------------------------------------------------------------------------
# Provenance summary — trace canonical data back to its origins
# ---------------------------------------------------------------------------


class ProvenanceSummary(BaseModel):
    """Provenance summary for one or more partitions.

    Answers: where did this data come from, what processed it, and
    can we rebuild it?

    This is the return type of ``Neptune.provenance()``.
    """

    dataset: str
    source: str | None = None
    date_from: str | None = None
    date_to: str | None = None

    partitions_scanned: int = 0

    # --- version info ---

    schema_versions: list[str] = Field(
        default_factory=list,
        description="Distinct schema versions found across partitions.",
    )
    adapter_versions: list[str] = Field(
        default_factory=list,
        description="Distinct adapter versions found.",
    )
    transform_versions: list[str] = Field(
        default_factory=list,
        description="Distinct transform versions found.",
    )

    # --- raw artifact summary ---

    total_raw_artifacts: int = Field(
        default=0,
        description="Total number of raw artifacts across all partitions.",
    )
    raw_policies: list[str] = Field(
        default_factory=list,
        description="Distinct raw policies found.",
    )
    artifacts_with_local_copy: int = Field(
        default=0,
        description="Raw artifacts that have a retained local file.",
    )
    artifacts_without_local_copy: int = Field(
        default=0,
        description="Raw artifacts with no local file (metadata or none policy).",
    )

    # --- rebuild assessment ---

    @property
    def can_rebuild_locally(self) -> bool:
        """True if all raw artifacts have retained local copies."""
        return (
            self.total_raw_artifacts > 0
            and self.artifacts_without_local_copy == 0
        )

    @property
    def has_mixed_versions(self) -> bool:
        """True if partitions were written with different schema/adapter versions."""
        return (
            len(self.schema_versions) > 1
            or len(self.adapter_versions) > 1
            or len(self.transform_versions) > 1
        )
