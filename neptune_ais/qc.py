"""QC — data quality checks and confidence scoring.

Layered quality model: hard-invalid rows are quarantined (removed from the
canonical store, kept in a quarantine table with reasons), and suspicious
rows are retained with ``qc_flags``/``qc_severity``.

QC does not compute ``confidence_score``. Any score here would be a
heuristic, and storing it in a column that reads like a probability would
overstate its calibration. Source-provided or fusion-assigned scores pass
through unchanged.

Module role — cross-cutting infrastructure
------------------------------------------
QC runs after normalization (on adapter output) and can also run after
fusion. It is invoked by ``api`` as part of the ingest pipeline.

**Owns:**
- The QC check registry and rule execution engine (``run_checks``).
- Built-in checks: lat/lon range, MMSI format, implausible speed, stale
  positions, timestamp monotonicity.
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

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import polars as pl
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
    interface. ``run_checks`` uses them to:
    - Build the ``qc_flags`` and ``qc_severity`` columns.
    - Decide which rows to quarantine (for HARD_INVALID checks).
    - Populate ``QCSummary`` in the manifest.

    ``expr(df)`` returns a boolean expression that is true for rows that
    fail the check (null counts as pass). The engine evaluates it on a
    frame sorted by ``mmsi``, ``timestamp`` and arrival order, with
    ``timestamp`` cast to ``Datetime(UTC)`` and an ``_qc_arrival`` column
    holding each row's original (arrival) index, so order-dependent checks
    can use ``shift``/``over("mmsi")``. Suspicious checks only see rows
    that passed every hard-invalid check. SOURCE_QUIRK checks are not
    executed: quirks are normalized by adapters.
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

    def expr(self, df: pl.DataFrame) -> pl.Expr:
        """Boolean expression, true where a row fails this check."""
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

    def expr(self, df: pl.DataFrame) -> pl.Expr:
        if self.column not in df.columns:
            return pl.lit(False)
        c = pl.col(self.column)
        out = (c < self.min_val) | (c > self.max_val)
        return out | c.is_nan() if df.schema[self.column].is_float() else out

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

    def expr(self, df: pl.DataFrame) -> pl.Expr:
        if self.column not in df.columns:
            return pl.lit(True)
        c = pl.col(self.column)
        return c.is_null() | c.is_nan() if df.schema[self.column].is_float() else c.is_null()


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

    def expr(self, df: pl.DataFrame) -> pl.Expr:
        if "mmsi" not in df.columns:
            return pl.lit(False)
        return ~pl.col("mmsi").is_between(100_000_000, 999_999_999)


class SpeedPlausibilityCheck:
    """Check for implausible speed, reported or implied.

    Rows whose reported ``sog`` or whose implied speed from the previous
    position of the same vessel exceeds the threshold (default 50 knots)
    are flagged as suspicious. High speed is context-dependent (fast craft,
    aircraft), so these rows are kept.
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
        return f"Reported and implied speed must be <= {self.max_knots} knots"

    def expr(self, df: pl.DataFrame) -> pl.Expr:
        out = pl.col("sog") > self.max_knots if "sog" in df.columns else pl.lit(False)
        if not {"mmsi", "timestamp", "lat", "lon"} <= set(df.columns):
            return out
        lat, lon = pl.col("lat").radians(), pl.col("lon").radians()
        a = ((lat - lat.shift(1)) / 2).sin() ** 2 + lat.cos() * lat.shift(1).cos() * (
            (lon - lon.shift(1)) / 2
        ).sin() ** 2
        nm = 2 * _EARTH_RADIUS_NM * a.sqrt().arcsin()
        hours = pl.col("timestamp").diff().dt.total_microseconds() / 3.6e9
        # ponytail: same-timestamp pairs are skipped (hours == 0) rather than
        # treated as infinite speed; duplicates are fusion's concern.
        implied = ((hours > 0) & (nm / hours > self.max_knots)).over("mmsi")
        return out | implied.fill_null(False)


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

    def expr(self, df: pl.DataFrame) -> pl.Expr:
        if not {"mmsi", "lat", "lon"} <= set(df.columns):
            return pl.lit(False)
        run_id = pl.struct("mmsi", "lat", "lon").rle_id()
        return pl.int_range(1, pl.len() + 1).over(run_id) > self.max_repeats


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

    def expr(self, df: pl.DataFrame) -> pl.Expr:
        # In time order, a row arrived out of order iff some later-timestamped
        # row of the same vessel arrived before it.
        row = pl.col("_qc_arrival")
        later_min = row.reverse().cum_min().reverse().shift(-1)
        return (row > later_min).over("mmsi").fill_null(False)


_EARTH_RADIUS_NM = 3440.065


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
# QC engine — execute checks, split accepted from quarantined rows
# ---------------------------------------------------------------------------


@dataclass
class QCOutcome:
    """Result of ``run_checks`` on one batch.

    ``accepted`` rows carry ``qc_flags`` (names of suspicious checks they
    failed) and ``qc_severity``. ``quarantined`` rows failed at least one
    hard-invalid check; their ``qc_flags`` lists those checks as reasons.
    """

    accepted: pl.DataFrame
    quarantined: pl.DataFrame
    check_results: list[CheckResult]

    @property
    def total_rows(self) -> int:
        return len(self.accepted) + len(self.quarantined)

    def severity_count(self, severity: Severity) -> int:
        return int((self.accepted["qc_severity"] == severity.value).sum())


def _flag_columns(df: pl.DataFrame, checks: list[QCCheck]) -> pl.DataFrame:
    return df.with_columns(
        c.expr(df).fill_null(False).alias(f"_qc__{c.name}") for c in checks
    )


def _annotate(df: pl.DataFrame, checks: list[QCCheck]) -> pl.DataFrame:
    """Set ``qc_flags``/``qc_severity`` from the ``_qc__*`` flag columns.

    Rows are encoded as a bitmask of failed checks; flags and severity are
    built once per distinct mask (few in practice) and joined back, which
    is much faster than building a list per row.
    """
    # ponytail: Int64 mask caps a single run at 63 checks.
    mask = pl.sum_horizontal(
        pl.lit(0, pl.Int64),
        *(pl.col(f"_qc__{c.name}").cast(pl.Int64) * (1 << i) for i, c in enumerate(checks)),
    )
    df = df.with_columns(mask.alias("_qc_mask"))
    masks = df["_qc_mask"].unique().to_list()
    failed = [[c for i, c in enumerate(checks) if m >> i & 1] for m in masks]
    lookup = pl.DataFrame(
        {
            "_qc_mask": masks,
            "qc_flags": [[c.name for c in f] for f in failed],
            "qc_severity": [
                next(
                    (s.value for s in (Severity.ERROR, Severity.WARNING)
                     if any(c.severity == s for c in f)),
                    Severity.OK.value,
                )
                for f in failed
            ],
        },
        schema={"_qc_mask": pl.Int64, "qc_flags": pl.List(pl.String), "qc_severity": pl.String},
    )
    df = df.drop([c for c in ("qc_flags", "qc_severity") if c in df.columns])
    return df.join(lookup, on="_qc_mask", how="left").sort("_qc_row")


def _results(df: pl.DataFrame, checks: list[QCCheck]) -> list[CheckResult]:
    return [
        CheckResult(
            check_name=c.name,
            rows_checked=len(df),
            rows_flagged=int(df[f"_qc__{c.name}"].sum()),
            severity=c.severity,
            description=c.description,
        )
        for c in checks
    ]


def run_checks(df: pl.DataFrame, checks: list[QCCheck] | None = None) -> QCOutcome:
    """Execute QC checks on a positions batch.

    Hard-invalid checks run first; rows failing any of them are quarantined.
    Suspicious checks then run on the remaining rows, which are retained
    with flags. Both outputs are sorted by ``mmsi``, ``timestamp``.
    String timestamps are parsed to ``Datetime(UTC)`` (unparseable → null,
    which ``timestamp_not_null`` quarantines).
    """
    checks = BUILTIN_POSITIONS_CHECKS if checks is None else checks
    hard = [c for c in checks if c.qc_class == QCClass.HARD_INVALID]
    soft = [c for c in checks if c.qc_class == QCClass.SUSPICIOUS]

    df = df.with_row_index("_qc_row")
    if df.schema.get("timestamp") == pl.String:
        df = df.with_columns(
            pl.col("timestamp").str.to_datetime(strict=False, time_zone="UTC")
        )
    # _qc_arrival: original row index (monotonicity check, sort tie-break).
    # _qc_row: index in sorted order, which _annotate restores after its join.
    df = (
        df.sort([c for c in ("mmsi", "timestamp", "_qc_row") if c in df.columns])
        .rename({"_qc_row": "_qc_arrival"})
        .with_row_index("_qc_row")
    )

    df = _annotate(_flag_columns(df, hard), hard)
    results = _results(df, hard)
    invalid = pl.col("qc_flags").list.len() > 0
    quarantined = df.filter(invalid).with_columns(
        pl.lit(Severity.ERROR.value).alias("qc_severity")
    )
    accepted = _flag_columns(df.filter(~invalid), soft)
    results += _results(accepted, soft)
    accepted = _annotate(accepted, soft)

    def clean(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.drop([c for c in frame.columns if c.startswith("_qc_")])

    return QCOutcome(clean(accepted), clean(quarantined), results)


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
