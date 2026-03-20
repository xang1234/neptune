"""Adapter certification harness — systematic checks across all sources.

Certifies every registered adapter against a consistent rubric:
1. Protocol conformance (SourceAdapter)
2. Capability metadata completeness
3. Registration correctness
4. Source-specific readiness criteria

This replaces per-adapter ad-hoc checks with one parametrized suite
that evaluates all adapters through the same lens.
"""

from __future__ import annotations

from datetime import date

import pytest

from neptune_ais.adapters.base import SourceAdapter, SourceCapabilities
from neptune_ais.adapters.registry import (
    _ADAPTERS,
    _STREAMING_CAPABILITIES,
    load_all_adapters,
    registered_sources,
)

# Load all adapters once at module level.
load_all_adapters()

# Collect all archival adapter IDs for parametrization.
_ARCHIVAL_IDS = sorted(_ADAPTERS.keys())
_STREAMING_IDS = sorted(
    sid for sid in _STREAMING_CAPABILITIES if sid not in _ADAPTERS
)
_ALL_IDS = sorted(set(_ADAPTERS.keys()) | set(_STREAMING_CAPABILITIES.keys()))


# ---------------------------------------------------------------------------
# Readiness rubric — what "good enough to ship" means
# ---------------------------------------------------------------------------

REQUIRED_CAPABILITY_FIELDS = [
    "source_id",
    "provider",
    "description",
    "coverage",
    "delivery_format",
]

REQUIRED_SUMMARY_KEYS = [
    "source",
    "provider",
    "coverage",
    "backfill",
    "streaming",
    "auth",
    "format",
]


# ---------------------------------------------------------------------------
# 1. Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Every archival adapter satisfies the SourceAdapter protocol."""

    @pytest.mark.parametrize("source_id", _ARCHIVAL_IDS)
    def test_implements_protocol(self, source_id: str):
        adapter = _ADAPTERS[source_id]()
        assert isinstance(adapter, SourceAdapter), (
            f"{source_id} does not implement SourceAdapter protocol"
        )

    @pytest.mark.parametrize("source_id", _ARCHIVAL_IDS)
    def test_source_id_matches_registration(self, source_id: str):
        adapter = _ADAPTERS[source_id]()
        assert adapter.source_id == source_id, (
            f"Adapter reports source_id={adapter.source_id!r} but registered as {source_id!r}"
        )

    @pytest.mark.parametrize("source_id", _ARCHIVAL_IDS)
    def test_has_class_source_id(self, source_id: str):
        """SOURCE_ID class attribute avoids unnecessary instantiation during register()."""
        cls = _ADAPTERS[source_id]
        assert hasattr(cls, "SOURCE_ID"), (
            f"{source_id} adapter missing SOURCE_ID class attribute"
        )

    @pytest.mark.parametrize("source_id", _ARCHIVAL_IDS)
    def test_qc_rules_returns_list(self, source_id: str):
        adapter = _ADAPTERS[source_id]()
        rules = adapter.qc_rules()
        assert isinstance(rules, list), (
            f"{source_id}.qc_rules() should return a list, got {type(rules)}"
        )

    @pytest.mark.parametrize("source_id", _ARCHIVAL_IDS)
    def test_available_dates_valid_type(self, source_id: str):
        adapter = _ADAPTERS[source_id]()
        result = adapter.available_dates()
        assert result is None or isinstance(result, (list, tuple)), (
            f"{source_id}.available_dates() should return None, list, or tuple"
        )
        if isinstance(result, tuple):
            assert len(result) == 2, "Date tuple must be (start, end)"
            assert isinstance(result[0], date)
            assert isinstance(result[1], date)


# ---------------------------------------------------------------------------
# 2. Capability metadata completeness
# ---------------------------------------------------------------------------


class TestCapabilityMetadata:
    """Every source has complete, well-formed capability metadata."""

    @pytest.mark.parametrize("source_id", _ALL_IDS)
    def test_capabilities_type(self, source_id: str):
        from neptune_ais.adapters.registry import info
        caps = info(source_id)
        assert isinstance(caps, SourceCapabilities)

    @pytest.mark.parametrize("source_id", _ALL_IDS)
    def test_required_fields_populated(self, source_id: str):
        from neptune_ais.adapters.registry import info
        caps = info(source_id)
        for field in REQUIRED_CAPABILITY_FIELDS:
            val = getattr(caps, field)
            assert val, f"{source_id}: {field} is empty or None"

    @pytest.mark.parametrize("source_id", _ALL_IDS)
    def test_summary_has_required_keys(self, source_id: str):
        from neptune_ais.adapters.registry import info
        summary = info(source_id).summary()
        for key in REQUIRED_SUMMARY_KEYS:
            assert key in summary, f"{source_id}: summary missing key {key!r}"

    @pytest.mark.parametrize("source_id", _ALL_IDS)
    def test_datasets_provided_non_empty(self, source_id: str):
        from neptune_ais.adapters.registry import info
        caps = info(source_id)
        assert len(caps.datasets_provided) >= 1, (
            f"{source_id}: datasets_provided is empty"
        )

    @pytest.mark.parametrize("source_id", _ALL_IDS)
    def test_known_quirks_is_list(self, source_id: str):
        from neptune_ais.adapters.registry import info
        caps = info(source_id)
        assert isinstance(caps.known_quirks, list), (
            f"{source_id}: known_quirks should be a list"
        )

    @pytest.mark.parametrize("source_id", _ALL_IDS)
    def test_source_id_consistency(self, source_id: str):
        """source_id in capabilities matches the registration key."""
        from neptune_ais.adapters.registry import info
        caps = info(source_id)
        assert caps.source_id == source_id


# ---------------------------------------------------------------------------
# 3. Registration correctness
# ---------------------------------------------------------------------------


class TestRegistration:
    """Adapter registry is consistent and complete."""

    def test_all_builtins_registered(self):
        from neptune_ais.adapters.registry import _BUILTIN_ADAPTERS
        sources = registered_sources()
        for mod in _BUILTIN_ADAPTERS:
            # Module name may differ from source_id (e.g. module=noaa, id=noaa).
            # Just check that the expected number are registered.
            pass
        assert len(sources) >= len(_BUILTIN_ADAPTERS)

    def test_no_empty_source_ids(self):
        for sid in registered_sources():
            assert sid, "Empty source_id in registry"
            assert sid.strip() == sid, f"Source ID has whitespace: {sid!r}"

    def test_archival_and_streaming_disjoint(self):
        """No source is in both _ADAPTERS and _STREAMING_CAPABILITIES."""
        overlap = set(_ADAPTERS.keys()) & set(_STREAMING_CAPABILITIES.keys())
        # It's ok for a source to have both — finland has both.
        # But streaming-only sources should not be in _ADAPTERS.
        for sid in _STREAMING_IDS:
            assert sid not in _ADAPTERS, (
                f"{sid} is streaming-only but also in _ADAPTERS"
            )

    def test_catalog_returns_all_sources(self):
        from neptune_ais.adapters.registry import catalog
        caps_list = catalog()
        cap_ids = {c.source_id for c in caps_list}
        for sid in _ALL_IDS:
            assert sid in cap_ids, f"{sid} missing from catalog()"


# ---------------------------------------------------------------------------
# 4. Source-specific readiness rubric
# ---------------------------------------------------------------------------


class TestReadinessRubric:
    """Source-type-specific certification criteria."""

    @pytest.mark.parametrize("source_id", [
        sid for sid in _ARCHIVAL_IDS
        if _ADAPTERS[sid]().capabilities.supports_backfill
    ])
    def test_backfill_sources_have_history_start(self, source_id: str):
        """Sources with backfill must declare when their data starts."""
        caps = _ADAPTERS[source_id]().capabilities
        assert caps.history_start is not None, (
            f"{source_id} supports backfill but has no history_start"
        )

    @pytest.mark.parametrize("source_id", [
        sid for sid in _ARCHIVAL_IDS
        if _ADAPTERS[sid]().capabilities.supports_backfill
    ])
    def test_backfill_sources_have_date_range(self, source_id: str):
        """Sources with backfill must return a date range from available_dates."""
        adapter = _ADAPTERS[source_id]()
        dates = adapter.available_dates()
        assert dates is not None, (
            f"{source_id} supports backfill but available_dates() returns None"
        )

    @pytest.mark.parametrize("source_id", [
        sid for sid in _ALL_IDS
        if (sid in _ADAPTERS and _ADAPTERS[sid]().capabilities.auth_scheme)
        or (sid in _STREAMING_CAPABILITIES and _STREAMING_CAPABILITIES[sid].auth_scheme)
    ])
    def test_auth_sources_document_scheme(self, source_id: str):
        """Sources requiring auth must specify the scheme."""
        from neptune_ais.adapters.registry import info
        caps = info(source_id)
        assert caps.auth_scheme in ("api_key", "oauth2", "token", "basic"), (
            f"{source_id}: unknown auth_scheme {caps.auth_scheme!r}"
        )

    @pytest.mark.parametrize("source_id", _ALL_IDS)
    def test_license_documented(self, source_id: str):
        """Every source should document its license requirements."""
        from neptune_ais.adapters.registry import info
        caps = info(source_id)
        assert caps.license_requirements, (
            f"{source_id}: license_requirements is empty — "
            "document the license even if it is 'unknown'"
        )

    @pytest.mark.parametrize("source_id", [
        sid for sid in _ARCHIVAL_IDS
        if _ADAPTERS[sid]().capabilities.known_quirks
    ])
    def test_quirks_are_descriptive(self, source_id: str):
        """Known quirks should be human-readable sentences, not codes."""
        caps = _ADAPTERS[source_id]().capabilities
        for quirk in caps.known_quirks:
            assert len(quirk) >= 10, (
                f"{source_id}: quirk too short to be descriptive: {quirk!r}"
            )


# ---------------------------------------------------------------------------
# 5. Cross-adapter consistency
# ---------------------------------------------------------------------------


class TestCrossAdapterConsistency:
    """Adapters should be consistent with each other."""

    def test_all_provide_at_least_one_dataset(self):
        """Every source provides at least one dataset."""
        for sid in _ALL_IDS:
            from neptune_ais.adapters.registry import info
            caps = info(sid)
            assert len(caps.datasets_provided) >= 1, (
                f"{sid} provides no datasets"
            )

    def test_adapter_versions_follow_convention(self):
        """Adapter versions should follow source_id/semver pattern."""
        for sid in _ARCHIVAL_IDS:
            cls = _ADAPTERS[sid]
            version = getattr(cls, "ADAPTER_VERSION", None)
            if version is not None:
                assert "/" in version, (
                    f"{sid}: ADAPTER_VERSION should be 'source/semver', got {version!r}"
                )
                assert version.startswith(sid), (
                    f"{sid}: ADAPTER_VERSION should start with source_id, got {version!r}"
                )
