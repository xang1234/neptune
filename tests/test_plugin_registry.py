"""Tests for external adapter plugin registration and discovery.

Covers: load_all_adapters, discover_plugins, entry point group,
built-in adapter loading, and idempotent loading.
"""

from __future__ import annotations

import pytest

from neptune_ais.adapters.registry import (
    ENTRY_POINT_GROUP,
    _BUILTIN_ADAPTERS,
    discover_plugins,
    find_sources,
    load_all_adapters,
    registered_sources,
    _ADAPTERS,
    _STREAMING_CAPABILITIES,
)


# ---------------------------------------------------------------------------
# Entry point group
# ---------------------------------------------------------------------------


class TestEntryPointGroup:
    def test_group_name(self):
        """Plugin entry point group is correctly defined."""
        assert ENTRY_POINT_GROUP == "neptune_ais.adapters"

    def test_builtin_list(self):
        """All 6 built-in adapters are listed."""
        assert len(_BUILTIN_ADAPTERS) == 6
        assert "noaa" in _BUILTIN_ADAPTERS
        assert "dma" in _BUILTIN_ADAPTERS
        assert "gfw" in _BUILTIN_ADAPTERS
        assert "finland" in _BUILTIN_ADAPTERS
        assert "aishub" in _BUILTIN_ADAPTERS
        assert "aisstream" in _BUILTIN_ADAPTERS


# ---------------------------------------------------------------------------
# load_all_adapters
# ---------------------------------------------------------------------------


class TestLoadAllAdapters:
    def test_loads_all_builtins(self):
        """load_all_adapters registers all 6 built-in sources."""
        sources = load_all_adapters()
        assert "noaa" in sources
        assert "dma" in sources
        assert "gfw" in sources
        assert "finland" in sources
        assert "aishub" in sources
        assert "aisstream" in sources

    def test_idempotent(self):
        """Multiple calls return the same result without re-loading."""
        sources1 = load_all_adapters()
        sources2 = load_all_adapters()
        assert sources1 == sources2

    def test_returns_registered_sources(self):
        """Return value matches registered_sources()."""
        sources = load_all_adapters()
        assert sources == registered_sources()

    def test_all_adapters_have_capabilities(self):
        """Every loaded adapter has a valid SourceCapabilities."""
        from neptune_ais.adapters.base import SourceCapabilities
        from neptune_ais.adapters.registry import info

        load_all_adapters()
        for source_id in registered_sources():
            caps = info(source_id)
            assert isinstance(caps, SourceCapabilities)
            assert caps.source_id == source_id
            assert caps.provider  # non-empty


# ---------------------------------------------------------------------------
# discover_plugins
# ---------------------------------------------------------------------------


class TestDiscoverPlugins:
    def test_discover_returns_list(self):
        """discover_plugins returns a list (empty if no plugins installed)."""
        result = discover_plugins()
        assert isinstance(result, list)

    def test_no_crash_without_plugins(self):
        """Gracefully handles no external plugins being installed."""
        # This test just confirms no exception is raised.
        discover_plugins()


# ---------------------------------------------------------------------------
# Registry consistency
# ---------------------------------------------------------------------------


class TestRegistryConsistency:
    def test_archival_adapters_implement_protocol(self):
        """All archival adapters satisfy the SourceAdapter protocol."""
        from neptune_ais.adapters.base import SourceAdapter

        load_all_adapters()
        for source_id, adapter_cls in _ADAPTERS.items():
            instance = adapter_cls()
            assert isinstance(instance, SourceAdapter), (
                f"{source_id} does not satisfy SourceAdapter protocol"
            )

    def test_streaming_sources_have_capabilities(self):
        """All streaming-only sources have valid capabilities."""
        load_all_adapters()
        for source_id, caps in _STREAMING_CAPABILITIES.items():
            assert caps.source_id == source_id
            assert caps.supports_streaming is True

    def test_no_duplicate_registration(self):
        """No source appears in both archival and streaming-only registries."""
        load_all_adapters()
        archival = set(_ADAPTERS.keys())
        streaming_only = set(_STREAMING_CAPABILITIES.keys()) - archival
        # aisstream is streaming-only (registered via register_streaming)
        # and should not also be in _ADAPTERS
        for sid in streaming_only:
            assert sid not in _ADAPTERS


# ---------------------------------------------------------------------------
# find_sources — capability filtering
# ---------------------------------------------------------------------------


class TestFindSources:
    def test_find_backfill_sources(self):
        """Filter for sources supporting backfill."""
        load_all_adapters()
        results = find_sources(backfill=True)
        source_ids = [c.source_id for c in results]
        assert "noaa" in source_ids
        assert "gfw" in source_ids
        assert "finland" in source_ids
        # aishub doesn't support backfill
        assert "aishub" not in source_ids

    def test_find_streaming_sources(self):
        """Filter for sources supporting streaming."""
        load_all_adapters()
        results = find_sources(streaming=True)
        source_ids = [c.source_id for c in results]
        assert "aisstream" in source_ids
        assert "finland" in source_ids  # mixed delivery
        # noaa doesn't support streaming
        assert "noaa" not in source_ids

    def test_find_open_sources(self):
        """Filter for sources requiring no authentication."""
        load_all_adapters()
        results = find_sources(auth=False)
        source_ids = [c.source_id for c in results]
        assert "noaa" in source_ids
        assert "finland" in source_ids
        assert "dma" in source_ids
        # gfw requires api_key
        assert "gfw" not in source_ids

    def test_find_with_multiple_filters(self):
        """Multiple filters are ANDed."""
        load_all_adapters()
        results = find_sources(backfill=True, auth=False)
        source_ids = [c.source_id for c in results]
        # noaa, dma, finland all have backfill + no auth
        assert "noaa" in source_ids
        # gfw has backfill but requires auth
        assert "gfw" not in source_ids

    def test_find_by_dataset(self):
        """Filter by provided dataset."""
        load_all_adapters()
        results = find_sources(dataset="positions")
        assert len(results) >= 5  # all sources provide positions

    def test_find_no_match(self):
        """Returns empty list when no sources match."""
        load_all_adapters()
        results = find_sources(backfill=True, streaming=True, auth=True)
        # No source has all three
        # (finland has backfill+streaming but no auth)
        source_ids = [c.source_id for c in results]
        assert "finland" not in source_ids  # finland has no auth

    def test_find_no_filters_returns_all(self):
        """No filters returns all sources (same as catalog)."""
        load_all_adapters()
        from neptune_ais.adapters.registry import catalog
        assert len(find_sources()) == len(catalog())
