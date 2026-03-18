"""Registry — adapter discovery, catalog, and capability queries.

Provides ``catalog()``, ``available(date)``, ``info(source_id)``, and
``capabilities(source_id)`` for programmatic source inspection.

This module is importable as ``neptune_ais.sources``.
"""

from __future__ import annotations

from datetime import date

from neptune_ais.adapters.base import SourceAdapter, SourceCapabilities

# ---------------------------------------------------------------------------
# Adapter registry — maps source_id to adapter class
# ---------------------------------------------------------------------------

_ADAPTERS: dict[str, type[SourceAdapter]] = {}
_STREAMING_CAPABILITIES: dict[str, SourceCapabilities] = {}


def register_streaming(caps: SourceCapabilities) -> None:
    """Register capabilities for a streaming-only source.

    Streaming sources don't implement the full SourceAdapter protocol
    (no fetch_raw/normalize_positions). This registers their capabilities
    so they appear in ``catalog()``, ``info()``, and ``sources`` CLI.
    """
    _STREAMING_CAPABILITIES[caps.source_id] = caps


def register(adapter_cls: type[SourceAdapter]) -> type[SourceAdapter]:
    """Register an adapter class. Can be used as a decorator.

    Usage::

        @register
        class NOAAAdapter:
            ...
    """
    # Instantiate briefly to read source_id, or read from class attribute.
    source_id = getattr(adapter_cls, "SOURCE_ID", None)
    if source_id is None:
        instance = adapter_cls()
        source_id = instance.source_id
    _ADAPTERS[source_id] = adapter_cls
    return adapter_cls


def get_adapter(source_id: str) -> SourceAdapter:
    """Return an adapter instance for the given source.

    Raises ``KeyError`` if the source is not registered.
    """
    if source_id not in _ADAPTERS:
        raise KeyError(
            f"Unknown source {source_id!r}. "
            f"Available: {sorted(_ADAPTERS.keys())}"
        )
    return _ADAPTERS[source_id]()


# ---------------------------------------------------------------------------
# Public query API — used as neptune_ais.sources.*
# ---------------------------------------------------------------------------


def catalog() -> list[SourceCapabilities]:
    """Return capabilities for all registered sources (archival + streaming)."""
    caps = [get_adapter(sid).capabilities for sid in sorted(_ADAPTERS)]
    caps.extend(
        _STREAMING_CAPABILITIES[sid]
        for sid in sorted(_STREAMING_CAPABILITIES)
        if sid not in _ADAPTERS
    )
    return caps


def available(date_str: str) -> list[str]:
    """Return source IDs that may have data for the given date.

    This is a best-effort check based on adapter metadata — it does
    not guarantee the source actually has data for that date.
    """
    target = date.fromisoformat(date_str)
    result: list[str] = []
    for sid in sorted(_ADAPTERS):
        adapter = get_adapter(sid)
        dates = adapter.available_dates()
        if dates is None:
            # Source doesn't enumerate dates — include it optimistically.
            result.append(sid)
        elif isinstance(dates, tuple):
            start, end = dates
            if start <= target <= end:
                result.append(sid)
        elif isinstance(dates, list):
            if target in dates:
                result.append(sid)
    return result


def info(source_id: str) -> SourceCapabilities:
    """Return capabilities for a specific source (archival or streaming).

    Archival adapters take priority over streaming-only registrations
    (consistent with ``catalog()``).

    Raises ``KeyError`` if the source is not registered in either registry.
    """
    if source_id in _ADAPTERS:
        return get_adapter(source_id).capabilities
    if source_id in _STREAMING_CAPABILITIES:
        return _STREAMING_CAPABILITIES[source_id]
    raise KeyError(
        f"Unknown source {source_id!r}. "
        f"Available: {registered_sources()}"
    )


def capabilities(source_id: str) -> SourceCapabilities:
    """Alias for ``info()``."""
    return info(source_id)


def compare(*source_ids: str) -> list[dict[str, str]]:
    """Return side-by-side capability summaries for the given sources.

    If no source_ids are given, compares all registered sources
    (archival + streaming). Returns a list of dicts suitable for
    table display.
    """
    ids = list(source_ids) if source_ids else registered_sources()
    return [info(sid).summary() for sid in ids]


def registered_sources() -> list[str]:
    """Return sorted list of all registered source IDs (archival + streaming)."""
    return sorted(set(_ADAPTERS.keys()) | set(_STREAMING_CAPABILITIES.keys()))
