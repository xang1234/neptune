"""AIS destination field resolver.

Fuzzy-matches the free-text ``destination`` field from AIS Message
Type 5 to canonical ports in the WPI/UNLOCODE database.

The destination field is noisy — misspelled, abbreviated, sometimes
stale — but it provides a validation signal independent of position.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neptune_ais.ports._index import PortIndex
    from neptune_ais.ports._models import Port


def resolve_destination(
    destination: str,
    port_index: PortIndex,
    *,
    threshold: float = 0.6,
) -> Port | None:
    """Fuzzy-match an AIS destination string to a canonical port.

    Matching strategy (in priority order):

    1. **Exact UNLOCODE**: ``"NLRTM"`` → direct lookup.
    2. **UNLOCODE with space**: ``"NL RTM"`` → normalize and lookup.
    3. **Exact name match**: ``"ROTTERDAM"`` → case-insensitive lookup.
    4. **Token-overlap scoring**: ``"ROTT"`` or ``"PORT OF ROTTERDAM"``
       → Jaccard overlap on name tokens above ``threshold``.

    Args:
        destination: Raw AIS destination string.
        port_index: PortIndex to resolve against.
        threshold: Minimum token-overlap score (0.0–1.0) for
            fuzzy matches. Default 0.6.

    Returns:
        The best matching ``Port``, or None if no match found.
    """
    if not destination:
        return None

    clean = _normalize(destination)
    if not clean:
        return None

    # Strategy 1+2: Exact UNLOCODE (with or without space)
    if len(clean) <= 6:
        result = port_index.by_unlocode(clean)
        if result is not None:
            return result

    # Strategy 3: Exact name match against WPI ports and UNLOCODE
    result = _exact_name_match(clean, port_index)
    if result is not None:
        return result

    # Strategy 4: Name-contains match (destination is a substring of port name
    # or port name is a substring of destination)
    result = _substring_match(clean, port_index)
    if result is not None:
        return result

    # Strategy 5: Token-overlap scoring
    return _token_overlap_match(clean, port_index, threshold)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Characters to strip from destination strings
_STRIP_RE = re.compile(r"[^A-Z0-9 ]")


def _normalize(s: str) -> str:
    """Normalize a destination string for matching.

    Strips punctuation, collapses whitespace, uppercases.
    Returns empty string for garbage inputs (all symbols, arrows, etc.).
    """
    s = s.strip().upper()
    s = _STRIP_RE.sub("", s)
    s = " ".join(s.split())  # collapse whitespace
    return s


def _exact_name_match(clean: str, port_index: PortIndex) -> Port | None:
    """Try exact case-insensitive match against WPI + UNLOCODE names."""
    import polars as pl

    # WPI port names
    matches = port_index.ports.filter(
        pl.col("name").str.to_uppercase() == clean
    )
    if len(matches) > 0:
        from neptune_ais.ports._models import Port
        return Port(**matches.row(0, named=True))

    # WPI alternate names
    matches = port_index.ports.filter(
        pl.col("alternate_name").str.to_uppercase() == clean
    )
    if len(matches) > 0:
        from neptune_ais.ports._models import Port
        return Port(**matches.row(0, named=True))

    # UNLOCODE names → resolve to WPI port via UNLOCODE code
    unl_matches = port_index.unlocodes.filter(
        pl.col("name").str.to_uppercase() == clean
    )
    if len(unl_matches) > 0:
        code = unl_matches["code"][0]
        return port_index.by_unlocode(code)

    return None


def _substring_match(clean: str, port_index: PortIndex) -> Port | None:
    """Match where destination contains a port name or vice versa.

    Handles cases like 'PORT OF ROTTERDAM' (contains 'ROTTERDAM')
    or 'YOKOHAMA' (contained in 'YOKOHAMA KO').
    Prefers larger ports (by harbor_size) when multiple matches exist.
    """
    import polars as pl

    if len(clean) < 4:
        return None  # too short for substring matching

    name_upper = pl.col("name").str.to_uppercase()

    # Direction 1: destination is a substring of port name (fast, Polars-native)
    combined = port_index.ports.filter(
        name_upper.str.contains(clean, literal=True)
    )

    # Direction 2: port name is a substring of destination (Python fallback)
    # Only run if direction 1 found nothing — avoids O(n) Python loop per call.
    # Cannot use pl.lit(str).str.contains(Expr) — Polars broadcasting bug (#12632).
    if len(combined) == 0:
        port_names_upper = port_index.ports["name"].str.to_uppercase().to_list()
        reverse_mask = [pn in clean for pn in port_names_upper]
        combined = port_index.ports.filter(pl.Series(reverse_mask))
    if len(combined) == 0:
        return None

    # Prefer largest harbor
    size_rank = (
        pl.when(pl.col("harbor_size") == "L").then(0)
        .when(pl.col("harbor_size") == "M").then(1)
        .when(pl.col("harbor_size") == "S").then(2)
        .when(pl.col("harbor_size") == "V").then(3)
        .otherwise(4)
    )
    best = combined.with_columns(size_rank.alias("_rank")).sort("_rank").drop("_rank").head(1)
    from neptune_ais.ports._models import Port
    return Port(**best.row(0, named=True))


def _token_overlap_match(
    clean: str,
    port_index: PortIndex,
    threshold: float,
) -> Port | None:
    """Score destination against port names using token (Jaccard) overlap.

    Splits both the destination and each port name into token sets,
    computes Jaccard similarity, returns the best match above threshold.
    """
    dest_tokens = set(clean.split())
    if not dest_tokens:
        return None

    best_score = 0.0
    best_row: dict | None = None

    for row in port_index.ports.iter_rows(named=True):
        name_tokens = set(row["name"].upper().split())
        alt = row.get("alternate_name")
        if alt:
            name_tokens |= set(alt.upper().split())

        intersection = dest_tokens & name_tokens
        if not intersection:
            continue

        union = dest_tokens | name_tokens
        score = len(intersection) / len(union)
        if score > best_score:
            best_score = score
            best_row = row

    if best_score >= threshold and best_row is not None:
        from neptune_ais.ports._models import Port
        return Port(**best_row)

    return None


# ---------------------------------------------------------------------------
# Vectorized column resolver
# ---------------------------------------------------------------------------


def resolve_destination_column(
    destinations: pl.Series,
    port_index: PortIndex,
    *,
    threshold: float = 0.6,
) -> pl.DataFrame:
    """Resolve an entire column of AIS destination strings.

    Uses the "distinct → resolve → join" pattern: extracts unique
    destination strings (~1K–5K unique values in a million rows),
    resolves each once, then maps back to the original column.

    Args:
        destinations: A String Series of raw AIS destination values.
        port_index: PortIndex to resolve against.
        threshold: Token-overlap threshold for fuzzy matching.

    Returns:
        A DataFrame with columns: ``resolved_unlocode`` (String|None),
        ``resolved_port_name`` (String|None), ``match_confidence``
        (Float64: 1.0 for exact, 0.5 for substring/token match).
    """
    import polars as pl

    n = len(destinations)
    if n == 0:
        return pl.DataFrame({
            "resolved_unlocode": pl.Series([], dtype=pl.String),
            "resolved_port_name": pl.Series([], dtype=pl.String),
            "match_confidence": pl.Series([], dtype=pl.Float64),
        })

    # Extract unique non-null, non-empty destination strings
    non_null = destinations.drop_nulls()
    unique_dests = (
        non_null
        .filter(non_null.str.strip_chars().str.len_chars() > 0)
        .unique()
        .to_list()
    )

    # Resolve each unique string once
    lookup: dict[str, tuple[str | None, str | None, float]] = {}
    for dest in unique_dests:
        result = resolve_destination(dest, port_index, threshold=threshold)
        if result is not None:
            # Confidence: 1.0 for UNLOCODE/exact name, 0.5 for substring/token
            clean = _normalize(dest)
            is_exact = (
                (result.unlocode is not None and clean == result.unlocode)
                or clean == result.name.upper()
            )
            conf = 1.0 if is_exact else 0.5
            lookup[dest] = (result.unlocode, result.name, conf)
        else:
            lookup[dest] = (None, None, 0.0)

    # Map back to original column
    resolved_unlocode: list[str | None] = []
    resolved_port_name: list[str | None] = []
    match_confidence: list[float] = []

    for dest in destinations.to_list():
        if dest is None or dest not in lookup:
            resolved_unlocode.append(None)
            resolved_port_name.append(None)
            match_confidence.append(0.0)
        else:
            unl, name, conf = lookup[dest]
            resolved_unlocode.append(unl)
            resolved_port_name.append(name)
            match_confidence.append(conf)

    return pl.DataFrame({
        "resolved_unlocode": pl.Series(resolved_unlocode, dtype=pl.String),
        "resolved_port_name": pl.Series(resolved_port_name, dtype=pl.String),
        "match_confidence": pl.Series(match_confidence, dtype=pl.Float64),
    })
