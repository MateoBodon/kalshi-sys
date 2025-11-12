from __future__ import annotations

from typing import Sequence, Tuple

INDEX_CANONICAL_SERIES: tuple[str, ...] = ("INXU", "NASDAQ100U", "INX", "NASDAQ100")
_INDEX_ALIAS_MAP: dict[str, str] = {
    "INX": "INX",
    "INXU": "INXU",
    "NASDAQ100": "NASDAQ100",
    "NASDAQ100U": "NASDAQ100U",
    "SPX": "INX",
    "SPXU": "INXU",
    "NDX": "NASDAQ100",
    "NDXU": "NASDAQ100U",
    "I:SPX": "INX",
    "I:SPXU": "INXU",
    "I:NDX": "NASDAQ100",
    "I:NDXU": "NASDAQ100U",
}


def _clean(label: str | None) -> str:
    return (label or "").strip().upper()


def normalize_index_series(label: str | None) -> str:
    """Return canonical INX/NDX series for the provided alias."""

    normalized = _clean(label)
    if normalized.startswith("KX") and len(normalized) > 2:
        normalized = normalized[2:]
    return _INDEX_ALIAS_MAP.get(normalized, normalized)


def normalize_index_series_list(series: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize a sequence of index series and drop duplicates while preserving order."""

    ordered: list[str] = []
    seen: set[str] = set()
    inputs = series or INDEX_CANONICAL_SERIES
    for entry in inputs:
        canonical = normalize_index_series(entry)
        if not canonical or canonical in seen:
            continue
        ordered.append(canonical)
        seen.add(canonical)
    return tuple(ordered)


def index_series_query_candidates(label: str | None) -> tuple[str, ...]:
    """Return possible API query candidates for a series alias."""

    normalized_input = _clean(label)
    canonical = normalize_index_series(label)
    candidates: list[str] = []
    if normalized_input:
        candidates.append(normalized_input)
    if normalized_input.startswith("KX") and len(normalized_input) > 2:
        stripped = normalized_input[2:]
        if stripped not in candidates:
            candidates.append(stripped)
    if canonical and canonical not in candidates:
        candidates.append(canonical)
    if canonical:
        prefixed = f"KX{canonical}"
        if prefixed not in candidates:
            candidates.append(prefixed)
    deduped: dict[str, None] = {}
    for candidate in candidates:
        if candidate:
            deduped.setdefault(candidate, None)
    return tuple(deduped.keys())
