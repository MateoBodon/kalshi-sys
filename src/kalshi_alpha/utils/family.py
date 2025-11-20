"""Family helpers for focusing execution on index ladders."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence

INDEX_SERIES: tuple[str, ...] = ("INX", "INXU", "NASDAQ100", "NASDAQ100U")
DEFAULT_FAMILY = "index"


def resolve_family(value: str | None = None) -> str:
    """Resolve the requested family from an explicit value or the FAMILY env var.

    Normalizes common aliases so ``index``, ``indices``, ``spx``, and ``ndx`` all
    map to the canonical ``index`` family. ``all`` passes everything through.
    """

    candidate = value or os.getenv("FAMILY") or DEFAULT_FAMILY
    normalized = candidate.strip().lower()
    if normalized in {"index", "indices", "index-only", "spx", "ndx"}:
        return "index"
    if normalized in {"macro", "macros"}:
        return "macro"
    if normalized in {"all", "*", "any"}:
        return "all"
    return normalized


def filter_index_series(series: Sequence[str] | Iterable[str]) -> list[str]:
    """Return only index ladder tickers from *series*."""

    return [item.upper() for item in series if str(item).upper() in INDEX_SERIES]


def is_index_family(value: str | None) -> bool:
    """Shortcut to test whether *value* resolves to the index family."""

    return resolve_family(value) == "index"


__all__ = ["INDEX_SERIES", "DEFAULT_FAMILY", "filter_index_series", "is_index_family", "resolve_family"]
