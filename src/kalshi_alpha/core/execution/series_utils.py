"""Helpers for canonicalizing execution series names."""

from __future__ import annotations

_ALIASES: dict[str, str] = {
    "CPI": "CPI",
    "CLAIMS": "CLAIMS",
    "UNEMPLOYMENT": "CLAIMS",
    "TENY": "TENY",
    "10Y": "TENY",
    "T10Y": "TENY",
    "YIELDS": "TENY",
    "WEATHER": "WEATHER",
    "WX": "WEATHER",
}


def canonical_series_family(label: str | None) -> str:
    """Return the canonical family identifier for a series label."""

    if not label:
        return "UNKNOWN"
    normalized = label.strip().upper()
    return _ALIASES.get(normalized, normalized)
