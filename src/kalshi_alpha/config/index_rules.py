"""Load index ladder rule semantics from the markdown rulebook."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RULEBOOK_PATH = ROOT / "docs" / "rules" / "index_rules.md"


@dataclass(frozen=True)
class IndexRule:
    """Structured view of a Kalshi index ladder rule entry."""

    series: str
    display_name: str
    evaluation_time_et: str
    evaluation_clause: str
    timing_clause: str
    fallback_clause: str
    reference_source: str
    primary_window_et: str
    tick_size_usd: float
    position_limit_usd: int


@dataclass(frozen=True)
class IndexRuleBook:
    """Container for all series rule entries plus shared metadata."""

    updated: str
    tick_size_usd: float
    position_limit_usd: int
    series: Mapping[str, IndexRule]


def load_index_rulebook(path: Path | None = None) -> IndexRuleBook:
    """Parse the markdown rule summary and return a structured rulebook."""

    resolved = (path or DEFAULT_RULEBOOK_PATH).resolve()
    return _load_index_rulebook_cached(str(resolved))


@lru_cache(maxsize=4)
def _load_index_rulebook_cached(resolved_path: str) -> IndexRuleBook:
    path = Path(resolved_path)
    if not path.exists():
        raise FileNotFoundError(f"Index rulebook not found at {resolved_path}")
    front_matter = _extract_front_matter(path)
    tick_size = float(front_matter.get("tick_size_usd", 0.01))
    position_limit = int(front_matter.get("position_limit_usd", 0))
    updated = str(front_matter.get("updated", ""))
    series_entries: dict[str, IndexRule] = {}
    series_section = front_matter.get("series", {})
    if not isinstance(series_section, Mapping):
        raise ValueError("index rulebook front matter must include a 'series' mapping")

    for key, raw_entry in series_section.items():
        if not isinstance(raw_entry, Mapping):
            continue
        series_code = str(key).upper()
        entry = IndexRule(
            series=series_code,
            display_name=str(raw_entry.get("display_name", series_code)),
            evaluation_time_et=str(raw_entry.get("evaluation_time_et", "")),
            evaluation_clause=str(raw_entry.get("evaluation_clause", "")),
            timing_clause=str(raw_entry.get("timing_clause", "")),
            fallback_clause=str(raw_entry.get("fallback_clause", "")),
            reference_source=str(raw_entry.get("reference_source", "")),
            primary_window_et=str(raw_entry.get("primary_window_et", "")),
            tick_size_usd=tick_size,
            position_limit_usd=position_limit,
        )
        series_entries[series_code] = entry

    if not series_entries:
        raise ValueError("index rulebook contained no series entries")

    return IndexRuleBook(
        updated=updated,
        tick_size_usd=tick_size,
        position_limit_usd=position_limit,
        series=series_entries,
    )


def lookup_index_rule(series: str, *, path: Path | None = None) -> IndexRule:
    """Return the rule metadata for a specific Kalshi series."""

    rulebook = load_index_rulebook(path)
    try:
        return rulebook.series[series.upper()]
    except KeyError as exc:
        supported = ", ".join(sorted(rulebook.series))
        raise KeyError(f"Unknown index series '{series}'. Supported: {supported}") from exc


def _extract_front_matter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError("index rulebook must start with YAML front matter")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("index rulebook front matter is malformed")
    front_matter = parts[1]
    try:
        data = yaml.safe_load(front_matter) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ValueError("failed to parse index rulebook front matter") from exc
    if not isinstance(data, Mapping):
        raise ValueError("index rulebook front matter must resolve to a mapping")
    return dict(data)


__all__ = ["IndexRule", "IndexRuleBook", "load_index_rulebook", "lookup_index_rule"]

