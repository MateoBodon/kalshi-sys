"""Execution defaults for index ladder maker behaviour (alpha & slippage)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
DEFAULTS_PATH = ROOT / "data" / "reference" / "index_execution_defaults.json"


def _resolve_path(path: Path | None = None) -> Path:
    resolved = (path or DEFAULTS_PATH).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Execution defaults not found: {resolved}")
    return resolved


@lru_cache(maxsize=1)
def _load_defaults(resolved_path: str) -> dict[str, Any]:
    payload = json.loads(Path(resolved_path).read_text(encoding="utf-8"))
    data = payload.get("series")
    if not isinstance(data, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        normalized[key.upper()] = entry
    return normalized


def default_alpha(series: str, *, path: Path | None = None) -> float | None:
    series_key = series.strip().upper()
    defaults = _load_defaults(str(_resolve_path(path)))
    entry = defaults.get(series_key)
    if not entry:
        return None
    value = entry.get("alpha")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def slippage_config(series: str, *, path: Path | None = None) -> dict[str, Any] | None:
    series_key = series.strip().upper()
    defaults = _load_defaults(str(_resolve_path(path)))
    entry = defaults.get(series_key)
    if not entry:
        return None
    config = entry.get("slippage")
    return config if isinstance(config, dict) else None


__all__ = ["default_alpha", "slippage_config"]
