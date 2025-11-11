"""Load conservative fill probability curves derived from TOB snapshots."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from kalshi_alpha.datastore.paths import PROC_ROOT

FILL_CURVE_PATH = PROC_ROOT / "fill" / "index_fill_curve.json"


@lru_cache(maxsize=4)
def _load_payload(path: str) -> Mapping[str, object]:
    resolved = Path(path)
    if not resolved.exists():
        return {}
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupted file
        return {}


def probability(series: str, *, seconds_to_event: float | None = None, path: Path | None = None) -> float | None:
    payload = _load_payload(str(path or FILL_CURVE_PATH))
    series_block = payload.get("series") if isinstance(payload, Mapping) else None
    if not isinstance(series_block, Mapping):
        return None
    entry = series_block.get(series.upper())
    if not isinstance(entry, Mapping):
        return None
    late_threshold = float(entry.get("late_threshold_seconds") or 0.0)
    late_prob = entry.get("late_probability")
    default_prob = entry.get("default_probability")
    candidate: float | None = None
    if seconds_to_event is not None and late_prob is not None and seconds_to_event <= late_threshold:
        candidate = float(late_prob)
    elif default_prob is not None:
        candidate = float(default_prob)
    if candidate is None or candidate <= 0.0:
        return None
    return min(1.0, max(0.1, candidate))


def adjust_alpha(series: str, base_alpha: float, *, seconds_to_event: float | None = None, path: Path | None = None) -> float:
    clamp = probability(series, seconds_to_event=seconds_to_event, path=path)
    if clamp is None:
        return base_alpha
    return min(base_alpha, clamp)


__all__ = ["probability", "adjust_alpha", "FILL_CURVE_PATH"]
