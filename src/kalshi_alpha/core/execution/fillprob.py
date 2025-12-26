"""Load conservative fill probability curves derived from TOB snapshots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from kalshi_alpha.datastore.paths import PROC_ROOT

FILL_CURVE_DIR = PROC_ROOT / "fillcalib"
FILL_CURVE_PATH = FILL_CURVE_DIR / "curves_latest.json"
LEGACY_FILL_CURVE_PATH = PROC_ROOT / "fill" / "index_fill_curve.json"
_CURVE_GLOB = "curves_*.json"


@dataclass(frozen=True)
class FillCurveStatus:
    path: Path | None
    uncalibrated: bool
    reason: str | None
    asof_date: str | None = None
    generated_at: str | None = None
    version: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path) if self.path is not None else None,
            "uncalibrated": self.uncalibrated,
            "reason": self.reason,
            "asof_date": self.asof_date,
            "generated_at": self.generated_at,
            "version": self.version,
        }


def resolve_curve_path(path: Path | None = None) -> Path | None:
    if path is not None:
        return Path(path)
    if FILL_CURVE_PATH.exists():
        return FILL_CURVE_PATH
    latest = _latest_curve_path(FILL_CURVE_DIR)
    if latest is not None:
        return latest
    if LEGACY_FILL_CURVE_PATH.exists():
        return LEGACY_FILL_CURVE_PATH
    return None


@lru_cache(maxsize=4)
def _load_payload(path_str: str) -> tuple[Mapping[str, object] | None, str | None]:
    resolved = Path(path_str)
    if not resolved.exists():
        return None, "missing"
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupted file
        return None, "invalid"
    if not isinstance(payload, Mapping):
        return None, "invalid"
    return payload, None


def curve_status(path: Path | None = None) -> FillCurveStatus:
    resolved = resolve_curve_path(path)
    if resolved is None:
        return FillCurveStatus(path=None, uncalibrated=True, reason="fill_curve_missing")
    payload, error = _load_payload(str(resolved))
    if payload is None:
        reason = "fill_curve_missing" if error == "missing" else "fill_curve_invalid"
        return FillCurveStatus(path=resolved, uncalibrated=True, reason=reason)
    return FillCurveStatus(
        path=resolved,
        uncalibrated=False,
        reason=None,
        asof_date=_safe_str(payload.get("asof_date")),
        generated_at=_safe_str(payload.get("generated_at")),
        version=_safe_int(payload.get("version")),
    )


def probability(
    series: str,
    *,
    seconds_to_event: float | None = None,
    window_id: str | None = None,
    side: str | None = None,
    quote_distance_to_touch_bin: str | None = None,
    time_to_expiry_bin: str | None = None,
    path: Path | None = None,
) -> float:
    resolved = resolve_curve_path(path)
    if resolved is None:
        return 0.0
    payload, error = _load_payload(str(resolved))
    if payload is None or error is not None:
        return 0.0
    series_block = payload.get("series") if isinstance(payload, Mapping) else None
    if not isinstance(series_block, Mapping):
        return 0.0
    entry = series_block.get(series.strip().upper())
    if not isinstance(entry, Mapping):
        return 0.0

    if all(
        value is not None
        for value in (window_id, side, quote_distance_to_touch_bin, time_to_expiry_bin)
    ):
        buckets = entry.get("buckets")
        if isinstance(buckets, list):
            target_side = str(side).upper()
            for bucket in buckets:
                if not isinstance(bucket, Mapping):
                    continue
                if str(bucket.get("window_id")) != str(window_id):
                    continue
                if str(bucket.get("side")).upper() != target_side:
                    continue
                if str(bucket.get("quote_distance_to_touch_bin")) != str(quote_distance_to_touch_bin):
                    continue
                if str(bucket.get("time_to_expiry_bin")) != str(time_to_expiry_bin):
                    continue
                return _clamp_probability(bucket.get("p_fill"))
        return 0.0

    late_threshold = _safe_float(entry.get("late_threshold_seconds")) or 0.0
    late_prob = entry.get("late_probability")
    default_prob = entry.get("default_probability")
    candidate: float | None = None
    if seconds_to_event is not None and late_prob is not None and seconds_to_event <= late_threshold:
        candidate = _safe_float(late_prob)
    elif default_prob is not None:
        candidate = _safe_float(default_prob)
    if candidate is None:
        return 0.0
    return _clamp_probability(candidate)


def adjust_alpha(
    series: str,
    base_alpha: float,
    *,
    seconds_to_event: float | None = None,
    window_id: str | None = None,
    side: str | None = None,
    quote_distance_to_touch_bin: str | None = None,
    time_to_expiry_bin: str | None = None,
    path: Path | None = None,
) -> float:
    base_alpha = max(0.0, min(1.0, float(base_alpha)))
    clamp = probability(
        series,
        seconds_to_event=seconds_to_event,
        window_id=window_id,
        side=side,
        quote_distance_to_touch_bin=quote_distance_to_touch_bin,
        time_to_expiry_bin=time_to_expiry_bin,
        path=path,
    )
    return min(base_alpha, clamp)


def _latest_curve_path(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates: list[tuple[date, Path]] = []
    fallbacks: list[tuple[float, Path]] = []
    for path in root.glob(_CURVE_GLOB):
        if not path.is_file():
            continue
        asof = _parse_asof_date(path.name)
        if asof is not None:
            candidates.append((asof, path))
        else:
            try:
                fallbacks.append((path.stat().st_mtime, path))
            except OSError:
                continue
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    if fallbacks:
        return max(fallbacks, key=lambda item: item[0])[1]
    return None


def _parse_asof_date(name: str) -> date | None:
    if not (name.startswith("curves_") and name.endswith(".json")):
        return None
    token = name[len("curves_") : -len(".json")]
    try:
        return date.fromisoformat(token)
    except ValueError:
        try:
            return datetime.strptime(token, "%Y%m%d").date()
        except ValueError:
            return None


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _clamp_probability(value: object) -> float:
    parsed = _safe_float(value)
    if parsed is None:
        return 0.0
    return max(0.0, min(1.0, parsed))


__all__ = [
    "FILL_CURVE_DIR",
    "FILL_CURVE_PATH",
    "LEGACY_FILL_CURVE_PATH",
    "FillCurveStatus",
    "resolve_curve_path",
    "curve_status",
    "probability",
    "adjust_alpha",
]
