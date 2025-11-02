"""Utilities for evaluating pre-event freeze windows per series family."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.exec.pipelines.calendar import resolve_run_window

ET = ZoneInfo("America/New_York")

SERIES_MODE_MAP: dict[str, str] = {
    "CPI": "pre_cpi",
    "CLAIMS": "pre_claims",
    "TENY": "teny_close",
    "WEATHER": "weather_cycle",
}


@dataclass(slots=True)
class FreezeEvaluation:
    series: str
    mode: str | None
    freeze_start: datetime | None
    freeze_active: bool
    scan_open: datetime | None
    scan_close: datetime | None
    reference: datetime | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "series": self.series,
            "mode": self.mode,
            "freeze_start": _serialize_dt(self.freeze_start),
            "freeze_active": self.freeze_active,
            "scan_open": _serialize_dt(self.scan_open),
            "scan_close": _serialize_dt(self.scan_close),
            "reference": _serialize_dt(self.reference),
            "notes": list(self.notes),
        }


def evaluate_freeze_for_series(
    series: str,
    *,
    now: datetime,
    proc_root: str | None = None,
) -> FreezeEvaluation:
    series_upper = series.upper()
    mode = _resolve_mode(series_upper)
    if mode is None:
        return FreezeEvaluation(series_upper, None, None, False, None, None, None, ["no_mode"])

    target_date = now.astimezone(ET).date()
    root = PROC_ROOT if proc_root is None else proc_root
    try:
        window = resolve_run_window(mode=mode, target_date=target_date, now=now, proc_root=root)
    except Exception as exc:  # pragma: no cover - defensive fallback
        return FreezeEvaluation(
            series_upper,
            mode,
            None,
            False,
            None,
            None,
            None,
            [f"error:{exc}"],
        )

    freeze_active = window.freeze_active(now)
    return FreezeEvaluation(
        series_upper,
        mode,
        window.freeze_start,
        freeze_active,
        window.scan_open,
        window.scan_close,
        window.reference,
        window.notes,
    )


def _resolve_mode(series: str) -> str | None:
    for prefix, mode in SERIES_MODE_MAP.items():
        if series.startswith(prefix):
            return mode
    return None


def _serialize_dt(value: datetime | None) -> dict[str, str] | None:
    if value is None:
        return None
    utc_value = value.astimezone(UTC)
    et_value = value.astimezone(ET)
    return {"utc": utc_value.isoformat(), "et": et_value.isoformat()}
