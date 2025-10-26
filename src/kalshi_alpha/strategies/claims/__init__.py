"""Initial jobless claims strategy stub using SARIMA-style heuristics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import sqrt
from statistics import mean as stat_mean
from statistics import pstdev
from typing import Any

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.strategies import base


@dataclass(frozen=True)
class ClaimsInputs:
    history: Sequence[int] | None = None
    holiday_next: bool = False
    freeze_active: bool | None = None
    latest_initial_claims: int | None = None
    four_week_avg: int | None = None


def freeze_window(active_at: datetime | None = None) -> bool:
    """Return True once the Wednesday data freeze has begun (UTC)."""
    active_at = active_at or datetime.now(tz=UTC)
    return active_at.weekday() >= 2  # Wednesday (2) through Friday


def pmf(strikes: Sequence[float], inputs: ClaimsInputs | None = None) -> list[LadderBinProbability]:
    """Return a distribution over initial claims ladder strikes."""
    inputs = inputs or ClaimsInputs()
    if inputs.history and len(inputs.history) >= 2:
        mean = _damped_trend_forecast(inputs.history, holiday_adjust=inputs.holiday_next)
        std = _claims_std(inputs.history)
        freeze_flag = inputs.freeze_active if inputs.freeze_active is not None else freeze_window()
        if freeze_flag:
            std = max(std * 0.5, 3_000.0)
    else:
        mean = _mean_claims(inputs)
        std = max(mean * 0.05, 5_000.0)
    calibration = _load_calibration()
    if calibration:
        if inputs.holiday_next:
            mean += calibration.get("holiday_lift", 0.0)
        std = max(std, calibration.get("std", std))
    return base.pmf_from_gaussian(strikes, mean=mean, std=std)


def _damped_trend_forecast(history: Sequence[int], *, holiday_adjust: bool) -> float:
    recent = list(history)
    last = float(recent[-1])
    prev = float(recent[-2])
    trend = last - prev
    trailing = recent[-4:]
    level = 0.5 * last + 0.3 * (sum(trailing) / len(trailing)) + 0.2 * prev
    forecast = level + 0.6 * trend
    if holiday_adjust:
        forecast += 3_500.0
    return forecast


def _claims_std(history: Sequence[int]) -> float:
    window = list(history)[-6:]
    mean = sum(window) / len(window)
    variance = sum((value - mean) ** 2 for value in window) / len(window)
    return max(sqrt(variance), 4_000.0)


def _mean_claims(inputs: ClaimsInputs) -> float:
    if inputs.four_week_avg:
        return float(inputs.four_week_avg)
    if inputs.latest_initial_claims:
        return float(inputs.latest_initial_claims)
    return 230_000.0


CALIBRATION_DIR = PROC_ROOT / "calibration"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_PATH = CALIBRATION_DIR / "claims.json"


def calibrate(
    history: Sequence[Mapping[str, Any]],
    window: int = 12,
) -> dict[str, float]:
    if len(history) < 2:
        raise ValueError("claims history requires at least two points")
    tail = list(history)[-window:]
    diffs: list[float] = []
    holiday_diffs: list[float] = []
    for prev, current in zip(tail, tail[1:], strict=False):
        diff = float(current["claims"]) - float(prev["claims"])
        diffs.append(diff)
        if bool(current.get("holiday")):
            holiday_diffs.append(diff)
    holiday_lift = stat_mean(holiday_diffs) if holiday_diffs else 0.0
    std = max(pstdev(diffs) if len(diffs) > 1 else 3_000.0, 3_000.0)
    payload = {"holiday_lift": holiday_lift, "std": std}
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _load_calibration() -> dict[str, float] | None:
    if not CALIBRATION_PATH.exists():
        return None
    try:
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    result: dict[str, float] = {}
    if "holiday_lift" in data:
        result["holiday_lift"] = float(data["holiday_lift"])
    if "std" in data:
        result["std"] = float(data["std"])
    return result or None
