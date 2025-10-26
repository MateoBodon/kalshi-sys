"""Initial jobless claims strategy stub using SARIMA-style heuristics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import sqrt

from kalshi_alpha.core.pricing import LadderBinProbability
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
