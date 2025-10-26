"""10-year Treasury yield strategy stub."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.strategies import base


@dataclass(frozen=True)
class TenYInputs:
    latest_yield: float | None = None
    trailing_vol: float | None = None
    prior_close: float | None = None
    macro_shock: float = 0.0
    trailing_history: Sequence[float] | None = None


def pmf(strikes: Sequence[float], inputs: TenYInputs | None = None) -> list[LadderBinProbability]:
    inputs = inputs or TenYInputs()
    if inputs.prior_close is not None:
        mean = inputs.prior_close + 1.4 * inputs.macro_shock
        history = list(inputs.trailing_history or [])
        history.append(inputs.prior_close)
        spread = max(_sample_std(history[-6:]), 0.05)
        spread = min(spread, 0.2)
        distribution = {
            round(mean - spread, 3): 0.25,
            round(mean, 3): 0.5,
            round(mean + spread, 3): 0.25,
        }
        return base.grid_distribution_to_pmf(distribution)

    mean = inputs.latest_yield if inputs.latest_yield is not None else 4.50
    std = inputs.trailing_vol if inputs.trailing_vol is not None else 0.35
    std = max(std, 0.05)
    return base.pmf_from_gaussian(strikes, mean=mean, std=std)


def _sample_std(values: Sequence[float]) -> float:
    if not values:
        return 0.10
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return sqrt(variance)
