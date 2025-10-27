"""Ladder mispricing analytics and spread detection."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from kalshi_alpha.core.pricing import LadderBinProbability


@dataclass(frozen=True)
class KinkMetrics:
    max_kink: float
    mean_abs_kink: float
    monotonicity_penalty: float
    kink_count: int


def implied_cdf_kinks(survival: Sequence[float]) -> KinkMetrics:
    """Quantify curvature and monotonicity violations for an implied survival curve."""
    if len(survival) < 3:
        return KinkMetrics(max_kink=0.0, mean_abs_kink=0.0, monotonicity_penalty=0.0, kink_count=0)

    curvature: list[float] = []
    for idx in range(1, len(survival) - 1):
        prev_val = float(survival[idx - 1])
        current = float(survival[idx])
        next_val = float(survival[idx + 1])
        curvature.append(prev_val - 2.0 * current + next_val)

    max_kink = max(curvature, default=0.0)
    mean_abs = statistics.mean(abs(value) for value in curvature) if curvature else 0.0
    monotonicity_penalty = sum(max(0.0, float(survival[idx]) - float(survival[idx - 1])) for idx in range(1, len(survival)))
    kink_count = sum(1 for value in curvature if value > 0)
    return KinkMetrics(
        max_kink=max_kink,
        mean_abs_kink=mean_abs,
        monotonicity_penalty=monotonicity_penalty,
        kink_count=kink_count,
    )


def prob_sum_gap(pmf: Sequence[object]) -> float:
    """Return the absolute probability mass deficit/excess for a PMF."""
    if not pmf:
        return 1.0
    first = pmf[0]
    if hasattr(first, "probability"):
        total = sum(float(getattr(bin_prob, "probability", 0.0)) for bin_prob in pmf)
    else:
        total = sum(float(value) for value in pmf)  # type: ignore[arg-type]
    return abs(float(total) - 1.0)


def kink_spreads(
    pmf: Sequence["LadderBinProbability"],
    market_pmf: Sequence["LadderBinProbability"],
    *,
    max_legs: int = 4,
    min_abs_delta: float = 1e-6,
) -> list[dict[str, object]]:
    """Identify adjacent-bin spread candidates based on probability deltas."""
    max_legs = max(2, int(max_legs))
    length = min(len(pmf), len(market_pmf))
    if length < 2:
        return []

    deltas = [
        float(pmf[idx].probability) - float(market_pmf[idx].probability) for idx in range(length)
    ]
    candidates: list[dict[str, object]] = []

    for start in range(length - 1):
        for legs in range(2, min(max_legs, length - start) + 1):
            segment = deltas[start : start + legs]
            significant = [value for value in segment if abs(value) > min_abs_delta]
            if len(significant) < 2:
                continue
            has_pos = any(value > 0 for value in significant)
            has_neg = any(value < 0 for value in significant)
            if not (has_pos and has_neg):
                continue
            abs_sum = sum(abs(value) for value in segment)
            if abs_sum <= min_abs_delta:
                continue
            net = sum(segment)
            candidate = {
                "start_index": start,
                "end_index": start + legs - 1,
                "legs": legs,
                "direction": "model_over" if net > 0 else "model_under",
                "delta_sum": net,
                "abs_delta_sum": abs_sum,
                "bins": [
                    {
                        "index": start + offset,
                        "lower": pmf[start + offset].lower,
                        "upper": pmf[start + offset].upper,
                        "model_prob": pmf[start + offset].probability,
                        "market_prob": market_pmf[start + offset].probability,
                        "delta": segment[offset],
                    }
                    for offset in range(legs)
                ],
            }
            candidates.append(candidate)

    candidates.sort(key=lambda item: item["abs_delta_sum"], reverse=True)
    return candidates
