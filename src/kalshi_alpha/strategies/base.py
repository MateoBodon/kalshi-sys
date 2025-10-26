"""Shared helpers for strategy distributions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import erf, sqrt

from kalshi_alpha.core.pricing import LadderBinProbability


def ladder_bins(strikes: Sequence[float]) -> list[tuple[float | None, float | None]]:
    ordered = sorted(strikes)
    bins: list[tuple[float | None, float | None]] = []
    previous: float | None = None
    for strike in ordered:
        bins.append((previous, strike))
        previous = strike
    bins.append((previous, None))
    return bins


def normalize(weights: Iterable[float]) -> list[float]:
    weights_list = list(weights)
    total = sum(weights_list)
    if total <= 0:
        raise ValueError("weights must have positive sum")
    return [weight / total for weight in weights_list]


def gaussian_weight(mean: float, std: float, lower: float | None, upper: float | None) -> float:
    if std <= 0:
        raise ValueError("std must be positive")
    lower_bound = float("-inf") if lower is None else lower
    upper_bound = float("inf") if upper is None else upper
    return _gaussian_cdf(upper_bound, mean, std) - _gaussian_cdf(lower_bound, mean, std)


def _gaussian_cdf(x: float, mean: float, std: float) -> float:
    return 0.5 * (1.0 + erf((x - mean) / (std * sqrt(2.0))))


def pmf_from_gaussian(
    strikes: Sequence[float],
    mean: float,
    std: float,
) -> list[LadderBinProbability]:
    bins = ladder_bins(strikes)
    weights = [gaussian_weight(mean, std, lower, upper) for lower, upper in bins]
    normalized = normalize(weights)
    return [
        LadderBinProbability(lower=lower, upper=upper, probability=prob)
        for (lower, upper), prob in zip(bins, normalized, strict=True)
    ]


def grid_distribution_to_pmf(distribution: Mapping[float, float]) -> list[LadderBinProbability]:
    if not distribution:
        return []
    sorted_points = sorted(distribution.items())
    total = sum(weight for _, weight in sorted_points)
    if total <= 0:
        raise ValueError("distribution weights must sum to a positive value")

    normalized = [(point, weight / total) for point, weight in sorted_points]
    pmf: list[LadderBinProbability] = []
    for index, (point, weight) in enumerate(normalized):
        lower = None
        if index > 0:
            previous_point = normalized[index - 1][0]
            lower = (previous_point + point) / 2
        upper = None
        if index < len(normalized) - 1:
            next_point = normalized[index + 1][0]
            upper = (point + next_point) / 2
        pmf.append(LadderBinProbability(lower=lower, upper=upper, probability=weight))
    return pmf
