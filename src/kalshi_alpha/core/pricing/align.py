"""Utilities for aligning model PMFs to ladder strike grids."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from kalshi_alpha.core.pricing import LadderBinProbability, project_simplex


@dataclass(slots=True)
class SkipScan(Exception):
    """Indicates the scan should be skipped and surfaced as a NO-GO."""

    reason: str

    def __str__(self) -> str:
        return self.reason


def cdf_from_pmf(
    grid_breaks: Sequence[float],
    pmf: Sequence[LadderBinProbability],
) -> Callable[[float], float]:
    """Build a piecewise-linear CDF from arbitrary ladder bin probabilities."""

    if not pmf:
        raise SkipScan("degenerate ladder/strike set")

    xs = sorted({float(break_point) for break_point in grid_breaks})
    if not xs:
        boundaries: set[float] = set()
        for bin_prob in pmf:
            if bin_prob.lower is not None and math.isfinite(bin_prob.lower):
                boundaries.add(float(bin_prob.lower))
            if bin_prob.upper is not None and math.isfinite(bin_prob.upper):
                boundaries.add(float(bin_prob.upper))
        if not boundaries:
            raise SkipScan("degenerate ladder/strike set")
        xs = sorted(boundaries)

    values = [_cdf_at(x, pmf) for x in xs]
    values = [_clamp(value) for value in values]

    def cdf_fn(point: float) -> float:
        x = float(point)
        if x <= xs[0]:
            return 0.0 if x < xs[0] else values[0]
        if x >= xs[-1]:
            return 1.0 if x > xs[-1] else values[-1]
        index = bisect_right(xs, x)
        left_x = xs[index - 1]
        right_x = xs[index]
        left_y = values[index - 1]
        right_y = values[index]
        if math.isclose(right_x, left_x):
            return _clamp(right_y)
        weight = (x - left_x) / (right_x - left_x)
        interpolated = left_y + weight * (right_y - left_y)
        return _clamp(interpolated)

    return cdf_fn


def pmf_on_strikes(
    cdf_fn: Callable[[float], float],
    strikes: Sequence[float],
) -> list[float]:
    """Project a CDF onto ladder strikes, returning simplex-projected probabilities."""

    ordered_strikes = sorted({float(strike) for strike in strikes})
    if len(ordered_strikes) <= 1:
        raise SkipScan("degenerate ladder/strike set")

    cumulative = [cdf_fn(strike) for strike in ordered_strikes]
    cumulative = [_clamp(value) for value in cumulative]

    masses: list[float] = []
    previous = 0.0
    for value in cumulative:
        masses.append(max(value - previous, 0.0))
        previous = value
    masses.append(max(1.0 - previous, 0.0))

    if sum(masses) <= 1e-12:
        raise SkipScan("degenerate ladder/strike set")

    projected = project_simplex(masses)
    if sum(projected) <= 0.0:
        raise SkipScan("degenerate ladder/strike set")
    return projected


def align_pmf_to_strikes(
    pmf: Sequence[LadderBinProbability],
    strikes: Sequence[float],
) -> list[LadderBinProbability]:
    """Align an arbitrary PMF to the ladder grid defined by strikes."""

    if not pmf:
        raise SkipScan("degenerate ladder/strike set")
    ordered_strikes = sorted({float(strike) for strike in strikes})
    if len(ordered_strikes) <= 1:
        raise SkipScan("degenerate ladder/strike set")

    grid_points = list(ordered_strikes)
    for bin_prob in pmf:
        if bin_prob.lower is not None and math.isfinite(bin_prob.lower):
            grid_points.append(float(bin_prob.lower))
        if bin_prob.upper is not None and math.isfinite(bin_prob.upper):
            grid_points.append(float(bin_prob.upper))

    cdf_fn = cdf_from_pmf(grid_points, pmf)
    aligned_probs = pmf_on_strikes(cdf_fn, ordered_strikes)

    bins = _ladder_bins(ordered_strikes)
    return [
        LadderBinProbability(lower=lower, upper=upper, probability=probability)
        for (lower, upper), probability in zip(bins, aligned_probs, strict=True)
    ]


def _ladder_bins(strikes: Sequence[float]) -> list[tuple[float | None, float | None]]:
    ordered = list(strikes)
    bins: list[tuple[float | None, float | None]] = []
    previous: float | None = None
    for strike in ordered:
        bins.append((previous, strike))
        previous = strike
    bins.append((previous, None))
    return bins


def _cdf_at(x: float, pmf: Sequence[LadderBinProbability]) -> float:
    total = 0.0
    for bin_prob in pmf:
        probability = max(float(bin_prob.probability), 0.0)
        if probability <= 0.0:
            continue
        lower = -math.inf if bin_prob.lower is None else float(bin_prob.lower)
        upper = math.inf if bin_prob.upper is None else float(bin_prob.upper)
        if x < lower:
            continue
        if x >= upper or math.isclose(x, upper):
            total += probability
            continue
        if math.isinf(upper):
            total += probability
            continue
        if upper <= lower:
            if x >= lower:
                total += probability
            continue
        span = upper - lower
        fraction = max(min((x - lower) / span, 1.0), 0.0)
        total += probability * fraction
    return _clamp(total)


def _clamp(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
