from __future__ import annotations

import math
import random

import pytest

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.core.pricing.align import SkipScan, align_pmf_to_strikes, cdf_from_pmf, pmf_on_strikes
from kalshi_alpha.strategies import base


def _random_distribution(points: int) -> dict[float, float]:
    support_set = {round(random.uniform(-5, 5), 3) for _ in range(points)}
    while len(support_set) < points:
        support_set.add(round(random.uniform(-5, 5), 3))
    support = sorted(support_set)
    weights = [random.random() + 0.01 for _ in range(len(support))]
    total = sum(weights)
    return {point: weight / total for point, weight in zip(sorted(support), weights, strict=True)}


def _grid_breaks(strikes: list[float], pmf: list[LadderBinProbability]) -> list[float]:
    breaks = list(strikes)
    for bin_prob in pmf:
        if bin_prob.lower is not None and math.isfinite(bin_prob.lower):
            breaks.append(float(bin_prob.lower))
        if bin_prob.upper is not None and math.isfinite(bin_prob.upper):
            breaks.append(float(bin_prob.upper))
    return breaks


def test_pmf_alignment_simplex_properties() -> None:
    random.seed(1234)
    for strike_count in range(1, 21):
        strikes = sorted(random.uniform(-3, 3) for _ in range(strike_count))
        distribution = _random_distribution(random.randint(2, 6))
        raw_pmf = base.grid_distribution_to_pmf(distribution)
        breaks = _grid_breaks(strikes, raw_pmf)
        cdf_fn = cdf_from_pmf(breaks, raw_pmf)
        if strike_count <= 1:
            with pytest.raises(SkipScan):
                pmf_on_strikes(cdf_fn, strikes)
            continue
        aligned = pmf_on_strikes(cdf_fn, strikes)
        assert len(aligned) == strike_count + 1
        assert all(prob >= 0.0 for prob in aligned)
        assert abs(sum(aligned) - 1.0) < 1e-9


def test_align_pmf_to_strikes_matches_bin_count() -> None:
    strikes = [3.7, 3.8, 4.0, 4.25]
    distribution = {
        3.65: 0.25,
        3.8: 0.5,
        4.1: 0.25,
    }
    raw_pmf = base.grid_distribution_to_pmf(distribution)
    aligned = align_pmf_to_strikes(raw_pmf, strikes)
    assert len(aligned) == len(strikes) + 1
    total_prob = sum(bin_prob.probability for bin_prob in aligned)
    assert abs(total_prob - 1.0) < 1e-9
    # Ensure bins map to ladder intervals
    bounds = [(bin_prob.lower, bin_prob.upper) for bin_prob in aligned]
    assert bounds[0][1] == strikes[0]
    assert bounds[-1][0] == strikes[-1]


def test_align_pmf_to_strikes_raises_on_degenerate_strikes() -> None:
    raw_pmf = base.grid_distribution_to_pmf({0.0: 1.0})
    with pytest.raises(SkipScan):
        align_pmf_to_strikes(raw_pmf, [])
    with pytest.raises(SkipScan):
        align_pmf_to_strikes(raw_pmf, [0.0])
