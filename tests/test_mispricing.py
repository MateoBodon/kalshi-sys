from __future__ import annotations

import pytest

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.core.pricing.mispricing import implied_cdf_kinks, kink_spreads, prob_sum_gap


def test_implied_cdf_kinks_detects_curvature() -> None:
    survival = [1.0, 0.85, 0.7, 0.4, 0.35, 0.2]
    metrics = implied_cdf_kinks(survival)
    assert metrics.max_kink > 0
    assert metrics.kink_count >= 1
    assert metrics.monotonicity_penalty == 0.0


def test_prob_sum_gap_handles_simplex() -> None:
    pmf = [
        LadderBinProbability(lower=None, upper=1, probability=0.25),
        LadderBinProbability(lower=1, upper=2, probability=0.25),
        LadderBinProbability(lower=2, upper=3, probability=0.25),
        LadderBinProbability(lower=3, upper=None, probability=0.30),
    ]
    assert prob_sum_gap(pmf) == pytest.approx(0.05, abs=1e-9)


def test_kink_spreads_finds_alternating_segment() -> None:
    model = [
        LadderBinProbability(lower=None, upper=1, probability=0.40),
        LadderBinProbability(lower=1, upper=2, probability=0.30),
        LadderBinProbability(lower=2, upper=3, probability=0.20),
        LadderBinProbability(lower=3, upper=None, probability=0.10),
    ]
    market = [
        LadderBinProbability(lower=None, upper=1, probability=0.20),
        LadderBinProbability(lower=1, upper=2, probability=0.40),
        LadderBinProbability(lower=2, upper=3, probability=0.20),
        LadderBinProbability(lower=3, upper=None, probability=0.20),
    ]
    spreads = kink_spreads(model, market, max_legs=4)
    assert spreads, "Expected mispricing spreads"
    top = spreads[0]
    assert top["legs"] >= 2
    assert top["abs_delta_sum"] > 0
    assert any(entry["delta"] > 0 for entry in top["bins"])
    assert any(entry["delta"] < 0 for entry in top["bins"])
