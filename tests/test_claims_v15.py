from __future__ import annotations

import pytest

from kalshi_alpha.strategies.claims import ClaimsInputs, pmf, pmf_v15


def test_claims_v15_improves_brier_with_holiday_short_week_partials() -> None:
    strikes = [205_000, 210_000, 215_000, 220_000, 225_000]
    inputs = ClaimsInputs(
        history=[233_000, 231_000, 228_000, 224_000],
        holiday_next=True,
        short_week=True,
        continuing_claims=[1_780_000, 1_772_000],
        gt_topic_score=-0.015,
        latest_initial_claims=216_500,
        freeze_active=True,
    )
    actual_claims = 217_000.0

    pmf_v0 = pmf(strikes, inputs=inputs)
    pmf_v15_vals = pmf_v15(strikes, inputs=inputs)

    assert _brier_score(pmf_v15_vals, actual_claims) < _brier_score(pmf_v0, actual_claims)


def test_claims_v15_matches_baseline_without_partials() -> None:
    strikes = [200_000, 205_000, 210_000, 215_000]
    inputs = ClaimsInputs(history=[228_000, 226_000], freeze_active=False)

    pmf_v0 = pmf(strikes, inputs=inputs)
    pmf_v15_vals = pmf_v15(strikes, inputs=inputs)

    assert len(pmf_v0) == len(pmf_v15_vals)
    for base_bin, upgraded_bin in zip(pmf_v0, pmf_v15_vals, strict=True):
        assert upgraded_bin.lower == base_bin.lower
        assert upgraded_bin.upper == base_bin.upper
        assert upgraded_bin.probability == pytest.approx(base_bin.probability, rel=1e-6, abs=1e-6)


def _brier_score(pmf_values, actual: float) -> float:
    total = 0.0
    for bin_prob in pmf_values:
        lower = float("-inf") if bin_prob.lower is None else bin_prob.lower
        upper = float("inf") if bin_prob.upper is None else bin_prob.upper
        indicator = 1.0 if lower <= actual < upper else 0.0
        total += (bin_prob.probability - indicator) ** 2
    return total
