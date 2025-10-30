from __future__ import annotations

from kalshi_alpha.core.pricing.align import align_pmf_to_strikes
from kalshi_alpha.strategies.teny import TenYInputs, pmf_v15


def test_teny_pmf_alignment_with_prior_close() -> None:
    strikes = [3.5, 3.6, 3.7, 3.8, 4.0]
    inputs = TenYInputs(
        prior_close=3.65,
        macro_shock=0.05,
        trailing_history=[3.4, 3.45, 3.55, 3.6, 3.62],
    )
    raw_pmf = pmf_v15(strikes, inputs=inputs)
    pmf_values = align_pmf_to_strikes(raw_pmf, strikes)
    assert len(pmf_values) == len(strikes) + 1
    total = sum(bin_prob.probability for bin_prob in pmf_values)
    assert abs(total - 1.0) < 1e-9
    assert pmf_values[0].upper == strikes[0]
    assert pmf_values[-1].lower == strikes[-1]
