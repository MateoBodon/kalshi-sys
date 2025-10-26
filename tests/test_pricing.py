from __future__ import annotations

from math import isclose

from hypothesis import given
from hypothesis import strategies as st

from kalshi_alpha.core.pricing import (
    LadderRung,
    Liquidity,
    OrderSide,
    expected_value_after_fees,
    pmf_from_quotes,
    pmf_from_survival,
    project_simplex,
    project_survival,
)


def _monotone(values: list[float]) -> list[float]:
    projected = project_survival(values)
    return projected


@given(st.lists(st.floats(min_value=0, max_value=1), min_size=2, max_size=5))
def test_pmf_properties(values: list[float]) -> None:
    strikes = [float(index) for index in range(len(values))]
    survival = _monotone(values)
    pmf = pmf_from_survival(strikes, survival)
    total = sum(bin_prob.probability for bin_prob in pmf)
    assert all(bin_prob.probability >= -1e-9 for bin_prob in pmf)
    assert isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_known_distribution_round_trip() -> None:
    strikes = [1.0, 2.0, 3.0]
    pmf_truth = [0.1, 0.2, 0.3, 0.4]
    survival = [sum(pmf_truth[i + 1 :]) for i in range(len(strikes))]
    rungs = [
        LadderRung(strike=strike, yes_price=survival[idx]) for idx, strike in enumerate(strikes)
    ]
    implied = pmf_from_quotes(rungs)
    recovered = [bin_prob.probability for bin_prob in implied]
    for expected, actual in zip(pmf_truth, recovered, strict=True):
        assert isclose(expected, actual, rel_tol=1e-6, abs_tol=1e-6)


def test_expected_value_after_fees_symmetry() -> None:
    contracts = 10
    yes_price = 0.45
    probability = 0.55
    yes_ev = expected_value_after_fees(
        contracts=contracts,
        yes_price=yes_price,
        event_probability=probability,
        side=OrderSide.YES,
        liquidity=Liquidity.MAKER,
    )
    no_ev = expected_value_after_fees(
        contracts=contracts,
        yes_price=yes_price,
        event_probability=probability,
        side=OrderSide.NO,
        liquidity=Liquidity.MAKER,
    )
    assert yes_ev > no_ev


@given(st.lists(st.floats(min_value=-1, max_value=2), min_size=3, max_size=8))
def test_project_survival_isotonic(values: list[float]) -> None:
    projected = project_survival(values)
    assert len(projected) == len(values)
    assert all(0.0 <= value <= 1.0 for value in projected)
    assert all(projected[i] >= projected[i + 1] for i in range(len(projected) - 1))


@given(st.lists(st.floats(min_value=0, max_value=1), min_size=3, max_size=8))
def test_project_survival_keeps_monotone(values: list[float]) -> None:
    monotone = sorted(values, reverse=True)
    projected = project_survival(monotone)
    assert projected == monotone


@given(
    st.lists(
        st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    )
)
def test_simplex_projection(values: list[float]) -> None:
    projected = project_simplex(values)
    assert len(projected) == len(values)
    assert all(value >= -1e-12 for value in projected)
    assert isclose(sum(projected), 1.0, rel_tol=1e-9, abs_tol=1e-9)
