from __future__ import annotations

import pytest

from kalshi_alpha.core.pricing import Liquidity, OrderSide
from kalshi_alpha.core.risk import OrderProposal, PALGuard, PALPolicy
from kalshi_alpha.exec.limits import LossBudget, LimitViolation, ProposalLimitChecker


def _guard() -> PALGuard:
    policy = PALPolicy(series="TEST", default_max_loss=100.0, per_strike={})
    return PALGuard(policy)


def test_limit_checker_consumes_budgets() -> None:
    guard = _guard()
    daily = LossBudget(50.0)
    checker = ProposalLimitChecker(guard, daily_budget=daily)
    order = OrderProposal(
        strike_id="TEST-1",
        yes_price=0.5,
        contracts=10,
        side=OrderSide.YES,
        liquidity=Liquidity.MAKER,
        market_name="TEST",
        series="TEST",
    )
    checker.try_accept(order)
    assert daily.remaining is not None and daily.remaining < 50.0
    with pytest.raises(LimitViolation):
        checker.try_accept(order_maker := OrderProposal(
            strike_id="TEST-1",
            yes_price=0.5,
            contracts=500,
            side=OrderSide.YES,
            liquidity=Liquidity.MAKER,
            market_name="TEST",
            series="TEST",
        ))


def test_limit_checker_respects_weekly_cap() -> None:
    guard = _guard()
    weekly = LossBudget(10.0)
    checker = ProposalLimitChecker(guard, weekly_budget=weekly)
    order = OrderProposal(
        strike_id="TEST-2",
        yes_price=0.2,
        contracts=1,
        side=OrderSide.YES,
        liquidity=Liquidity.MAKER,
        market_name="TEST",
        series="TEST",
    )
    checker.try_accept(order)
    assert weekly.remaining < 10.0
    order_large = OrderProposal(
        strike_id="TEST-3",
        yes_price=0.9,
        contracts=20,
        side=OrderSide.YES,
        liquidity=Liquidity.MAKER,
        market_name="TEST",
        series="TEST",
    )
    with pytest.raises(LimitViolation):
        checker.try_accept(order_large)
