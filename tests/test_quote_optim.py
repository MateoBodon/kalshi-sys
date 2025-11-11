from __future__ import annotations

from kalshi_alpha.core.pricing import OrderSide
from kalshi_alpha.exec.quote_optim import QuoteContext, QuoteOptimizer


def test_penalty_reflects_skew_and_freshness() -> None:
    optimizer = QuoteOptimizer(skew_floor=0.02, skew_weight=10.0, freshness_soft_ms=1000.0, freshness_slope=0.001)
    context = QuoteContext(
        market_id="m1",
        strike=5000.0,
        side=OrderSide.YES,
        pmf_probability=0.51,
        market_probability=0.50,
        microprice=None,
        best_bid=None,
        best_ask=None,
        freshness_ms=1500.0,
        maker_ev_per_contract=0.08,
    )
    penalty = optimizer.penalty(context)
    # Gap of 0.01 below floor contributes 0.1; freshness adds 0.5
    assert penalty > 0.5
