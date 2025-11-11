from __future__ import annotations

from kalshi_alpha.core.kalshi_api import Orderbook
from kalshi_alpha.core.pricing import OrderSide
from kalshi_alpha.exec.quote_optim import QuoteContext, QuoteOptimizer, microprice_from_orderbook


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


def test_microprice_helper_uses_depth_weighting() -> None:
    orderbook = Orderbook(
        market_id="m1",
        bids=[{"price": 0.48, "size": 40}],
        asks=[{"price": 0.52, "size": 10}],
    )
    microprice, best_bid, best_ask = microprice_from_orderbook(orderbook)
    assert best_bid == 0.48
    assert best_ask == 0.52
    assert microprice is not None
    assert best_bid < microprice < best_ask


def test_replacement_throttle_enforces_limit() -> None:
    optimizer = QuoteOptimizer(max_replacements_per_bin=2)
    key = optimizer.key_for_order("m1", 5000.0, OrderSide.YES)
    assert not optimizer.should_throttle(key)
    optimizer.record_submission(key)
    assert not optimizer.should_throttle(key)
    optimizer.record_submission(key)
    assert optimizer.should_throttle(key)
