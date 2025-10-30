from __future__ import annotations

import pytest

from kalshi_alpha.core.execution.fillratio import FillRatioEstimator
from kalshi_alpha.core.kalshi_api import Orderbook
from kalshi_alpha.exec.ledger import simulate_fills
from kalshi_alpha.exec.runners.scan_ladders import Proposal


def _proposal(contracts: int) -> Proposal:
    return Proposal(
        market_id="MKT",
        market_ticker="TENY-TEST",
        strike=4.2,
        side="YES",
        contracts=contracts,
        maker_ev=2.0,
        taker_ev=0.0,
        maker_ev_per_contract=0.2,
        taker_ev_per_contract=0.0,
        strategy_probability=0.6,
        market_yes_price=0.4,
        survival_market=0.4,
        survival_strategy=0.6,
        max_loss=5.0,
        strategy="TNEY",
        metadata={},
    )


def _orderbook(depth: float) -> Orderbook:
    return Orderbook(
        market_id="MKT",
        bids=[{"price": 0.4, "size": depth}],
        asks=[{"price": 0.4, "size": depth}],
    )


def test_size_throttle_engages_under_weak_depth() -> None:
    proposal = _proposal(contracts=10)
    ledger = simulate_fills(
        [proposal],
        {"MKT": _orderbook(depth=2.0)},
        fill_estimator=FillRatioEstimator(alpha=0.6),
        mode="mid",
    )
    record = ledger.records[0]
    assert record.size_throttled is True
    assert record.expected_contracts <= 2


def test_size_throttle_not_triggered_when_depth_sufficient() -> None:
    proposal = _proposal(contracts=10)
    ledger = simulate_fills(
        [proposal],
        {"MKT": _orderbook(depth=20.0)},
        fill_estimator=FillRatioEstimator(alpha=0.6),
        mode="mid",
    )
    record = ledger.records[0]
    assert record.size_throttled is False
    assert record.expected_contracts == 6
