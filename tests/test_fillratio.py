from __future__ import annotations

from datetime import UTC, datetime

import pytest

from kalshi_alpha.core.execution.fillratio import FillRatioEstimator, expected_fills
from kalshi_alpha.core.fees import DEFAULT_FEE_SCHEDULE
from kalshi_alpha.core.kalshi_api import Orderbook
from kalshi_alpha.exec.ledger import simulate_fills
from kalshi_alpha.exec.runners.scan_ladders import Proposal
from kalshi_alpha.exec.scanners.utils import expected_value_summary


def test_fill_ratio_estimator_depth() -> None:
    orderbook = Orderbook(
        market_id="M1",
        bids=[{"price": 0.45, "size": 12}],
        asks=[{"price": 0.55, "size": 8}],
    )
    estimator = FillRatioEstimator(alpha=0.5)
    expected, ratio = estimator.estimate(
        side="YES",
        price=0.55,
        contracts=10,
        orderbook=orderbook,
    )
    assert expected == 4  # floor(0.5 * 8)
    assert ratio == pytest.approx(0.4)


def test_simulate_fills_with_estimator() -> None:
    proposal = Proposal(
        market_id="M1",
        market_ticker="CPI_X",
        strike=100.0,
        side="YES",
        contracts=10,
        maker_ev=0.0,
        taker_ev=0.0,
        maker_ev_per_contract=0.0,
        taker_ev_per_contract=0.0,
        strategy_probability=0.6,
        market_yes_price=0.55,
        survival_market=0.4,
        survival_strategy=0.6,
        max_loss=5.5,
        strategy="CPI",
        metadata=None,
    )
    orderbook = Orderbook(
        market_id="M1",
        bids=[{"price": 0.45, "size": 15}],
        asks=[{"price": 0.55, "size": 8}],
    )
    estimator = FillRatioEstimator(0.5)
    ledger = simulate_fills(
        [proposal],
        {"M1": orderbook},
        fill_estimator=estimator,
    )
    record = ledger.records[0]
    assert record.expected_contracts == 4
    assert record.expected_fills == 4
    assert record.fill_ratio == pytest.approx(0.4)

    expected_summary = expected_value_summary(
        contracts=4,
        yes_price=record.fill_price,
        event_probability=proposal.strategy_probability,
        schedule=DEFAULT_FEE_SCHEDULE,
        market_name=proposal.market_ticker,
    )
    assert record.expected_value == pytest.approx(expected_summary["maker_yes"])


def test_expected_fills_caps_to_size() -> None:
    expected, ratio = expected_fills(size=5, visible_depth=100, alpha=0.9)
    assert expected == 5
    assert ratio == pytest.approx(1.0)
