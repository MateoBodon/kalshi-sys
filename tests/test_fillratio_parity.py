from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.core.execution.fillratio import FillRatioEstimator
from kalshi_alpha.core.kalshi_api import Orderbook
from kalshi_alpha.exec.ledger import simulate_fills
from kalshi_alpha.exec.runners.scan_ladders import Proposal


def _proposal(contracts: int = 12) -> Proposal:
    return Proposal(
        market_id="M1",
        market_ticker="CLAIMS_X",
        strike=250.0,
        side="YES",
        contracts=contracts,
        maker_ev=0.0,
        taker_ev=0.0,
        maker_ev_per_contract=0.0,
        taker_ev_per_contract=0.0,
        strategy_probability=0.55,
        market_yes_price=0.6,
        survival_market=0.48,
        survival_strategy=0.55,
        max_loss=contracts * 0.6,
        strategy="CLAIMS",
        series="CLAIMS",
        metadata=None,
    )


def _orderbook(visible: int) -> Orderbook:
    return Orderbook(
        market_id="M1",
        bids=[{"price": 0.6, "size": visible}],
        asks=[{"price": 0.6, "size": visible}],
    )


def test_fillratio_toggle_changes_expected(tmp_path: Path) -> None:
    proposal = _proposal(contracts=20)
    orderbook = _orderbook(visible=8)

    ledger_full = simulate_fills([proposal], {"M1": orderbook})
    record_full = ledger_full.records[0]
    assert record_full.expected_fills == proposal.contracts
    assert record_full.fill_ratio == pytest.approx(1.0)

    estimator = FillRatioEstimator(alpha=0.4)
    ledger_est = simulate_fills([proposal], {"M1": orderbook}, fill_estimator=estimator)
    record_est = ledger_est.records[0]
    assert record_est.expected_fills == 0
    assert record_est.fill_ratio == pytest.approx(0.0)
    assert record_est.size_throttled is True

    csv_path = ledger_est.write_artifacts(tmp_path)[1]
    frame = pl.read_csv(csv_path)
    row = frame.row(0, named=True)
    assert row["expected_fills"] == 0
    assert row["fill_ratio"] == pytest.approx(record_est.fill_ratio)


def test_fillratio_caps_respected(tmp_path: Path) -> None:
    proposal = _proposal(contracts=15)
    orderbook = _orderbook(visible=20)
    estimator = FillRatioEstimator(alpha=0.9)
    ledger = simulate_fills([proposal], {"M1": orderbook}, fill_estimator=estimator)
    record = ledger.records[0]
    assert record.expected_fills <= proposal.contracts
    assert record.expected_fills == 13
    assert record.fill_ratio == pytest.approx(13 / proposal.contracts)

    shallow_orderbook = _orderbook(visible=4)
    ledger_shallow = simulate_fills(
        [proposal],
        {"M1": shallow_orderbook},
        fill_estimator=estimator,
    )
    record_shallow = ledger_shallow.records[0]
    assert record_shallow.expected_fills == 1
    assert record_shallow.size_throttled is True
    assert 0.0 <= record_shallow.fill_ratio <= 1.0
    assert record_shallow.expected_fills <= proposal.contracts
