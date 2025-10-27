from __future__ import annotations

from pathlib import Path

import json
import polars as pl
import pytest

from kalshi_alpha.core.execution.slippage import SlippageModel
from kalshi_alpha.core.kalshi_api import Orderbook
from kalshi_alpha.exec.ledger import simulate_fills
from kalshi_alpha.exec.runners.scan_ladders import Proposal


def _proposal(side: str, contracts: int) -> Proposal:
    return Proposal(
        market_id="M1",
        market_ticker="M1",
        strike=0.5,
        side=side,
        contracts=contracts,
        maker_ev=0.0,
        taker_ev=0.0,
        maker_ev_per_contract=0.0,
        taker_ev_per_contract=0.0,
        strategy_probability=0.6,
        market_yes_price=0.55,
        survival_market=0.5,
        survival_strategy=0.6,
        max_loss=contracts * 0.55,
        strategy="CPI",
        metadata=None,
    )


def _orderbook() -> Orderbook:
    return Orderbook(
        market_id="M1",
        bids=[{"price": 0.54, "size": 80}, {"price": 0.50, "size": 120}],
        asks=[{"price": 0.55, "size": 50}, {"price": 0.6, "size": 150}],
    )


def test_depth_slippage_exceeds_top_of_book() -> None:
    proposal = _proposal("YES", 150)
    book = _orderbook()

    ledger_top = simulate_fills(
        [proposal],
        {"M1": book},
        slippage_model=SlippageModel(mode="top"),
    )
    ledger_depth = simulate_fills(
        [proposal],
        {"M1": book},
        slippage_model=SlippageModel(mode="depth", impact_cap=0.02),
    )

    assert ledger_top.records[0].fill_price == pytest.approx(0.55)
    assert ledger_depth.records[0].fill_price > ledger_top.records[0].fill_price
    assert ledger_depth.records[0].fill_price == pytest.approx(0.57, abs=1e-6)
    assert ledger_depth.records[0].slippage == pytest.approx(0.02, abs=1e-6)


def test_slippage_artifacts_written(tmp_path: Path) -> None:
    proposal = _proposal("NO", 120)
    book = _orderbook()
    ledger = simulate_fills(
        [proposal],
        {"M1": book},
        slippage_model=SlippageModel(mode="depth", impact_cap=0.01),
    )
    json_path, csv_path = ledger.write_artifacts(tmp_path)
    assert json_path and csv_path
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["trades"] == 1
    frame = pl.read_csv(csv_path)
    assert frame.height == 1
    assert frame["slippage"][0] <= 0.0  # NO side improves price
