from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.core.kalshi_api import Orderbook
from kalshi_alpha.exec.ledger import simulate_fills
from kalshi_alpha.exec.ledger.aggregate import main as aggregate_main
from kalshi_alpha.exec.runners.scan_ladders import Proposal


def _proposal(market_id: str, strike: float, side: str = "YES") -> Proposal:
    return Proposal(
        market_id=market_id,
        market_ticker=f"{market_id}_T",
        strike=strike,
        side=side,
        contracts=10,
        maker_ev=0.5,
        taker_ev=-0.5,
        maker_ev_per_contract=0.05,
        taker_ev_per_contract=-0.05,
        strategy_probability=0.55,
        market_yes_price=0.45,
        survival_market=0.4,
        survival_strategy=0.55,
        max_loss=4.5,
        strategy="CPI",
        series="CPI",
        metadata=None,
    )


def _orderbook(market_id: str) -> Orderbook:
    return Orderbook(
        market_id=market_id,
        bids=[{"price": 0.44, "size": 50}],
        asks=[{"price": 0.45, "size": 50}],
    )


def test_ledger_aggregate_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    reports_dir = Path("reports/_artifacts")
    reports_dir.mkdir(parents=True, exist_ok=True)

    ledger = simulate_fills([
        _proposal("M1", 270.0),
        _proposal("M2", 275.0, side="NO"),
    ], {"M1": _orderbook("M1"), "M2": _orderbook("M2")})
    ledger.write_artifacts(reports_dir)

    aggregate_main([])

    output_path = Path("data/proc/ledger_all.parquet")
    assert output_path.exists()
    frame = pl.read_parquet(output_path)
    assert frame.height == len(ledger.records)
    for column in [
        "t_fill_ms",
        "size_partial",
        "slippage_ticks",
        "ev_expected_bps",
        "ev_realized_bps",
        "fees_bps",
    ]:
        assert column in frame.columns
    assert set(frame["ledger_schema_version"].unique().to_list()) == {2}


def test_ledger_aggregate_rejects_wrong_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    reports_dir = Path("reports/_artifacts")
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "bad_ledger.csv"
    csv_path.write_text(
        "series,event,bin,side,price,model_p,market_p,delta_p,size,expected_contracts,expected_fills,fill_ratio,slippage_mode,impact_cap,fees_maker,ev_after_fees,pnl_simulated,timestamp_et,manifest_path,ledger_schema_version\n"
        "CPI,E1,270.0,YES,0.45,0.55,0.40,0.15,10,10,1.0,top,0.02,0.12,0.23,0.23,2025-01-01T08:00:00-05:00,,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        aggregate_main([])
