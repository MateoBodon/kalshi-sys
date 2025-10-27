from __future__ import annotations

from pathlib import Path
import math

import polars as pl

from kalshi_alpha.exec.ledger import PaperLedger, simulate_fills
from kalshi_alpha.exec.ledger.schema import LedgerRowV1
from kalshi_alpha.exec.runners.scan_ladders import Proposal
from kalshi_alpha.core.kalshi_api import Orderbook


def _proposal() -> Proposal:
    return Proposal(
        market_id="M1",
        market_ticker="CPI_X1",
        strike=101.0,
        side="YES",
        contracts=12,
        maker_ev=1.25,
        taker_ev=-1.40,
        maker_ev_per_contract=0.11,
        taker_ev_per_contract=-0.12,
        strategy_probability=0.58,
        market_yes_price=0.47,
        survival_market=0.42,
        survival_strategy=0.58,
        max_loss=5.64,
        strategy="cpi",
        metadata=None,
    )


def _orderbook() -> Orderbook:
    return Orderbook(
        market_id="M1",
        bids=[{"price": 0.46, "size": 50}],
        asks=[{"price": 0.47, "size": 40}],
    )


def test_ledger_schema_v1_columns_and_types(tmp_path: Path) -> None:
    proposal = _proposal()
    ledger = simulate_fills(
        [proposal],
        {"M1": _orderbook()},
        ledger_series="cpi",
        market_event_lookup={"M1": 202510},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    json_path, csv_path = ledger.write_artifacts(tmp_path, manifest_path=manifest_path)

    assert isinstance(ledger, PaperLedger)
    assert json_path and csv_path
    assert csv_path.exists()

    frame = pl.read_csv(csv_path, try_parse_dates=False)
    expected_columns = list(LedgerRowV1.canonical_fields())
    assert frame.columns == expected_columns

    row = frame.row(0, named=True)
    for name, value in row.items():
        assert value is not None, f"{name} unexpectedly None"
        if isinstance(value, float):
            assert not math.isnan(value), f"{name} is NaN"
    assert row["ledger_schema_version"] == 1
    assert row["series"] == "CPI"
    assert str(row["event"]) == "202510"
    assert str(row["manifest_path"]).endswith("manifest.json")

    parsed = LedgerRowV1.model_validate({**row, "event": str(row["event"])})
    assert parsed.series == "CPI"
    assert parsed.manifest_path.endswith("manifest.json")
