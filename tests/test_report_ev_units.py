from __future__ import annotations

from pathlib import Path
import polars as pl
import pytest

from kalshi_alpha.core.pricing import Liquidity
from kalshi_alpha.exec.reports import write_markdown_report
from kalshi_alpha.exec.ledger import FillRecord, PaperLedger
from kalshi_alpha.exec.runners.scan_ladders import Proposal


def _make_proposal(strike: float, maker_ev: float, maker_ev_per_contract: float) -> Proposal:
    return Proposal(
        market_id="MKT",
        market_ticker="TENY-TEST",
        strike=strike,
        side="YES",
        contracts=2,
        maker_ev=maker_ev,
        taker_ev=0.0,
        maker_ev_per_contract=maker_ev_per_contract,
        taker_ev_per_contract=0.0,
        strategy_probability=0.6,
        market_yes_price=0.4,
        survival_market=0.4,
        survival_strategy=0.6,
        max_loss=2.0,
        strategy="TNEY",
        metadata={},
    )


def test_report_ev_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proposals = [
        _make_proposal(4.1, maker_ev=0.38, maker_ev_per_contract=0.19),
        _make_proposal(4.2, maker_ev=0.38, maker_ev_per_contract=0.19),
    ]

    table = pl.DataFrame(
        {
            "market_id": ["MKT", "MKT"],
            "market_ticker": ["TENY-TEST", "TENY-TEST"],
            "strike": [4.1, 4.2],
            "fill_price": [0.4, 0.4],
            "maker_ev_original": [0.38, 0.38],
            "maker_ev_per_contract_original": [0.19, 0.19],
            "maker_ev_replay": [0.18, 0.18],
            "maker_ev_per_contract_replay": [0.09, 0.09],
        }
    )
    artifacts_dir = tmp_path / "reports/_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    replay_path = artifacts_dir / "replay_ev.parquet"
    table.write_parquet(replay_path)

    ledger = PaperLedger()
    ledger.set_series("TNEY")
    ledger.record(
        FillRecord(
            proposal=proposals[0],
            fill_price=0.4,
            expected_value=1.8,
            liquidity=Liquidity.MAKER,
            slippage=0.0,
            expected_contracts=1,
            expected_fills=1,
            fill_ratio=0.5,
            slippage_mode="top",
            impact_cap=0.0,
            fees_maker=0.0,
            pnl_simulated=1.8,
            alpha_row=0.6,
            size_throttled=False,
        )
    )

    monkeypatch.chdir(tmp_path)
    report_dir = tmp_path / "reports/TNEY"
    monitors: dict[str, object] = {}
    path = write_markdown_report(
        series="TNEY",
        proposals=proposals,
        ledger=ledger,
        output_dir=report_dir,
        monitors=monitors,
        exposure_summary={},
        manifest_path=Path("manifest.json"),
        go_status=True,
        pilot_metadata={},
    )
    contents = path.read_text(encoding="utf-8")
    assert "EV Honesty" in contents
    assert "| Market | Strike | EV_per_contract_original | EV_per_contract_replay | EV_total_original | EV_total_replay | Delta |" in contents
    assert "| TENY-TEST | 4.10 | 0.19 | 0.09 | 1.80 | 0.18 | 0.10 |" in contents
    assert "Max per-contract delta: 0.10" in contents
    assert monitors.get("ev_per_contract_diff_max") == pytest.approx(0.10, abs=1e-9)
