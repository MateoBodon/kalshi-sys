from __future__ import annotations

from pathlib import Path

import polars as pl

from kalshi_alpha.exec.reports import write_markdown_report
from kalshi_alpha.exec.runners.scan_ladders import (
    Proposal,
    _compute_exposure_summary,
    _write_cdf_diffs,
)


def _sample_proposals() -> list[Proposal]:
    return [
        Proposal(
            market_id="M1",
            market_ticker="CPI_X",
            strike=100.0,
            side="YES",
            contracts=10,
            maker_ev=5.0,
            taker_ev=0.0,
            maker_ev_per_contract=0.5,
            taker_ev_per_contract=0.0,
            strategy_probability=0.6,
            market_yes_price=0.55,
            survival_market=0.45,
            survival_strategy=0.60,
            max_loss=45.0,
            strategy="CPI",
            metadata=None,
        ),
        Proposal(
            market_id="M2",
            market_ticker="TNEY_X",
            strike=95.0,
            side="NO",
            contracts=8,
            maker_ev=4.0,
            taker_ev=0.0,
            maker_ev_per_contract=0.5,
            taker_ev_per_contract=0.0,
            strategy_probability=0.4,
            market_yes_price=0.48,
            survival_market=0.62,
            survival_strategy=0.40,
            max_loss=40.0,
            strategy="TENY",
            metadata=None,
        ),
    ]


def test_exposure_summary_and_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    project_root = Path(__file__).resolve().parents[1]
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    portfolio_src = project_root / "configs" / "portfolio.yaml"
    configs_dir.joinpath("portfolio.yaml").write_text(portfolio_src.read_text(encoding="utf-8"), encoding="utf-8")
    proposals = _sample_proposals()
    summary = _compute_exposure_summary(proposals)
    assert summary["total_max_loss"] == 85.0
    assert summary["per_series"]["CPI"] == 45.0
    assert summary["per_series"]["TENY"] == 40.0
    factors = summary["factors"]
    assert factors["INFLATION"] == 45.0
    assert factors["RATES"] == 40.0
    assert summary["net_contracts"]["CPI_X"] == 10
    assert summary["net_contracts"]["TNEY_X"] == -8

    report_path = write_markdown_report(
        series="CPI",
        proposals=proposals,
        ledger=None,
        output_dir=tmp_path / "reports" / "CPI",
        monitors={},
        exposure_summary=summary,
        manifest_path=None,
        go_status=True,
        fill_alpha=0.6,
    )
    text = report_path.read_text(encoding="utf-8")
    assert "GO/NO-GO" in text
    assert "Portfolio Exposure" in text
    assert "INFLATION" in text
    assert "Fill Alpha: 0.60" in text
    assert "Net Ladder Exposure" in text
    assert "| CPI | CPI_X |" in text


def test_cdf_diffs_parquet(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    diffs = [
        {
            "market_id": "M1",
            "market_ticker": "CPI_X",
            "bin_index": 0,
            "strike": 100.0,
            "p_model": 0.60,
            "p_market": 0.55,
            "delta": 0.05,
        },
        {
            "market_id": "M1",
            "market_ticker": "CPI_X",
            "bin_index": 1,
            "strike": 105.0,
            "p_model": 0.40,
            "p_market": 0.45,
            "delta": -0.05,
        },
    ]
    path = _write_cdf_diffs(diffs)
    assert path is not None
    frame = pl.read_parquet(path)
    assert set(frame.columns) == {
        "market_id",
        "market_ticker",
        "bin_index",
        "strike",
        "p_model",
        "p_market",
        "delta",
    }
    assert frame.height == 2
