from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.exec.runners import risk_preview


def test_risk_preview_reports_and_fails_on_no_go(monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(tmp_path)

    # Configs
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "portfolio.yaml").write_text(
        "factor_vols:\n  TOTAL: 1.0\nstrategy_betas:\n  CPI:\n    TOTAL: 1.0\n",
        encoding="utf-8",
    )
    (configs / "pal_policy.yaml").write_text(
        "series: CPI\ndefault_max_loss: 1000\nper_strike:\n  CPI_X@270.0: 250\n",
        encoding="utf-8",
    )
    (configs / "quality_gates.yaml").write_text(
        "metrics:\n  series:\n    cpi:\n      crps_advantage_min: 0.02\n      brier_advantage_min: 0.01\n"
        "data_freshness: []\nreconciliation: []\nmonitors: {}\n",
        encoding="utf-8",
    )

    # Calibration data with poor performance -> NO-GO
    proc_root = tmp_path / "data" / "proc"
    proc_root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "record_type": ["params"],
            "crps": [0.15],
            "baseline_crps": [0.10],
            "brier": [0.20],
            "baseline_brier": [0.15],
        }
    ).write_parquet(proc_root / "cpi_calib.parquet")

    # Ledger snapshot
    ledger = pl.DataFrame(
        {
            "series": ["CPI"],
            "market": ["CPI_X"],
            "bin": [270.0],
            "side": ["YES"],
            "price": [0.45],
            "size": [20],
            "fees_maker": [0.12],
            "ev_after_fees": [2.4],
            "pnl_simulated": [2.0],
            "expected_fills": [16],
            "fill_ratio": [0.8],
            "timestamp_et": [datetime.now(tz=UTC)],
        }
    )
    ledger.write_parquet(proc_root / "ledger_all.parquet")

    with pytest.raises(SystemExit) as excinfo:
        risk_preview.main(["--mode", "pre_cpi", "--offline"])

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "VaR" in out
    assert "Quality gates verdict: NO-GO" in out
    assert "Per-strike max loss projections" in out
