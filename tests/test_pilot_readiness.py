from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.exec import scoreboard


def test_pilot_readiness_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    data_dir = tmp_path / "data" / "proc"
    data_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=UTC)

    ledger = pl.DataFrame(
        {
            "series": ["CPI", "CPI", "CLAIMS"],
            "ev_after_fees": [12.0, -3.0, 5.0],
            "pnl_simulated": [11.0, -2.5, 4.0],
            "expected_fills": [80, 40, 30],
            "size": [100, 50, 40],
            "fill_ratio": [0.8, 0.6, 0.7],
            "t_fill_ms": [100.0, 140.0, 180.0],
            "size_partial": [0, 5, 3],
            "slippage_ticks": [1.0, -0.8, 0.5],
            "ev_expected_bps": [110.0, -45.0, 60.0],
            "ev_realized_bps": [105.0, -40.0, 55.0],
            "fees_bps": [3.0, 4.0, 5.0],
            "timestamp_et": [
                now - timedelta(days=1),
                now - timedelta(days=2),
                now - timedelta(days=3),
            ],
        }
    )
    ledger.write_parquet(data_dir / "ledger_all.parquet")

    alpha_state = {
        "series": {
            "CPI": {"alpha": 0.65, "ts": (now - timedelta(hours=1)).isoformat()},
            "CLAIMS": {"alpha": 0.55, "ts": (now - timedelta(hours=1)).isoformat()},
        }
    }
    state_dir = data_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_dir.joinpath("fill_alpha.json").write_text(json.dumps(alpha_state), encoding="utf-8")

    artifacts_dir = tmp_path / "reports" / "_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    go_events = [
        {"go": True, "series": "CPI", "timestamp": (now - timedelta(days=1)).isoformat()},
        {"go": False, "series": "CPI", "timestamp": (now - timedelta(days=1)).isoformat()},
        {"go": True, "series": "CLAIMS", "timestamp": (now - timedelta(days=2)).isoformat()},
    ]
    for idx, payload in enumerate(go_events, start=1):
        artifacts_dir.joinpath(f"go_no_go_{idx}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    scorecard_dir = artifacts_dir / "scorecards"
    scorecard_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "market_id": ["M1"],
            "market_ticker": ["CPI-T"],
            "mean_abs_cdf_delta": [0.025],
            "max_abs_cdf_delta": [0.080],
            "prob_sum_gap": [0.01],
            "max_kink": [0.02],
            "mean_abs_kink": [0.01],
            "kink_count": [2],
            "monotonicity_penalty": [0.0],
            "model_version": ["v15"],
        }
    ).write_parquet(scorecard_dir / "CPI_summary.parquet")

    scoreboard.main(["--window", "7"])

    pilot_report = tmp_path / "reports" / "pilot_readiness.md"
    assert pilot_report.exists()
    text = pilot_report.read_text(encoding="utf-8")
    assert "GO Rate" in text
    assert "Fill realism" in text
    assert "Replay mean" in text
    assert "| CPI" in text
