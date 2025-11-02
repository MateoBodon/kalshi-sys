from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.core.risk import drawdown
from kalshi_alpha.exec.reports.ramp import RampPolicyConfig, compute_ramp_policy, write_ramp_outputs


def test_pilot_ramp_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    now = datetime(2025, 11, 2, tzinfo=UTC)

    ledger_path = tmp_path / "data" / "proc" / "ledger_all.parquet"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = pl.DataFrame(
        {
            "series": ["CPI"] * 6 + ["CLAIMS"] * 6,
            "ev_expected_bps": [10.0] * 6 + [8.0] * 6,
            "ev_realized_bps": [18.0, 17.5, 18.6, 18.1, 17.9, 18.4, 6.0, 6.2, 5.8, 6.1, 5.9, 6.0],
            "expected_fills": [60, 55, 50, 52, 58, 62, 20, 18, 22, 19, 21, 23],
            "timestamp_et": [now - timedelta(days=idx) for idx in range(1, 7)]
            + [now - timedelta(days=idx) for idx in range(1, 7)],
        }
    )
    ledger.write_parquet(ledger_path)

    artifacts_dir = tmp_path / "reports" / "_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.joinpath("go_no_go_1.json").write_text(
        json.dumps(
            {
                "go": False,
                "series": "CLAIMS",
                "timestamp": (now - timedelta(days=2)).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    proc_root = tmp_path / "data" / "proc"
    drawdown_state_dir = proc_root
    drawdown.record_pnl(500.0, timestamp=now, state_dir=drawdown_state_dir)

    config = RampPolicyConfig(
        lookback_days=30,
        min_fills=200,
        min_delta_bps=6.0,
        min_t_stat=1.5,
        go_multiplier=1.5,
        base_multiplier=1.0,
    )

    policy = compute_ramp_policy(
        ledger_path=ledger_path,
        artifacts_dir=artifacts_dir,
        drawdown_state_dir=drawdown_state_dir,
        config=config,
        now=now,
    )

    assert policy["drawdown"]["ok"] is True
    series = {entry["series"]: entry for entry in policy["series"]}
    assert series["CPI"]["recommendation"] == "GO"
    assert series["CPI"]["size_multiplier"] == config.go_multiplier
    assert series["CLAIMS"]["recommendation"] == "NO_GO"
    assert "guardrail_breaches" in series["CLAIMS"]
    assert series["CLAIMS"]["size_multiplier"] == config.base_multiplier

    json_path = tmp_path / "reports" / "pilot_ready.json"
    markdown_path = tmp_path / "reports" / "pilot_readiness.md"
    write_ramp_outputs(policy, json_path=json_path, markdown_path=markdown_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["series"]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Pilot Ramp Readiness" in markdown
    assert "| CPI" in markdown
