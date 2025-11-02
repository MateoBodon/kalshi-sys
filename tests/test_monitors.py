from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.core.risk import drawdown
from kalshi_alpha.exec.monitors import (
    RuntimeMonitorConfig,
    build_report_summary,
    compute_runtime_monitors,
    write_monitor_artifacts,
)
from kalshi_alpha.exec.pipelines import daily
from kalshi_alpha.exec.policy import freeze as freeze_policy
from kalshi_alpha.exec.policy.freeze import FreezeEvaluation
from kalshi_alpha.exec.reports import write_markdown_report
from kalshi_alpha.exec.runners import scan_ladders


def test_model_drift_flag(
    tmp_path: Path,
    isolated_data_roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, proc_root = isolated_data_roots
    calib_path = proc_root / "cpi_calib.parquet"
    frame = pl.DataFrame(
        {
            "record_type": ["params"],
            "crps": [0.25],
            "baseline_crps": [0.2],
            "brier": [0.6],
            "baseline_brier": [0.5],
        }
    )
    frame.write_parquet(calib_path)

    assert scan_ladders._model_drift_flag("CPI") is True

    monitors = {"model_drift": True, "tz_not_et": False, "non_monotone_ladders": 2}
    report_path = write_markdown_report(
        series="CPI",
        proposals=[],
        ledger=None,
        output_dir=tmp_path,
        monitors=monitors,
    )
    contents = report_path.read_text(encoding="utf-8")
    assert "Monitors" in contents
    assert "model_drift: True" in contents

    log = {"monitors": monitors}
    monkeypatch.setattr(daily, "PROC_ROOT", proc_root)
    timestamp = datetime(2025, 10, 25, 12, 0, tzinfo=UTC)
    daily.write_log("pre_cpi", log, timestamp)
    logs = list((proc_root / "logs").rglob("*.json"))
    assert logs, "Expected monitor log file"
    assert "model_drift" in logs[0].read_text(encoding="utf-8")


def test_runtime_monitors_emit_alerts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2025, 11, 2, 18, 0, tzinfo=UTC)

    telemetry_root = tmp_path / "telemetry"
    telemetry_path = telemetry_root / "2025" / "11" / "02" / "exec.jsonl"
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_events = [
        {
            "timestamp": (now - timedelta(minutes=50)).isoformat(),
            "event_type": "ws_disconnect",
            "data": {"error": "network"},
        },
        {
            "timestamp": (now - timedelta(minutes=40)).isoformat(),
            "event_type": "reject",
            "data": {
                "error": "Failed to execute Kalshi trading API request",
                "error_cause": "Kalshi API returned 401 for POST /orders: unauthorized",
            },
        },
        {
            "timestamp": (now - timedelta(minutes=30)).isoformat(),
            "event_type": "reject",
            "data": {
                "error": "Failed to execute Kalshi trading API request",
                "error_cause": "Kalshi API returned 401 for POST /orders: unauthorized",
            },
        },
        {
            "timestamp": (now - timedelta(minutes=20)).isoformat(),
            "event_type": "reject",
            "data": {
                "error": "Failed to execute Kalshi trading API request",
                "error_cause": "Kalshi API returned 401 for POST /orders: unauthorized",
            },
        },
    ]
    telemetry_path.write_text("\n".join(json.dumps(entry) for entry in telemetry_events), encoding="utf-8")

    ledger_path = tmp_path / "data" / "proc" / "ledger_all.parquet"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_frame = pl.DataFrame(
        {
            "series": ["CPI"] * 8,
            "ev_expected_bps": [25.0] * 8,
            "ev_realized_bps": [8.0] * 8,
            "expected_fills": [20, 15, 18, 10, 12, 15, 16, 14],
            "size": [40, 35, 36, 30, 28, 34, 30, 32],
            "timestamp_et": [now - timedelta(hours=idx + 1) for idx in range(8)],
        }
    )
    ledger_frame.write_parquet(ledger_path)

    alpha_path = tmp_path / "data" / "proc" / "state" / "fill_alpha.json"
    alpha_path.parent.mkdir(parents=True, exist_ok=True)
    alpha_payload = {"series": {"CPI": {"alpha": 0.65}}}
    alpha_path.write_text(json.dumps(alpha_payload), encoding="utf-8")

    proc_root = tmp_path / "data" / "proc"
    drawdown.record_pnl(-2500.0, timestamp=now, state_dir=proc_root)

    def _fake_freeze(series: str, *, now: datetime, proc_root: str | None = None) -> FreezeEvaluation:
        return FreezeEvaluation(series.upper(), "mock", now, True, None, None, None, ["test-freeze"])

    monkeypatch.setattr(freeze_policy, "evaluate_freeze_for_series", _fake_freeze)

    config = RuntimeMonitorConfig(
        telemetry_lookback_hours=1,
        ledger_lookback_days=2,
        ws_disconnect_rate_threshold=0.5,
        auth_error_streak_threshold=3,
        fill_min_contracts=50,
        daily_loss_cap=2000.0,
        weekly_loss_cap=6000.0,
        kill_switch_path=proc_root / "state" / "kill_switch",
        seq_min_sample=5,
        seq_cusum_threshold=5.0,
    )

    results = compute_runtime_monitors(
        config=config,
        telemetry_root=telemetry_root,
        ledger_path=ledger_path,
        alpha_state_path=alpha_path,
        drawdown_state_dir=proc_root,
        now=now,
    )

    by_name = {result.name: result for result in results}
    assert by_name["ev_gap"].status == "ALERT"
    assert by_name["fill_vs_alpha"].status == "ALERT"
    assert by_name["drawdown"].status == "ALERT"
    assert by_name["ws_disconnect_rate"].status == "ALERT"
    assert by_name["auth_error_streak"].status == "ALERT"
    assert by_name["auth_error_streak"].metrics["max_streak"] == 3
    assert by_name["kill_switch"].status == "OK"
    assert by_name["ev_seq_guard"].status == "ALERT"
    seq_metrics = by_name["ev_seq_guard"].metrics
    assert seq_metrics["triggers"], "Expected sequential guard triggers"
    freeze_monitor = by_name["freeze_window"]
    assert freeze_monitor.metrics["evaluations"], "Expected freeze evaluations"
    assert freeze_monitor.status in {"OK", "ALERT"}

    artifacts_dir = tmp_path / "reports" / "_artifacts" / "monitors"
    written = write_monitor_artifacts(results, artifacts_dir=artifacts_dir, generated_at=now)
    assert written
    sample_payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert sample_payload["status"] in {"ALERT", "OK", "NO_DATA"}

    summary_lines = build_report_summary(results)
    assert summary_lines
    assert any("⚠️" in line for line in summary_lines)
