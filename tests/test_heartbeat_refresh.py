from __future__ import annotations

import json
from argparse import Namespace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from kalshi_alpha.exec.pipelines import daily


def test_daily_refresh_prevents_heartbeat_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kalshi_alpha.datastore import paths as datastore_paths

    monkeypatch.chdir(tmp_path)
    proc_root = tmp_path / "data" / "proc"
    raw_root = tmp_path / "data" / "raw"
    monkeypatch.setattr(datastore_paths, "PROC_ROOT", proc_root)
    monkeypatch.setattr(datastore_paths, "RAW_ROOT", raw_root)
    monkeypatch.setattr(daily, "PROC_ROOT", proc_root)
    monkeypatch.setattr(daily, "RAW_ROOT", raw_root)

    old_timestamp = datetime.now(tz=UTC) - timedelta(hours=2)
    original_write = daily.write_heartbeat
    calls: list[tuple[str, dict[str, object] | None]] = []

    def record_write(**kwargs):
        calls.append((kwargs["mode"], kwargs.get("extra")))
        return original_write(**kwargs)

    monkeypatch.setattr(daily, "write_heartbeat", record_write)
    original_write(mode="daily:pre_cpi", now=old_timestamp, extra={"stage": "stale"})

    monkeypatch.setattr(daily, "run_ingest", lambda *_: None)

    def fake_calibrations(args, log, heartbeat_cb=None):
        if heartbeat_cb:
            heartbeat_cb("post_calibrate", {"calibration": "dummy"})

    monkeypatch.setattr(daily, "run_calibrations", fake_calibrations)

    def fake_run_scan(mode, args, log, series, fill_alpha_value, fill_alpha_auto):
        log.setdefault("scan_results", {})[mode] = {"proposals": 0}
        daily.write_heartbeat(mode=f"daily:{mode}", extra={"stage": "scan_stub"})

    monkeypatch.setattr(daily, "run_scan", fake_run_scan)

    class DummyWindow:
        notes: list[str] = []

        def scan_allowed(self, _now):
            return True

        def to_dict(self, _now):
            return {"mode": "pre_cpi"}

    monkeypatch.setattr(daily, "resolve_run_window", lambda **_: DummyWindow())

    monkeypatch.setattr(
        daily,
        "run_quality_gates",
        lambda **_: daily.QualityGateResult(go=True, reasons=[], details={}),
    )
    monkeypatch.setattr(
        daily.drawdown,
        "check_limits",
        lambda *_, **__: SimpleNamespace(ok=True, reasons=[], metrics={}),
    )
    monkeypatch.setattr(daily, "load_quality_gate_config", lambda *_: SimpleNamespace())
    monkeypatch.setattr(daily, "resolve_quality_gate_config_path", lambda: Path("dummy"))

    args = Namespace(
        mode="pre_cpi",
        offline=True,
        online=False,
        driver_fixtures=str(tmp_path / "fixtures"),
        scanner_fixtures=str(tmp_path / "scanner"),
        kelly_cap=0.1,
        fill_alpha="0.6",
        slippage_mode="top",
        impact_cap=0.02,
        report=False,
        paper_ledger=False,
        broker="dry",
        allow_no_go=False,
        mispricing_only=False,
        max_legs=4,
        prob_sum_gap_threshold=0.0,
        model_version="v15",
        kill_switch_file=None,
        when=None,
        daily_loss_cap=None,
        weekly_loss_cap=None,
        force_refresh=False,
        paper=False,
    )

    daily.run_mode("pre_cpi", args)

    payload = json.loads((tmp_path / "reports" / "_artifacts" / "go_no_go.json").read_text())
    assert payload["go"] is True
    assert "heartbeat_stale" not in payload["reasons"]

    stages = {extra.get("stage") for _, extra in calls if extra}
    assert "start" in stages
    assert "post_ingest" in stages
    assert "pre_scan" in stages
