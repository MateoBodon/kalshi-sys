from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

from kalshi_alpha.core.risk import drawdown
from kalshi_alpha.exec.pipelines import daily
from kalshi_alpha.exec.runners import scan_ladders


def test_drawdown_daily_cap(isolated_data_roots: tuple[Path, Path]) -> None:
    _, proc_root = isolated_data_roots
    timestamp = datetime(2025, 10, 27, 15, 0, tzinfo=UTC)
    drawdown.record_pnl(-120.0, timestamp=timestamp, state_dir=proc_root / "state")

    status = drawdown.check_limits(100.0, None, now=timestamp, state_dir=proc_root / "state")
    assert status.ok is False
    assert any(reason.startswith("drawdown.daily") for reason in status.reasons)


def test_drawdown_weekly_cap(isolated_data_roots: tuple[Path, Path]) -> None:
    _, proc_root = isolated_data_roots
    monday = datetime(2025, 10, 27, 15, 0, tzinfo=UTC)
    wednesday = monday + timedelta(days=2)
    drawdown.record_pnl(-60.0, timestamp=monday, state_dir=proc_root / "state")
    drawdown.record_pnl(-70.0, timestamp=wednesday, state_dir=proc_root / "state")

    status = drawdown.check_limits(None, 100.0, now=wednesday, state_dir=proc_root / "state")
    assert status.ok is False
    assert any(reason.startswith("drawdown.weekly") for reason in status.reasons)


def test_scan_ladders_halts_on_drawdown(monkeypatch, fixtures_root: Path) -> None:
    monkeypatch.setattr(drawdown, "check_limits", lambda *args, **kwargs: drawdown.DrawdownStatus(False, ["drawdown.daily:-150.00<=-100.00"], {"daily_pnl": -150.0}))

    def fail_scan_series(*args, **kwargs):  # pragma: no cover - ensure not called
        raise AssertionError("scan_series should not be invoked when drawdown breached")

    monkeypatch.setattr(scan_ladders, "scan_series", fail_scan_series)
    scan_ladders.main(
        [
            "--series",
            "CPI",
            "--offline",
            "--fixtures-root",
            str(fixtures_root),
        ]
    )


def test_daily_quality_gate_drawdown(monkeypatch, isolated_data_roots: tuple[Path, Path]) -> None:
    _, proc_root = isolated_data_roots

    monkeypatch.setattr(drawdown, "check_limits", lambda *args, **kwargs: drawdown.DrawdownStatus(False, ["drawdown.weekly:-250.00<=-200.00"], {"weekly_pnl": -250.0}))

    args = argparse.Namespace(
        daily_loss_cap=100.0,
        weekly_loss_cap=200.0,
    )
    log: dict[str, object] = {}

    now = datetime(2025, 10, 27, 18, 0, tzinfo=UTC)
    gate_result = daily.run_quality_gate_step(args, now, log)
    assert gate_result.go is False
    assert log["quality_gates"]["go"] is False
    assert any("drawdown.weekly" in reason for reason in log["quality_gates"]["reasons"])
