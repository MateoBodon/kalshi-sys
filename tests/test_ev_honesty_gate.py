from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from kalshi_alpha.exec.runners import scan_ladders


def test_ev_honesty_gate_flags_stale_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monitors_dir = tmp_path / "reports" / "_artifacts" / "monitors"
    monitors_dir.mkdir(parents=True, exist_ok=True)
    monitors_dir.joinpath("ev_gap.json").write_text(
        json.dumps(
            {
                "name": "ev_gap",
                "status": "OK",
                "metrics": {},
                "generated_at": datetime.now(tz=UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    monitors: dict[str, object] = {
        "ev_honesty_max_delta_cents": 25.0,
        "book_latency_ms": 1250.0,
    }
    scan_ladders._apply_ev_honesty_gate(monitors, threshold_cents=10.0)
    assert monitors["ev_honesty_no_go"] is True
    assert monitors["ev_honesty_threshold_cents"] == 10.0

    dummy_result = scan_ladders.QualityGateResult(go=True, reasons=[], details={})

    monkeypatch.setattr(scan_ladders, "load_quality_gate_config", lambda _: {})
    monkeypatch.setattr(scan_ladders, "run_quality_gates", lambda **kwargs: dummy_result)
    monkeypatch.setattr(scan_ladders, "resolve_quality_gate_config_path", lambda: Path("dummy-config.yaml"))

    class _DrawdownStatus:
        ok = True
        reasons: list[str] = []
        metrics = None

    monkeypatch.setattr(scan_ladders.drawdown, "check_limits", lambda *a, **k: _DrawdownStatus())
    monkeypatch.setattr(scan_ladders, "kill_switch_engaged", lambda path: False)
    monkeypatch.setattr(scan_ladders, "heartbeat_stale", lambda threshold: (False, None))

    args = SimpleNamespace(
        daily_loss_cap=None,
        weekly_loss_cap=None,
        broker="dry",
        kill_switch_file=None,
    )

    result = scan_ladders._quality_gate_for_broker(args, monitors)
    assert result.go is False
    assert "ev_honesty_stale" in result.reasons
    ev_details = result.details.get("ev_honesty")
    assert ev_details is not None
    assert pytest.approx(ev_details.get("max_delta_cents"), rel=0, abs=1e-9) == 25.0
    assert pytest.approx(ev_details.get("threshold_cents"), rel=0, abs=1e-9) == 10.0
    assert ev_details.get("book_latency_ms") == monitors["book_latency_ms"]
