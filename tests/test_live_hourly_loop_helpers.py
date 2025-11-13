from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

from zoneinfo import ZoneInfo

from scripts import live_hourly_loop


def test_close_phase_transitions() -> None:
    et = ZoneInfo("America/New_York")
    start = datetime(2025, 11, 13, 15, 50, tzinfo=et)
    target = datetime(2025, 11, 13, 16, 0, tzinfo=et)
    cancel_buffer = timedelta(seconds=2)
    dummy = SimpleNamespace(
        start_et=start,
        target_et=target,
        freeze_et=target - cancel_buffer,
        freshness_strict_et=max(start, target - timedelta(minutes=1)),
    )
    phase_idle = live_hourly_loop._close_phase(dummy, start - timedelta(minutes=1), None)
    assert phase_idle["state"] == "IDLE"
    phase_prep = live_hourly_loop._close_phase(dummy, start + timedelta(minutes=1), None)
    assert phase_prep["state"] == "PREP"
    phase_make = live_hourly_loop._close_phase(dummy, target - timedelta(minutes=2), None)
    assert phase_make["state"] == "MAKE"
    phase_freeze = live_hourly_loop._close_phase(dummy, target, None)
    assert phase_freeze["state"] == "FREEZE"


def test_freshness_breach_detects_series(tmp_path, monkeypatch):
    slo_path = tmp_path / "slo.json"
    slo_path.write_text(
        "\n".join(
            [
                '{"series": "INX", "freshness_p95_ms": 900}',
                '{"series": "NAS", "freshness_p95_ms": 200}',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(live_hourly_loop, "SLO_METRICS_PATH", slo_path)
    breaches = live_hourly_loop._freshness_breach(threshold_ms=700)
    assert breaches == {"INX": 900.0}
