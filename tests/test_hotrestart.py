from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from kalshi_alpha.sched.hotrestart import HotRestartManager, HotRestartSnapshot, summarize_orders_state
from kalshi_alpha.sched.windows import TradingWindow


def _window(label: str) -> TradingWindow:
    now = datetime.now(tz=UTC)
    et = ZoneInfo("America/New_York")
    return TradingWindow(
        label=label,
        target_type="hourly",
        series=("INXU",),
        target_et=now.astimezone(et),
        start_et=(now - timedelta(minutes=5)).astimezone(et),
        freeze_et=(now - timedelta(minutes=1)).astimezone(et),
        freshness_strict_et=(now - timedelta(minutes=1)).astimezone(et),
    )


def test_capture_and_restore_roundtrip(tmp_path) -> None:
    path = tmp_path / "hot.json"
    manager = HotRestartManager(path=path, max_age_seconds=10.0)
    summary = {"dry": 1, "live": 0, "dry_contracts": 10, "live_contracts": 0, "total": 1, "total_contracts": 10}
    snapshot = manager.capture(window=_window("hourly-1000"), outstanding_summary=summary, upcoming=[], now=datetime.now(tz=UTC))
    assert isinstance(snapshot, HotRestartSnapshot)
    restored = manager.restore()
    assert restored is not None
    assert restored.outstanding["dry"] == 1
    assert restored.active_window["label"] == "hourly-1000"


def test_restore_rejects_stale_snapshot(tmp_path) -> None:
    path = tmp_path / "hot.json"
    manager = HotRestartManager(path=path, max_age_seconds=0.1)
    stale_payload = {
        "captured_at": (datetime.now(tz=UTC) - timedelta(seconds=1)).isoformat(),
        "active_window": None,
        "upcoming_windows": [],
        "outstanding": {},
    }
    import json

    path.write_text(json.dumps(stale_payload), encoding="utf-8")
    assert manager.restore() is None


def test_summarize_orders_state_handles_missing_methods() -> None:
    class Dummy:
        def outstanding_for(self, mode: str):
            if mode == "dry":
                return {"a": {"contracts": 5}}
            return {}

    summary = summarize_orders_state(Dummy())
    assert summary["total_contracts"] == 5
    assert summary["dry"] == 1
