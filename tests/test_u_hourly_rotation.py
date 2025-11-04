from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from kalshi_alpha.brokers.kalshi.base import BrokerOrder
from kalshi_alpha.core.kalshi_api import KalshiPublicClient
from kalshi_alpha.core.risk import PALGuard, PALPolicy
from kalshi_alpha.exec.runners import scan_ladders


def _copy_index_calibration(proc_root: Path, symbol: str, horizon: str) -> None:
    fixture = Path("tests/fixtures/index") / symbol / horizon / "params.json"
    target = proc_root / "calib" / "index" / symbol / horizon / "params.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(fixture.read_bytes())


def test_u_series_roll_decision_flags_next_hour() -> None:
    et_zone = ZoneInfo("America/New_York")
    now_et = datetime(2025, 11, 3, 11, 55, tzinfo=et_zone)
    decision = scan_ladders._u_series_roll_decision(now_et.astimezone(UTC))
    assert decision["rolled"] is True
    assert decision["target_hour"] == (now_et.hour + 1) % 24
    assert decision["cancel_required"] is False


def test_u_series_roll_decision_triggers_cancel() -> None:
    et_zone = ZoneInfo("America/New_York")
    now_et = datetime(2025, 11, 3, 12, 59, 59, tzinfo=et_zone)
    decision = scan_ladders._u_series_roll_decision(now_et.astimezone(UTC))
    assert decision["rolled"] is True
    assert decision["cancel_required"] is True


def test_scan_series_targets_next_hour_event(
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
) -> None:
    _, proc_root = isolated_data_roots
    _copy_index_calibration(proc_root, "spx", "hourly")
    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    pal_guard = PALGuard(PALPolicy(series="INXU", default_max_loss=10_000.0))
    now_et = datetime(2025, 11, 3, 12, 55, tzinfo=ZoneInfo("America/New_York"))
    result = scan_ladders.scan_series(
        series="INXU",
        client=client,
        min_ev=0.0,
        contracts=1,
        pal_guard=pal_guard,
        driver_fixtures=fixtures_root / "drivers",
        strategy_name="auto",
        maker_only=True,
        allow_tails=False,
        risk_manager=None,
        max_var=None,
        offline=True,
        sizing_mode="kelly",
        kelly_cap=0.25,
        now_override=now_et.astimezone(UTC),
    )
    assert result.events, "expected events to be selected"
    event_ids = {event.id for event in result.events}
    assert event_ids == {"EVT_INXU_H1300"}
    assert result.roll_info is not None
    assert result.roll_info["rolled"] is True
    assert result.roll_info["target_hour_label"] == "H1300"


def test_main_emits_roll_log_and_cancel(
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, proc_root = isolated_data_roots
    _copy_index_calibration(proc_root, "spx", "hourly")

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            base = datetime(2025, 11, 3, 12, 59, 59, tzinfo=ZoneInfo("America/New_York"))
            return base.astimezone(tz) if tz is not None else base

    monkeypatch.setattr(scan_ladders, "datetime", FixedDateTime)
    fixtures_path = fixtures_root.resolve()
    pal_policy_path = Path("configs/pal_policy.example.yaml").resolve()
    quality_example = Path("configs/quality_gates.example.yaml").resolve()
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    (configs_dir / "quality_gates.example.yaml").write_text(
        quality_example.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    scan_ladders.main(
        [
            "--series",
            "INXU",
            "--fixtures-root",
            str(fixtures_path),
            "--offline",
            "--min-ev",
            "0.0",
            "--contracts",
            "1",
            "--maker-only",
            "--pal-policy",
            str(pal_policy_path),
        ]
    )
    captured = capsys.readouterr().out
    assert "Cancel-all requested for INXU ahead of H1300" in captured
    assert "[u-roll] H1200→H1300" in captured

    orders_state = proc_root / "state" / "orders.json"
    payload = json.loads(orders_state.read_text(encoding="utf-8"))
    cancel_entry = payload.get("cancel_all")
    assert cancel_entry is not None


def test_u_series_guardrail_blocks_when_next_hour_missing(
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, proc_root = isolated_data_roots
    _copy_index_calibration(proc_root, "spx", "hourly")
    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    original_get_events = client.get_events

    def _only_current(event_series_id: str):
        events = original_get_events(event_series_id)
        return events[:1]

    monkeypatch.setattr(client, "get_events", _only_current)
    pal_guard = PALGuard(PALPolicy(series="INXU", default_max_loss=10_000.0))
    now_et = datetime(2025, 11, 3, 12, 58, tzinfo=ZoneInfo("America/New_York"))
    result = scan_ladders.scan_series(
        series="INXU",
        client=client,
        min_ev=0.0,
        contracts=1,
        pal_guard=pal_guard,
        driver_fixtures=fixtures_root / "drivers",
        strategy_name="auto",
        maker_only=True,
        allow_tails=False,
        risk_manager=None,
        max_var=None,
        offline=True,
        sizing_mode="kelly",
        kelly_cap=0.25,
        now_override=now_et.astimezone(UTC),
    )
    assert not result.proposals
    guardrails = tuple(result.monitors.get("u_series_guardrails", ()))
    assert "next_hour_missing" in guardrails
    assert result.monitors.get("u_series_guardrail_blocked") is True


def test_u_series_guardrail_pre_boundary_cooldown(
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
) -> None:
    _, proc_root = isolated_data_roots
    _copy_index_calibration(proc_root, "spx", "hourly")
    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    pal_guard = PALGuard(PALPolicy(series="INXU", default_max_loss=10_000.0))
    now_et = datetime(2025, 11, 3, 12, 59, 55, tzinfo=ZoneInfo("America/New_York"))
    result = scan_ladders.scan_series(
        series="INXU",
        client=client,
        min_ev=0.0,
        contracts=1,
        pal_guard=pal_guard,
        driver_fixtures=fixtures_root / "drivers",
        strategy_name="auto",
        maker_only=True,
        allow_tails=False,
        risk_manager=None,
        max_var=None,
        offline=True,
        sizing_mode="kelly",
        kelly_cap=0.25,
        now_override=now_et.astimezone(UTC),
    )
    assert not result.proposals
    guardrails = tuple(result.monitors.get("u_series_guardrails", ()))
    assert "pre_boundary_cooldown" in guardrails


def test_u_series_guardrail_post_roll_delay(
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
) -> None:
    _, proc_root = isolated_data_roots
    _copy_index_calibration(proc_root, "spx", "hourly")
    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    pal_guard = PALGuard(PALPolicy(series="INXU", default_max_loss=10_000.0))
    now_et = datetime(2025, 11, 3, 13, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    result = scan_ladders.scan_series(
        series="INXU",
        client=client,
        min_ev=0.0,
        contracts=1,
        pal_guard=pal_guard,
        driver_fixtures=fixtures_root / "drivers",
        strategy_name="auto",
        maker_only=True,
        allow_tails=False,
        risk_manager=None,
        max_var=None,
        offline=True,
        sizing_mode="kelly",
        kelly_cap=0.25,
        now_override=now_et.astimezone(UTC),
    )
    assert not result.proposals
    guardrails = tuple(result.monitors.get("u_series_guardrails", ()))
    assert "post_roll_delay" in guardrails


def test_u_series_guardrail_cancel_ack_pending(
    fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
) -> None:
    _, proc_root = isolated_data_roots
    _copy_index_calibration(proc_root, "spx", "hourly")
    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    pal_guard = PALGuard(PALPolicy(series="INXU", default_max_loss=10_000.0))
    state = scan_ladders.OutstandingOrdersState.load()
    state.record_submission(
        "dry",
        [
            BrokerOrder(
                idempotency_key="test-order",
                market_id="MKT_INXU_H1200_A",
                strike=5000.0,
                side="YES",
                price=0.45,
                contracts=1,
                probability=0.45,
                metadata=None,
            )
        ],
    )
    now_et = datetime(2025, 11, 3, 12, 59, 59, tzinfo=ZoneInfo("America/New_York"))
    result = scan_ladders.scan_series(
        series="INXU",
        client=client,
        min_ev=0.0,
        contracts=1,
        pal_guard=pal_guard,
        driver_fixtures=fixtures_root / "drivers",
        strategy_name="auto",
        maker_only=True,
        allow_tails=False,
        risk_manager=None,
        max_var=None,
        offline=True,
        sizing_mode="kelly",
        kelly_cap=0.25,
        now_override=now_et.astimezone(UTC),
    )
    assert not result.proposals
    guardrails = tuple(result.monitors.get("u_series_guardrails", ()))
    assert "cancel_ack_pending" in guardrails
