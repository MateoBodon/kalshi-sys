from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from kalshi_alpha.core.pricing import Liquidity
from kalshi_alpha.exec.ledger import FillRecord, PaperLedger
from kalshi_alpha.exec.monitors.summary import MonitorArtifactsSummary
from kalshi_alpha.exec.pilot import (
    PilotConfig,
    PilotSession,
    build_pilot_session_payload,
    write_pilot_session_artifact,
)
from kalshi_alpha.exec.runners.scan_ladders import Proposal, ScanOutcome


def _sample_proposal() -> Proposal:
    return Proposal(
        market_id="M1",
        market_ticker="CPI-TEST",
        strike=270.0,
        side="YES",
        contracts=1,
        maker_ev=0.25,
        taker_ev=-0.20,
        maker_ev_per_contract=0.0005,
        taker_ev_per_contract=-0.0004,
        strategy_probability=0.56,
        market_yes_price=0.44,
        survival_market=0.53,
        survival_strategy=0.56,
        max_loss=75.0,
        strategy="CPI",
        series="CPI",
        metadata={},
    )


def test_build_pilot_session_payload_with_metrics() -> None:
    session = PilotSession(
        session_id="pilot-cpi-20251102T120000Z-abc",
        series="CPI",
        started_at=datetime(2025, 11, 2, 12, 0, tzinfo=UTC),
        config=PilotConfig(),
    )
    ledger = PaperLedger()
    proposal = _sample_proposal()
    record = FillRecord(
        proposal=proposal,
        fill_price=0.44,
        expected_value=0.18,
        liquidity=Liquidity.MAKER,
        slippage=0.0,
        expected_contracts=1,
        expected_fills=1,
        fill_ratio=1.0,
        slippage_mode="top",
        impact_cap=0.0,
        fees_maker=0.01,
        pnl_simulated=0.18,
        alpha_row=1.0,
        size_throttled=False,
        t_fill_ms=0.0,
        size_partial=0,
        slippage_ticks=0.0,
        ev_expected_bps=15.0,
        ev_realized_bps=19.0,
        fees_bps=2.0,
        fill_ratio_realized=1.0,
    )
    ledger.record(record)

    monitor_summary = MonitorArtifactsSummary(
        max_age_minutes=5.0,
        latest_generated_at=datetime(2025, 11, 2, 11, 55, tzinfo=UTC),
        file_count=3,
        statuses={"ev_seq_guard": "OK", "kill_switch": "OK"},
        metrics={"fill_vs_alpha": {"worst_gap_pp": -4.2}},
        alerts_recent={"kill_switch"},
    )
    monitors = {
        "fill_realism_median": 0.08,
        "ev_honesty_table": [
            {
                "market_ticker": "CPI-TEST",
                "market_id": "M1",
                "strike": 270.0,
                "side": "YES",
                "delta": 0.2,
                "maker_ev_per_contract_original": 0.5,
                "maker_ev_per_contract_replay": 0.3,
                "maker_ev_per_contract_proposal": 0.4,
            }
        ],
        "ev_honesty_threshold_cents": 10.0,
        "ev_honesty_max_delta_cents": 20.0,
        "ev_honesty_no_go": True,
    }
    broker_status = {"mode": "live", "orders_recorded": 1}

    payload = build_pilot_session_payload(
        session=session,
        ledger=ledger,
        monitors=monitors,
        monitor_summary=monitor_summary,
        broker_status=broker_status,
        generated_at=datetime(2025, 11, 2, 12, 5, tzinfo=UTC),
    )

    assert payload["session_id"] == session.session_id
    assert payload["family"] == session.series
    assert payload["n_trades"] == 1
    assert payload["mean_delta_bps_after_fees"] == 4.0
    assert payload["t_stat"] is None
    assert payload["cusum_state"] == "OK"
    assert payload["cusum_status"] == "OK"
    assert payload["fill_realism_gap"] == -4.2
    assert payload["alerts_summary"]["recent_alerts"] == ["kill_switch"]
    assert payload["broker_mode"] == "live"
    assert payload["ev_honesty_threshold_cents"] == 10.0
    assert payload["ev_honesty_no_go"] is True
    assert payload["ev_honesty_table"][0]["market_ticker"] == "CPI-TEST"


def test_write_pilot_session_artifact(tmp_path: Path) -> None:
    session = PilotSession(
        session_id="pilot-cpi-1",
        series="CPI",
        started_at=datetime(2025, 11, 2, 10, 0, tzinfo=UTC),
        config=PilotConfig(),
    )
    ledger_stub = SimpleNamespace(records=[])
    summary = MonitorArtifactsSummary(None, None, 0, {}, {}, set())

    path = write_pilot_session_artifact(
        session=session,
        ledger=ledger_stub,  # type: ignore[arg-type]
        monitors={},
        monitor_summary=summary,
        broker_status={"mode": "dry", "orders_recorded": 0},
        artifacts_dir=tmp_path / "reports" / "_artifacts",
        generated_at=datetime(2025, 11, 2, 10, 5, tzinfo=UTC),
    )

    assert path.exists()
    payload_text = path.read_text(encoding="utf-8")
    payload = json.loads(payload_text)
    assert payload["session_id"] == "pilot-cpi-1"
    assert payload["family"] == "CPI"
    assert payload["broker_mode"] == "dry"
    assert "alerts_summary" in payload

def test_scan_ladders_invokes_pilot_session_writer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from kalshi_alpha.exec.runners import scan_ladders

    monkeypatch.chdir(tmp_path)
    fixtures_root = tmp_path / "fixtures"
    fixtures_root.mkdir()
    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(
        """
        pilot:
          require_live_broker: false
          require_acknowledgement: false
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(scan_ladders, "_build_client", lambda *a, **k: object())
    monkeypatch.setattr(scan_ladders, "_build_pal_guard", lambda *_a, **_k: object())
    monkeypatch.setattr(scan_ladders, "_build_risk_manager", lambda *_a, **_k: None)
    monkeypatch.setattr(scan_ladders, "_clear_dry_orders_start", lambda **kwargs: {})
    monkeypatch.setattr(scan_ladders, "_maybe_simulate_ledger", lambda *a, **k: None)
    monkeypatch.setattr(scan_ladders, "_compute_exposure_summary", lambda *a, **k: {})
    monkeypatch.setattr(scan_ladders, "_write_cdf_diffs", lambda *a, **k: None)
    monkeypatch.setattr(scan_ladders, "_attach_series_metadata", lambda **kwargs: None)

    def _fake_write_proposals(*_args, **_kwargs):
        path = tmp_path / "proposals.json"
        path.write_text("[]", encoding="utf-8")
        return path

    monkeypatch.setattr(scan_ladders, "write_proposals", _fake_write_proposals)
    monkeypatch.setattr(scan_ladders, "_archive_and_replay", lambda **kwargs: (None, None))
    monkeypatch.setattr(scan_ladders.drawdown, "record_pnl", lambda *_: None)

    monitor_snapshot = MonitorArtifactsSummary(0.0, None, 0, {}, {}, set())
    monkeypatch.setattr(scan_ladders, "summarize_monitor_artifacts", lambda *a, **k: monitor_snapshot)

    class _StubOutstanding:
        @classmethod
        def load(cls, *_args, **_kwargs):
            return cls()

        def summary(self) -> dict[str, int]:
            return {}

        def outstanding_for(self, *_args, **_kwargs):  # pragma: no cover - defensive
            return {}

        def remove(self, *_args, **_kwargs):  # pragma: no cover - defensive
            return []

        def clear_cancel_all(self) -> None:  # pragma: no cover - defensive
            return None

    monkeypatch.setattr(scan_ladders, "OutstandingOrdersState", _StubOutstanding)

    outcome = ScanOutcome(
        proposals=[],
        monitors={"fill_realism_median": 0.07},
        cdf_diffs=[],
        mispricings=[],
        series=None,
        events=[],
        markets=[],
        model_metadata={},
        books_at_scan={},
        book_snapshot_started_at=datetime(2025, 11, 2, 12, 0, tzinfo=UTC),
        book_snapshot_completed_at=datetime(2025, 11, 2, 12, 1, tzinfo=UTC),
    )
    monkeypatch.setattr(scan_ladders, "scan_series", lambda **kwargs: outcome)

    captured: dict[str, object] = {}

    def _capture_writer(**kwargs):
        captured["payload"] = kwargs
        artifact = tmp_path / "reports" / "_artifacts" / "pilot_session.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}", encoding="utf-8")
        return artifact

    monkeypatch.setattr(scan_ladders, "write_pilot_session_artifact", _capture_writer)

    from kalshi_alpha.datastore import paths as datastore_paths

    monkeypatch.setattr(datastore_paths, "PROC_ROOT", tmp_path / "data" / "proc")
    monkeypatch.setattr(datastore_paths, "RAW_ROOT", tmp_path / "data" / "raw")

    scan_ladders.main(
        [
            "--series",
            "CPI",
            "--pilot",
            "--pilot-config",
            str(config_path),
            "--fixtures-root",
            str(fixtures_root),
            "--broker",
            "dry",
        ]
    )

    assert "payload" in captured
    payload = captured["payload"]
    assert payload["session"].series == "CPI"
    assert payload["ledger"] is None
    assert payload["monitors"] == outcome.monitors
    assert payload["monitor_summary"] is monitor_snapshot

def test_pilot_mode_respects_kill_switch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from kalshi_alpha.exec.runners import scan_ladders

    monkeypatch.chdir(tmp_path)
    fixtures_root = tmp_path / "fixtures"
    fixtures_root.mkdir()
    kill_switch = tmp_path / "kill_switch"
    kill_switch.write_text("halt", encoding="utf-8")

    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(
        """
        pilot:
          require_live_broker: false
          require_acknowledgement: false
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(scan_ladders, "_build_client", lambda *a, **k: object())
    monkeypatch.setattr(scan_ladders, "_build_pal_guard", lambda *_a, **_k: object())
    monkeypatch.setattr(scan_ladders, "_build_risk_manager", lambda *_a, **_k: None)
    monkeypatch.setattr(scan_ladders, "_clear_dry_orders_start", lambda **kwargs: {})
    monkeypatch.setattr(scan_ladders, "_maybe_simulate_ledger", lambda *a, **k: None)
    monkeypatch.setattr(scan_ladders, "_compute_exposure_summary", lambda *a, **k: {})
    monkeypatch.setattr(scan_ladders, "_write_cdf_diffs", lambda *a, **k: None)
    monkeypatch.setattr(scan_ladders, "_attach_series_metadata", lambda **kwargs: None)

    def _fake_write_proposals(*_args, **_kwargs):
        path = tmp_path / "proposals.json"
        path.write_text("[]", encoding="utf-8")
        return path

    monkeypatch.setattr(scan_ladders, "write_proposals", _fake_write_proposals)
    monkeypatch.setattr(scan_ladders, "_archive_and_replay", lambda **kwargs: (None, None))
    monkeypatch.setattr(scan_ladders.drawdown, "record_pnl", lambda *_: None)

    proposals = [_sample_proposal()]
    outcome = ScanOutcome(
        proposals=proposals,
        monitors={"fill_realism_median": 0.06},
        cdf_diffs=[],
        mispricings=[],
        series=None,
        events=[],
        markets=[],
        model_metadata={},
        books_at_scan={},
        book_snapshot_started_at=datetime(2025, 11, 2, 12, 0, tzinfo=UTC),
        book_snapshot_completed_at=datetime(2025, 11, 2, 12, 1, tzinfo=UTC),
    )
    monkeypatch.setattr(scan_ladders, "scan_series", lambda **kwargs: outcome)

    monitor_snapshot = MonitorArtifactsSummary(0.0, None, 0, {}, {}, set())
    monkeypatch.setattr(scan_ladders, "summarize_monitor_artifacts", lambda *a, **k: monitor_snapshot)

    class _RecordingOutstanding:
        instances: list[_RecordingOutstanding] = []

        def __init__(self) -> None:
            self.cancel_calls: list[tuple[str, tuple[str, ...]]] = []

        @classmethod
        def load(cls, *_args, **_kwargs) -> _RecordingOutstanding:
            instance = cls()
            cls.instances.append(instance)
            return instance

        def summary(self) -> dict[str, int]:  # pragma: no cover - defensive
            return {}

        def mark_cancel_all(self, reason: str, modes: list[str]) -> dict[str, object]:
            self.cancel_calls.append((reason, tuple(modes)))
            return {"reason": reason, "modes": modes}

        def record_submission(self, *_args, **_kwargs) -> None:  # pragma: no cover - defensive
            return None

    monkeypatch.setattr(scan_ladders, "OutstandingOrdersState", _RecordingOutstanding)

    captured: dict[str, object] = {}

    def _capture_writer(**kwargs):
        captured["payload"] = kwargs
        artifact = tmp_path / "reports" / "_artifacts" / "pilot_session.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}", encoding="utf-8")
        return artifact

    monkeypatch.setattr(scan_ladders, "write_pilot_session_artifact", _capture_writer)

    def _reject_broker(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("broker should not arm")

    monkeypatch.setattr(scan_ladders, "create_broker", _reject_broker)

    monkeypatch.setattr(
        scan_ladders,
        "_quality_gate_for_broker",
        lambda args, monitors, **_: scan_ladders.QualityGateResult(
            go=False,
            reasons=["kill_switch"],
            details={},
        ),
    )

    from kalshi_alpha.datastore import paths as datastore_paths

    monkeypatch.setattr(datastore_paths, "PROC_ROOT", tmp_path / "data" / "proc")
    monkeypatch.setattr(datastore_paths, "RAW_ROOT", tmp_path / "data" / "raw")

    scan_ladders.main(
        [
            "--series",
            "CPI",
            "--pilot",
            "--pilot-config",
            str(config_path),
            "--fixtures-root",
            str(fixtures_root),
            "--broker",
            "dry",
            "--kill-switch-file",
            str(kill_switch),
        ]
    )

    assert "payload" in captured
    payload = captured["payload"]
    broker_status = payload["broker_status"]
    assert isinstance(broker_status, dict)
    assert broker_status.get("error", "").startswith("Kill switch engaged")
    assert any(instance.cancel_calls for instance in _RecordingOutstanding.instances)

def test_pilot_mode_respects_freeze_alert(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from kalshi_alpha.exec.runners import scan_ladders

    monkeypatch.chdir(tmp_path)
    fixtures_root = tmp_path / "fixtures"
    fixtures_root.mkdir()

    config_path = tmp_path / "pilot.yaml"
    config_path.write_text(
        """
        pilot:
          require_live_broker: false
          require_acknowledgement: false
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(scan_ladders, "_build_client", lambda *a, **k: object())
    monkeypatch.setattr(scan_ladders, "_build_pal_guard", lambda *_a, **_k: object())
    monkeypatch.setattr(scan_ladders, "_build_risk_manager", lambda *_a, **_k: None)
    monkeypatch.setattr(scan_ladders, "_clear_dry_orders_start", lambda **kwargs: {})
    monkeypatch.setattr(scan_ladders, "_maybe_simulate_ledger", lambda *a, **k: None)
    monkeypatch.setattr(scan_ladders, "_compute_exposure_summary", lambda *a, **k: {})
    monkeypatch.setattr(scan_ladders, "_write_cdf_diffs", lambda *a, **k: None)
    monkeypatch.setattr(scan_ladders, "_attach_series_metadata", lambda **kwargs: None)
    monkeypatch.setattr(scan_ladders.drawdown, "record_pnl", lambda *_: None)

    def _ok_drawdown(*_args: object, **_kwargs: object) -> scan_ladders.drawdown.DrawdownStatus:
        return scan_ladders.drawdown.DrawdownStatus(True, [], {})

    monkeypatch.setattr(scan_ladders.drawdown, "check_limits", _ok_drawdown)

    def _fake_write_proposals(*_args, **_kwargs):
        path = tmp_path / "proposals.json"
        path.write_text("[]", encoding="utf-8")
        return path

    monkeypatch.setattr(scan_ladders, "write_proposals", _fake_write_proposals)
    monkeypatch.setattr(scan_ladders, "_archive_and_replay", lambda **kwargs: (None, None))

    proposals = [_sample_proposal()]
    outcome = ScanOutcome(
        proposals=proposals,
        monitors={"fill_realism_median": 0.05},
        cdf_diffs=[],
        mispricings=[],
        series=None,
        events=[],
        markets=[],
        model_metadata={},
        books_at_scan={},
        book_snapshot_started_at=datetime(2025, 11, 2, 12, 0, tzinfo=UTC),
        book_snapshot_completed_at=datetime(2025, 11, 2, 12, 1, tzinfo=UTC),
    )
    monkeypatch.setattr(scan_ladders, "scan_series", lambda **kwargs: outcome)

    monitor_snapshot = MonitorArtifactsSummary(
        max_age_minutes=2.0,
        latest_generated_at=datetime(2025, 11, 2, 11, 58, tzinfo=UTC),
        file_count=4,
        statuses={"freeze_window": "ALERT"},
        metrics={},
        alerts_recent={"freeze_window"},
    )
    monkeypatch.setattr(scan_ladders, "summarize_monitor_artifacts", lambda *a, **k: monitor_snapshot)

    class _RecordingOutstanding:
        instances: list[_RecordingOutstanding] = []

        def __init__(self) -> None:
            self.cancel_calls: list[tuple[str, tuple[str, ...]]] = []

        @classmethod
        def load(cls, *_args, **_kwargs) -> _RecordingOutstanding:
            instance = cls()
            cls.instances.append(instance)
            return instance

        def summary(self) -> dict[str, int]:  # pragma: no cover - defensive
            return {}

        def mark_cancel_all(self, reason: str, modes: list[str]) -> dict[str, object]:
            self.cancel_calls.append((reason, tuple(modes)))
            return {"reason": reason, "modes": modes}

        def record_submission(self, *_args, **_kwargs) -> None:  # pragma: no cover - defensive
            return None

    monkeypatch.setattr(scan_ladders, "OutstandingOrdersState", _RecordingOutstanding)

    captured: dict[str, object] = {}

    def _capture_writer(**kwargs):
        captured["payload"] = kwargs
        artifact = tmp_path / "reports" / "_artifacts" / "pilot_session.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}", encoding="utf-8")
        return artifact

    monkeypatch.setattr(scan_ladders, "write_pilot_session_artifact", _capture_writer)

    from kalshi_alpha.exec.gate_utils import QualityGateResult

    def _fake_quality_config(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {}

    def _fake_quality_result(**_kwargs: object) -> QualityGateResult:
        return QualityGateResult(go=True, reasons=[], details={})

    monkeypatch.setattr(scan_ladders, "load_quality_gate_config", _fake_quality_config)
    monkeypatch.setattr(scan_ladders, "run_quality_gates", _fake_quality_result)

    from kalshi_alpha.datastore import paths as datastore_paths

    monkeypatch.setattr(datastore_paths, "PROC_ROOT", tmp_path / "data" / "proc")
    monkeypatch.setattr(datastore_paths, "RAW_ROOT", tmp_path / "data" / "raw")

    scan_ladders.main(
        [
            "--series",
            "CPI",
            "--pilot",
            "--pilot-config",
            str(config_path),
            "--fixtures-root",
            str(fixtures_root),
            "--broker",
            "dry",
        ]
    )

    assert "payload" in captured
    payload = captured["payload"]
    broker_status = payload["broker_status"]
    assert isinstance(broker_status, dict)
    assert "Quality gates returned NO-GO" in broker_status.get("error", "")
    assert any(instance.cancel_calls for instance in _RecordingOutstanding.instances)
