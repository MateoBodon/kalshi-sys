from __future__ import annotations

import json
from pathlib import Path

from kalshi_alpha.exec.reports import write_markdown_report
from kalshi_alpha.exec.runners import scan_ladders
from kalshi_alpha.exec.runners.scan_ladders import Proposal


def _sample_proposals() -> list[Proposal]:
    return [
        Proposal(
            market_id="M1",
            market_ticker="TENY_X",
            strike=100.0,
            side="YES",
            contracts=1,
            maker_ev=1.0,
            taker_ev=0.0,
            maker_ev_per_contract=0.5,
            taker_ev_per_contract=0.0,
            strategy_probability=0.55,
            market_yes_price=0.5,
            survival_market=0.45,
            survival_strategy=0.55,
            max_loss=0.5,
            strategy="TENY",
            series="TENY",
            metadata=None,
        )
    ]


def test_report_uses_artifact_go_status(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "reports" / "_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "go_no_go.json"
    artifact_path.write_text(json.dumps({"go": False, "reasons": ["test"]}), encoding="utf-8")

    report_dir = tmp_path / "reports" / "TENY"
    report_path = write_markdown_report(
        series="TENY",
        proposals=_sample_proposals(),
        ledger=None,
        output_dir=report_dir,
        monitors={},
        exposure_summary={},
        manifest_path=None,
        go_status=True,
        go_artifact_path=artifact_path,
    )

    first_line = report_path.read_text(encoding="utf-8").splitlines()[0]
    assert "**GO/NO-GO:** NO-GO" in first_line


def test_scan_report_matches_latest_gate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    artifacts_dir = Path("reports/_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    latest_artifact = artifacts_dir / "go_no_go_20250201.json"
    latest_artifact.write_text(
        json.dumps({"go": False, "reasons": ["test"], "timestamp": "2025-02-01T20:00:00Z"}),
        encoding="utf-8",
    )

    dummy_outcome = scan_ladders.ScanOutcome(proposals=[], monitors={})

    monkeypatch.setattr(scan_ladders, "apply_pilot_mode", lambda args: None)
    monkeypatch.setattr(scan_ladders, "scan_series", lambda **kwargs: dummy_outcome)
    monkeypatch.setattr(scan_ladders, "_build_client", lambda *a, **k: object())
    monkeypatch.setattr(scan_ladders, "_build_pal_guard", lambda *a, **k: object())
    monkeypatch.setattr(scan_ladders, "_build_risk_manager", lambda *a, **k: None)
    monkeypatch.setattr(scan_ladders, "_maybe_simulate_ledger", lambda *a, **k: None)
    monkeypatch.setattr(scan_ladders, "_write_cdf_diffs", lambda *a, **k: None)
    monkeypatch.setattr(scan_ladders, "_attach_series_metadata", lambda **kwargs: None)
    monkeypatch.setattr(scan_ladders, "_archive_and_replay", lambda **kwargs: (None, None))

    def _fake_write_proposals(*, series: str, output_dir: str | Path, **_: object) -> Path:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{series.lower()}_proposals.json"
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(scan_ladders, "write_proposals", _fake_write_proposals)
    monkeypatch.setattr(scan_ladders, "_clear_dry_orders_start", lambda **kwargs: {"dry": 0})
    monkeypatch.setattr(scan_ladders, "write_heartbeat", lambda **kwargs: None)
    monkeypatch.setattr(scan_ladders, "write_pilot_session_artifact", lambda **kwargs: None)

    class _DummyMonitorSummary:
        statuses: dict[str, str] = {}
        alerts_recent: list[str] = []
        max_age_minutes: int = 0
        file_count: int = 0
        metrics: dict[str, object] = {}

    monkeypatch.setattr(
        scan_ladders,
        "summarize_monitor_artifacts",
        lambda *a, **k: _DummyMonitorSummary(),
    )

    class _DummyOutstandingState:
        @classmethod
        def load(cls) -> _DummyOutstandingState:
            return cls()

        def summary(self) -> dict[str, int]:
            return {}

        def record_submission(self, *args: object, **kwargs: object) -> None:
            return None

        def mark_cancel_all(self, *args: object, **kwargs: object) -> None:
            return None

    monkeypatch.setattr(scan_ladders, "OutstandingOrdersState", _DummyOutstandingState)

    class _DummyDrawdownStatus:
        ok = True
        reasons: list[str] = []
        metrics: dict[str, float] = {}

    monkeypatch.setattr(scan_ladders.drawdown, "check_limits", lambda *a, **k: _DummyDrawdownStatus())
    monkeypatch.setattr(scan_ladders.drawdown, "record_pnl", lambda *a, **k: None)

    monkeypatch.setattr(
        scan_ladders,
        "_quality_gate_for_broker",
        lambda args, monitors: scan_ladders.QualityGateResult(go=False, reasons=["test"], details={}),
    )
    monkeypatch.setattr(scan_ladders, "write_go_no_go", lambda result: latest_artifact)
    monkeypatch.setattr(scan_ladders, "resolve_kill_switch_path", lambda *_: Path("kill_switch"))
    monkeypatch.setattr(scan_ladders, "kill_switch_engaged", lambda *_: False)
    monkeypatch.setattr(scan_ladders, "heartbeat_stale", lambda **kwargs: (False, None))

    scan_ladders.main(["--series", "TENY", "--report", "--offline", "--fixtures-root", str(tmp_path)])

    report_dir = Path("reports") / "TENY"
    report_files = sorted(report_dir.glob("*.md"))
    assert report_files, "expected scan_ladders to write a report"
    first_line = report_files[-1].read_text(encoding="utf-8").splitlines()[0]

    artifact_paths = sorted(artifacts_dir.glob("go_no_go*.json"))
    assert artifact_paths, "expected at least one go_no_go artifact"
    latest_payload = json.loads(artifact_paths[-1].read_text(encoding="utf-8"))
    assert latest_payload.get("go") is False
    assert "**GO/NO-GO:** GO" not in first_line
    assert "**GO/NO-GO:** NO-GO" in first_line
