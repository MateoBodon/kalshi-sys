from __future__ import annotations

from argparse import Namespace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from kalshi_alpha.exec.pipelines import daily


class _DummyOutcome:
    def __init__(self, series: str) -> None:
        self.proposals: list[object] = []
        self.monitors: dict[str, object] = {}
        self.cdf_diffs: list[object] = []
        self.series = series
        self.events: list[object] = []
        self.markets: list[object] = []
        self.mispricings: list[dict[str, object]] = []
        self.model_metadata: dict[str, object] = {}


def test_force_run_allows_scan_and_marks_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kalshi_alpha.datastore import paths as datastore_paths

    monkeypatch.chdir(tmp_path)
    proc_root = tmp_path / "data" / "proc"
    raw_root = tmp_path / "data" / "raw"
    monkeypatch.setattr(datastore_paths, "PROC_ROOT", proc_root)
    monkeypatch.setattr(datastore_paths, "RAW_ROOT", raw_root)
    monkeypatch.setattr(daily, "PROC_ROOT", proc_root)
    monkeypatch.setattr(daily, "RAW_ROOT", raw_root)

    monkeypatch.setattr(daily, "run_ingest", lambda *_, **__: None)

    def fake_calibration(args, log, heartbeat_cb=None):
        if heartbeat_cb:
            heartbeat_cb("post_calibrate", {"calibration": "dummy"})

    monkeypatch.setattr(daily, "run_calibrations", fake_calibration)

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

    class DummyClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def get_orderbook(self, _market_id: str) -> object:  # pragma: no cover - not used
            return SimpleNamespace()

    monkeypatch.setattr(daily, "KalshiPublicClient", DummyClient)
    monkeypatch.setattr(daily.PALPolicy, "from_yaml", staticmethod(lambda *_: object()))
    monkeypatch.setattr(daily, "PALGuard", lambda *_: object())

    scan_labels: list[str] = []

    def fake_scan_series(series: str, **_kwargs):
        scan_labels.append(current_label["value"])
        return _DummyOutcome(series)

    monkeypatch.setattr(daily, "scan_series", fake_scan_series)
    monkeypatch.setattr(daily, "_compute_exposure_summary", lambda *_: {})
    monkeypatch.setattr(daily, "_write_cdf_diffs", lambda *_: tmp_path / "cdf.json")
    monkeypatch.setattr(daily, "_archive_and_replay", lambda **_: None)
    monkeypatch.setattr(daily, "write_proposals", lambda **_: tmp_path / "proposals.json")

    window = SimpleNamespace(
        notes=[],
        to_dict=lambda _now: {"mode": "pre_cpi"},
        scan_allowed=lambda _now: False,
    )
    monkeypatch.setattr(daily, "resolve_run_window", lambda **_: window)

    report_records: dict[str, dict[str, object] | None] = {}
    report_contents: dict[str, str] = {}
    original_report = daily.write_markdown_report
    current_label = {"value": "init"}

    def record_report(**kwargs):
        label = current_label["value"]
        report_records[label] = kwargs.get("pilot_metadata")
        path = original_report(**kwargs)
        report_contents[label] = path.read_text(encoding="utf-8")
        return path

    monkeypatch.setattr(daily, "write_markdown_report", record_report)

    args_template = dict(
        offline=True,
        online=False,
        driver_fixtures=str(tmp_path / "fixtures"),
        scanner_fixtures=str(tmp_path / "scanner"),
        kelly_cap=0.1,
        fill_alpha="0.6",
        slippage_mode="top",
        impact_cap=0.02,
        report=True,
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

    def run(force_run: bool, label: str) -> None:
        current_label["value"] = label
        args = Namespace(force_run=force_run, **args_template)
        daily.run_mode("pre_cpi", args)

    run(force_run=False, label="no_force")
    assert "no_force" not in report_records
    assert all(entry != "no_force" for entry in scan_labels)

    run(force_run=True, label="force")
    assert "force" in report_records
    assert "force" in report_contents
    assert "FORCE-RUN (DRY)" in report_contents["force"]
    metadata = report_records["force"] or {}
    assert metadata.get("force_run") is True
    assert scan_labels.count("force") == 1
