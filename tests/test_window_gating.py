from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from kalshi_alpha.core.gates import QualityGateResult
from kalshi_alpha.core.risk import PALPolicy
from kalshi_alpha.exec.pipelines import daily


class _FixedDateTime(datetime):
    _now = datetime(2025, 10, 30, 18, 0, tzinfo=UTC)  # 14:00 ET, before TenY window

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        if tz is None:
            return cls._now
        return cls._now.astimezone(tz)


def _minimal_args(
    mode: str,
    fixtures_root: Path,
    offline_root: Path,
    *,
    force_run: bool = False,
    report: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        mode=mode,
        offline=True,
        online=False,
        force_run=force_run,
        report=report,
        paper_ledger=False,
        driver_fixtures=str(offline_root),
        scanner_fixtures=str(fixtures_root),
        kelly_cap=0.25,
        force_refresh=False,
        allow_no_go=False,
        daily_loss_cap=None,
        weekly_loss_cap=None,
        fill_alpha="0.6",
        slippage_mode="top",
        impact_cap=0.02,
        broker="dry",
        model_version="v15",
        kill_switch_file=None,
        when=None,
        window_et=None,
        mispricing_only=False,
        max_legs=4,
        prob_sum_gap_threshold=0.0,
        clear_dry_orders_start=False,
        snap_to_window="off",
    )


def test_out_of_window_without_force_run_skips_scan(
    tmp_path: Path,
    fixtures_root: Path,
    offline_fixtures_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_data_roots: tuple[Path, Path],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(daily, "datetime", _FixedDateTime)
    raw_root, proc_root = isolated_data_roots
    proc_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(daily, "PROC_ROOT", proc_root)
    monkeypatch.setattr(daily, "RAW_ROOT", raw_root)

    scan_called = {"value": False}

    monkeypatch.setattr(daily, "run_ingest", lambda *_: None)
    monkeypatch.setattr(daily, "run_calibrations", lambda *_, **__: None)
    monkeypatch.setattr(
        daily,
        "run_quality_gate_step",
        lambda *_, **__: QualityGateResult(go=True, reasons=[], details={}),
    )

    def _fail_run_scan(*_args, **_kwargs):
        scan_called["value"] = True

    monkeypatch.setattr(daily, "run_scan", _fail_run_scan)

    args = _minimal_args("teny_close", fixtures_root, offline_fixtures_root, report=True)
    daily.run_mode("teny_close", args)

    assert scan_called["value"] is False
    assert not (Path("exec") / "proposals").exists()
    assert not (Path("reports")).exists()

    logs = list(proc_root.joinpath("logs").rglob("*_teny_close.json"))
    assert logs, "Expected a log file to be written"
    payload = json.loads(logs[-1].read_text(encoding="utf-8"))
    assert payload.get("scan_notes", {}).get("teny_close") == "outside window"


def test_force_run_dry_always_produces_report(
    tmp_path: Path,
    fixtures_root: Path,
    offline_fixtures_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_data_roots: tuple[Path, Path],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(daily, "datetime", _FixedDateTime)
    monkeypatch.setattr(
        daily,
        "run_quality_gate_step",
        lambda *_, **__: QualityGateResult(go=True, reasons=[], details={}),
    )
    raw_root, proc_root = isolated_data_roots
    proc_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(daily, "PROC_ROOT", proc_root)
    monkeypatch.setattr(daily, "RAW_ROOT", raw_root)
    monkeypatch.setattr(
        daily.PALPolicy,
        "from_yaml",
        classmethod(lambda cls, path: PALPolicy(series="TNEY", default_max_loss=1_000.0, per_strike={})),
    )

    args = _minimal_args("teny_close", fixtures_root, offline_fixtures_root, force_run=True, report=False)
    daily.run_mode("teny_close", args)

    report_dir = Path("reports") / "TNEY"
    reports = sorted(report_dir.glob("*.md"))
    assert reports, "Expected force-run to generate a report"
    contents = reports[-1].read_text(encoding="utf-8")
    assert "FORCE-RUN (DRY)" in contents

    proposals_dir = Path("exec") / "proposals" / "TNEY"
    assert proposals_dir.exists()
    assert any(proposals_dir.iterdir()), "Expected proposals to be written"

    logs = list((Path("data") / "proc" / "logs").rglob("*_teny_close.json"))
    payload = json.loads(logs[-1].read_text(encoding="utf-8"))
    assert payload.get("scan_notes", {}).get("teny_close") == "force_run_dry"
