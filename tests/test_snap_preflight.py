from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from kalshi_alpha.core.gates import QualityGateResult
from kalshi_alpha.exec.pipelines import daily, preflight


class _FixedDateTime(datetime):
    _now = datetime(2025, 10, 30, 18, 5, tzinfo=UTC)  # 14:05 ET

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        if tz is None:
            return cls._now
        return cls._now.astimezone(tz)


class _SequenceDateTime(datetime):
    _times = [
        datetime(2025, 10, 30, 18, 29, 30, tzinfo=UTC),  # 14:29:30 ET
        datetime(2025, 10, 30, 18, 30, 30, tzinfo=UTC),  # 14:30:30 ET
    ]

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        value = cls._times[0]
        if len(cls._times) > 1:
            cls._times = cls._times[1:]
        if tz is None:
            return value
        return value.astimezone(tz)


def _snap_args(fixtures_root: Path, offline_root: Path, snap: str) -> SimpleNamespace:
    return SimpleNamespace(
        mode="teny_close",
        offline=True,
        online=False,
        force_run=False,
        report=False,
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
        snap_to_window=snap,
    )


@pytest.mark.usefixtures("isolated_data_roots")
def test_snap_print_skips_execution(
    tmp_path: Path,
    fixtures_root: Path,
    offline_fixtures_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(daily, "datetime", _FixedDateTime)

    ingest_called = {"value": False}

    def _mark_ingest(*_args, **_kwargs) -> None:
        ingest_called["value"] = True

    monkeypatch.setattr(daily, "run_ingest", _mark_ingest)
    monkeypatch.setattr(daily, "run_calibrations", lambda *_, **__: None)
    monkeypatch.setattr(
        daily,
        "run_quality_gate_step",
        lambda *_, **__: QualityGateResult(go=True, reasons=[], details={}),
    )
    monkeypatch.setattr(daily, "run_scan", lambda *_, **__: None)

    args = _snap_args(fixtures_root, offline_fixtures_root, snap="print")
    daily.run_mode("teny_close", args)

    assert ingest_called["value"] is False
    output = capsys.readouterr().out
    assert "[snap] Next window" in output


@pytest.mark.usefixtures("isolated_data_roots")
def test_snap_wait_sleeps_until_window(
    tmp_path: Path,
    fixtures_root: Path,
    offline_fixtures_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    # reset sequence for each test run
    _SequenceDateTime._times = [
        datetime(2025, 10, 30, 18, 29, 30, tzinfo=UTC),
        datetime(2025, 10, 30, 18, 30, 30, tzinfo=UTC),
    ]
    monkeypatch.setattr(daily, "datetime", _SequenceDateTime)

    sleep_calls: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(daily.time, "sleep", _fake_sleep)
    monkeypatch.setattr(daily, "run_ingest", lambda *_: None)
    monkeypatch.setattr(daily, "run_calibrations", lambda *_, **__: None)
    monkeypatch.setattr(
        daily,
        "run_quality_gate_step",
        lambda *_, **__: QualityGateResult(go=True, reasons=[], details={}),
    )

    scan_called = {"value": False}

    def _marker_run_scan(*_args, **_kwargs) -> None:
        scan_called["value"] = True

    monkeypatch.setattr(daily, "run_scan", _marker_run_scan)

    args = _snap_args(fixtures_root, offline_fixtures_root, snap="wait")
    daily.run_mode("teny_close", args)

    assert sleep_calls and sleep_calls[0] > 0
    assert scan_called["value"] is True


def test_preflight_outputs_summary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(preflight, "datetime", _FixedDateTime)

    def _fake_evaluate(*_args, **_kwargs):
        return QualityGateResult(go=False, reasons=["test_warning"], details={}), {}

    monkeypatch.setattr(preflight, "_evaluate_quality_gates", _fake_evaluate)

    preflight.main(["--mode", "teny_close"])
    output = capsys.readouterr().out
    assert "[preflight] Mode: teny_close" in output
    assert "Now (ET)" in output
    assert "Quality gates: NO-GO" in output
    assert "test_warning" in output
