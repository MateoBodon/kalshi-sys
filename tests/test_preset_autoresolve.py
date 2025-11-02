from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta

import pytest

from kalshi_alpha.exec.pipelines import week
from kalshi_alpha.exec.pipelines.calendar import RunWindow


def _make_window(
    mode: str,
    reference_date: date,
    *,
    reference_time: time,
    scan_open_time: time,
    scan_close_time: time,
    freeze_time: time,
) -> RunWindow:
    reference_et = datetime.combine(reference_date, reference_time, tzinfo=week.ET)
    return RunWindow(
        mode=mode,
        freeze_start=datetime.combine(reference_date, freeze_time, tzinfo=week.ET).astimezone(UTC),
        scan_open=datetime.combine(reference_date, scan_open_time, tzinfo=week.ET).astimezone(UTC),
        scan_close=datetime.combine(reference_date, scan_close_time, tzinfo=week.ET).astimezone(UTC),
        reference=reference_et.astimezone(UTC),
        notes=[],
    )


@pytest.fixture(autouse=True)
def _patch_resolve_run_window(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_resolve_run_window(*, mode: str, target_date: date, now: datetime, proc_root=None) -> RunWindow:
        if mode == "teny_close":
            if target_date.weekday() >= 5:
                target_date = target_date + timedelta(days=(7 - target_date.weekday()))
            return _make_window(
                mode,
                target_date,
                reference_time=time(15, 30),
                scan_open_time=time(14, 30),
                scan_close_time=time(15, 25),
                freeze_time=time(13, 30),
            )
        if mode == "pre_claims":
            return _make_window(
                mode,
                target_date,
                reference_time=time(8, 30),
                scan_open_time=time(6, 0),
                scan_close_time=time(8, 25),
                freeze_time=time(18, 0),
            )
        if mode == "pre_cpi":
            return _make_window(
                mode,
                target_date,
                reference_time=time(8, 30),
                scan_open_time=time(6, 0),
                scan_close_time=time(8, 20),
                freeze_time=time(6, 0),
            )
        # weather fallback
        window = _make_window(
            mode,
            target_date,
            reference_time=time(12, 0),
            scan_open_time=time(12, 0),
            scan_close_time=time(12, 45),
            freeze_time=time(12, 0),
        )
        return window

    monkeypatch.setattr(week, "resolve_run_window", fake_resolve_run_window)


def _mode_runs(schedule: Iterable[week.WeekRun], mode: str) -> list[week.WeekRun]:
    return [run for run in schedule if run.mode == mode]


def test_preset_autoresolve_advances_to_next_window() -> None:
    now = datetime(2025, 10, 30, 18, 0, tzinfo=UTC)
    schedule = week._paper_live_schedule(now, include_weather=False)

    ten_y_runs = _mode_runs(schedule, "teny_close")
    assert ten_y_runs, "expected TenY runs in schedule"
    first_ten_y = ten_y_runs[0]
    assert first_ten_y.auto_resolved is True
    assert first_ten_y.original_date is not None
    assert first_ten_y.run_date >= date(2025, 10, 30)
    assert first_ten_y.window_et and "ET" in first_ten_y.window_et

    claims_release = next(run for run in schedule if run.mode == "pre_claims" and run.window_et)
    assert claims_release.run_date > date(2025, 10, 30)
    assert claims_release.auto_resolved is True
    assert claims_release.window_et.endswith("ET")
