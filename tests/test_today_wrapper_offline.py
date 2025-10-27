from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from kalshi_alpha.exec.pipelines import daily, today


def test_plan_runs_cpi_and_claims(
    tmp_path: Path,
    isolated_data_roots: tuple[Path, Path],
) -> None:
    _, proc_root = isolated_data_roots
    calendar_dir = proc_root / "bls_cpi" / "calendar"
    calendar_dir.mkdir(parents=True, exist_ok=True)
    release_dt = datetime(2025, 10, 29, 13, 30, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "release_datetime": [release_dt],
            "release_date": [release_dt.date()],
        }
    )
    frame.write_parquet(calendar_dir / "release.parquet")

    now_eve = datetime(2025, 10, 28, 16, 0, tzinfo=UTC)
    release_date = release_dt.astimezone(today.ET).date()
    runs_eve = today._plan_runs(now_eve, include_weather=False, proc_root=proc_root)
    assert today.ScheduledRun("pre_cpi", release_date) in runs_eve
    assert today.ScheduledRun("pre_claims", release_date) not in runs_eve

    now_wed_pm = datetime(2025, 10, 29, 23, 30, tzinfo=UTC)
    runs_wed = today._plan_runs(now_wed_pm, include_weather=False, proc_root=proc_root)
    next_release = (release_dt + timedelta(days=1)).astimezone(today.ET).date()
    assert today.ScheduledRun("pre_claims", next_release) in runs_wed
    assert today.ScheduledRun("teny_close", now_wed_pm.astimezone(today.ET).date()) in runs_wed


def test_plan_runs_weather_toggle(
    isolated_data_roots: tuple[Path, Path],
) -> None:
    _, proc_root = isolated_data_roots
    now = datetime(2025, 10, 27, 12, 0, tzinfo=UTC)
    runs = today._plan_runs(now, include_weather=True, proc_root=proc_root)
    assert today.ScheduledRun("weather_cycle", now.astimezone(today.ET).date()) in runs


def test_today_cli_invokes_daily(monkeypatch, fixtures_root: Path, offline_fixtures_root: Path) -> None:
    run_date = datetime(2025, 10, 29, 0, 0, tzinfo=UTC).astimezone(today.ET).date()
    scheduled = [today.ScheduledRun("pre_cpi", run_date)]

    monkeypatch.setattr(today, "_plan_runs", lambda now, include_weather, proc_root=None: scheduled)
    captured: list[list[str]] = []

    def fake_daily_main(argv: list[str]) -> None:
        captured.append(argv)

    monkeypatch.setattr(daily, "main", fake_daily_main)
    monkeypatch.setattr(today, "_now", lambda: datetime(2025, 10, 28, 12, 0, tzinfo=UTC))

    today.main(
        [
            "--offline",
            "--report",
            "--driver-fixtures",
            str(offline_fixtures_root),
            "--scanner-fixtures",
            str(fixtures_root),
            "--kelly-cap",
            "0.3",
            "--daily-loss-cap",
            "150",
            "--weekly-loss-cap",
            "400",
            "--fill-alpha",
            "0.7",
        ]
    )

    assert captured, "Expected daily pipeline invocation"
    args = captured[0]
    assert "--mode" in args and "pre_cpi" in args
    assert "--offline" in args
    assert "--report" in args
    assert "--driver-fixtures" in args
    assert "--kelly-cap" in args and "0.3" in args
    assert "--daily-loss-cap" in args and "150" in args
    assert "--weekly-loss-cap" in args and "400" in args
    assert "--fill-alpha" in args and "0.7" in args
