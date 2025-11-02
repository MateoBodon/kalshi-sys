from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from kalshi_alpha.exec.pipelines.calendar import resolve_run_window


def _write_parquet(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def test_pre_cpi_window_uses_calendar(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    release_ts = datetime(2025, 11, 12, 13, 30, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "release_datetime": [release_ts],
            "release_date": [release_ts.date()],
        }
    )
    _write_parquet(proc_root / "bls_cpi/calendar/20251112T000000.parquet", frame)

    now = datetime(2025, 11, 12, 12, 0, tzinfo=UTC)
    window = resolve_run_window(
        mode="pre_cpi",
        target_date=date(2025, 11, 12),
        now=now,
        proc_root=proc_root,
    )

    assert window.reference is not None
    assert window.scan_open is not None and window.scan_close is not None
    assert window.freeze_start is not None
    assert window.scan_allowed(now) is True
    # 06:00 ET -> 11:00 UTC
    assert window.scan_open.hour == 11 and window.scan_open.minute == 0
    # Freeze starts 24h prior (06:00 ET previous day -> 11:00 UTC)
    assert window.freeze_start.hour == 11


def test_pre_claims_holiday_shift(tmp_path: Path) -> None:
    now = datetime(2025, 11, 26, 12, 30, tzinfo=UTC)
    window = resolve_run_window(
        mode="pre_claims",
        target_date=date(2025, 11, 27),
        now=now,
        proc_root=tmp_path / "proc",
    )

    assert any(note.startswith("holiday_shift") for note in window.notes)
    # Release shifted to Wednesday (Nov 26)
    assert window.reference is not None
    reference_et = window.reference.astimezone(ZoneInfo("America/New_York"))
    assert reference_et.weekday() == 2  # Wednesday
    assert window.scan_allowed(now) is True


def test_teny_close_weekend_rolls_forward(tmp_path: Path) -> None:
    window = resolve_run_window(
        mode="teny_close",
        target_date=date(2025, 10, 26),  # Sunday
        now=datetime(2025, 10, 27, 19, 0, tzinfo=UTC),
        proc_root=tmp_path / "proc",
    )

    assert window.reference is not None
    ref = window.reference.astimezone(ZoneInfo("America/New_York"))
    assert ref.weekday() == 0  # Monday
    assert ref.hour == 15 and ref.minute == 30
    scan_open_et = window.scan_open.astimezone(ZoneInfo("America/New_York"))
    assert scan_open_et.hour == 14 and scan_open_et.minute == 30


def test_weather_cycle_nearest(tmp_path: Path) -> None:
    now = datetime(2025, 10, 27, 5, 10, tzinfo=UTC)
    window = resolve_run_window(
        mode="weather_cycle",
        target_date=date(2025, 10, 27),
        now=now,
        proc_root=tmp_path / "proc",
    )

    assert window.reference.hour == 6
    assert "cycle_hour:06Z" in window.notes
    assert window.scan_open == window.reference
    assert window.scan_close == window.reference + timedelta(minutes=45)
