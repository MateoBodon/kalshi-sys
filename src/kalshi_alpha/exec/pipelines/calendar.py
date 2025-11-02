"""Calendar-aware run window resolution for daily pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from kalshi_alpha.datastore.paths import PROC_ROOT

ET = ZoneInfo("America/New_York")


@dataclass
class RunWindow:
    mode: str
    freeze_start: datetime | None
    scan_open: datetime | None
    scan_close: datetime | None
    reference: datetime | None
    notes: list[str] = field(default_factory=list)

    def freeze_active(self, now: datetime) -> bool:
        return self.freeze_start is not None and now >= self.freeze_start

    def scan_allowed(self, now: datetime) -> bool:
        if self.scan_open is None or self.scan_close is None:
            return False
        return self.scan_open <= now <= self.scan_close

    def to_dict(self, now: datetime) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "freeze_start": _serialize_dt(self.freeze_start),
            "scan_open": _serialize_dt(self.scan_open),
            "scan_close": _serialize_dt(self.scan_close),
            "reference": _serialize_dt(self.reference),
            "freeze_active": self.freeze_active(now),
            "scan_allowed": self.scan_allowed(now),
            "notes": list(self.notes),
        }


def resolve_run_window(
    *,
    mode: str,
    target_date: date,
    now: datetime,
    proc_root: Path | None = None,
) -> RunWindow:
    proc_root = proc_root or PROC_ROOT
    mode_lower = mode.lower()
    if mode_lower == "pre_cpi":
        return _resolve_pre_cpi(target_date, now, proc_root)
    if mode_lower == "pre_claims":
        return _resolve_pre_claims(target_date, now)
    if mode_lower == "teny_close":
        return _resolve_teny_close(target_date)
    if mode_lower == "weather_cycle":
        return _resolve_weather_cycle(target_date, now)
    raise ValueError(f"Unsupported mode for calendar resolution: {mode}")


# --- mode resolvers ---------------------------------------------------------


def _resolve_pre_cpi(target_date: date, now: datetime, proc_root: Path) -> RunWindow:
    calendar_dir = proc_root / "bls_cpi" / "calendar"
    if not calendar_dir.exists():
        return RunWindow("pre_cpi", None, None, None, None, notes=["missing_calendar"])
    latest = _latest_parquet(calendar_dir)
    if latest is None:
        return RunWindow("pre_cpi", None, None, None, None, notes=["missing_calendar_snapshot"])
    frame = pl.read_parquet(latest)
    if frame.is_empty():
        return RunWindow("pre_cpi", None, None, None, None, notes=["empty_calendar"])
    candidates = frame.filter(pl.col("release_date") >= target_date).sort("release_date")
    if candidates.is_empty():
        # fall back to latest known release before target_date
        candidates = frame.sort("release_date")
        if candidates.is_empty():  # pragma: no cover - double guard
            return RunWindow("pre_cpi", None, None, None, None, notes=["no_releases"])
    release_ts = candidates.row(0, named=True)["release_datetime"]
    release_dt = _ensure_datetime(release_ts)
    release_et = release_dt.astimezone(ET)
    scan_open_et = release_et.replace(hour=6, minute=0, second=0, microsecond=0)
    scan_close_et = release_et - timedelta(minutes=10)
    freeze_start_et = scan_open_et - timedelta(hours=24)
    return RunWindow(
        mode="pre_cpi",
        freeze_start=freeze_start_et.astimezone(UTC),
        scan_open=scan_open_et.astimezone(UTC),
        scan_close=scan_close_et.astimezone(UTC),
        reference=release_dt.astimezone(UTC),
        notes=[],
    )


def _resolve_pre_claims(target_date: date, now: datetime) -> RunWindow:
    release_date = target_date
    while release_date.weekday() != 3:
        release_date += timedelta(days=1)
    holidays = _us_holidays(release_date.year)
    adjusted = release_date
    while adjusted in holidays or adjusted.weekday() >= 5:
        adjusted -= timedelta(days=1)
    release_et = datetime.combine(adjusted, time(8, 30), tzinfo=ET)
    if adjusted != release_date:
        note = f"holiday_shift:{release_date.isoformat()}->{adjusted.isoformat()}"
        notes = [note]
    else:
        notes = []
    scan_open_et = release_et.replace(hour=6, minute=0, second=0, microsecond=0)
    scan_close_et = release_et - timedelta(minutes=5)
    freeze_day = (release_et - timedelta(days=1)).date()
    freeze_start_et = datetime.combine(freeze_day, time(18, 0), tzinfo=ET)
    return RunWindow(
        mode="pre_claims",
        freeze_start=freeze_start_et.astimezone(UTC),
        scan_open=scan_open_et.astimezone(UTC),
        scan_close=scan_close_et.astimezone(UTC),
        reference=release_et.astimezone(UTC),
        notes=notes,
    )


def _resolve_teny_close(target_date: date) -> RunWindow:
    adjusted = target_date
    while adjusted.weekday() >= 5:
        adjusted += timedelta(days=1)
    reference_et = datetime.combine(adjusted, time(15, 30), tzinfo=ET)
    scan_open_et = reference_et.replace(hour=14, minute=30)
    scan_close_et = reference_et.replace(hour=15, minute=25)
    freeze_start_et = reference_et.replace(hour=13, minute=30)
    return RunWindow(
        mode="teny_close",
        freeze_start=freeze_start_et.astimezone(UTC),
        scan_open=scan_open_et.astimezone(UTC),
        scan_close=scan_close_et.astimezone(UTC),
        reference=reference_et.astimezone(UTC),
        notes=[],
    )


def _resolve_weather_cycle(target_date: date, now: datetime) -> RunWindow:
    cycle_hours = [0, 6, 12, 18]
    candidates = [datetime.combine(target_date, time(hour), tzinfo=UTC) for hour in cycle_hours]
    if not candidates:
        return RunWindow("weather_cycle", None, None, None, None, notes=["no_cycles"])
    selected = min(candidates, key=lambda dt: abs((dt - now).total_seconds()))
    scan_open = selected
    scan_close = selected + timedelta(minutes=45)
    freeze_start = selected
    note = f"cycle_hour:{selected.hour:02d}Z"
    return RunWindow(
        mode="weather_cycle",
        freeze_start=freeze_start,
        scan_open=scan_open,
        scan_close=scan_close,
        reference=selected,
        notes=[note],
    )


# --- helpers ----------------------------------------------------------------


def _latest_parquet(directory: Path) -> Path | None:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return None
    return files[-1]


def _ensure_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    raise TypeError(f"Expected datetime, got {type(value)!r}")


def _serialize_dt(value: datetime | None) -> dict[str, str] | None:
    if value is None:
        return None
    value_utc = value.astimezone(UTC)
    value_et = value.astimezone(ET)
    return {"utc": value_utc.isoformat(), "et": value_et.isoformat()}


def _us_holidays(year: int) -> set[date]:
    holidays: set[date] = set()

    def observe(dt: date) -> date:
        if dt.weekday() == 5:  # Saturday
            return dt - timedelta(days=1)
        if dt.weekday() == 6:  # Sunday
            return dt + timedelta(days=1)
        return dt

    # Fixed-date holidays
    holidays.add(observe(date(year, 1, 1)))  # New Year's Day
    holidays.add(observe(date(year, 6, 19)))  # Juneteenth
    holidays.add(observe(date(year, 7, 4)))  # Independence Day
    holidays.add(observe(date(year, 11, 11)))  # Veterans Day
    holidays.add(observe(date(year, 12, 25)))  # Christmas Day

    # Floating holidays
    holidays.add(_nth_weekday_of_month(year, 1, 0, 3))  # MLK Day
    holidays.add(_nth_weekday_of_month(year, 2, 0, 3))  # Presidents' Day
    holidays.add(_last_weekday_of_month(year, 5, 0))  # Memorial Day
    holidays.add(_nth_weekday_of_month(year, 9, 0, 1))  # Labor Day
    holidays.add(_nth_weekday_of_month(year, 10, 0, 2))  # Columbus Day
    holidays.add(_nth_weekday_of_month(year, 11, 3, 4))  # Thanksgiving (4th Thursday)

    # Ensure Thanksgiving Thursday itself (already above) and add the Friday after? not federal.
    holidays.add(date(year, 11, 11))  # ensure actual Veterans Day

    return holidays


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Return the date of the nth weekday (Monday=0) for the given month."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    day = 1 + offset + (n - 1) * 7
    return date(year, month, day)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    day = next_month - timedelta(days=1)
    while day.weekday() != weekday:
        day -= timedelta(days=1)
    return day
