"""Trading calendar helpers for index backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import Enum
from functools import lru_cache
from typing import Iterable, Iterator, Sequence
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


class TargetType(str, Enum):
    """Label for target horizons."""

    HOURLY = "hourly"
    CLOSE = "close"


_DEFAULT_TARGET_HOURS: tuple[tuple[int, TargetType], ...] = (
    (10, TargetType.HOURLY),
    (11, TargetType.HOURLY),
    (12, TargetType.HOURLY),
    (13, TargetType.HOURLY),
    (14, TargetType.HOURLY),
    (15, TargetType.HOURLY),
    (16, TargetType.CLOSE),
)


@dataclass(frozen=True)
class TargetSpec:
    """Target timestamp metadata for a given trading day."""

    trading_day: date
    timestamp_et: datetime
    target_type: TargetType

    @property
    def timestamp_utc(self) -> datetime:
        return self.timestamp_et.astimezone(UTC)


def trading_days(start: date, end: date) -> list[date]:
    """Return all NYSE trading days in the inclusive [start, end] window."""

    if end < start:
        raise ValueError("end date must be on or after start date")
    days: list[date] = []
    current = start
    while current <= end:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days


def enumerate_targets(
    start: date,
    end: date,
    *,
    target_hours: Sequence[tuple[int, TargetType]] | None = None,
) -> Iterator[TargetSpec]:
    """Yield target specifications between the provided bounds."""

    for trading_day in trading_days(start, end):
        yield from targets_for_day(trading_day, target_hours=target_hours)


def targets_for_day(
    trading_day: date,
    *,
    target_hours: Sequence[tuple[int, TargetType]] | None = None,
) -> list[TargetSpec]:
    """Resolve all backtest targets for the provided trading day."""

    if not is_trading_day(trading_day):
        return []
    resolved = target_hours or _DEFAULT_TARGET_HOURS
    specs: list[TargetSpec] = []
    for hour_value, label in resolved:
        target_type = TargetType.CLOSE if hour_value >= 16 else TargetType.HOURLY
        if isinstance(label, TargetType):
            target_type = label
        elif isinstance(label, str):
            normalized = label.lower()
            if normalized == TargetType.CLOSE:
                target_type = TargetType.CLOSE
            else:
                target_type = TargetType.HOURLY
        timestamp = datetime.combine(trading_day, time(hour_value, 0), tzinfo=ET)
        specs.append(TargetSpec(trading_day=trading_day, timestamp_et=timestamp, target_type=target_type))
    return specs


def is_trading_day(day: date) -> bool:
    """Return whether the supplied date is a regular NYSE trading session."""

    if day.weekday() >= 5:  # Saturday & Sunday
        return False
    holidays = _us_equity_holidays(day.year)
    return day not in holidays


@lru_cache(maxsize=4)
def _us_equity_holidays(year: int) -> set[date]:
    """Approximate NYSE holiday calendar for the provided year."""

    holidays: set[date] = set()

    def observed(dt: date) -> date:
        if dt.weekday() == 5:  # Saturday observed Friday
            return dt - timedelta(days=1)
        if dt.weekday() == 6:  # Sunday observed Monday
            return dt + timedelta(days=1)
        return dt

    # Fixed-date holidays
    holidays.add(observed(date(year, 1, 1)))  # New Year's Day
    holidays.add(observed(date(year, 6, 19)))  # Juneteenth
    holidays.add(observed(date(year, 7, 4)))  # Independence Day
    holidays.add(observed(date(year, 12, 25)))  # Christmas Day

    # Floating holidays
    holidays.add(_nth_weekday(year, month=1, weekday=0, n=3))  # Martin Luther King Jr. Day
    holidays.add(_nth_weekday(year, month=2, weekday=0, n=3))  # Presidents' Day
    holidays.add(_last_weekday(year, month=5, weekday=0))  # Memorial Day
    holidays.add(_nth_weekday(year, month=9, weekday=0, n=1))  # Labor Day
    holidays.add(_nth_weekday(year, month=11, weekday=3, n=4))  # Thanksgiving Day

    # Good Friday (two days before Easter Sunday)
    easter_sunday = _easter_date(year)
    good_friday = easter_sunday - timedelta(days=2)
    holidays.add(good_friday)

    # Additional historical full-closure days
    holidays.add(observed(date(year, 11, 11)))  # Veterans Day (rare closures but conservative)

    return holidays


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    day = 1 + offset + (n - 1) * 7
    return date(year, month, day)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    day = next_month - timedelta(days=1)
    while day.weekday() != weekday:
        day -= timedelta(days=1)
    return day


def _easter_date(year: int) -> date:
    """Compute Easter Sunday using Anonymous Gregorian algorithm."""

    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


__all__ = [
    "TargetSpec",
    "TargetType",
    "enumerate_targets",
    "is_trading_day",
    "targets_for_day",
    "trading_days",
]
