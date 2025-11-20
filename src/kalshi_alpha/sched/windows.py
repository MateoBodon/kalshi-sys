"""US/Eastern-aware scheduler for hourly and close index ladder windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

from kalshi_alpha.backtest import index_calendar
from kalshi_alpha.config import IndexOpsWindow, load_index_ops_config

ET = ZoneInfo("America/New_York")
FINAL_MINUTE = timedelta(minutes=1)
HOURLY_TARGET_HOURS: tuple[int, ...] = (10, 11, 12, 13, 14, 15)
CLOSE_HOUR = 16
SERIES_BY_TYPE: dict[str, tuple[str, ...]] = {
    "hourly": ("INXU", "NASDAQ100U"),
    "close": ("INX", "NASDAQ100"),
}


@dataclass(frozen=True)
class TradingWindow:
    """Resolved execution window with start/target/freeze metadata."""

    label: str
    target_type: str
    series: tuple[str, ...]
    target_et: datetime
    start_et: datetime
    freeze_et: datetime
    freshness_strict_et: datetime

    @property
    def target_utc(self) -> datetime:
        return self.target_et.astimezone(UTC)

    @property
    def start_utc(self) -> datetime:
        return self.start_et.astimezone(UTC)

    def seconds_to_freeze(self, moment: datetime) -> float:
        reference = _ensure_et(moment)
        return (self.freeze_et - reference).total_seconds()

    def in_final_minute(self, moment: datetime) -> bool:
        reference = _ensure_et(moment)
        return self.freshness_strict_et <= reference < self.target_et

    def contains(self, moment: datetime) -> bool:
        reference = _ensure_et(moment)
        return self.start_et <= reference <= self.target_et


@lru_cache(maxsize=2)
def _ops_config() -> tuple[IndexOpsWindow, IndexOpsWindow]:
    cfg = load_index_ops_config()
    return cfg.window_hourly, cfg.window_close


def windows_for_day(trading_day: date) -> list[TradingWindow]:
    """Return all ladder windows for the provided trading day."""

    if not index_calendar.is_trading_day(trading_day):
        return []
    hour_window, close_window = _ops_config()
    windows: list[TradingWindow] = []
    for hour in HOURLY_TARGET_HOURS:
        windows.append(
            _build_window(
                trading_day=trading_day,
                target_hour=hour,
                label=f"hourly-{hour:02d}00",
                window=hour_window,
                target_type="hourly",
            )
        )
    windows.append(
        _build_window(
            trading_day=trading_day,
            target_hour=CLOSE_HOUR,
            label="close-1600",
            window=close_window,
            target_type="close",
        )
    )
    return windows


def current_window(series: str, moment: datetime | None = None) -> TradingWindow | None:
    """Return the active window for *series* at *moment*, if any."""

    if not series:
        return None
    reference = _ensure_et(moment or datetime.now(tz=UTC))
    for window in windows_for_day(reference.date()):
        if series.upper() not in window.series:
            continue
        if window.contains(reference):
            return window
    return None


def next_window_for_series(series: str, moment: datetime | None = None) -> TradingWindow | None:
    """Return the next upcoming window for *series* after *moment* (default: now)."""

    target = str(series or "").upper()
    for window in next_windows(moment, limit=8):
        if target in window.series:
            return window
    return None


def next_windows(now: datetime | None = None, *, limit: int = 4) -> list[TradingWindow]:
    """Return the next *limit* upcoming windows from *now*."""

    reference = _ensure_et(now or datetime.now(tz=UTC))
    targets: list[TradingWindow] = []
    day = reference.date()
    attempts = 0
    while len(targets) < limit and attempts < 10:
        for window in windows_for_day(day):
            if window.target_et >= reference:
                targets.append(window)
                if len(targets) >= limit:
                    break
        day += timedelta(days=1)
        attempts += 1
    return targets[:limit]


def _build_window(
    *,
    trading_day: date,
    target_hour: int,
    label: str,
    window: IndexOpsWindow,
    target_type: str,
) -> TradingWindow:
    target_et = datetime.combine(trading_day, time(target_hour % 24, 0), tzinfo=ET)
    start_et = _resolve_window_start(window, target_et)
    freeze_et = target_et - timedelta(seconds=float(window.cancel_buffer_seconds))
    freshness_et = max(start_et, target_et - FINAL_MINUTE)
    series = SERIES_BY_TYPE.get(target_type, ())
    return TradingWindow(
        label=label,
        target_type=target_type,
        series=series,
        target_et=target_et,
        start_et=start_et,
        freeze_et=freeze_et,
        freshness_strict_et=freshness_et,
    )


def _resolve_window_start(window: IndexOpsWindow, target_et: datetime) -> datetime:
    if window.start is not None:
        return datetime.combine(target_et.date(), window.start, tzinfo=ET)
    offset = window.start_offset_minutes or 0
    return target_et - timedelta(minutes=offset)


def _ensure_et(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC).astimezone(ET)
    return moment.astimezone(ET)


__all__ = ["TradingWindow", "current_window", "next_window_for_series", "next_windows", "windows_for_day"]
