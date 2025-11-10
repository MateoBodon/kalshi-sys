from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from kalshi_alpha.sched import current_window, next_windows, windows_for_day

ET = ZoneInfo("America/New_York")


def test_windows_for_day_produces_expected_targets() -> None:
    trading_day = date(2025, 11, 10)  # Monday
    windows = windows_for_day(trading_day)
    labels = [window.label for window in windows]
    assert labels[0] == "hourly-1000"
    assert labels[-1] == "close-1600"
    assert windows[-1].series == ("INX", "NASDAQ100")
    assert len(windows) == 7


def test_current_window_identifies_active_hourly_window() -> None:
    now_et = datetime(2025, 11, 10, 14, 50, tzinfo=ET)
    window = current_window("INXU", now_et.astimezone(UTC))
    assert window is not None
    assert window.label == "hourly-1500"
    assert window.in_final_minute(datetime(2025, 11, 10, 14, 59, 30, tzinfo=ET)) is True


def test_next_windows_includes_future_targets() -> None:
    now_utc = datetime(2025, 11, 10, 20, 30, tzinfo=UTC)
    upcoming = next_windows(now_utc, limit=2)
    assert len(upcoming) == 2
    assert upcoming[0].label == "close-1600"
    assert upcoming[1].label == "hourly-1000"
