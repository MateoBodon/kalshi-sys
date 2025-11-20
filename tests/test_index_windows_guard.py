from __future__ import annotations

from datetime import UTC, datetime

from kalshi_alpha.exec.window_guard import guard_series_window
from kalshi_alpha.sched import current_window


def test_guard_allows_inside_hourly_window() -> None:
    reference = datetime(2025, 1, 7, 16, 55, tzinfo=UTC)  # 11:55 ET, inside noon window
    allowed, window, next_window = guard_series_window("INXU", now=reference, quiet=True)
    assert allowed
    assert window is not None
    assert "hourly" in window.label
    assert next_window is None


def test_guard_blocks_outside_window_and_returns_next() -> None:
    reference = datetime(2025, 1, 7, 12, 5, tzinfo=UTC)  # 07:05 ET, pre-market
    allowed, window, next_window = guard_series_window("INX", now=reference, quiet=True)
    assert not allowed
    assert window is None
    assert next_window is not None
    assert "close" in next_window.label


def test_final_minute_detection_for_close_window() -> None:
    reference = datetime(2025, 1, 7, 20, 59, 30, tzinfo=UTC)  # 15:59:30 ET
    window = current_window("INX", reference)
    assert window is not None
    assert window.in_final_minute(reference)
