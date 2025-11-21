from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from kalshi_alpha.exec.supervisor import Supervisor, SupervisorConfig
from kalshi_alpha.exec.window_guard import guard_series_window

ET = ZoneInfo("America/New_York")


def test_guard_window_blocks_outside_hours() -> None:
    now = datetime(2025, 1, 5, 3, 0, tzinfo=ET)
    allowed, active, next_window = guard_series_window("INXU", now=now.astimezone(UTC), quiet=True)
    assert not allowed
    assert active is None
    assert next_window is not None


def test_guard_window_allows_inside_hourly_window() -> None:
    now = datetime(2025, 1, 6, 11, 50, tzinfo=ET)
    allowed, active, next_window = guard_series_window("INXU", now=now.astimezone(UTC), quiet=True)
    assert allowed
    assert active is not None
    assert next_window is None
    assert active.target_et.hour == 12


@pytest.mark.asyncio
async def test_final_minute_rerun_triggers_close_scan() -> None:
    config = SupervisorConfig(offline=True, poll_seconds=0.1)
    supervisor = Supervisor(config)
    supervisor._run_scan_close = AsyncMock()
    supervisor._run_scan_ladders = AsyncMock()

    now_et = datetime(2025, 1, 6, 15, 59, 30, tzinfo=ET)
    await supervisor._maybe_trigger_windows(now_et)

    supervisor._run_scan_close.assert_awaited()
    assert "final" in supervisor._run_scan_close.await_args.kwargs.get("label", "")
    assert supervisor._run_scan_ladders.await_count == 0
