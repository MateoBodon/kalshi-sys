from __future__ import annotations

from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

import polars as pl

from jobs._index_calibration import build_sigma_curve
from kalshi_alpha.drivers.polygon_index.client import MinuteBar


def _minute_bar(et_timestamp: datetime, close: float) -> MinuteBar:
    ts = et_timestamp.astimezone(UTC)
    return MinuteBar(
        timestamp=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
        vwap=close,
        trades=10,
    )


def test_build_sigma_curve_computes_minutes() -> None:
    et = ZoneInfo("America/New_York")
    day = datetime(2024, 1, 2, tzinfo=et)
    bars = [
        _minute_bar(day.replace(hour=11, minute=58), 5000.0),
        _minute_bar(day.replace(hour=11, minute=59), 5001.0),
        _minute_bar(day.replace(hour=12, minute=0), 5002.0),
    ]
    frame = build_sigma_curve({"I:SPX": bars}, target_time=time(12, 0), residual_window=2)
    assert isinstance(frame, pl.DataFrame)
    assert not frame.is_empty()
    minutes = frame.get_column("minutes_to_target").to_list()
    assert 0 in minutes
    assert max(minutes) == 2
