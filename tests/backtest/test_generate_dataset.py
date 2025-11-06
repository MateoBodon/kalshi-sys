from __future__ import annotations

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from kalshi_alpha.backtest.index_calendar import TargetSpec, TargetType, targets_for_day
from kalshi_alpha.backtest.generate_dataset import _rows_for_symbol
from kalshi_alpha.drivers.polygon_index.client import MinuteBar
from kalshi_alpha.drivers.polygon_index.snapshots import micro_drift

ET = ZoneInfo("America/New_York")


def _bar(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    *,
    close: float,
) -> MinuteBar:
    timestamp_et = datetime(year, month, day, hour, minute, tzinfo=ET)
    return MinuteBar(
        timestamp=timestamp_et.astimezone(UTC),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=100.0,
    )


def test_on_before_uses_prior_minute_when_target_missing() -> None:
    trading_day = date(2025, 3, 14)
    targets = targets_for_day(trading_day)
    symbol = "I:SPX"

    bars = [
        _bar(2025, 3, 14, 11, 58, close=4989.0),
        _bar(2025, 3, 14, 11, 59, close=4991.0),
        _bar(2025, 3, 14, 12, 1, close=4995.0),
    ]
    rows = _rows_for_symbol(trading_day, symbol, bars, targets)
    noon_rows = [row for row in rows if row.target_timestamp.hour == 12]
    assert noon_rows, "expected rows targeting 12:00 ET"
    on_before_values = {row.target_on_before for row in noon_rows}
    assert on_before_values == {4991.0}


def test_minutes_to_target_and_micro_drift_alignment() -> None:
    trading_day = date(2025, 6, 3)
    targets = [
        TargetSpec(trading_day=trading_day, timestamp_et=datetime(2025, 6, 3, 12, 0, tzinfo=ET), target_type=TargetType.HOURLY),
    ]
    symbol = "I:NDX"
    closes = [19000.0, 19005.0, 19002.0, 19008.0, 19010.0, 19015.0]
    bars = [
        _bar(2025, 6, 3, 11, minute, close=close)
        for minute, close in enumerate(closes, start=54)
    ]
    rows = _rows_for_symbol(trading_day, symbol, bars, targets)
    assert rows, "expected dataset rows"
    minutes = [row.minutes_to_target for row in rows]
    assert minutes[0] == 6  # 11:54 -> 6 minutes to noon
    assert minutes[-1] == 1  # 11:59 -> 1 minute to noon

    drift_values = [row.micro_drift for row in rows]
    expected = [micro_drift(bars[: idx + 1], window=5) for idx in range(len(rows))]
    assert drift_values == expected
