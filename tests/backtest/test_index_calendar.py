from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from kalshi_alpha.backtest import index_calendar


@pytest.mark.parametrize(
    ("trading_day", "expected_hour", "expected_utc_hour"),
    [
        (date(2025, 3, 7), 12, 17),  # pre-DST (EST, UTC-5)
        (date(2025, 3, 10), 12, 16),  # post-DST (EDT, UTC-4)
        (date(2025, 10, 31), 12, 16),  # final EDT trading day before fallback
        (date(2025, 11, 3), 12, 17),  # first trading day after fallback
    ],
)
def test_targets_handle_dst_transitions(trading_day: date, expected_hour: int, expected_utc_hour: int) -> None:
    specs = index_calendar.targets_for_day(trading_day)
    assert specs, "expected trading targets"
    noon_specs = [spec for spec in specs if spec.timestamp_et.hour == expected_hour]
    assert noon_specs, "expected noon target"
    utc_offsets = {spec.timestamp_et.astimezone(UTC).hour for spec in noon_specs}
    assert utc_offsets == {expected_utc_hour}


def test_enumerate_skips_weekends_and_holidays() -> None:
    days = index_calendar.trading_days(date(2025, 1, 17), date(2025, 1, 21))
    assert days == [date(2025, 1, 17), date(2025, 1, 21)]  # Jan 20 = MLK (market closed)
