from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from kalshi_alpha.core.kalshi_api import Event
from kalshi_alpha.exec.runners import scan_ladders


def _event(ticker: str) -> Event:
    return Event(id=ticker, series_id="INXU", ticker=ticker, title=ticker)


def test_filter_u_series_events_prefers_target_trading_day() -> None:
    now_et = datetime(2025, 11, 12, 13, 30, tzinfo=ZoneInfo("America/New_York"))
    decision = {
        "target_hour": 14,
        "current_hour": 13,
        "rolled": False,
        "now_et": now_et,
    }
    events = [
        _event("KXINXU-25NOV11H1400"),
        _event("KXINXU-25NOV12H1400"),
        _event("KXINXU-25NOV13H1400"),
    ]

    filtered = scan_ladders._filter_u_series_events(events, decision=decision)

    assert [event.ticker for event in filtered] == ["KXINXU-25NOV12H1400"]


def test_filter_u_series_events_rolls_to_next_day() -> None:
    now_et = datetime(2025, 11, 12, 23, 45, tzinfo=ZoneInfo("America/New_York"))
    decision = {
        "target_hour": 0,
        "current_hour": 23,
        "rolled": True,
        "now_et": now_et,
    }
    events = [
        _event("KXINXU-25NOV12H0000"),
        _event("KXINXU-25NOV13H0000"),
    ]

    filtered = scan_ladders._filter_u_series_events(events, decision=decision)

    assert [event.ticker for event in filtered] == ["KXINXU-25NOV13H0000"]
