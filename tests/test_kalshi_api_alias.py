from __future__ import annotations

from datetime import datetime

from kalshi_alpha.core.kalshi_api import Market, _matches_filter


def _fake_market(series: str, status: str) -> Market:
    return Market(
        id="mkt",
        event_id="evt",
        ticker=f"{series}-TEST",
        title="test",
        ladder_strikes=[100.0],
        ladder_yes_prices=[0.5],
        event_ticker=f"{series}-EVT",
        series_ticker=series,
        status=status,
        close_time=datetime.now(),
        rung_tickers=None,
    )


def test_matches_filter_accepts_kx_alias_series() -> None:
    market = _fake_market("KXINXU", "active")
    assert _matches_filter(market, "INXU", "open", None)
    assert not _matches_filter(market, "NASDAQ100U", "open", None)
