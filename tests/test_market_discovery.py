from __future__ import annotations

from datetime import date
from pathlib import Path

import kalshi_alpha.markets.discovery as discovery
from kalshi_alpha.core.kalshi_api import KalshiPublicClient


def _client() -> KalshiPublicClient:
    fixtures_root = Path("tests/data_fixtures/kalshi")
    return KalshiPublicClient(offline_dir=fixtures_root, use_offline=True)


def test_discover_markets_aligns_with_scheduler() -> None:
    client = _client()
    trading_day = date(2025, 11, 10)
    windows = discovery.discover_markets_for_day(client, trading_day=trading_day)
    labels = {window.label for window in windows}
    assert "hourly-1000" in labels
    hourly_10 = next(window for window in windows if window.label == "hourly-1000")
    assert not hourly_10.missing_series
    assert {market.series for market in hourly_10.markets} == {"INXU", "NASDAQ100U"}
    close_window = next(window for window in windows if window.label == "close-1600")
    assert {market.series for market in close_window.markets} == {"INX", "NASDAQ100"}


def test_discover_markets_derives_windows_when_scheduler_empty(monkeypatch) -> None:
    client = _client()
    trading_day = date(2025, 11, 10)
    monkeypatch.setattr(discovery, "windows_for_day", lambda _day: [])
    windows = discovery.discover_markets_for_day(client, trading_day=trading_day)
    assert any(window.label.startswith("derived-") for window in windows)


def test_discover_markets_accepts_aliases() -> None:
    client = _client()
    trading_day = date(2025, 11, 10)
    windows = discovery.discover_markets_for_day(
        client,
        trading_day=trading_day,
        series=("SPXU", "NDXU"),
    )
    hourly_window = next(window for window in windows if window.label == "hourly-1000")
    assert not hourly_window.missing_series
    assert {market.series for market in hourly_window.markets} == {"INXU", "NASDAQ100U"}
