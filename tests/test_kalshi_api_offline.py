from pathlib import Path

from kalshi_alpha.core.kalshi_api import KalshiPublicClient


def test_offline_client_series(fixtures_root: Path) -> None:
    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    series = client.get_series()
    assert series[0].ticker == "CPI"


def test_offline_client_events_and_markets(fixtures_root: Path) -> None:
    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    series = client.get_series()[0]
    events = client.get_events(series.id)
    markets = client.get_markets(events[0].id)
    ladder = markets[0].ladder_strikes
    assert ladder
    diff_1 = ladder[1] - ladder[0]
    diff_2 = ladder[-1] - ladder[-2]
    assert diff_1 != diff_2
    orderbook = client.get_orderbook(markets[0].id)
    assert orderbook.market_id == markets[0].id
    mid = orderbook.depth_weighted_mid()
    assert 0 < mid < 1
