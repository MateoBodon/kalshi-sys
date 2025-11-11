from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kalshi_alpha.core.kalshi_api import Orderbook
from kalshi_alpha.exec import quote_microprice


def test_compute_signal_returns_offset() -> None:
    orderbook = Orderbook(
        market_id="m1",
        bids=[{"price": 0.48, "size": 40}],
        asks=[{"price": 0.52, "size": 10}],
    )
    signal = quote_microprice.compute_signal(orderbook, tick_size=0.02)
    assert signal.best_bid == 0.48
    assert signal.best_ask == 0.52
    assert signal.microprice is not None
    assert signal.offset_ticks is not None
    assert signal.best_bid < signal.microprice < signal.best_ask


def test_replacement_throttle_window() -> None:
    throttle = quote_microprice.ReplacementThrottle(max_per_window=2, window_seconds=0.5)
    key = "m1:5000:YES"
    base = datetime.now(tz=UTC)
    assert not throttle.should_block(key, now=base)
    throttle.record(key, now=base)
    assert not throttle.should_block(key, now=base + timedelta(milliseconds=10))
    throttle.record(key, now=base + timedelta(milliseconds=10))
    assert throttle.should_block(key, now=base + timedelta(milliseconds=20))
    # After the window, replacements are allowed again.
    assert not throttle.should_block(key, now=base + timedelta(seconds=1))
