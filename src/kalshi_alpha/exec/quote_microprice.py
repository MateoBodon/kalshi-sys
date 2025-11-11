"""Microprice calculator and replacement throttle for ladder quoting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Deque

from kalshi_alpha.core.kalshi_api import Orderbook

__all__ = [
    "MicropriceSignal",
    "ReplacementThrottle",
    "compute_signal",
]


@dataclass(frozen=True)
class MicropriceSignal:
    microprice: float | None
    best_bid: float | None
    best_ask: float | None
    imbalance: float | None
    offset_ticks: float | None


def compute_signal(orderbook: Orderbook | None, *, tick_size: float = 0.01) -> MicropriceSignal:
    """Return microprice-derived signal using top-of-book depth weighting."""

    if orderbook is None or not orderbook.bids or not orderbook.asks:
        return MicropriceSignal(None, None, None, None, None)
    try:
        best_bid_entry = orderbook.bids[0]
        best_ask_entry = orderbook.asks[0]
        best_bid = float(best_bid_entry.get("price"))
        best_ask = float(best_ask_entry.get("price"))
        bid_size = max(float(best_bid_entry.get("size", 0.0)), 0.0)
        ask_size = max(float(best_ask_entry.get("size", 0.0)), 0.0)
    except (TypeError, ValueError, KeyError):
        return MicropriceSignal(None, None, None, None, None)
    depth = bid_size + ask_size
    if depth <= 0.0 or best_ask <= best_bid:
        return MicropriceSignal(None, best_bid, best_ask, None, None)
    microprice = (best_ask * bid_size + best_bid * ask_size) / depth
    mid = 0.5 * (best_bid + best_ask)
    imbalance = (bid_size - ask_size) / depth if depth > 0 else None
    tick = max(float(tick_size), 1e-4)
    offset_ticks = (microprice - mid) / tick
    return MicropriceSignal(microprice, best_bid, best_ask, imbalance, offset_ticks)


class ReplacementThrottle:
    """Simple time-window throttle that bounds replacements per ladder bin."""

    def __init__(self, *, max_per_window: int = 3, window_seconds: float = 0.75) -> None:
        self.max_per_window = max(int(max_per_window), 1)
        self.window = timedelta(seconds=max(window_seconds, 0.1))
        self._events: dict[str, Deque[datetime]] = {}

    def should_block(self, key: str, now: datetime | None = None) -> bool:
        now_ts = now or datetime.now(tz=UTC)
        window = self._events.setdefault(key, deque())
        self._prune(window, now_ts)
        return len(window) >= self.max_per_window

    def record(self, key: str, now: datetime | None = None) -> None:
        now_ts = now or datetime.now(tz=UTC)
        window = self._events.setdefault(key, deque())
        self._prune(window, now_ts)
        window.append(now_ts)

    def snapshot(self) -> dict[str, int]:
        return {key: len(events) for key, events in self._events.items() if events}

    def _prune(self, window: Deque[datetime], now: datetime) -> None:
        cutoff = now - self.window
        while window and window[0] < cutoff:
            window.popleft()

