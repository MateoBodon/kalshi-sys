"""Utilities for consuming Kalshi orderbook websockets and deriving imbalance metrics."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import websockets

from kalshi_alpha.datastore.paths import PROC_ROOT, RAW_ROOT

WS_ENDPOINT = "wss://trading-api.kalshi.com/trade-api/ws"
RAW_ORDERBOOK_ROOT = RAW_ROOT / "kalshi" / "orderbook"
PROC_IMBALANCE_ROOT = PROC_ROOT / "kalshi" / "orderbook_imbalance"


@dataclass(slots=True)
class OrderbookLevel:
    price: float
    quantity: float


def compute_imbalance(
    bids: Sequence[OrderbookLevel] | Sequence[dict[str, float]],
    asks: Sequence[OrderbookLevel] | Sequence[dict[str, float]],
    *,
    depth: int = 3,
) -> float:
    """Return the normalized bid/ask imbalance for the top-of-book."""

    def _volume(levels: Sequence[OrderbookLevel] | Sequence[dict[str, float]]) -> float:
        total = 0.0
        for idx, level in enumerate(levels):
            if idx >= depth:
                break
            quantity = getattr(level, "quantity", None)
            if quantity is None:
                quantity = float(level.get("quantity", 0.0))
            total += max(float(quantity), 0.0)
        return total

    bid_volume = _volume(bids)
    ask_volume = _volume(asks)
    denom = bid_volume + ask_volume
    if denom <= 0.0:
        return 0.0
    return (bid_volume - ask_volume) / denom


class OrderbookImbalanceTracker:
    """Track rolling orderbook imbalance for a single market."""

    def __init__(self, *, depth: int = 3, window_seconds: int = 30) -> None:
        self._depth = max(1, depth)
        self._window = max(1, window_seconds)
        self._series: deque[tuple[datetime, float]] = deque()

    def update(self, snapshot: dict[str, Any], *, timestamp: datetime | None = None) -> float:
        ts = timestamp or datetime.now(tz=UTC)
        bids = snapshot.get("bids", [])
        asks = snapshot.get("asks", [])
        imbalance = compute_imbalance(bids, asks, depth=self._depth)
        self._series.append((ts, imbalance))
        self._prune(ts)
        return imbalance

    def rolling_average(self, *, now: datetime | None = None) -> float | None:
        if not self._series:
            return None
        reference = now or self._series[-1][0]
        self._prune(reference)
        if not self._series:
            return None
        total = sum(value for _, value in self._series)
        return total / len(self._series)

    def _prune(self, reference: datetime) -> None:
        threshold = reference - timedelta(seconds=self._window)
        while self._series and self._series[0][0] < threshold:
            self._series.popleft()


def persist_orderbook_snapshot(ticker: str, snapshot: dict[str, Any], *, timestamp: datetime | None = None) -> Path:
    ts = timestamp or datetime.now(tz=UTC)
    output_dir = RAW_ORDERBOOK_ROOT / ticker.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"{ts.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    payload = {
        "ticker": ticker.upper(),
        "updated_at": ts.astimezone(UTC).isoformat(),
        "snapshot": snapshot,
    }
    filename.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return filename


def persist_imbalance_metric(ticker: str, imbalance: float, *, timestamp: datetime | None = None) -> Path:
    ts = timestamp or datetime.now(tz=UTC)
    PROC_IMBALANCE_ROOT.mkdir(parents=True, exist_ok=True)
    path = PROC_IMBALANCE_ROOT / f"{ticker.upper()}.json"
    payload = {
        "ticker": ticker.upper(),
        "imbalance": float(imbalance),
        "updated_at": ts.astimezone(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_latest_imbalance(ticker: str) -> tuple[float, datetime] | None:
    path = PROC_IMBALANCE_ROOT / f"{ticker.upper()}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = payload.get("imbalance")
    updated = payload.get("updated_at")
    if value is None or updated is None:
        return None
    try:
        moment = datetime.fromisoformat(str(updated))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
    except ValueError:
        moment = datetime.now(tz=UTC)
    return float(value), moment


async def stream_orderbook_imbalance(
    tickers: Sequence[str],
    *,
    depth: int = 3,
    window_seconds: int = 30,
    ws_url: str = WS_ENDPOINT,
    auth_token: str | None = None,
) -> None:
    """Consume orderbook deltas for ``tickers`` and persist rolling imbalance metrics."""

    trackers = {
        ticker.upper(): OrderbookImbalanceTracker(depth=depth, window_seconds=window_seconds)
        for ticker in tickers
    }

    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    async with websockets.connect(ws_url, extra_headers=headers, ping_interval=30) as websocket:
        subscribe_message = {
            "type": "subscribe",
            "channels": [
                {
                    "name": "orderbook",
                    "tickers": [ticker.upper() for ticker in tickers],
                }
            ],
        }
        await websocket.send(json.dumps(subscribe_message))

        async for message in websocket:
            payload = json.loads(message)
            ticker = str(payload.get("ticker", "")).upper()
            if ticker not in trackers:
                continue
            data = payload.get("data") or {}
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            snapshot = {"bids": bids, "asks": asks}
            now = datetime.now(tz=UTC)
            tracker = trackers[ticker]
            tracker.update(snapshot, timestamp=now)
            average = tracker.rolling_average(now=now)
            persist_orderbook_snapshot(ticker, snapshot, timestamp=now)
            if average is not None:
                persist_imbalance_metric(ticker, average, timestamp=now)


def replay_snapshots(paths: Iterable[Path], *, depth: int = 3, window_seconds: int = 30) -> dict[str, float]:
    """Utility to recompute imbalance metrics from existing snapshots."""

    trackers: dict[str, OrderbookImbalanceTracker] = {}
    for path in sorted(paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        ticker = str(payload.get("ticker", "")).upper()
        if not ticker:
            continue
        tracker = trackers.setdefault(ticker, OrderbookImbalanceTracker(depth=depth, window_seconds=window_seconds))
        snapshot = payload.get("snapshot") or {}
        timestamp_raw = payload.get("updated_at")
        try:
            timestamp = datetime.fromisoformat(str(timestamp_raw)) if timestamp_raw else None
        except ValueError:
            timestamp = None
        tracker.update(snapshot, timestamp=timestamp)
    results: dict[str, float] = {}
    for ticker, tracker in trackers.items():
        average = tracker.rolling_average()
        if average is not None:
            results[ticker] = average
    return results


__all__ = [
    "OrderbookImbalanceTracker",
    "compute_imbalance",
    "load_latest_imbalance",
    "persist_orderbook_snapshot",
    "persist_imbalance_metric",
    "replay_snapshots",
    "stream_orderbook_imbalance",
]
