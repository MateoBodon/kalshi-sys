"""Utilities for consuming Kalshi orderbook websockets and deriving imbalance metrics."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import websockets

from kalshi_alpha.core.ws import DEFAULT_WS_URL, KalshiWebsocketClient
from kalshi_alpha.datastore.paths import PROC_ROOT, RAW_ROOT

WS_ENDPOINT = DEFAULT_WS_URL
ET_ZONE = ZoneInfo("America/New_York")
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


class OrderbookSnapshotWriter:
    """Append orderbook snapshots to newline-delimited JSON files."""

    def __init__(
        self,
        ticker: str,
        *,
        root: Path = RAW_ORDERBOOK_ROOT,
        started_at: datetime | None = None,
        run_id: str | None = None,
    ) -> None:
        self._ticker = ticker.upper()
        baseline = started_at or datetime.now(tz=UTC)
        label = run_id or self._format_label(baseline)
        directory = root / self._ticker
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{label}.jsonl"
        self._file = self.path.open("a", encoding="utf-8")

    @staticmethod
    def _format_label(moment: datetime) -> str:
        reference = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
        return reference.astimezone(ET_ZONE).strftime("%Y%m%d_%H%M%S")

    def append(
        self,
        snapshot: dict[str, Any],
        *,
        timestamp: datetime | None = None,
        imbalance: float | None = None,
    ) -> None:
        ts = timestamp or datetime.now(tz=UTC)
        payload: dict[str, Any] = {
            "ticker": self._ticker,
            "updated_at": ts.astimezone(UTC).isoformat(),
            "snapshot": snapshot,
        }
        if imbalance is not None:
            payload["rolling_imbalance"] = float(imbalance)
        self._file.write(json.dumps(payload) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def persist_orderbook_snapshot(ticker: str, snapshot: dict[str, Any], *, timestamp: datetime | None = None) -> Path:
    ts = timestamp or datetime.now(tz=UTC)
    writer = OrderbookSnapshotWriter(ticker, root=RAW_ORDERBOOK_ROOT, started_at=ts)
    writer.append(snapshot, timestamp=ts)
    writer.close()
    return writer.path


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


async def stream_orderbook_imbalance(  # noqa: PLR0913
    tickers: Sequence[str],
    *,
    depth: int = 3,
    window_seconds: int = 30,
    ws_url: str = WS_ENDPOINT,
    client: KalshiWebsocketClient | None = None,
    run_seconds: float | None = None,
    now_fn: Callable[[], datetime] | None = None,
    writer_root: Path | None = None,
    reader_timeout: float = 5.0,
    auth_token: str | None = None,
) -> dict[str, float]:
    """Consume orderbook deltas for ``tickers`` and persist rolling imbalance metrics."""

    if auth_token is not None:  # pragma: no cover - retained for backwards compatibility
        raise ValueError("Token-based authentication is no longer supported for Kalshi websockets.")

    normalized = [ticker.upper() for ticker in tickers if ticker]
    if not normalized:
        return {}

    factory = client or KalshiWebsocketClient(base_url=ws_url)
    clock = now_fn or (lambda: datetime.now(tz=UTC))
    started_at = clock()
    end_time = started_at + timedelta(seconds=float(run_seconds)) if run_seconds else None
    root = writer_root or RAW_ORDERBOOK_ROOT

    trackers = {
        ticker: OrderbookImbalanceTracker(depth=depth, window_seconds=window_seconds)
        for ticker in normalized
    }
    writers: dict[str, OrderbookSnapshotWriter] = {}
    latest: dict[str, float | None] = {ticker: None for ticker in normalized}

    async with factory.session() as websocket:
        subscribe_message = {
            "type": "subscribe",
            "channel": "orderbook_delta",
            "tickers": normalized,
        }
        await websocket.send(json.dumps(subscribe_message))

        while True:
            if end_time is not None and clock() >= end_time:
                break
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=reader_timeout)
            except TimeoutError:
                continue
            except websockets.ConnectionClosed:
                break

            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue

            ticker = _extract_ticker(payload)
            if ticker not in trackers:
                continue
            snapshot = _extract_snapshot(payload)
            now = clock()
            tracker = trackers[ticker]
            tracker.update(snapshot, timestamp=now)
            average = tracker.rolling_average(now=now)
            writer = writers.setdefault(
                ticker,
                OrderbookSnapshotWriter(ticker, root=root, started_at=started_at),
            )
            writer.append(snapshot, timestamp=now, imbalance=average)
            if average is not None:
                persist_imbalance_metric(ticker, average, timestamp=now)
                latest[ticker] = average

    for writer in writers.values():
        writer.close()

    return {ticker: value for ticker, value in latest.items() if value is not None}


def replay_snapshots(paths: Iterable[Path], *, depth: int = 3, window_seconds: int = 30) -> dict[str, float]:
    """Utility to recompute imbalance metrics from existing snapshots."""

    trackers: dict[str, OrderbookImbalanceTracker] = {}
    for path in sorted(paths):
        try:
            contents = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in contents.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            ticker = str(payload.get("ticker", "")).upper()
            if not ticker:
                continue
            tracker = trackers.setdefault(
                ticker,
                OrderbookImbalanceTracker(depth=depth, window_seconds=window_seconds),
            )
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


def _extract_ticker(payload: dict[str, Any]) -> str:
    ticker = payload.get("ticker")
    if isinstance(ticker, str) and ticker:
        return ticker.upper()
    data = payload.get("data")
    if isinstance(data, dict):
        ticker_value = data.get("ticker")
        if isinstance(ticker_value, str) and ticker_value:
            return ticker_value.upper()
    return ""


def _extract_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict):
        bids = data.get("bids") or data.get("bid_levels") or []
        asks = data.get("asks") or data.get("ask_levels") or []
    else:
        bids = payload.get("bids") or []
        asks = payload.get("asks") or []
    return {"bids": list(bids), "asks": list(asks)}


__all__ = [
    "OrderbookImbalanceTracker",
    "OrderbookSnapshotWriter",
    "compute_imbalance",
    "load_latest_imbalance",
    "persist_orderbook_snapshot",
    "persist_imbalance_metric",
    "replay_snapshots",
    "stream_orderbook_imbalance",
]
