from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import websockets

from kalshi_alpha.core import kalshi_ws


def _sample_levels() -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    bids = [
        {"price": 0.45, "quantity": 12},
        {"price": 0.44, "quantity": 8},
    ]
    asks = [
        {"price": 0.55, "quantity": 6},
        {"price": 0.56, "quantity": 9},
    ]
    return bids, asks


def test_compute_imbalance() -> None:
    bids, asks = _sample_levels()
    value = kalshi_ws.compute_imbalance(bids, asks, depth=2)
    expected = (12 + 8 - (6 + 9)) / (12 + 8 + 6 + 9)
    assert value == expected


def test_tracker_updates_and_window(monkeypatch) -> None:
    tracker = kalshi_ws.OrderbookImbalanceTracker(depth=2, window_seconds=60)
    bids, asks = _sample_levels()
    now = datetime(2025, 10, 24, 19, 0, tzinfo=UTC)
    tracker.update({"bids": bids, "asks": asks}, timestamp=now)
    later = now + timedelta(seconds=30)
    tracker.update({"bids": bids, "asks": asks}, timestamp=later)
    average = tracker.rolling_average(now=later)
    assert average is not None
    assert abs(average - tracker.rolling_average()) < 1e-9


def test_persist_snapshot_and_metric(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(kalshi_ws, "RAW_ORDERBOOK_ROOT", tmp_path / "raw")
    monkeypatch.setattr(kalshi_ws, "PROC_IMBALANCE_ROOT", tmp_path / "proc")
    bids, asks = _sample_levels()
    now = datetime.now(tz=UTC)
    snapshot_path = kalshi_ws.persist_orderbook_snapshot("TNEY", {"bids": bids, "asks": asks}, timestamp=now)
    assert snapshot_path.exists()
    assert snapshot_path.suffix == ".jsonl"
    payload = pl.read_ndjson(snapshot_path)
    assert payload.height == 1

    metric_path = kalshi_ws.persist_imbalance_metric("TNEY", imbalance=0.25, timestamp=now)
    assert metric_path.exists()
    loaded = kalshi_ws.load_latest_imbalance("TNEY")
    assert loaded is not None
    value, moment = loaded
    assert value == 0.25
    assert isinstance(moment, datetime)


def test_replay_snapshots(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(kalshi_ws, "RAW_ORDERBOOK_ROOT", tmp_path / "raw")
    monkeypatch.setattr(kalshi_ws, "PROC_IMBALANCE_ROOT", tmp_path / "proc")
    bids, asks = _sample_levels()
    root = kalshi_ws.RAW_ORDERBOOK_ROOT / "TNEY"
    root.mkdir(parents=True, exist_ok=True)
    for offset in range(3):
        ts = datetime(2025, 10, 24, 19, 0, tzinfo=UTC) + timedelta(seconds=offset * 10)
        kalshi_ws.persist_orderbook_snapshot("TNEY", {"bids": bids, "asks": asks}, timestamp=ts)
    values = kalshi_ws.replay_snapshots(root.glob("*.jsonl"))
    assert "TNEY" in values
    assert abs(values["TNEY"]) <= 1.0


class _FakeWebsocket:
    def __init__(self, messages: list[str]) -> None:
        self._messages = messages
        self._index = 0
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        if self._index >= len(self._messages):
            raise websockets.ConnectionClosedOK(0, "done")
        message = self._messages[self._index]
        self._index += 1
        return message

    async def close(self) -> None:
        self.closed = True


@asynccontextmanager
async def _fake_session(messages: list[str]):
    websocket = _FakeWebsocket(messages)
    try:
        yield websocket
    finally:
        await websocket.close()


class _FakeClient:
    def __init__(self, messages: list[str]) -> None:
        self._messages = messages

    @asynccontextmanager
    async def session(self):
        async with _fake_session(self._messages) as websocket:
            yield websocket


@pytest.mark.asyncio
async def test_stream_orderbook_imbalance_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kalshi_ws, "RAW_ORDERBOOK_ROOT", tmp_path / "raw")
    monkeypatch.setattr(kalshi_ws, "PROC_IMBALANCE_ROOT", tmp_path / "proc")

    bids, asks = _sample_levels()
    messages = [
        json.dumps(
            {
                "type": "orderbook_delta",
                "ticker": "TNEY",
                "data": {"bids": bids, "asks": asks},
            }
        )
    ]

    start = datetime(2025, 10, 24, 19, 0, tzinfo=UTC)
    timeline = [start, start, start + timedelta(seconds=10)]

    def _clock_factory(times: list[datetime]) -> Callable[[], datetime]:
        iterator = iter(times)
        last = times[-1]

        def _inner() -> datetime:
            nonlocal last
            try:
                last = next(iterator)
            except StopIteration:
                pass
            return last

        return _inner

    clock = _clock_factory(timeline)
    client = _FakeClient(messages)
    result = await kalshi_ws.stream_orderbook_imbalance(
        ["TNEY"],
        client=client,
        run_seconds=0.5,
        now_fn=clock,
        reader_timeout=0.05,
    )

    assert "TNEY" in result
    raw_files = list((tmp_path / "raw" / "TNEY").glob("*.jsonl"))
    assert raw_files
    lines = raw_files[0].read_text(encoding="utf-8").splitlines()
    assert lines and json.loads(lines[0])["ticker"] == "TNEY"

    metric = kalshi_ws.load_latest_imbalance("TNEY")
    assert metric is not None
