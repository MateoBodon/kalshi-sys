from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

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
    payload = pl.read_json(snapshot_path)
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
    values = kalshi_ws.replay_snapshots(root.glob("*.json"))
    assert "TNEY" in values
    assert abs(values["TNEY"]) <= 1.0
