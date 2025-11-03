from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from kalshi_alpha.core import kalshi_ws
from kalshi_alpha.dev import ws_smoke


async def _fake_stream(tickers: list[str], **_: Any) -> dict[str, float]:
    results: dict[str, float] = {}
    for ticker in tickers:
        writer = kalshi_ws.OrderbookSnapshotWriter(
            ticker,
            root=kalshi_ws.RAW_ORDERBOOK_ROOT,
            started_at=datetime.now(tz=UTC),
        )
        snapshot = {
            "bids": [{"price": 0.51, "quantity": 5}],
            "asks": [{"price": 0.52, "quantity": 4}],
        }
        writer.append(snapshot, imbalance=0.25)
        writer.close()
        kalshi_ws.persist_imbalance_metric(ticker, 0.25)
        results[ticker] = 0.25
    return results


def test_ws_smoke_writes_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    raw_root = tmp_path / "raw"
    proc_root = tmp_path / "proc"
    raw_root.mkdir()
    proc_root.mkdir()
    monkeypatch.setattr(kalshi_ws, "RAW_ORDERBOOK_ROOT", raw_root)
    monkeypatch.setattr(kalshi_ws, "PROC_IMBALANCE_ROOT", proc_root)
    monkeypatch.setattr(kalshi_ws, "stream_orderbook_imbalance", _fake_stream)

    result = ws_smoke.main(["--tickers", "TNEY-TEST", "--run-seconds", "0.1"])
    assert result == {"TNEY-TEST": 0.25}

    metric = kalshi_ws.PROC_IMBALANCE_ROOT / "TNEY-TEST.json"
    assert metric.exists()
    loaded = kalshi_ws.load_latest_imbalance("TNEY-TEST")
    assert loaded is not None
    value, updated_at = loaded
    assert value == pytest.approx(0.25)
    assert updated_at.tzinfo is not None

    snapshot_dir = kalshi_ws.RAW_ORDERBOOK_ROOT / "TNEY-TEST"
    files = list(snapshot_dir.glob("*.jsonl"))
    assert len(files) == 1

    out = capsys.readouterr().out
    assert "Recorded imbalance metrics" in out
    assert "metric=" in out
