from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from kalshi_alpha.core.execution.fillratio import tune_alpha


def test_tune_alpha_uses_archives_and_updates_state(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    archives_dir = tmp_path / "data" / "raw" / "kalshi"
    manifest_dir = archives_dir / "2025-01-01" / "000000" / "CPI"
    orderbooks_dir = manifest_dir / "orderbooks"
    orderbooks_dir.mkdir(parents=True, exist_ok=True)
    market_id = "CALSHI_CPI_1"
    orderbook_payload = {
        "market_id": market_id,
        "bids": [{"price": 0.45, "size": 30}],
        "asks": [{"price": 0.45, "size": 50}],
    }
    (orderbooks_dir / f"{market_id}.json").write_text(json.dumps(orderbook_payload), encoding="utf-8")
    manifest = {
        "paths": {
            "orderbooks": [f"orderbooks/{market_id}.json"],
        }
    }
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # Ledger entries referencing manifest
    proc_dir = tmp_path / "data" / "proc"
    proc_dir.mkdir(parents=True, exist_ok=True)
    timestamp_recent = datetime.now(tz=UTC) - timedelta(days=1)
    ledger = pl.DataFrame(
        {
            "series": ["CPI", "CPI"],
            "market": [market_id, market_id],
            "bin": [270.0, 271.0],
            "side": ["YES", "YES"],
            "price": [0.45, 0.45],
            "size": [40, 50],
            "expected_fills": [20, 35],
            "fees_maker": [0.1, 0.1],
            "manifest_path": [manifest_path.as_posix(), manifest_path.as_posix()],
            "timestamp_et": [timestamp_recent, timestamp_recent],
        }
    )
    ledger.write_parquet(proc_dir / "ledger_all.parquet")

    alpha = tune_alpha("CPI", archives_dir)
    assert alpha is not None
    assert 0.4 <= alpha <= 0.8

    state_path = proc_dir / "state" / "fill_alpha.json"
    assert state_path.exists()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["series"]["CPI"]["alpha"] == round(alpha, 4)
