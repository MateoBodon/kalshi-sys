from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from scripts import proof_of_fill


def _write_manifest(tmp_path: Path, proposals_path: Path) -> Path:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"proposals_path": str(proposals_path)}),
        encoding="utf-8",
    )
    return manifest_path


def _write_proposals(path: Path, market_ticker: str, market_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "proposals": [
                {
                    "market_ticker": market_ticker,
                    "market_id": market_id,
                }
            ]
        }),
        encoding="utf-8",
    )


def test_proof_of_fill_generates_parquet_and_tables(tmp_path, capsys):
    trading_day = "2025-11-10"
    market_ticker = "KXINXU-25NOV10H1200-T5000"
    market_id = "MKT_INXU_H1200_A"
    manifest_dir = tmp_path / "data" / "raw" / "kalshi" / "2025-11-10" / "005000" / "INXU"
    proposals_path = tmp_path / "reports" / "index_ladders" / "INXU" / f"{trading_day}.json"
    manifest_path = _write_manifest(manifest_dir, proposals_path)
    _write_proposals(proposals_path, market_ticker, market_id)

    timestamp = datetime(2025, 11, 10, 11, 55, tzinfo=ZoneInfo("America/New_York"))
    ledger = pl.DataFrame(
        {
            "series": ["INXU"],
            "market": [market_ticker],
            "price": [0.42],
            "side": ["YES"],
            "size": [1],
            "ev_after_fees": [0.12],
            "pnl_simulated": [0.15],
            "fill_ratio": [0.5],
            "fill_ratio_observed": [0.5],
            "timestamp_et": [timestamp.astimezone(ZoneInfo("UTC"))],
            "manifest_path": [manifest_path.as_posix()],
            "ledger_schema_version": [2],
        }
    )
    ledger_path = tmp_path / "ledger.parquet"
    ledger.write_parquet(ledger_path)

    orders_path = tmp_path / "orders.json"
    orders_path.write_text(
        json.dumps(
            [
                {
                    "order_id": "ORD-1",
                    "market_id": market_id,
                    "side": "YES",
                    "price": 0.42,
                    "contracts": 1,
                    "filled_contracts": 1,
                    "status": "filled",
                    "created_time": timestamp.astimezone(ZoneInfo("UTC")).isoformat(),
                }
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "artifacts"
    exit_code = proof_of_fill.main(
        [
            f"--start={trading_day}",
            f"--end={trading_day}",
            f"--ledger={ledger_path}",
            f"--orders-json={orders_path}",
            f"--output-dir={output_dir}",
            f"--reports-root={tmp_path}",
        ]
    )
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "Window Summaries" in out
    assert "Orders & Fills" in out

    parquet_path = output_dir / f"pnl_window_{trading_day}.parquet"
    assert parquet_path.exists()
    data = pl.read_parquet(parquet_path)
    assert set(data["scope"].to_list()) == {"window", "day"}
    window_rows = data.filter(pl.col("scope") == "window")
    assert window_rows.height == 1
    assert window_rows["ev_after_fees"].item() == 0.12
