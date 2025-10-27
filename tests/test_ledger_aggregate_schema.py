from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.exec.ledger.aggregate import main as aggregate_main
from kalshi_alpha.exec.ledger.schema import LedgerRowV1


def _ledger_frame() -> pl.DataFrame:
    now = datetime.now(tz=UTC)
    return pl.DataFrame(
        {
            "series": ["CPI"],
            "event": ["EV1"],
            "market": ["MKT"],
            "bin": [270.0],
            "side": ["YES"],
            "price": [0.45],
            "model_p": [0.55],
            "market_p": [0.40],
            "delta_p": [0.15],
            "size": [20],
            "expected_contracts": [18],
            "expected_fills": [16],
            "fill_ratio": [0.8],
            "slippage_mode": ["top"],
            "impact_cap": [0.02],
            "fees_maker": [0.12],
            "ev_after_fees": [2.4],
            "pnl_simulated": [2.1],
            "timestamp_et": [now],
            "manifest_path": ["data/raw/kalshi/manifest.json"],
            "ledger_schema_version": [1],
        }
    )


def test_aggregate_reorders_to_canonical(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    reports_dir = tmp_path / "reports" / "_artifacts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    frame = _ledger_frame()
    shuffled = frame.select(list(reversed(frame.columns)))
    shuffled.write_csv(reports_dir / "sample_ledger.csv")

    aggregate_main([])

    parquet_path = tmp_path / "data" / "proc" / "ledger_all.parquet"
    assert parquet_path.exists()
    combined = pl.read_parquet(parquet_path)
    assert combined.columns == list(LedgerRowV1.canonical_fields())


def test_aggregate_rejects_unknown_columns(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    reports_dir = tmp_path / "reports" / "_artifacts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    frame = _ledger_frame().with_columns(pl.lit(1).alias("extra"))
    frame.write_csv(reports_dir / "bad_ledger.csv")

    with pytest.raises(ValueError):
        aggregate_main([])
