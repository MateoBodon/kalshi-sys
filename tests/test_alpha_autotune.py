from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.core.execution.fillratio import load_alpha, tune_alpha


def test_tune_alpha_uses_ledger_and_updates_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    proc_dir = tmp_path / "data" / "proc"
    proc_dir.mkdir(parents=True, exist_ok=True)
    timestamp_recent = datetime.now(tz=UTC) - timedelta(days=1)
    ledger = pl.DataFrame(
        {
            "series": ["CPI", "CPI"],
            "size": [100, 40],
            "size_partial": [20, 8],
            "timestamp_et": [timestamp_recent, timestamp_recent],
        }
    )
    ledger.write_parquet(proc_dir / "ledger_all.parquet")

    alpha = tune_alpha("CPI", lookback_days=7, min_observations=1)
    assert alpha is not None
    assert alpha == pytest.approx(0.8, rel=1e-3)

    state_path = proc_dir / "state" / "fill_alpha.json"
    assert state_path.exists()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["series"]["CPI"]["alpha"] == round(alpha, 4)
    assert load_alpha("CPI") == pytest.approx(alpha, rel=1e-3)
