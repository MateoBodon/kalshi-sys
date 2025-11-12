from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from tools import replay as replay_tool


def test_discover_manifests_finds_all_series(tmp_path: Path) -> None:
    base = tmp_path / "kalshi"
    manifest_dir = base / "2025-11-10" / "175613" / "INXU"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    paths = replay_tool._discover_manifests(base, date(2025, 11, 10), ["INXU"])
    assert paths == [manifest_path]


def test_normalize_series_column_handles_aliases() -> None:
    frame = pl.DataFrame({"series": ["SPX", "NDX", "INXU"]})
    normalized = replay_tool._normalize_series_column(frame)
    assert normalized["series"].to_list() == ["INX", "NASDAQ100", "INXU"]


def test_apply_epsilon_filter_drops_small_deltas() -> None:
    frame = pl.DataFrame(
        {
            "maker_ev_original": [1.0, 2.0],
            "contracts": [2, 2],
            "maker_ev_per_contract_replay": [0.4, 1.0],
        }
    )
    filtered = replay_tool._apply_epsilon_filter(frame, epsilon_cents=15.0)
    # First row delta = |0.5-0.4|*100 = 10 < 15, second delta = |1.0-1.0|*100 = 0
    assert filtered.is_empty()
