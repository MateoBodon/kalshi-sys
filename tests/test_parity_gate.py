from __future__ import annotations

from pathlib import Path

import polars as pl

import scripts.parity_gate as parity_gate


def test_parity_gate_allows_small_delta(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "maker_ev_per_contract_replay": [0.12, 0.05],
            "maker_ev_original": [0.11, 0.05],
            "contracts": [10, 5],
        }
    )
    path = tmp_path / "replay.parquet"
    frame.write_parquet(path)
    parity_gate.main(["--threshold", "0.2", "--path", str(path)])
