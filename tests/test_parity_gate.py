from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import scripts.parity_gate as parity_gate


def test_parity_gate_allows_small_delta(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "maker_ev_per_contract_replay": [0.12, 0.05],
            "maker_ev_original": [0.11, 0.05],
            "contracts": [10, 5],
            "window_type": ["hourly", "close"],
            "window_label": ["INXU 2025-11-10 12:00", "INX 2025-11-10 16:00"],
        }
    )
    path = tmp_path / "replay.parquet"
    frame.write_parquet(path)
    output = tmp_path / "ev_gap.json"
    parity_gate.main(
        [
            "--threshold",
            "0.2",
            "--path",
            str(path),
            "--output-json",
            str(output),
        ]
    )
    payload = json.loads(output.read_text())
    assert set(payload["by_window_type"]) == {"hourly", "close"}
    assert "INXU 2025-11-10 12:00" in payload["by_window"]


def test_parity_gate_blocks_per_window(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "maker_ev_per_contract_replay": [0.5, 0.02],
            "maker_ev_original": [0.1, 0.02],
            "contracts": [1, 1],
            "window_type": ["hourly", "close"],
            "window_label": ["INXU noon", "INX close"],
        }
    )
    path = tmp_path / "replay.parquet"
    frame.write_parquet(path)
    with pytest.raises(SystemExit):
        parity_gate.main(["--threshold", "0.3", "--path", str(path)])
