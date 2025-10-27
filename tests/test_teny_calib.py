from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from kalshi_alpha.strategies.teny import TenYInputs, calibrate, pmf

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


def test_teny_calibration_factors(isolated_data_roots: tuple[Path, Path]) -> None:
    _, proc_root = isolated_data_roots
    history = json.loads((FIXTURE_ROOT / "teny" / "history.json").read_text(encoding="utf-8"))["history"]

    frame = calibrate(history)

    path = proc_root / "teny_calib.parquet"
    assert path.exists()

    summary = frame.filter(pl.col("record_type") == "params")
    row = summary.row(0, named=True)
    assert row["shock_beta"] is not None
    assert row["residual_std"] is not None
    assert row["crps"] < row["baseline_crps"]

    strikes = [4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9]
    last_entry = history[-1]
    trailing_history = [record["actual_close"] for record in history[:-1]]
    inputs = TenYInputs(
        prior_close=history[-2]["actual_close"],
        macro_shock=last_entry["macro_shock"],
        trailing_history=trailing_history,
    )
    ladder = pmf(strikes, inputs=inputs)
    probs = [segment.probability for segment in ladder]
    assert abs(sum(probs) - 1.0) < 1e-6
