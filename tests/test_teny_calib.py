from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl

from kalshi_alpha.drivers.macro_calendar import emit_day_dummies
from kalshi_alpha.strategies.teny import TenYInputs, calibrate, pmf

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


def test_teny_calibration_factors(
    isolated_data_roots: tuple[Path, Path],
    offline_fixtures_root: Path,
) -> None:
    _, proc_root = isolated_data_roots
    history = json.loads((FIXTURE_ROOT / "teny" / "history.json").read_text(encoding="utf-8"))["history"]

    start = date.fromisoformat(history[0]["date"])
    end = date.fromisoformat(history[-1]["date"])
    emit_day_dummies(start, end, offline=True, fixtures_dir=offline_fixtures_root)

    frame = calibrate(history)

    path = proc_root / "teny_calib.parquet"
    assert path.exists()

    summary = frame.filter(pl.col("record_type") == "params")
    row = summary.row(0, named=True)
    assert row["shock_beta"] is not None
    assert row["residual_std"] is not None
    assert row["crps"] < row["baseline_crps"]

    evaluation = frame.filter(pl.col("record_type") == "evaluation")
    assert "is_claims" in evaluation.columns
    claims_sum = evaluation.select(pl.col("is_claims").sum()).to_series().item()
    assert claims_sum >= 1.0

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
