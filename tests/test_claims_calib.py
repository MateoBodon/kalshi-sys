from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from kalshi_alpha.strategies.claims import ClaimsInputs, calibrate, pmf

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


def test_claims_calibration_metrics(isolated_data_roots: tuple[Path, Path]) -> None:
    _, proc_root = isolated_data_roots
    history = json.loads((FIXTURE_ROOT / "claims" / "history.json").read_text(encoding="utf-8"))["history"]

    frame = calibrate(history)

    path = proc_root / "claims_calib.parquet"
    assert path.exists()

    summary = frame.filter(pl.col("record_type") == "params")
    row = summary.row(0, named=True)
    assert row["brier"] < row["baseline_brier"]
    assert row["crps"] is not None
    assert row["std"] and row["holiday_lift"] is not None

    # Ensure PMF picks up calibration parameters
    observed = [entry["claims"] for entry in history]
    inputs = ClaimsInputs(history=observed[:-1], holiday_next=True, freeze_active=False)
    strikes = [200_000, 205_000, 210_000, 215_000, 220_000, 225_000]
    ladder = pmf(strikes, inputs=inputs)
    assert len(ladder) == len(strikes) + 1
    probs = [bin_prob.probability for bin_prob in ladder]
    assert abs(sum(probs) - 1.0) < 1e-6
