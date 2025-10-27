from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from kalshi_alpha.strategies.cpi import CPIInputs, calibrate, nowcast

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


def _expected_mean(distribution: dict[float, float]) -> float:
    return sum(point * weight for point, weight in distribution.items())


def test_cpi_calibration_writes_parquet(isolated_data_roots: tuple[Path, Path]) -> None:
    _, proc_root = isolated_data_roots
    history = json.loads((FIXTURE_ROOT / "cpi" / "history.json").read_text(encoding="utf-8"))["history"]

    frame = calibrate(history)

    path = proc_root / "cpi_calib.parquet"
    assert path.exists()

    summary = frame.filter(pl.col("record_type") == "params")
    assert not summary.is_empty()
    row = summary.row(0, named=True)
    assert row["crps"] < row["baseline_crps"]
    assert row["bias"] is not None and row["std"] is not None

    evaluations = frame.filter(pl.col("record_type") == "evaluation")
    assert evaluations.height == len(history)
    assert evaluations["yoy_prediction"].drop_nulls().len() > 0

    # Ensure calibration effects are applied in nowcast
    distribution = nowcast(
        CPIInputs(cleveland_nowcast=0.32, latest_release_mom=0.30, aaa_delta=0.02)
    )
    mean = _expected_mean(distribution)
    assert 0.25 <= mean <= 0.45
