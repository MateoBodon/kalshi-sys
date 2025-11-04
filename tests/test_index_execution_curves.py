from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.core.execution.index_models import load_alpha_curve, load_slippage_curve

INDEX_SERIES = ("INXU", "NASDAQ100U", "INX", "NASDAQ100")
HOLDOUT_PATH = Path("tests/data_fixtures/index_exec_holdout.parquet")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_abs_error(predictions: list[float], observations: list[float]) -> float:
    if not predictions:
        return 0.0
    total = sum(abs(p - o) for p, o in zip(predictions, observations, strict=True))
    return total / len(predictions)


def _load_holdout() -> pl.DataFrame:
    assert HOLDOUT_PATH.exists(), "Holdout fixture missing"
    return pl.read_parquet(HOLDOUT_PATH)


def test_alpha_curve_training_metrics_improve() -> None:
    for series in INDEX_SERIES:
        alpha_path = Path(f"data/proc/index_exec/{series}/alpha.json")
        assert alpha_path.exists(), f"Expected alpha curve for {series}"
        payload = _load_json(alpha_path)
        assert payload["model_mse"] <= payload["baseline_mse"]


def test_slippage_curve_training_metrics_improve() -> None:
    for series in INDEX_SERIES:
        slip_path = Path(f"data/proc/index_exec/{series}/slippage.json")
        assert slip_path.exists(), f"Expected slippage curve for {series}"
        payload = _load_json(slip_path)
        assert payload["model_mse"] <= payload["baseline_mse"]


@pytest.mark.parametrize("series", INDEX_SERIES)
def test_alpha_curve_reduces_holdout_gap(series: str) -> None:
    frame = _load_holdout()
    subset = frame.filter(pl.col("series") == series)
    if "alpha_target" in subset.columns:
        subset = subset.filter(pl.col("alpha_target") > 0)
    assert not subset.is_empty(), f"No holdout rows for {series}"
    curve = load_alpha_curve(series)
    assert curve is not None
    observed: list[float] = []
    baseline: list[float] = []
    predicted: list[float] = []
    for row in subset.iter_rows(named=True):
        observed.append(float(row.get("fill_ratio_observed") or 0.0))
        alpha_target = row.get("alpha_target")
        if alpha_target is None:
            alpha_target = curve.baseline_alpha
        baseline.append(float(alpha_target))
        depth = float(row.get("depth_fraction") or 0.0)
        delta_p = float(row.get("delta_p") or 0.0)
        tau = float(row.get("minutes_to_event") or 0.0)
        predicted.append(curve.predict(depth_fraction=depth, delta_p=delta_p, tau_minutes=tau))
    mae_pred = _mean_abs_error(predicted, observed)
    mae_base = _mean_abs_error(baseline, observed)
    assert mae_pred <= mae_base + 1e-3


@pytest.mark.parametrize("series", INDEX_SERIES)
def test_slippage_curve_reduces_holdout_error(series: str) -> None:
    frame = _load_holdout()
    subset = frame.filter(pl.col("series") == series)
    assert not subset.is_empty(), f"No holdout rows for {series}"
    curve = load_slippage_curve(series)
    assert curve is not None
    observed: list[float] = []
    predicted: list[float] = []
    for row in subset.iter_rows(named=True):
        observed.append(abs(float(row.get("slippage_ticks") or 0.0)))
        depth = float(row.get("depth_fraction") or 0.0)
        spread = float(row.get("spread") or 0.0)
        tau = float(row.get("minutes_to_event") or 0.0)
        predicted.append(curve.predict_ticks(depth_fraction=depth, spread=spread, tau_minutes=tau))
    mae_pred = _mean_abs_error(predicted, observed)
    mae_base = _mean_abs_error([0.0] * len(observed), observed)
    assert mae_pred <= mae_base + 1e-3


def test_alpha_curve_monotonic_in_depth() -> None:
    curve = load_alpha_curve("INXU")
    assert curve is not None
    depths = [0.1, 0.5, 0.9]
    predictions = [curve.predict(depth_fraction=d, delta_p=0.0, tau_minutes=60.0) for d in depths]
    assert predictions[0] <= predictions[1] <= predictions[2]


def test_slippage_curve_increases_with_spread() -> None:
    curve = load_slippage_curve("INXU")
    assert curve is not None
    narrow = curve.predict_ticks(depth_fraction=0.5, spread=0.01, tau_minutes=30.0)
    wide = curve.predict_ticks(depth_fraction=0.5, spread=0.10, tau_minutes=30.0)
    assert wide >= narrow
