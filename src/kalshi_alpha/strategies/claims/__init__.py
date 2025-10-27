"""Initial jobless claims strategy with calibration utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import sqrt
from statistics import mean as stat_mean
from statistics import pstdev
from typing import Any

import polars as pl

from kalshi_alpha.core.backtest import crps_from_pmf
from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.strategies import base

CALIBRATION_PATH = PROC_ROOT / "claims_calib.parquet"


@dataclass(frozen=True)
class ClaimsInputs:
    history: Sequence[int] | None = None
    holiday_next: bool = False
    freeze_active: bool | None = None
    latest_initial_claims: int | None = None
    four_week_avg: int | None = None


def freeze_window(active_at: datetime | None = None) -> bool:
    """Return True once the Wednesday data freeze has begun (UTC)."""

    active_at = active_at or datetime.now(tz=UTC)
    return active_at.weekday() >= 2  # Wednesday (2) through Friday


def pmf(
    strikes: Sequence[float],
    inputs: ClaimsInputs | None = None,
    *,
    calibration: Mapping[str, float] | None = None,
) -> list[LadderBinProbability]:
    """Return a distribution over initial claims ladder strikes."""

    inputs = inputs or ClaimsInputs()
    if inputs.history and len(inputs.history) >= 2:
        mean = _damped_trend_forecast(inputs.history, holiday_adjust=inputs.holiday_next)
        std = _claims_std(inputs.history)
        freeze_flag = inputs.freeze_active if inputs.freeze_active is not None else freeze_window()
        if freeze_flag:
            std = max(std * 0.5, 3_000.0)
    else:
        mean = _mean_claims(inputs)
        std = max(mean * 0.05, 5_000.0)
    params = calibration or _load_calibration()
    if params:
        if inputs.holiday_next:
            mean += params.get("holiday_lift", 0.0)
        std = max(std, params.get("std", std))
    return base.pmf_from_gaussian(strikes, mean=mean, std=std)


def calibrate(
    history: Sequence[Mapping[str, Any]],
    window: int = 12,
    strikes: Sequence[float] | None = None,
) -> pl.DataFrame:
    """Calibrate claims distribution parameters and persist evaluation metrics."""

    if len(history) < 2:
        raise ValueError("claims history requires at least two points")

    strike_set = strikes or [200_000, 205_000, 210_000, 215_000, 220_000, 225_000]
    observed: list[int] = []
    records: list[dict[str, Any]] = []
    model_crps: list[float] = []
    baseline_crps: list[float] = []
    model_brier: list[float] = []
    baseline_brier: list[float] = []

    for idx, entry in enumerate(history):
        claims_value = int(entry["claims"])
        observed.append(claims_value)
        if idx == 0:
            continue

        history_slice = observed[:-1]
        segment = history[max(0, idx - window) : idx + 1]
        segment_params = _params_from_history(segment)
        inputs = ClaimsInputs(
            history=history_slice,
            holiday_next=bool(entry.get("holiday")),
            freeze_active=False,
        )
        pmf_values = pmf(strike_set, inputs=inputs, calibration=segment_params)
        crps_value = crps_from_pmf(pmf_values, claims_value)
        brier_value = _brier_score(pmf_values, claims_value)
        model_crps.append(crps_value)
        model_brier.append(brier_value)

        baseline_inputs = ClaimsInputs(latest_initial_claims=history_slice[-1])
        baseline_pmf_vals = pmf(
            strike_set,
            inputs=baseline_inputs,
            calibration={"holiday_lift": 0.0, "std": segment_params["std"]},
        )
        baseline_crps_val = crps_from_pmf(baseline_pmf_vals, claims_value)
        baseline_brier_val = _brier_score(baseline_pmf_vals, claims_value)
        baseline_crps.append(baseline_crps_val)
        baseline_brier.append(baseline_brier_val)

        mean_estimate = _damped_trend_forecast(history_slice, holiday_adjust=inputs.holiday_next)
        std_estimate = _claims_std(history_slice)

        records.append(
            {
                "record_type": "evaluation",
                "week": entry.get("week"),
                "window": min(idx, window),
                "mean": mean_estimate,
                "std": std_estimate,
                "actual": float(claims_value),
                "crps": crps_value,
                "baseline_crps": baseline_crps_val,
                "brier": brier_value,
                "baseline_brier": baseline_brier_val,
                "holiday": bool(entry.get("holiday")),
            }
        )

    params = _params_from_history(history[-min(window, len(history)) :])
    summary_row = {
        "record_type": "params",
        "week": None,
        "window": min(window, len(history)),
        "mean": None,
        "std": params["std"],
        "actual": None,
        "crps": sum(model_crps) / len(model_crps),
        "baseline_crps": sum(baseline_crps) / len(baseline_crps),
        "brier": sum(model_brier) / len(model_brier),
        "baseline_brier": sum(baseline_brier) / len(baseline_brier),
        "holiday": None,
        "holiday_lift": params["holiday_lift"],
    }
    records.insert(0, summary_row)

    frame = pl.DataFrame(records)
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(CALIBRATION_PATH)
    return frame


def _damped_trend_forecast(history: Sequence[int], *, holiday_adjust: bool) -> float:
    recent = list(history)
    last = float(recent[-1])
    prev = float(recent[-2]) if len(recent) > 1 else last
    trend = last - prev
    trailing = recent[-4:]
    level = 0.5 * last + 0.3 * (sum(trailing) / len(trailing)) + 0.2 * prev
    forecast = level + 0.6 * trend
    if holiday_adjust:
        forecast += 3_500.0
    return forecast


def _claims_std(history: Sequence[int]) -> float:
    window = list(history)[-6:]
    mean = sum(window) / len(window)
    variance = sum((value - mean) ** 2 for value in window) / len(window)
    return max(sqrt(variance), 4_000.0)


def _mean_claims(inputs: ClaimsInputs) -> float:
    if inputs.four_week_avg:
        return float(inputs.four_week_avg)
    if inputs.latest_initial_claims:
        return float(inputs.latest_initial_claims)
    return 230_000.0


def _params_from_history(history: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    tail = list(history)
    if len(tail) < 2:
        return {"holiday_lift": 0.0, "std": 3_000.0}
    diffs: list[float] = []
    holiday_diffs: list[float] = []
    for prev, current in zip(tail, tail[1:], strict=False):
        diff = float(current["claims"]) - float(prev["claims"])
        diffs.append(diff)
        if bool(current.get("holiday")):
            holiday_diffs.append(diff)
    holiday_lift = stat_mean(holiday_diffs) if holiday_diffs else 0.0
    std = max(pstdev(diffs) if len(diffs) > 1 else 3_000.0, 3_000.0)
    return {"holiday_lift": holiday_lift, "std": std}


def _brier_score(pmf_values: Sequence[LadderBinProbability], actual: float) -> float:
    probabilities = [bin_prob.probability for bin_prob in pmf_values]
    indicators = [_bin_indicator(bin_prob, actual) for bin_prob in pmf_values]
    return sum(
        (prob - indicator) ** 2 for prob, indicator in zip(probabilities, indicators, strict=True)
    )


def _bin_indicator(bin_prob: LadderBinProbability, value: float) -> float:
    lower = float("-inf") if bin_prob.lower is None else bin_prob.lower
    upper = float("inf") if bin_prob.upper is None else bin_prob.upper
    return 1.0 if lower <= value < upper else 0.0


def _load_calibration() -> dict[str, float] | None:
    if not CALIBRATION_PATH.exists():
        return None
    frame = pl.read_parquet(CALIBRATION_PATH)
    params = frame.filter(pl.col("record_type") == "params")
    if params.is_empty():
        return None
    row = params.row(0, named=True)
    result: dict[str, float] = {}
    if row.get("holiday_lift") is not None:
        result["holiday_lift"] = float(row["holiday_lift"])
    if row.get("std") is not None:
        result["std"] = float(row["std"])
    return result or None
