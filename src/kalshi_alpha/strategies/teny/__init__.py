"""10-year Treasury yield strategy with factor calibration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from math import sqrt
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from kalshi_alpha.core.backtest import crps_from_pmf
from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.drivers import macro_calendar
from kalshi_alpha.strategies import base

CALIBRATION_PATH = PROC_ROOT / "teny_calib.parquet"
DEFAULT_SHOCK_WEIGHTS: dict[str, float] = {
    "cpi": 0.08,
    "jobs": 0.06,
    "claims": 0.04,
    "fomc": 0.12,
    "default": 0.03,
}
IMBALANCE_THRESHOLD = 0.6
IMBALANCE_MULTIPLIER = 1.4
_IMBALANCE_WINDOW_START = time(15, 0)
_IMBALANCE_WINDOW_END = time(15, 25)

try:  # pragma: no cover - platform specific zoneinfo availability
    _ET_ZONE = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _ET_ZONE = None


@dataclass(frozen=True)
class TenYInputs:
    latest_yield: float | None = None
    trailing_vol: float | None = None
    prior_close: float | None = None
    macro_shock: float = 0.0
    trailing_history: Sequence[float] | None = None
    macro_shock_dummies: Mapping[str, float] | None = None
    orderbook_imbalance: float | None = None
    imbalance: float | None = None
    event_timestamp: datetime | time | None = None
    imbalance_feature_enabled: bool = True


def pmf(
    strikes: Sequence[float],
    inputs: TenYInputs | None = None,
    *,
    calibration: Mapping[str, float] | None = None,
) -> list[LadderBinProbability]:
    inputs = inputs or TenYInputs()
    params = calibration or _load_calibration()
    if inputs.prior_close is not None:
        beta_macro = params.get("shock_beta", 1.4) if params else 1.4
        beta_slope = params.get("slope_beta", 0.0) if params else 0.0
        slope_factor = _slope_factor(inputs)
        mean = inputs.prior_close + beta_macro * inputs.macro_shock + beta_slope * slope_factor
        history = list(inputs.trailing_history or [])
        history.append(inputs.prior_close)
        residual_std = params.get("residual_std") if params else None
        spread = max(_sample_std(history[-6:]), residual_std or 0.05)
        spread = min(spread, 0.2)
        distribution = {
            round(mean - spread, 3): 0.25,
            round(mean, 3): 0.5,
            round(mean + spread, 3): 0.25,
        }
        return base.grid_distribution_to_pmf(distribution)

    mean = inputs.latest_yield if inputs.latest_yield is not None else 4.50
    std = inputs.trailing_vol if inputs.trailing_vol is not None else 0.35
    std = max(std, 0.05)
    return base.pmf_from_gaussian(strikes, mean=mean, std=std)


def pmf_v15(
    strikes: Sequence[float],
    inputs: TenYInputs | None = None,
    *,
    calibration: Mapping[str, float] | None = None,
    dummy_weights: Mapping[str, float] | None = None,
    imbalance_threshold: float = IMBALANCE_THRESHOLD,
    imbalance_multiplier: float = IMBALANCE_MULTIPLIER,
) -> list[LadderBinProbability]:
    inputs = inputs or TenYInputs()
    params = calibration or _load_calibration()
    weights = dict(dummy_weights or DEFAULT_SHOCK_WEIGHTS)
    macro_adjustment = _shock_dummy_adjustment(inputs.macro_shock_dummies, weights)

    if inputs.prior_close is not None:
        beta_macro = params.get("shock_beta", 1.4) if params else 1.4
        beta_slope = params.get("slope_beta", 0.0) if params else 0.0
        slope_factor = _slope_factor(inputs)
        macro_value = inputs.macro_shock + macro_adjustment
        mean = inputs.prior_close + beta_macro * macro_value + beta_slope * slope_factor
        history = list(inputs.trailing_history or [])
        history.append(inputs.prior_close)
        residual_std = params.get("residual_std") if params else None
        spread = max(_sample_std(history[-6:]), residual_std or 0.05)
        spread = min(spread, 0.2)
        spread = _apply_imbalance_spread(spread, inputs, imbalance_threshold, imbalance_multiplier)
        spread = min(spread, 0.35)
        distribution = {
            round(mean - spread, 3): 0.25,
            round(mean, 3): 0.5,
            round(mean + spread, 3): 0.25,
        }
        return base.grid_distribution_to_pmf(distribution)

    mean = inputs.latest_yield if inputs.latest_yield is not None else 4.50
    mean += macro_adjustment
    std = inputs.trailing_vol if inputs.trailing_vol is not None else 0.35
    std = max(std, 0.05)
    std = _apply_imbalance_spread(std, inputs, imbalance_threshold, imbalance_multiplier)
    return base.pmf_from_gaussian(strikes, mean=mean, std=std)


def _shock_dummy_adjustment(
    dummies: Mapping[str, float] | None,
    weights: Mapping[str, float],
) -> float:
    if not dummies:
        return 0.0
    default_weight = float(weights.get("default", 0.0))
    adjustment = 0.0
    for key, value in dummies.items():
        weight = float(weights.get(key, default_weight))
        adjustment += weight * float(value)
    return adjustment


def _apply_imbalance_spread(  # noqa: PLR0911
    spread: float,
    inputs: TenYInputs,
    threshold: float,
    multiplier: float,
) -> float:
    if not inputs.imbalance_feature_enabled:
        return spread
    if multiplier <= 1.0:
        return spread
    imbalance_value = inputs.orderbook_imbalance
    if imbalance_value is None:
        imbalance_value = inputs.imbalance
    if imbalance_value is None or threshold <= 0:
        return spread
    if abs(float(imbalance_value)) <= threshold:
        return spread
    event_time = _extract_event_time(inputs.event_timestamp)
    if event_time is None:
        return spread
    if event_time < _IMBALANCE_WINDOW_START or event_time > _IMBALANCE_WINDOW_END:
        return spread
    severity = abs(float(imbalance_value)) - threshold
    severity = max(min(severity, 0.5), 0.0)
    scaling = 1.0 + (severity / 0.5) * (multiplier - 1.0)
    return spread * scaling


def _extract_event_time(value: datetime | time | None) -> time | None:
    if value is None:
        return None
    if isinstance(value, time):
        return value
    if isinstance(value, datetime):
        if value.tzinfo is not None and _ET_ZONE is not None:
            return value.astimezone(_ET_ZONE).time()
        return value.time()
    return None


def _sample_std(values: Sequence[float]) -> float:
    if not values:
        return 0.10
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return sqrt(variance)


def _slope_factor(inputs: TenYInputs) -> float:
    history = list(inputs.trailing_history or [])
    if not history:
        return 0.0
    window = history[-5:] if len(history) >= 5 else history
    trailing_mean = sum(window) / len(window)
    return float(inputs.prior_close or trailing_mean) - trailing_mean


def calibrate(history: Sequence[dict[str, float]]) -> pl.DataFrame:
    if len(history) < 2:
        raise ValueError("ten-year history requires at least two rows")

    deltas: list[float] = []
    macro: list[float] = []
    slope_terms: list[float] = []
    records: list[dict[str, Any]] = []
    model_crps: list[float] = []
    baseline_crps: list[float] = []
    macro_lookup, macro_columns = _macro_dummies_lookup(history)

    closes = [float(entry["actual_close"]) for entry in history]
    for idx in range(1, len(history)):
        prior = closes[idx - 1]
        current = closes[idx]
        deltas.append(current - prior)
        macro_value = float(history[idx]["macro_shock"])
        macro.append(macro_value)
        slope_terms.append(prior - sum(closes[max(0, idx - 5) : idx]) / min(idx, 5))

    beta_macro, beta_slope = _ols_betas(macro, slope_terms, deltas)
    residuals = [
        delta - beta_macro * m - beta_slope * s
        for delta, m, s in zip(deltas, macro, slope_terms, strict=True)
    ]
    residual_std = sqrt(sum(res**2 for res in residuals) / max(len(residuals), 1))

    params = {
        "shock_beta": beta_macro,
        "slope_beta": beta_slope,
        "residual_std": residual_std,
    }

    strikes = [4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9]
    for idx in range(1, len(history)):
        prior = closes[idx - 1]
        macro_value = float(history[idx]["macro_shock"])
        trailing = closes[:idx]
        date_key = _normalize_history_date(history[idx].get("date"))
        macro_dummies = macro_lookup.get(date_key, {}) if date_key is not None else {}
        inputs = TenYInputs(
            prior_close=prior,
            macro_shock=macro_value,
            trailing_history=trailing,
            macro_shock_dummies=macro_dummies,
        )
        pmf_values = pmf(strikes, inputs=inputs, calibration=params)
        crps_value = crps_from_pmf(pmf_values, closes[idx])
        model_crps.append(crps_value)

        baseline_distribution = {
            round(prior - 0.1, 2): 0.25,
            round(prior, 2): 0.5,
            round(prior + 0.1, 2): 0.25,
        }
        baseline_pmf = base.grid_distribution_to_pmf(baseline_distribution)
        baseline_value = crps_from_pmf(baseline_pmf, closes[idx])
        baseline_crps.append(baseline_value)

        record = {
            "record_type": "evaluation",
            "date": history[idx].get("date"),
            "macro_shock": macro_value,
            "slope_factor": slope_terms[idx - 1],
            "mean": prior + beta_macro * macro_value + beta_slope * slope_terms[idx - 1],
            "std": max(_sample_std(trailing[-6:]), residual_std or 0.05),
            "actual_close": closes[idx],
            "crps": crps_value,
            "baseline_crps": baseline_value,
        }
        for column in macro_columns:
            record[column] = float(macro_dummies.get(_strip_dummy_prefix(column), 0.0))
        records.append(record)

    summary_row = {
        "record_type": "params",
        "date": None,
        "macro_shock": None,
        "slope_factor": None,
        "mean": None,
        "std": params["residual_std"],
        "actual_close": None,
        "crps": sum(model_crps) / len(model_crps),
        "baseline_crps": sum(baseline_crps) / len(baseline_crps),
        "shock_beta": params["shock_beta"],
        "slope_beta": params["slope_beta"],
        "residual_std": params["residual_std"],
    }
    for column in macro_columns:
        summary_row[column] = None
    records.insert(0, summary_row)

    frame = pl.DataFrame(records)
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(CALIBRATION_PATH)
    return frame


def _ols_betas(
    macro: Sequence[float],
    slope_terms: Sequence[float],
    deltas: Sequence[float],
) -> tuple[float, float]:
    if not macro:
        return 0.0, 0.0
    sum_mm = sum(m * m for m in macro)
    sum_ss = sum(s * s for s in slope_terms)
    sum_ms = sum(m * s for m, s in zip(macro, slope_terms, strict=True))
    sum_my = sum(m * y for m, y in zip(macro, deltas, strict=True))
    sum_sy = sum(s * y for s, y in zip(slope_terms, deltas, strict=True))

    det = sum_mm * sum_ss - sum_ms * sum_ms
    if abs(det) < 1e-9:
        beta_macro = sum_my / sum_mm if sum_mm else 0.0
        return beta_macro, 0.0

    beta_macro = (sum_my * sum_ss - sum_sy * sum_ms) / det
    beta_slope = (sum_sy * sum_mm - sum_my * sum_ms) / det
    return beta_macro, beta_slope


def _load_calibration() -> dict[str, float] | None:
    if not CALIBRATION_PATH.exists():
        return None
    frame = pl.read_parquet(CALIBRATION_PATH)
    params = frame.filter(pl.col("record_type") == "params")
    if params.is_empty():
        return None
    row = params.row(0, named=True)
    result: dict[str, float] = {}
    for key in ("shock_beta", "slope_beta", "residual_std"):
        if row.get(key) is not None:
            result[key] = float(row[key])
    return result or None


def _macro_dummies_lookup(history: Sequence[dict[str, Any]]) -> tuple[dict[str, dict[str, float]], list[str]]:
    if not history:
        return {}, []
    path = macro_calendar.DEFAULT_OUTPUT
    if not path.exists():
        return {}, []
    try:
        frame = pl.read_parquet(path)
    except Exception:  # pragma: no cover - allow missing macro dummies
        return {}, []
    if frame.is_empty():
        return {}, []
    columns = [name for name in frame.columns if name != "date"]
    lookup: dict[str, dict[str, float]] = {}
    for row in frame.iter_rows(named=True):
        key = _normalize_history_date(row.get("date"))
        if key is None:
            continue
        lookup[key] = {
            _strip_dummy_prefix(column): float(row.get(column) or 0.0)
            for column in columns
        }
    return lookup, columns


def _normalize_history_date(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return None


def _strip_dummy_prefix(column: str) -> str:
    return column[3:] if column.startswith("is_") else column
