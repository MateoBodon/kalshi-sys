"""CPI strategy producing monthly and YoY distributions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean as stat_mean
from statistics import pstdev
from typing import Any

import yaml

import polars as pl

from kalshi_alpha.core.backtest import crps_from_pmf
from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.strategies import base
from kalshi_alpha.strategies.cpi import components

GRID_STEP = 0.05
GRID_SPAN = 4  # +/- span multiples around the mean
DEFAULT_STD = 0.12
CALIBRATION_PATH = PROC_ROOT / "cpi_calib.parquet"
DEFAULT_COMPONENT_WEIGHTS = components.ComponentWeights()
CONFIG_PATH = Path("configs/strategies/cpi.yaml")


@dataclass(frozen=True)
class CPIV15Config:
    component_weights: components.ComponentWeights
    blend_cleveland: float
    blend_components: float
    variance_scale: float


_CONFIG_CACHE: dict[Path, CPIV15Config] = {}


def _safe_float(value: Any, fallback: float) -> float:
    try:
        if value is None:
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _load_v15_config(config_path: Path | None = None) -> CPIV15Config:
    path = (config_path or CONFIG_PATH).resolve()
    cached = _CONFIG_CACHE.get(path)
    if cached is not None:
        return cached

    payload: dict[str, Any] = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        except Exception:  # pragma: no cover - defensive config parsing
            payload = {}

    component_payload = payload.get("component_weights") or {}
    weights = components.ComponentWeights(
        gas=_safe_float(component_payload.get("gas"), DEFAULT_COMPONENT_WEIGHTS.gas),
        shelter=_safe_float(
            component_payload.get("shelter"),
            DEFAULT_COMPONENT_WEIGHTS.shelter,
        ),
        autos=_safe_float(component_payload.get("autos"), DEFAULT_COMPONENT_WEIGHTS.autos),
    )

    blend_payload = payload.get("blend") or {}
    blend_cleveland = _safe_float(blend_payload.get("cleveland"), 1.0)
    blend_components = _safe_float(blend_payload.get("components"), 1.0)
    if blend_cleveland <= 0 and blend_components <= 0:
        blend_cleveland = 1.0
        blend_components = 1.0

    variance_payload = payload.get("variance") or {}
    variance_scale = _safe_float(variance_payload.get("component_scale"), 0.001)

    config = CPIV15Config(
        component_weights=weights,
        blend_cleveland=blend_cleveland,
        blend_components=blend_components,
        variance_scale=variance_scale,
    )
    _CONFIG_CACHE[path] = config
    return config


def load_v15_config(config_path: Path | None = None) -> CPIV15Config:
    """Expose cached v1.5 configuration for downstream consumers."""

    return _load_v15_config(config_path)


@dataclass(frozen=True)
class CPIInputs:
    cleveland_nowcast: float | None = None
    latest_release_mom: float | None = None
    aaa_delta: float = 0.0
    variance: float | None = None


def nowcast(
    inputs: CPIInputs | None = None,
    *,
    calibration: Mapping[str, float] | None = None,
) -> dict[float, float]:
    """Return a normalized distribution over 0.05pp grid points."""

    inputs = inputs or CPIInputs()
    base_mean = _select_mean(inputs)
    mean = base_mean + inputs.aaa_delta
    variance = (
        inputs.variance
        if inputs.variance is not None
        else max(0.0025, 0.0036 + 0.04 * abs(inputs.aaa_delta))
    )
    std = max(variance**0.5, 0.03)
    params = calibration or _load_calibration()
    if params:
        mean += float(params.get("bias", 0.0))
        std = max(std, float(params.get("std", std)))

    grid = _grid_around(mean)
    weights = {point: _gaussian_weight(point, mean, std) for point in grid}
    total = sum(weights.values())
    if total == 0:
        raise ValueError("degenerate CPI nowcast weights")
    return {point: weight / total for point, weight in weights.items()}


def nowcast_v15(
    inputs: CPIInputs | None = None,
    *,
    fixtures_dir: Path | None = None,
    weights: components.ComponentWeights | Mapping[str, float] | None = None,
    component_overrides: Mapping[str, float] | None = None,
    calibration: Mapping[str, float] | None = None,
    offline: bool = True,
    config_path: Path | None = None,
) -> dict[float, float]:
    base_inputs = inputs or CPIInputs()
    config = _load_v15_config(config_path)
    signals = components.component_signals(fixtures_dir=fixtures_dir, offline=offline)
    if component_overrides:
        signals.update(component_overrides)

    required_components = {"gas", "shelter", "autos"}
    if required_components - signals.keys():
        return nowcast(base_inputs, calibration=calibration)

    weight_map: dict[str, float] = {
        "gas": config.component_weights.gas,
        "shelter": config.component_weights.shelter,
        "autos": config.component_weights.autos,
    }
    if weights is not None:
        if isinstance(weights, components.ComponentWeights):
            weight_map.update(
                {
                    "gas": weights.gas,
                    "shelter": weights.shelter,
                    "autos": weights.autos,
                }
            )
        else:
            weight_map.update(weights)

    total_component_weight = sum(abs(value) for value in weight_map.values())
    blend_total = config.blend_cleveland + config.blend_components
    if total_component_weight <= 0 or config.blend_components <= 0 or blend_total <= 0:
        return nowcast(base_inputs, calibration=calibration)

    shift = components.blend_component_shift(signals, weight_map)

    base_weight = config.blend_cleveland / blend_total
    component_weight = config.blend_components / blend_total

    base_mean = _select_mean(base_inputs)
    baseline_mean = base_mean + base_inputs.aaa_delta
    blended_mean = base_weight * baseline_mean + component_weight * (baseline_mean + shift)
    adjusted_delta = blended_mean - base_mean

    if base_inputs.variance is not None:
        base_variance = base_inputs.variance
    else:
        base_variance = max(2e-4, 3e-4 + 0.002 * abs(adjusted_delta))

    variance_boost = config.variance_scale * sum(
        abs(signals.get(key, 0.0)) * abs(weight_map.get(key, 0.0))
        for key in weight_map
    )

    tuned_inputs = CPIInputs(
        cleveland_nowcast=base_inputs.cleveland_nowcast,
        latest_release_mom=base_inputs.latest_release_mom,
        aaa_delta=adjusted_delta,
        variance=base_variance + variance_boost,
    )
    return nowcast(tuned_inputs, calibration=calibration)


def map_to_ladder_bins(
    strikes: Sequence[float],
    distribution: Mapping[float, float],
) -> list[LadderBinProbability]:
    bins = base.ladder_bins(strikes)
    bin_probs: list[float] = []
    for lower, upper in bins:
        prob = 0.0
        for point, weight in distribution.items():
            if _belongs_to_bin(point, lower, upper):
                prob += weight
        bin_probs.append(prob)

    normalized = base.normalize(bin_probs)
    return [
        LadderBinProbability(lower=lower, upper=upper, probability=prob)
        for (lower, upper), prob in zip(bins, normalized, strict=True)
    ]


def calibrate(
    history: Sequence[Mapping[str, Any]],
    window: int = 12,
) -> pl.DataFrame:
    """Calibrate CPI nowcast bias/std and persist evaluation metrics to parquet."""

    if len(history) < 2:
        raise ValueError("history requires at least two observations")

    records: list[dict[str, Any]] = []
    model_crps: list[float] = []
    baseline_crps: list[float] = []

    for idx, entry in enumerate(history):
        actual_mom = float(entry["actual"])
        cleveland_value = float(entry.get("cleveland_nowcast", actual_mom))
        delta = float(entry.get("aaa_delta", 0.0))
        prev_actual = float(history[idx - 1]["actual"]) if idx > 0 else None

        tail_slice = history[max(0, idx - window) : idx]
        params = _params_from_history(tail_slice) if tail_slice else None

        inputs = CPIInputs(
            cleveland_nowcast=cleveland_value,
            latest_release_mom=prev_actual,
            aaa_delta=delta,
        )
        distribution = nowcast(inputs, calibration=params)
        pmf = base.grid_distribution_to_pmf(distribution)
        crps_value = crps_from_pmf(pmf, actual_mom)
        model_crps.append(crps_value)

        if prev_actual is not None:
            baseline_pmf = base.pmf_from_gaussian(
                strikes=_grid_around(prev_actual),
                mean=prev_actual,
                std=0.25,
            )
            baseline_value = crps_from_pmf(baseline_pmf, actual_mom)
        else:
            baseline_value = crps_value
        baseline_crps.append(baseline_value)

        mom_mean, mom_std = _distribution_stats(distribution)
        yoy_prediction, actual_yoy = _yoy_projection(entry, mom_mean, actual_mom)

        records.append(
            {
                "record_type": "evaluation",
                "period": entry.get("period"),
                "window": len(tail_slice),
                "mom_mean": mom_mean,
                "mom_std": mom_std,
                "actual_mom": actual_mom,
                "crps": crps_value,
                "baseline_crps": baseline_value,
                "yoy_prediction": yoy_prediction,
                "actual_yoy": actual_yoy,
                "bias": float(params["bias"]) if params else None,
                "std": float(params["std"]) if params else None,
            }
        )

    summary_params = _params_from_history(history[-min(window, len(history)) :])
    summary_row = {
        "record_type": "params",
        "period": None,
        "window": min(window, len(history)),
        "mom_mean": None,
        "mom_std": None,
        "actual_mom": None,
        "crps": sum(model_crps) / len(model_crps),
        "baseline_crps": sum(baseline_crps) / len(baseline_crps),
        "yoy_prediction": None,
        "actual_yoy": None,
        "bias": summary_params["bias"],
        "std": summary_params["std"],
    }
    records.insert(0, summary_row)

    frame = pl.DataFrame(records)
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(CALIBRATION_PATH)
    return frame


def _belongs_to_bin(point: float, lower: float | None, upper: float | None) -> bool:
    if lower is None and upper is None:
        return True
    if lower is None:
        return upper is not None and point < upper
    if upper is None:
        return point >= lower
    return lower <= point < upper


def _select_mean(inputs: CPIInputs) -> float:
    if inputs.cleveland_nowcast is not None:
        return inputs.cleveland_nowcast
    if inputs.latest_release_mom is not None:
        return inputs.latest_release_mom
    return 0.30  # placeholder anchor when no drivers are available


def _grid_around(center: float) -> list[float]:
    steps = range(-GRID_SPAN, GRID_SPAN + 1)
    return [round(center + step * GRID_STEP, 3) for step in steps]


def _gaussian_weight(point: float, mean: float, std: float) -> float:
    exponent = -((point - mean) ** 2) / (2 * std**2)
    return math.exp(exponent)


def _params_from_history(history: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not history:
        return {"bias": 0.0, "std": DEFAULT_STD}
    actuals = [float(entry["actual"]) for entry in history]
    anchors = [
        float(entry.get("cleveland_nowcast", entry["actual"]))
        + float(entry.get("aaa_delta", 0.0))
        for entry in history
    ]
    errors = [actual - anchor for actual, anchor in zip(actuals, anchors, strict=True)]
    bias = stat_mean(errors)
    std = max(pstdev(actuals) if len(actuals) > 1 else DEFAULT_STD, DEFAULT_STD / 2)
    return {"bias": bias, "std": std}


def _distribution_stats(distribution: Mapping[float, float]) -> tuple[float, float]:
    mean = sum(point * weight for point, weight in distribution.items())
    variance = sum(weight * (point - mean) ** 2 for point, weight in distribution.items())
    return mean, math.sqrt(max(variance, 0.0))


def _yoy_projection(
    entry: Mapping[str, Any],
    predicted_mom: float,
    actual_mom: float,
) -> tuple[float | None, float | None]:
    prev_yoy = entry.get("prev_yoy")
    base_effect = entry.get("base_effect")
    if prev_yoy is None or base_effect is None:
        return None, None
    predicted_yoy = float(prev_yoy) + predicted_mom - float(base_effect)
    actual_yoy = (
        float(entry.get("actual_yoy"))
        if entry.get("actual_yoy") is not None
        else float(prev_yoy) + actual_mom - float(base_effect)
    )
    return predicted_yoy, actual_yoy


def _load_calibration() -> dict[str, float] | None:
    if not CALIBRATION_PATH.exists():
        return None
    frame = pl.read_parquet(CALIBRATION_PATH)
    params = frame.filter(pl.col("record_type") == "params")
    if params.is_empty():
        return None
    row = params.row(0, named=True)
    result: dict[str, float] = {}
    if row.get("bias") is not None:
        result["bias"] = float(row["bias"])
    if row.get("std") is not None:
        result["std"] = float(row["std"])
    return result or None
