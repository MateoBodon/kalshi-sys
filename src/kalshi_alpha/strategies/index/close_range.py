"""End-of-day close range strategy for index ladders."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.drivers.polygon_index.symbols import IndexSymbol, resolve_series

from .cdf import SigmaCalibration, gaussian_pmf, load_calibration

INDEX_CALIBRATION_ROOT = PROC_ROOT / "calib" / "index"
CLOSE_CALIBRATION_PATH = INDEX_CALIBRATION_ROOT
LATE_DAY_WINDOW_MINUTES = 10


@dataclass(frozen=True)
class CloseInputs:
    series: str
    current_price: float
    minutes_to_close: int
    drift_override: float | None = None
    sigma_override: float | None = None
    residual_override: float | None = None
    event_tags: tuple[str, ...] = ()
    late_day_bump_override: float | None = None
    variance_multiplier_override: float | None = None
    event_multiplier_override: float | None = None


def pmf(
    strikes: Sequence[float],
    inputs: CloseInputs,
    *,
    calibration: SigmaCalibration | None = None,
) -> list[LadderBinProbability]:
    meta = _resolve_series(inputs.series)
    target_minutes = max(int(inputs.minutes_to_close), 0)
    calib = calibration or _load_default_calibration(meta)
    sigma = inputs.sigma_override if inputs.sigma_override is not None else calib.sigma(target_minutes)
    residual = inputs.residual_override if inputs.residual_override is not None else calib.residual_std
    base_sigma = max(float(sigma), float(residual or 0.0), 1.0)
    drift = inputs.drift_override if inputs.drift_override is not None else calib.drift(target_minutes)
    variance = base_sigma * base_sigma
    variance += _late_day_variance(inputs, calib, target_minutes)
    variance_multiplier = _event_multiplier(inputs, calib)
    if inputs.variance_multiplier_override is not None:
        variance_multiplier *= max(float(inputs.variance_multiplier_override), 0.0)
    if variance_multiplier <= 0.0:
        variance_multiplier = 1.0
    variance *= variance_multiplier
    effective_sigma = max(math.sqrt(variance), 1.0)
    mean = float(inputs.current_price) + float(drift)
    return gaussian_pmf(strikes, mean=mean, std=effective_sigma, min_std=1.0)


def _resolve_series(series: str) -> IndexSymbol:
    try:
        return resolve_series(series)
    except KeyError as exc:  # pragma: no cover - validated upstream
        raise ValueError(str(exc)) from exc


@lru_cache(maxsize=4)
def _load_default_calibration(meta: IndexSymbol) -> SigmaCalibration:
    return load_calibration(CLOSE_CALIBRATION_PATH, meta.polygon_ticker, horizon="close")


def _late_day_variance(inputs: CloseInputs, calibration: SigmaCalibration, minutes_to_target: int) -> float:
    override = inputs.late_day_bump_override
    if override is not None:
        return max(float(override), 0.0)
    config = calibration.late_day_variance
    if minutes_to_target > LATE_DAY_WINDOW_MINUTES:
        return 0.0
    if config is None or minutes_to_target > config.minutes_threshold:
        return 0.0
    if not inputs.event_tags:
        return 0.0
    tail = calibration.event_tail
    if tail is not None:
        normalized = {tag.strip().lower() for tag in inputs.event_tags if tag}
        if not normalized.intersection(tail.tags):
            return 0.0
    return max(float(config.lambda_value), 0.0)


def _event_multiplier(inputs: CloseInputs, calibration: SigmaCalibration) -> float:
    if inputs.event_multiplier_override is not None:
        return max(float(inputs.event_multiplier_override), 0.0)
    tail = calibration.event_tail
    if tail is None:
        return 1.0
    if not inputs.event_tags:
        return 1.0
    normalized = {tag.strip().lower() for tag in inputs.event_tags if tag}
    if normalized.intersection(tail.tags):
        return max(float(tail.kappa), 0.0)
    return 1.0


__all__ = ["CloseInputs", "pmf", "CLOSE_CALIBRATION_PATH"]
