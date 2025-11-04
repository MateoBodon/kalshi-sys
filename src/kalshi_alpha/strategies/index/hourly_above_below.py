"""Intraday hourly above/below strategy for index ladders."""

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
HOURLY_CALIBRATION_PATH = INDEX_CALIBRATION_ROOT
NOON_CALIBRATION_PATH = HOURLY_CALIBRATION_PATH  # backward compatibility
EVENT_MULTIPLIER_CAP = 1.8
MIN_EVENT_MULTIPLIER = 1.0


@dataclass(frozen=True)
class HourlyInputs:
    series: str
    current_price: float
    minutes_to_noon: int
    prev_close: float | None = None
    drift_override: float | None = None
    sigma_override: float | None = None
    residual_override: float | None = None
    event_tags: tuple[str, ...] = ()
    variance_multiplier_override: float | None = None


def pmf(
    strikes: Sequence[float],
    inputs: HourlyInputs,
    *,
    calibration: SigmaCalibration | None = None,
) -> list[LadderBinProbability]:
    meta = _resolve_series(inputs.series)
    target_minutes = max(int(inputs.minutes_to_noon), 0)
    calib = calibration or _load_default_calibration(meta)
    sigma = inputs.sigma_override if inputs.sigma_override is not None else calib.sigma(target_minutes)
    residual = inputs.residual_override if inputs.residual_override is not None else calib.residual_std
    effective_sigma = max(float(sigma), float(residual or 0.0), 0.5)
    drift = inputs.drift_override if inputs.drift_override is not None else calib.drift(target_minutes)
    variance = effective_sigma * effective_sigma
    variance *= _event_multiplier(inputs, calib)
    effective_sigma = max(math.sqrt(variance), 0.5)
    mean = float(inputs.current_price) + float(drift)
    return gaussian_pmf(strikes, mean=mean, std=effective_sigma, min_std=0.5)


def _resolve_series(series: str) -> IndexSymbol:
    try:
        return resolve_series(series)
    except KeyError as exc:  # pragma: no cover - validated upstream
        raise ValueError(str(exc)) from exc


@lru_cache(maxsize=4)
def _load_default_calibration(meta: IndexSymbol) -> SigmaCalibration:
    try:
        return load_calibration(HOURLY_CALIBRATION_PATH, meta.polygon_ticker, horizon="hourly")
    except FileNotFoundError:
        return load_calibration(HOURLY_CALIBRATION_PATH, meta.polygon_ticker, horizon="noon")


def _event_multiplier(inputs: HourlyInputs, calibration: SigmaCalibration) -> float:
    if inputs.variance_multiplier_override is not None:
        return max(float(inputs.variance_multiplier_override), 0.0)
    tail = calibration.event_tail
    if tail is None or not inputs.event_tags:
        return 1.0
    normalized = {tag.strip().lower() for tag in inputs.event_tags if tag}
    if normalized.intersection(tail.tags):
        value = max(float(tail.kappa), 0.0)
        value = max(value, MIN_EVENT_MULTIPLIER)
        return min(value, EVENT_MULTIPLIER_CAP)
    return 1.0


__all__ = ["HourlyInputs", "pmf", "HOURLY_CALIBRATION_PATH"]
