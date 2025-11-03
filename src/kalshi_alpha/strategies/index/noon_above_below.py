"""Intraday noon above/below strategy for index ladders."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.drivers.polygon_index.symbols import IndexSymbol, resolve_series

from .cdf import SigmaCalibration, gaussian_pmf, load_calibration

NOON_CALIBRATION_PATH = PROC_ROOT / "index_noon_calibration.parquet"


@dataclass(frozen=True)
class NoonInputs:
    series: str
    current_price: float
    minutes_to_noon: int
    prev_close: float | None = None
    drift_override: float | None = None
    sigma_override: float | None = None
    residual_override: float | None = None


def pmf(
    strikes: Sequence[float],
    inputs: NoonInputs,
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
    mean = float(inputs.current_price) + float(drift)
    return gaussian_pmf(strikes, mean=mean, std=effective_sigma, min_std=0.5)


def _resolve_series(series: str) -> IndexSymbol:
    try:
        return resolve_series(series)
    except KeyError as exc:  # pragma: no cover - validated upstream
        raise ValueError(str(exc)) from exc


@lru_cache(maxsize=4)
def _load_default_calibration(meta: IndexSymbol) -> SigmaCalibration:
    return load_calibration(NOON_CALIBRATION_PATH, meta.polygon_ticker)


__all__ = ["NoonInputs", "pmf", "NOON_CALIBRATION_PATH"]
