"""End-of-day close range strategy for index ladders."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.drivers.polygon_index.symbols import IndexSymbol, resolve_series

from .cdf import SigmaCalibration, gaussian_pmf, load_calibration

INDEX_CALIBRATION_ROOT = PROC_ROOT / "calib" / "index"
CLOSE_CALIBRATION_PATH = INDEX_CALIBRATION_ROOT


@dataclass(frozen=True)
class CloseInputs:
    series: str
    current_price: float
    minutes_to_close: int
    drift_override: float | None = None
    sigma_override: float | None = None
    residual_override: float | None = None


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
    effective_sigma = max(float(sigma), float(residual or 0.0), 1.0)
    drift = inputs.drift_override if inputs.drift_override is not None else calib.drift(target_minutes)
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


__all__ = ["CloseInputs", "pmf", "CLOSE_CALIBRATION_PATH"]
