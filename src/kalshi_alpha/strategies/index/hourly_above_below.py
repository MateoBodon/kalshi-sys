"""Intraday hourly above/below strategy for index ladders."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from math import inf

from scipy.stats import skewnorm

from kalshi_alpha.strategies import base
from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.drivers.polygon_index.symbols import IndexSymbol, resolve_series

from .cdf import SigmaCalibration, load_calibration

INDEX_CALIBRATION_ROOT = PROC_ROOT / "calib" / "index"
HOURLY_CALIBRATION_PATH = INDEX_CALIBRATION_ROOT
NOON_CALIBRATION_PATH = HOURLY_CALIBRATION_PATH  # backward compatibility
EVENT_MULTIPLIER_CAP = 1.8
MIN_EVENT_MULTIPLIER = 1.0


@dataclass(frozen=True, init=False)
class HourlyInputs:
    series: str = field(init=False)
    current_price: float = field(init=False)
    minutes_to_target: int = field(init=False)
    prev_close: float | None = field(init=False, default=None)
    drift_override: float | None = field(init=False, default=None)
    sigma_override: float | None = field(init=False, default=None)
    residual_override: float | None = field(init=False, default=None)
    event_tags: tuple[str, ...] = field(init=False, default=())
    variance_multiplier_override: float | None = field(init=False, default=None)
    skew: float = field(init=False, default=0.0)
    target_hour_et: int | None = field(init=False, default=None)

    def __init__(  # noqa: PLR0913
        self,
        series: str,
        current_price: float,
        *,
        minutes_to_target: int | None = None,
        minutes_to_noon: int | None = None,
        prev_close: float | None = None,
        drift_override: float | None = None,
        sigma_override: float | None = None,
        residual_override: float | None = None,
        event_tags: Sequence[str] | None = None,
        variance_multiplier_override: float | None = None,
        skew: float = 0.0,
        target_hour: int | None = None,
    ) -> None:
        if minutes_to_target is not None and minutes_to_noon is not None:
            raise TypeError("Specify only one of minutes_to_target or minutes_to_noon")
        if minutes_to_target is None and minutes_to_noon is None:
            raise TypeError("minutes_to_target is required")
        if minutes_to_noon is not None:
            warnings.warn(
                "minutes_to_noon is deprecated; use minutes_to_target",
                DeprecationWarning,
                stacklevel=2,
            )
            minutes_to_target = minutes_to_noon
        if minutes_to_target is None:
            raise TypeError("minutes_to_target is required")
        minutes_value = int(minutes_to_target)
        object.__setattr__(self, "series", series)
        object.__setattr__(self, "current_price", current_price)
        object.__setattr__(self, "minutes_to_target", minutes_value)
        object.__setattr__(self, "prev_close", prev_close)
        object.__setattr__(self, "drift_override", drift_override)
        object.__setattr__(self, "sigma_override", sigma_override)
        object.__setattr__(self, "residual_override", residual_override)
        tags = tuple(tag for tag in (event_tags or ()) if tag is not None)
        object.__setattr__(self, "event_tags", tags)
        object.__setattr__(self, "variance_multiplier_override", variance_multiplier_override)
        object.__setattr__(self, "skew", float(skew))
        hour_value = None if target_hour is None else int(target_hour) % 24
        object.__setattr__(self, "target_hour_et", hour_value)

    @property
    def minutes_to_noon(self) -> int:
        warnings.warn(
            "minutes_to_noon is deprecated; use minutes_to_target",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.minutes_to_target


def pmf(
    strikes: Sequence[float],
    inputs: HourlyInputs,
    *,
    calibration: SigmaCalibration | None = None,
) -> list[LadderBinProbability]:
    meta = _resolve_series(inputs.series)
    target_minutes = max(int(inputs.minutes_to_target), 0)
    variant = _target_hour_variant(inputs.target_hour_et)
    calib = calibration or _load_hourly_calibration(meta, variant)
    sigma = inputs.sigma_override if inputs.sigma_override is not None else calib.sigma(target_minutes)
    residual = inputs.residual_override if inputs.residual_override is not None else calib.residual_std
    effective_sigma = max(float(sigma), float(residual or 0.0), 0.5)
    drift = inputs.drift_override if inputs.drift_override is not None else calib.drift(target_minutes)
    variance = effective_sigma * effective_sigma
    variance *= _event_multiplier(inputs, calib)
    effective_sigma = max(math.sqrt(variance), 0.5)
    mean = float(inputs.current_price) + float(drift)
    return _skewnorm_pmf(strikes, mean=mean, std=effective_sigma, skew=float(inputs.skew), min_std=0.5)


def _resolve_series(series: str) -> IndexSymbol:
    try:
        return resolve_series(series)
    except KeyError as exc:  # pragma: no cover - validated upstream
        raise ValueError(str(exc)) from exc


@lru_cache(maxsize=16)
def _load_hourly_calibration(meta: IndexSymbol, variant: str | None) -> SigmaCalibration:
    try:
        return load_calibration(
            HOURLY_CALIBRATION_PATH,
            meta.polygon_ticker,
            horizon="hourly",
            variant=variant,
        )
    except FileNotFoundError:
        if variant:
            try:
                return load_calibration(
                    HOURLY_CALIBRATION_PATH,
                    meta.polygon_ticker,
                    horizon="hourly",
                    variant=None,
                )
            except FileNotFoundError:
                pass
        return load_calibration(
            HOURLY_CALIBRATION_PATH,
            meta.polygon_ticker,
            horizon="noon",
            variant=None,
        )


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


def _target_hour_variant(target_hour: int | None) -> str | None:
    if target_hour is None:
        return None
    return f"{int(target_hour) % 24:02d}00"


def _skewnorm_pmf(
    strikes: Sequence[float],
    *,
    mean: float,
    std: float,
    skew: float,
    min_std: float,
) -> list[LadderBinProbability]:
    scale = max(float(std), float(min_std))
    distribution = skewnorm(a=float(skew), loc=float(mean), scale=scale)
    bins = base.ladder_bins(strikes)
    weights = []
    for lower, upper in bins:
        lower_bound = -inf if lower is None else float(lower)
        upper_bound = inf if upper is None else float(upper)
        weight = float(distribution.cdf(upper_bound) - distribution.cdf(lower_bound))
        weights.append(max(weight, 0.0))
    normalized = base.normalize(weights)
    return [
        LadderBinProbability(lower=lower, upper=upper, probability=prob)
        for (lower, upper), prob in zip(bins, normalized, strict=True)
    ]


__all__ = ["HourlyInputs", "pmf", "HOURLY_CALIBRATION_PATH"]
