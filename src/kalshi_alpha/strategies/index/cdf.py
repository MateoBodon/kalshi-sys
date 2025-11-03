"""CDF helpers shared by index strategies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import polars as pl

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.strategies import base


@dataclass(frozen=True)
class SigmaCalibration:
    sigma_curve: Mapping[int, float]
    drift_curve: Mapping[int, float]
    residual_std: float

    def sigma(self, minutes_to_target: int) -> float:
        return _nearest(self.sigma_curve, minutes_to_target)

    def drift(self, minutes_to_target: int) -> float:
        return _nearest(self.drift_curve, minutes_to_target)


def gaussian_pmf(
    strikes: Sequence[float],
    *,
    mean: float,
    std: float,
    min_std: float = 1.0,
) -> list[LadderBinProbability]:
    std_value = max(float(std), float(min_std))
    return base.pmf_from_gaussian(strikes, mean, std_value)


def survival_map(
    strikes: Sequence[float],
    pmf: Sequence[LadderBinProbability],
) -> dict[float, float]:
    if len(pmf) != len(strikes) + 1:
        raise ValueError("PMF length must equal strikes length + 1")
    ordered_strikes = sorted(strikes)
    tail = 0.0
    survival_pairs: list[tuple[float, float]] = []
    for index in reversed(range(len(pmf))):
        tail += float(pmf[index].probability)
        if index > 0:
            survival_pairs.append((ordered_strikes[index - 1], tail))
    survival_pairs.reverse()
    return dict(survival_pairs)


def probability_at_or_above(
    strike: float,
    strikes: Sequence[float],
    pmf: Sequence[LadderBinProbability],
) -> float:
    mapping = survival_map(strikes, pmf)
    if strike not in mapping:
        raise KeyError(f"Strike {strike} not found in survival map")
    return mapping[strike]


def probability_between(
    lower: float,
    upper: float,
    strikes: Sequence[float],
    pmf: Sequence[LadderBinProbability],
) -> float:
    mapping = survival_map(strikes, pmf)
    lower_prob = mapping.get(lower)
    if lower_prob is None:
        raise KeyError(f"Lower strike {lower} not found in survival map")
    upper_prob = mapping.get(upper, 0.0)
    probability = lower_prob - upper_prob
    return max(probability, 0.0)


def load_calibration(path: Path, symbol: str) -> SigmaCalibration:
    resolved = path.resolve()
    return _load_calibration_cached(str(resolved), symbol.upper())


@lru_cache(maxsize=12)
def _load_calibration_cached(path: str, symbol: str) -> SigmaCalibration:
    frame = pl.read_parquet(path)
    filtered = frame.filter(pl.col("symbol") == symbol)
    if filtered.is_empty():
        raise FileNotFoundError(f"Calibration for {symbol} not found in {path}")
    rows = list(filtered.iter_rows(named=True))
    sigma_curve = {int(row["minutes_to_target"]): float(row["sigma"]) for row in rows}
    drift_curve = {int(row["minutes_to_target"]): float(row.get("drift", 0.0)) for row in rows}
    residual_values = [float(row.get("residual_std", 0.0)) for row in rows if row.get("residual_std") is not None]
    residual_std = max(residual_values) if residual_values else 0.0
    return SigmaCalibration(sigma_curve=sigma_curve, drift_curve=drift_curve, residual_std=residual_std)


def _nearest(curve: Mapping[int, float], minutes: int) -> float:
    if not curve:
        return 0.0
    keys = list(curve.keys())
    closest = min(keys, key=lambda key: abs(int(key) - int(minutes)))
    return float(curve[closest])


__all__ = [
    "SigmaCalibration",
    "gaussian_pmf",
    "survival_map",
    "probability_at_or_above",
    "probability_between",
    "load_calibration",
]
