"""CDF helpers shared by index strategies."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.strategies import base


@dataclass(frozen=True)
class SigmaCalibration:
    sigma_curve: Mapping[int, float]
    drift_curve: Mapping[int, float]
    residual_std: float
    pit_bias: float | None = None

    def sigma(self, minutes_to_target: int) -> float:
        return _nearest(self.sigma_curve, minutes_to_target)

    def drift(self, minutes_to_target: int) -> float:
        return _nearest(self.drift_curve, minutes_to_target)

    def apply_pit(self, probability: float) -> float:
        prob = min(max(float(probability), 1e-9), 1.0 - 1e-9)
        if self.pit_bias is None:
            return prob
        logit = math.log(prob / (1.0 - prob))
        adjusted = 1.0 / (1.0 + math.exp(-(logit + self.pit_bias)))
        return min(max(adjusted, 0.0), 1.0)


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


def load_calibration(path: Path, symbol: str, *, horizon: str) -> SigmaCalibration:
    resolved_file = _resolve_calibration_file(path, symbol, horizon)
    return _load_calibration_cached(str(resolved_file))


@lru_cache(maxsize=12)
def _load_calibration_cached(file_path: str) -> SigmaCalibration:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found at {file_path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    minutes_section = data.get("minutes_to_target", {})
    sigma_curve = {}
    drift_curve = {}
    for minutes, entry in minutes_section.items():
        try:
            minutes_int = int(minutes)
        except (TypeError, ValueError):  # pragma: no cover - defensive parsing
            continue
        sigma_curve[minutes_int] = float(entry.get("sigma", 0.0))
        drift_curve[minutes_int] = float(entry.get("drift", 0.0))
    if not sigma_curve:
        raise ValueError(f"Calibration payload missing sigma curve: {file_path}")
    residual_std = float(data.get("residual_std", 0.0))
    pit_bias = data.get("pit_bias")
    if pit_bias is not None:
        try:
            pit_bias = float(pit_bias)
        except (TypeError, ValueError):
            pit_bias = None
    return SigmaCalibration(
        sigma_curve=sigma_curve,
        drift_curve=drift_curve,
        residual_std=residual_std,
        pit_bias=pit_bias,
    )


def _resolve_calibration_file(path: Path, symbol: str, horizon: str) -> Path:
    resolved = path.resolve()
    if resolved.is_file():
        return resolved
    slug = _symbol_slug(symbol)
    file_path = resolved / slug / horizon / "params.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Calibration params not found: {file_path}")
    return file_path


def _symbol_slug(symbol: str) -> str:
    return symbol.split(":")[-1].lower()


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
