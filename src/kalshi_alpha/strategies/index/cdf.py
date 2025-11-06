"""CDF helpers shared by index strategies."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.strategies import base


@dataclass(frozen=True)
class LateDayVariance:
    minutes_threshold: int
    lambda_value: float


@dataclass(frozen=True)
class EventTail:
    tags: tuple[str, ...]
    kappa: float


@dataclass(frozen=True)
class SigmaCalibration:
    sigma_curve: Mapping[int, float]
    drift_curve: Mapping[int, float]
    residual_std: float
    pit_bias: float | None = None
    late_day_variance: LateDayVariance | None = None
    event_tail: EventTail | None = None
    sigma_now: float | None = None
    m_tod_curve: Mapping[int, float] | None = None
    micro_drift_curve: Mapping[int, float] | None = None
    metadata: Mapping[str, Any] | None = None

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
    return cast(list[LadderBinProbability], base.pmf_from_gaussian(strikes, mean, std_value))


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


def load_calibration(path: Path, symbol: str, *, horizon: str, variant: str | None = None) -> SigmaCalibration:
    resolved_file = _resolve_calibration_file(path, symbol, horizon, variant=variant)
    return _load_calibration_cached(str(resolved_file))


def _extract_minutes_curves(data: Mapping[str, Any], file_path: str) -> tuple[dict[int, float], dict[int, float]]:
    minutes_section = data.get("minutes_to_target", {})
    sigma_curve: dict[int, float] = {}
    drift_curve: dict[int, float] = {}
    for minutes, entry in minutes_section.items():
        try:
            minutes_int = int(minutes)
        except (TypeError, ValueError):  # pragma: no cover - defensive parsing
            continue
        sigma_curve[minutes_int] = float(entry.get("sigma", 0.0))
        drift_curve[minutes_int] = float(entry.get("drift", 0.0))
    if not sigma_curve:
        raise ValueError(f"Calibration payload missing sigma curve: {file_path}")
    return sigma_curve, drift_curve


def _extract_pit_bias(data: Mapping[str, Any]) -> float | None:
    pit_bias = data.get("pit_bias")
    if pit_bias is None:
        return None
    try:
        return float(pit_bias)
    except (TypeError, ValueError):
        return None


def _extract_late_day_variance(data: Mapping[str, Any]) -> LateDayVariance | None:
    variance_block = data.get("late_day_variance") or data.get("lambda_close")
    if not variance_block:
        return None
    try:
        threshold = int(variance_block.get("minutes_threshold", variance_block.get("minutes", 15)))
        lambda_value = float(variance_block.get("lambda", variance_block.get("lambda_value")))
    except (TypeError, ValueError):
        return None
    if threshold <= 0 or lambda_value <= 0.0:
        return None
    return LateDayVariance(minutes_threshold=threshold, lambda_value=lambda_value)


def _extract_event_tail(data: Mapping[str, Any]) -> EventTail | None:
    tail_block = data.get("event_tail") or data.get("event_day")
    if not tail_block:
        return None
    raw_tags = tail_block.get("tags") or []
    try:
        tags = tuple(str(tag).strip().lower() for tag in raw_tags if str(tag).strip())
    except Exception:  # pragma: no cover - defensive normalization
        tags = ()
    if not tags:
        return None
    try:
        kappa_value = float(tail_block.get("kappa") if "kappa" in tail_block else tail_block.get("multiplier"))
    except (TypeError, ValueError):
        return None
    if kappa_value <= 0.0:
        return None
    return EventTail(tags=tags, kappa=kappa_value)


def _extract_optional_curve(data: Mapping[str, Any], key: str) -> dict[int, float] | None:
    section = data.get(key)
    if not isinstance(section, Mapping):
        return None
    curve: dict[int, float] = {}
    for minute, value in section.items():
        try:
            minute_int = int(minute)
            curve[minute_int] = float(value)
        except (TypeError, ValueError):
            continue
    return curve or None


@lru_cache(maxsize=32)
def _load_calibration_cached(file_path: str) -> SigmaCalibration:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found at {file_path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    sigma_curve, drift_curve = _extract_minutes_curves(data, file_path)
    residual_std = float(data.get("residual_std", 0.0))
    return SigmaCalibration(
        sigma_curve=sigma_curve,
        drift_curve=drift_curve,
        residual_std=residual_std,
        pit_bias=_extract_pit_bias(data),
        late_day_variance=_extract_late_day_variance(data),
        event_tail=_extract_event_tail(data),
        sigma_now=_extract_sigma_now(data),
        m_tod_curve=_extract_optional_curve(data, "m_tod"),
        micro_drift_curve=_extract_optional_curve(data, "micro_drift"),
        metadata=_extract_metadata(data),
    )


def _extract_sigma_now(data: Mapping[str, Any]) -> float | None:
    value = data.get("sigma_now")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return None


def _extract_metadata(data: Mapping[str, Any]) -> Mapping[str, Any] | None:
    keys = ("source_window", "age", "checksum", "target", "series", "slug")
    metadata: dict[str, Any] = {}
    for key in keys:
        if key in data:
            metadata[key] = data[key]
    return metadata or None


def _resolve_calibration_file(path: Path, symbol: str, horizon: str, *, variant: str | None) -> Path:
    resolved = path.resolve()
    if resolved.is_file():
        return resolved
    slug = _symbol_slug(symbol)
    base_dir = resolved / slug / horizon
    if variant:
        candidate = base_dir / variant / "params.json"
        if candidate.exists():
            return candidate
    default_path = base_dir / "params.json"
    if default_path.exists():
        return default_path
    # Legacy noon fallback for intraday calibrations
    if horizon == "hourly":
        legacy = resolved / slug / "noon" / "params.json"
        if legacy.exists():
            return legacy
    raise FileNotFoundError(f"Calibration params not found for {symbol} horizon={horizon} variant={variant}")


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
