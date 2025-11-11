"""Index PMF utilities with σ_tod curves and optional EOD variance bumps."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.datastore.paths import INDEX_PMF_ROOT
from kalshi_alpha.strategies.index.cdf import gaussian_pmf

_MIN_STD_DEFAULT = 0.5


@dataclass(frozen=True)
class EODBump:
    minutes_threshold: int
    variance: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, object] | None) -> EODBump | None:
        if not isinstance(payload, Mapping):
            return None
        try:
            threshold = int(payload.get("minutes_threshold", payload.get("minutes")))
            variance = float(payload.get("variance") or payload.get("lambda"))
        except (TypeError, ValueError):
            return None
        if threshold < 0 or variance <= 0.0:
            return None
        return cls(minutes_threshold=threshold, variance=variance)


@dataclass(frozen=True)
class IndexPMFParameters:
    series: str
    target_type: str
    target_label: str
    sigma_curve: Mapping[int, float]
    drift_curve: Mapping[int, float]
    residual_std: float
    min_std: float
    metadata: Mapping[str, object]
    eod_bump: EODBump | None = None

    def sigma(self, minutes_to_target: int) -> float:
        return _nearest(self.sigma_curve, minutes_to_target)

    def drift(self, minutes_to_target: int) -> float:
        return _nearest(self.drift_curve, minutes_to_target)


class IndexPMFModel:
    """Produce ladder PMFs from stored σ_tod + drift curves."""

    def __init__(self, params: IndexPMFParameters) -> None:
        self.params = params

    def pmf(
        self,
        strikes: Sequence[float],
        *,
        minutes_to_target: int,
        current_price: float,
        drift_override: float | None = None,
        sigma_override: float | None = None,
    ) -> list[LadderBinProbability]:
        minutes = max(int(minutes_to_target), 0)
        sigma = sigma_override if sigma_override is not None else self.params.sigma(minutes)
        drift = drift_override if drift_override is not None else self.params.drift(minutes)
        base_std = max(float(sigma), float(self.params.residual_std), float(self.params.min_std))
        variance = base_std**2
        bump = self.params.eod_bump
        if bump is not None and minutes <= bump.minutes_threshold:
            variance += max(bump.variance, 0.0)
        std = math.sqrt(max(variance, self.params.min_std**2, 1e-9))
        mean = float(current_price) + float(drift)
        return gaussian_pmf(strikes, mean=mean, std=std, min_std=self.params.min_std)


def load_model(
    series: str,
    target_type: str,
    target_label: str,
    *,
    root: Path | None = None,
) -> IndexPMFModel:
    params = load_parameters(series, target_type, target_label, root=root)
    return IndexPMFModel(params)


@lru_cache(maxsize=64)
def load_parameters(
    series: str,
    target_type: str,
    target_label: str,
    *,
    root: Path | None = None,
) -> IndexPMFParameters:
    path = _params_path(series, target_type, target_label, root=root)
    if not path.exists():
        raise FileNotFoundError(f"PMF parameters missing for {series}/{target_type}/{target_label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    sigma_curve = _decode_curve(payload.get("sigma_curve") or payload.get("minutes_to_target"), component="sigma")
    drift_curve = _decode_curve(payload.get("drift_curve") or payload.get("minutes_to_target"), component="drift")
    residual_std = float(payload.get("residual_std", 0.0))
    min_std = float(payload.get("min_std", _MIN_STD_DEFAULT))
    eod_bump = EODBump.from_payload(payload.get("eod_bump"))
    metadata = payload.get("metadata") or {}
    return IndexPMFParameters(
        series=str(payload.get("series") or series).upper(),
        target_type=str(payload.get("target_type") or target_type).lower(),
        target_label=str(payload.get("target_label") or target_label),
        sigma_curve=sigma_curve,
        drift_curve=drift_curve,
        residual_std=residual_std,
        min_std=min_std,
        metadata=metadata,
        eod_bump=eod_bump,
    )


def available_targets(series: str, target_type: str, *, root: Path | None = None) -> list[str]:
    base = _series_root(series, target_type, root=root)
    if not base.exists():
        return []
    return sorted(path.stem for path in base.glob("*.json") if path.is_file())


def _params_path(series: str, target_type: str, target_label: str, *, root: Path | None) -> Path:
    base = _series_root(series, target_type, root=root)
    return base / f"{target_label}.json"


def _series_root(series: str, target_type: str, *, root: Path | None) -> Path:
    normalized_series = series.strip().upper()
    normalized_type = target_type.strip().lower()
    base = (root or INDEX_PMF_ROOT) / normalized_series / normalized_type
    base.mkdir(parents=True, exist_ok=True)
    return base


def _decode_curve(payload: Mapping[str, object] | None, *, component: str) -> dict[int, float]:
    curve: dict[int, float] = {}
    if not isinstance(payload, Mapping):
        return curve
    for key, value in payload.items():
        try:
            minute = int(key)
        except (TypeError, ValueError):
            continue
        numeric: float | None
        if isinstance(value, Mapping):
            raw = value.get(component)
            if raw is None and component == "sigma":
                raw = value.get("value")
        else:
            raw = value
        try:
            numeric = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            numeric = None
        if numeric is None:
            continue
        curve[minute] = numeric
    return curve


def _nearest(curve: Mapping[int, float], minutes: int) -> float:
    if not curve:
        return 0.0
    if minutes in curve:
        return float(curve[minutes])
    ordered = sorted(curve)
    if not ordered:
        return 0.0
    if minutes <= ordered[0]:
        return float(curve[ordered[0]])
    if minutes >= ordered[-1]:
        return float(curve[ordered[-1]])
    # binary search for nearest neighbor
    lo, hi = 0, len(ordered) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if ordered[mid] == minutes:
            return float(curve[ordered[mid]])
        if ordered[mid] < minutes:
            lo = mid + 1
        else:
            hi = mid - 1
    upper = ordered[lo] if lo < len(ordered) else ordered[-1]
    lower = ordered[lo - 1] if lo - 1 >= 0 else ordered[0]
    if abs(minutes - lower) <= abs(upper - minutes):
        return float(curve[lower])
    return float(curve[upper])


__all__ = [
    "IndexPMFModel",
    "IndexPMFParameters",
    "EODBump",
    "available_targets",
    "load_model",
    "load_parameters",
]
