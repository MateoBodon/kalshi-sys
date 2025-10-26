"""CPI strategy stubs producing monthly change distributions.

References:
- Cleveland Fed inflation nowcast (used as an anchor when available).
- BLS CPI release schedule (driver data feeds).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean as stat_mean
from statistics import pstdev

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.strategies import base

GRID_STEP = 0.05
GRID_SPAN = 4  # +/- span multiples around the mean
DEFAULT_STD = 0.12
CALIBRATION_DIR = PROC_ROOT / "calibration"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_PATH = CALIBRATION_DIR / "cpi.json"


@dataclass(frozen=True)
class CPIInputs:
    cleveland_nowcast: float | None = None
    latest_release_mom: float | None = None
    aaa_delta: float = 0.0
    variance: float | None = None


def nowcast(inputs: CPIInputs | None = None) -> dict[float, float]:
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
    calibration = _load_calibration()
    if calibration:
        mean += calibration.get("bias", 0.0)
        std = max(std, calibration.get("std", std))

    grid = _grid_around(mean)
    weights = {point: _gaussian_weight(point, mean, std) for point in grid}
    total = sum(weights.values())
    if total == 0:
        raise ValueError("degenerate CPI nowcast weights")
    return {point: weight / total for point, weight in weights.items()}


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


def calibrate(history: Sequence[Mapping[str, float]], window: int = 12) -> dict[str, float]:
    if not history:
        raise ValueError("history is required for calibration")
    tail = list(history)[-window:]
    actuals = [float(entry["actual"]) for entry in tail]
    predictions = [
        float(entry["cleveland_nowcast"]) + float(entry.get("aaa_delta", 0.0))
        for entry in tail
    ]
    errors = [actual - pred for actual, pred in zip(actuals, predictions, strict=True)]
    bias = stat_mean(errors)
    std = max(pstdev(actuals), 0.03) if len(actuals) > 1 else 0.03
    payload = {"bias": bias, "std": std, "window": len(tail)}
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _load_calibration() -> dict[str, float] | None:
    if not CALIBRATION_PATH.exists():
        return None
    try:
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    result: dict[str, float] = {}
    if "bias" in data:
        result["bias"] = float(data["bias"])
    if "std" in data:
        result["std"] = float(data["std"])
    return result or None
