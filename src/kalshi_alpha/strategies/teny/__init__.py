"""10-year Treasury yield strategy stub."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

from kalshi_alpha.core.pricing import LadderBinProbability
from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.strategies import base


@dataclass(frozen=True)
class TenYInputs:
    latest_yield: float | None = None
    trailing_vol: float | None = None
    prior_close: float | None = None
    macro_shock: float = 0.0
    trailing_history: Sequence[float] | None = None


def pmf(strikes: Sequence[float], inputs: TenYInputs | None = None) -> list[LadderBinProbability]:
    inputs = inputs or TenYInputs()
    if inputs.prior_close is not None:
        calibration = _load_calibration()
        beta = calibration.get("shock_beta", 1.4) if calibration else 1.4
        mean = inputs.prior_close + beta * inputs.macro_shock
        history = list(inputs.trailing_history or [])
        history.append(inputs.prior_close)
        residual_std = calibration.get("residual_std") if calibration else None
        spread = max(_sample_std(history[-6:]), residual_std or 0.05)
        spread = min(spread, 0.2)
        distribution = {
            round(mean - spread, 3): 0.25,
            round(mean, 3): 0.5,
            round(mean + spread, 3): 0.25,
        }
        return base.grid_distribution_to_pmf(distribution)

    mean = inputs.latest_yield if inputs.latest_yield is not None else 4.50
    std = inputs.trailing_vol if inputs.trailing_vol is not None else 0.35
    std = max(std, 0.05)
    return base.pmf_from_gaussian(strikes, mean=mean, std=std)


def _sample_std(values: Sequence[float]) -> float:
    if not values:
        return 0.10
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return sqrt(variance)


CALIBRATION_DIR = PROC_ROOT / "calibration"
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_PATH = CALIBRATION_DIR / "teny.json"


def calibrate(history: Sequence[dict[str, float]]) -> dict[str, float]:
    if len(history) < 2:
        raise ValueError("ten-year history requires at least two rows")
    shocks: list[float] = []
    diffs: list[float] = []
    for idx in range(1, len(history)):
        prior = history[idx - 1]["actual_close"]
        current = history[idx]["actual_close"]
        shocks.append(history[idx]["macro_shock"])
        diffs.append(current - prior)
    shock_beta = _ols_beta(shocks, diffs)
    residuals = [diff - shock_beta * shock for diff, shock in zip(diffs, shocks, strict=True)]
    residual_std = sqrt(sum(res**2 for res in residuals) / max(len(residuals), 1))
    payload = {"shock_beta": shock_beta, "residual_std": residual_std}
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _ols_beta(x: Sequence[float], y: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(x, y, strict=True))
    denominator = sum(a * a for a in x)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _load_calibration() -> dict[str, float] | None:
    if not CALIBRATION_PATH.exists():
        return None
    try:
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return {key: float(value) for key, value in data.items()}
