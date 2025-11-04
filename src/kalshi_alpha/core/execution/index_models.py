"""Execution curve loaders for index fills and slippage."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from kalshi_alpha.datastore.paths import PROC_ROOT

EXEC_ROOT = PROC_ROOT / "index_exec"
TICK_SIZE = 0.01


@dataclass(frozen=True)
class AlphaCurve:
    intercept: float
    depth_fraction: float
    abs_delta: float
    log_tau: float
    clip_min: float
    clip_max: float
    baseline_alpha: float

    def predict(
        self,
        *,
        depth_fraction: float,
        delta_p: float,
        tau_minutes: float,
    ) -> float:
        """Predict fill alpha using the fitted linear surface."""

        logits = (
            self.intercept
            + self.depth_fraction * depth_fraction
            + self.abs_delta * abs(delta_p)
            + self.log_tau * math.log1p(max(tau_minutes, 0.0) / 60.0)
        )
        return max(self.clip_min, min(self.clip_max, logits))


@dataclass(frozen=True)
class SlippageCurve:
    intercept: float
    depth_fraction: float
    spread: float
    depth_spread: float
    log_tau: float

    def predict_ticks(
        self,
        *,
        depth_fraction: float,
        spread: float,
        tau_minutes: float,
    ) -> float:
        """Predict absolute slippage in ticks."""

        raw = (
            self.intercept
            + self.depth_fraction * depth_fraction
            + self.spread * spread
            + self.depth_spread * (depth_fraction * spread)
            + self.log_tau * math.log1p(max(tau_minutes, 0.0) / 60.0)
        )
        return max(0.0, raw)


def _curve_path(series: str, name: str) -> Path:
    return EXEC_ROOT / series.upper() / f"{name}.json"


@lru_cache(maxsize=8)
def load_alpha_curve(series: str) -> AlphaCurve | None:
    path = _curve_path(series, "alpha")
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    coeffs = payload.get("coefficients", {})
    clip = payload.get("clip", {})
    try:
        return AlphaCurve(
            intercept=float(coeffs.get("intercept", 0.0)),
            depth_fraction=float(coeffs.get("depth_fraction", 0.0)),
            abs_delta=float(coeffs.get("abs_delta_p", 0.0)),
            log_tau=float(coeffs.get("log_tau_minutes", 0.0)),
            clip_min=float(clip.get("min", 0.0)),
            clip_max=float(clip.get("max", 1.0)),
            baseline_alpha=float(payload.get("baseline_alpha", 0.0)),
        )
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=8)
def load_slippage_curve(series: str) -> SlippageCurve | None:
    path = _curve_path(series, "slippage")
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    coeffs = payload.get("coefficients", {})
    try:
        return SlippageCurve(
            intercept=float(coeffs.get("intercept", 0.0)),
            depth_fraction=float(coeffs.get("depth_fraction", 0.0)),
            spread=float(coeffs.get("spread", 0.0)),
            depth_spread=float(coeffs.get("depth_fraction_x_spread", 0.0)),
            log_tau=float(coeffs.get("log_tau_minutes", 0.0)),
        )
    except (TypeError, ValueError):
        return None


def slippage_ticks_to_price(ticks: float) -> float:
    return max(0.0, float(ticks)) * TICK_SIZE


__all__ = [
    "AlphaCurve",
    "SlippageCurve",
    "load_alpha_curve",
    "load_slippage_curve",
    "slippage_ticks_to_price",
]
