"""Simple per-family VaR limiter for index ladders."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping

import yaml

DEFAULT_LIMITS = {
    "SPX": 7500.0,
    "NDX": 7500.0,
}
CONFIG_PATH = Path("configs/index_var.yaml")
SERIES_FAMILY = {
    "INX": "SPX",
    "INXU": "SPX",
    "NASDAQ100": "NDX",
    "NASDAQ100U": "NDX",
}


class FamilyVarLimiter:
    def __init__(self, limits: Mapping[str, float] | None = None) -> None:
        self.limits = {key.upper(): float(value) for key, value in (limits or DEFAULT_LIMITS).items()}
        self.exposure: defaultdict[str, float] = defaultdict(float)

    def can_accept(self, series: str, max_loss: float) -> bool:
        if max_loss <= 0:
            return True
        family = SERIES_FAMILY.get(series.upper(), series.upper())
        limit = self.limits.get(family)
        if limit is None or limit <= 0:
            return True
        return self.exposure[family] + max_loss <= limit

    def register(self, series: str, max_loss: float) -> None:
        if max_loss <= 0:
            return
        family = SERIES_FAMILY.get(series.upper(), series.upper())
        self.exposure[family] += max_loss

    def snapshot(self) -> dict[str, float]:
        return dict(self.exposure)


def load_family_limits(path: Path | None = None) -> Mapping[str, float]:
    config_path = path or CONFIG_PATH
    if not config_path.exists():
        return dict(DEFAULT_LIMITS)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:  # pragma: no cover - invalid config
        return dict(DEFAULT_LIMITS)
    limits = payload.get("limits") if isinstance(payload, dict) else None
    if not isinstance(limits, Mapping):
        return dict(DEFAULT_LIMITS)
    parsed: dict[str, float] = {}
    for key, value in limits.items():
        try:
            parsed[str(key).upper()] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed or dict(DEFAULT_LIMITS)


__all__ = ["FamilyVarLimiter", "load_family_limits", "CONFIG_PATH"]
