"""Size ladder configuration for staged lot/bin limits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

DEFAULT_SIZE_LADDER_PATH = Path("configs/size_ladder.yaml")


@dataclass(frozen=True)
class SeriesLimits:
    max_contracts: int
    max_bins: int


@dataclass(frozen=True)
class SizeStage:
    name: str
    description: str
    per_series: dict[str, SeriesLimits]

    def limits_for(self, series: str) -> SeriesLimits | None:
        return self.per_series.get(series.upper())


@dataclass(frozen=True)
class SizeLadderConfig:
    current_stage: str
    stages: dict[str, SizeStage]

    def stage(self, name: str | None = None) -> SizeStage:
        key = (name or self.current_stage).upper()
        if key not in self.stages:
            available = ", ".join(sorted(self.stages))
            raise KeyError(f"Unknown size stage '{key}'. Available: {available}")
        return self.stages[key]


def load_size_ladder(path: Path | None = None) -> SizeLadderConfig:
    config_path = path or DEFAULT_SIZE_LADDER_PATH
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    stages_payload = payload.get("stages")
    if not isinstance(stages_payload, Mapping):
        raise ValueError("size ladder configuration missing 'stages' mapping")
    stages: dict[str, SizeStage] = {}
    for stage_name, stage_config in stages_payload.items():
        if not isinstance(stage_config, Mapping):
            continue
        per_series_payload = stage_config.get("per_series")
        if not isinstance(per_series_payload, Mapping):
            continue
        series_limits: dict[str, SeriesLimits] = {}
        for series_name, limits in per_series_payload.items():
            if not isinstance(limits, Mapping):
                continue
            max_contracts = int(limits.get("max_contracts", 0))
            max_bins = int(limits.get("max_bins", 0))
            series_limits[series_name.upper()] = SeriesLimits(
                max_contracts=max(0, max_contracts),
                max_bins=max(0, max_bins),
            )
        description = str(stage_config.get("description") or stage_name)
        stages[stage_name.upper()] = SizeStage(
            name=stage_name.upper(),
            description=description,
            per_series=series_limits,
        )
    current_stage = str(payload.get("current_stage") or "A").upper()
    return SizeLadderConfig(current_stage=current_stage, stages=stages)


__all__ = ["SeriesLimits", "SizeStage", "SizeLadderConfig", "load_size_ladder", "DEFAULT_SIZE_LADDER_PATH"]
