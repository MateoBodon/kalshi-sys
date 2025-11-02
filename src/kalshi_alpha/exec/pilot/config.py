"""Utilities for loading pilot mode configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_CONFIG_CANDIDATES: tuple[Path, ...] = (
    Path("configs/pilot.yaml"),
    Path("configs/pilot.example.yaml"),
)


def _normalize_series(series: Iterable[str] | None) -> tuple[str, ...]:
    if not series:
        return tuple()
    normalized = {value.strip().upper() for value in series if value}
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class PilotConfig:
    """Immutable configuration for live pilot sessions."""

    allowed_series: tuple[str, ...] = ("CPI",)
    max_contracts_per_order: int = 1
    max_unique_bins: int = 2
    require_live_broker: bool = True
    enforce_maker_only: bool = True
    require_acknowledgement: bool = True
    session_prefix: str = "pilot"
    max_daily_loss: float | None = None
    max_weekly_loss: float | None = None

    @property
    def allow_any_series(self) -> bool:
        return not self.allowed_series

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PilotConfig":
        if not isinstance(payload, dict):
            raise TypeError("Pilot configuration payload must be a mapping")
        pilot_section = dict(payload)
        allowed_series = _normalize_series(pilot_section.get("allowed_series"))
        max_contracts = int(pilot_section.get("max_contracts_per_order", cls.max_contracts_per_order))
        if max_contracts <= 0:
            raise ValueError("max_contracts_per_order must be positive")
        max_bins_raw = pilot_section.get("max_unique_bins", cls.max_unique_bins)
        max_bins = int(max_bins_raw) if max_bins_raw is not None else cls.max_unique_bins
        if max_bins <= 0:
            raise ValueError("max_unique_bins must be positive")
        max_daily_loss_raw = pilot_section.get("max_daily_loss")
        max_weekly_loss_raw = pilot_section.get("max_weekly_loss")
        max_daily_loss = float(max_daily_loss_raw) if max_daily_loss_raw is not None else None
        max_weekly_loss = float(max_weekly_loss_raw) if max_weekly_loss_raw is not None else None
        return cls(
            allowed_series=allowed_series,
            max_contracts_per_order=max_contracts,
            max_unique_bins=max_bins,
            require_live_broker=bool(pilot_section.get("require_live_broker", cls.require_live_broker)),
            enforce_maker_only=bool(pilot_section.get("enforce_maker_only", cls.enforce_maker_only)),
            require_acknowledgement=bool(pilot_section.get("require_acknowledgement", cls.require_acknowledgement)),
            session_prefix=str(pilot_section.get("session_prefix", cls.session_prefix)).strip() or cls.session_prefix,
            max_daily_loss=max_daily_loss,
            max_weekly_loss=max_weekly_loss,
        )


def resolve_pilot_config_path(override: Path | None = None) -> Path | None:
    if override is not None:
        candidate = Path(override)
        if not candidate.exists():
            raise FileNotFoundError(f"Pilot config override not found: {candidate}")
        return candidate
    for candidate in DEFAULT_CONFIG_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def load_pilot_config(path: Path | None = None) -> PilotConfig:
    config_path = resolve_pilot_config_path(path)
    if config_path is None:
        return PilotConfig()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError("Pilot configuration root must be a mapping")
    payload = raw.get("pilot", raw)
    if not isinstance(payload, dict):
        raise TypeError("Pilot configuration must be a mapping under 'pilot'")
    return PilotConfig.from_dict(payload)
