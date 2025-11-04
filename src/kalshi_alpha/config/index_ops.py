"""Load shared operational window configuration for index ladder strategies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "index_ops.yaml"
_DEFAULT_TZ = ZoneInfo("America/New_York")
_SERIES_WINDOW_MAP = MappingProxyType(
    {
        "INXU": "window_noon",
        "NASDAQ100U": "window_noon",
        "INX": "window_close",
        "NASDAQ100": "window_close",
    }
)


@dataclass(frozen=True)
class IndexOpsWindow:
    """Operations window definition with cancel buffers."""

    name: str
    start: time
    end: time
    cancel_buffer_seconds: float


@dataclass(frozen=True)
class IndexOpsConfig:
    """Aggregated operations configuration for index ladder scanners and microlive."""

    window_noon: IndexOpsWindow
    window_close: IndexOpsWindow
    min_ev_usd: float
    max_bins_per_series: int

    @property
    def timezone(self) -> ZoneInfo:
        return _DEFAULT_TZ

    def window_for_series(self, series: str) -> IndexOpsWindow:
        series_key = series.upper()
        try:
            window_key = _SERIES_WINDOW_MAP[series_key]
        except KeyError as exc:
            supported = ", ".join(sorted(_SERIES_WINDOW_MAP.keys()))
            message = (
                f"No operations window configured for '{series_key}'. "
                f"Supported series: {supported}"
            )
            raise KeyError(message) from exc
        return getattr(self, window_key)


def load_index_ops_config(path: Path | None = None) -> IndexOpsConfig:
    """Read the index operations configuration from disk."""

    resolved = (path or DEFAULT_CONFIG_PATH).resolve()
    return _load_index_ops_config_cached(str(resolved))


@lru_cache(maxsize=4)
def _load_index_ops_config_cached(resolved_path: str) -> IndexOpsConfig:
    config_path = Path(resolved_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Index operations configuration not found at {resolved_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ValueError("Failed to parse index operations configuration") from exc

    noon_payload = payload.get("window_noon", {})
    close_payload = payload.get("window_close", {})
    window_noon = _parse_window(noon_payload, label="window_noon")
    window_close = _parse_window(close_payload, label="window_close")
    min_ev = float(payload.get("min_ev_usd", 0.05))
    max_bins = int(payload.get("max_bins_per_series", 2))

    return IndexOpsConfig(
        window_noon=window_noon,
        window_close=window_close,
        min_ev_usd=min_ev,
        max_bins_per_series=max(max_bins, 1),
    )


def _parse_window(payload: dict[str, object], *, label: str) -> IndexOpsWindow:
    start_raw = str(payload.get("start", "")).strip()
    end_raw = str(payload.get("end", "")).strip()
    if not start_raw or not end_raw:
        raise ValueError(f"{label} requires start and end times (HH:MM)")
    start_time = _parse_time(start_raw)
    end_time = _parse_time(end_raw)
    cancel_buffer = float(payload.get("cancel_buffer_seconds", 2.0))
    name = label.removeprefix("window_") if label.startswith("window_") else label
    return IndexOpsWindow(
        name=name,
        start=start_time,
        end=end_time,
        cancel_buffer_seconds=cancel_buffer,
    )


def _parse_time(value: str) -> time:
    parts = value.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid time value '{value}' in index operations config")
    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) == 3 else 0
    except ValueError as exc:
        raise ValueError(f"Invalid time component in '{value}' for index operations config") from exc
    return time(hour % 24, minute % 60, second % 60)


__all__ = ["IndexOpsConfig", "IndexOpsWindow", "load_index_ops_config"]
