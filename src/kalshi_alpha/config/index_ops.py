"""Load shared operational window configuration for index ladder strategies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "index_ops.yaml"


@dataclass(frozen=True)
class IndexOpsWindow:
    """Operations window definition for a group of index ladder series."""

    name: str
    start: time
    end: time
    series: tuple[str, ...]

    def contains_series(self, series: str) -> bool:
        return series.upper() in self.series


@dataclass(frozen=True)
class IndexOpsConfig:
    """Aggregated operations configuration for index ladder scanners and microlive."""

    timezone: ZoneInfo
    cancel_buffer_seconds: float
    windows: tuple[IndexOpsWindow, ...]
    series_to_window: Mapping[str, IndexOpsWindow]

    def window_for_series(self, series: str) -> IndexOpsWindow:
        series_key = series.upper()
        try:
            return self.series_to_window[series_key]
        except KeyError as exc:
            supported_codes: set[str] = set()
            for window in self.windows:
                supported_codes.update(window.series)
            supported = ", ".join(sorted(supported_codes))
            raise KeyError(f"No operations window configured for '{series_key}'. Supported series: {supported}") from exc


def load_index_ops_config(path: Path | None = None) -> IndexOpsConfig:
    """Read the index operations window configuration from disk."""

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
    timezone_name = str(payload.get("timezone", "America/New_York"))
    try:
        timezone = ZoneInfo(timezone_name)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported timezone '{timezone_name}' in index operations config") from exc
    cancel_buffer = float(payload.get("cancel_buffer_seconds", 2.0))
    windows_payload = payload.get("windows", [])
    if not isinstance(windows_payload, Sequence):
        raise ValueError("index operations config 'windows' entry must be a sequence")
    windows: list[IndexOpsWindow] = []
    series_map: dict[str, IndexOpsWindow] = {}
    for entry in windows_payload:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError("index operations window entry missing 'name'")
        start_raw = str(entry.get("start", "")).strip()
        end_raw = str(entry.get("end", "")).strip()
        if not start_raw or not end_raw:
            raise ValueError(f"index operations window '{name}' missing start/end times")
        start_time = _parse_time(start_raw)
        end_time = _parse_time(end_raw)
        series_values = entry.get("series", [])
        if not isinstance(series_values, Sequence):
            raise ValueError(f"index operations window '{name}' series list must be a sequence")
        codes = tuple(sorted({str(item).upper() for item in series_values if str(item).strip()}))
        if not codes:
            raise ValueError(f"index operations window '{name}' must include at least one series")
        window = IndexOpsWindow(name=name, start=start_time, end=end_time, series=codes)
        windows.append(window)
        for series in codes:
            series_map[series] = window
    if not windows:
        raise ValueError("index operations configuration contained no windows")
    return IndexOpsConfig(
        timezone=timezone,
        cancel_buffer_seconds=cancel_buffer,
        windows=tuple(windows),
        series_to_window=MappingProxyType(series_map),
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
