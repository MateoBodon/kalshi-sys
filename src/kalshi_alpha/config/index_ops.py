"""Load shared operational window configuration for index ladder strategies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
import warnings
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "index_ops.yaml"
_DEFAULT_TZ = ZoneInfo("America/New_York")
_SERIES_WINDOW_MAP = MappingProxyType(
    {
        "INXU": "window_hourly",
        "NASDAQ100U": "window_hourly",
        "INX": "window_close",
        "NASDAQ100": "window_close",
    }
)


@dataclass(frozen=True)
class IndexOpsWindow:
    """Operations window definition with cancel buffers and optional offsets."""

    name: str
    start: time | None
    end: time | None
    cancel_buffer_seconds: float
    start_offset_minutes: int | None = None
    end_at_target: bool = False

    def bounds_for(
        self,
        *,
        reference: datetime,
        target_time: time | None = None,
        timezone: ZoneInfo | None = None,
    ) -> tuple[datetime, datetime]:
        tz = timezone or reference.tzinfo or _DEFAULT_TZ
        reference_local = reference.astimezone(tz)
        base_date = reference_local.date()

        if self.start_offset_minutes is not None and target_time is not None:
            end_base = target_time if self.end_at_target else (self.end or target_time)
            if end_base is None:
                raise ValueError(f"{self.name} window requires target/end time to resolve bounds")
            end_dt = datetime.combine(base_date, end_base, tzinfo=tz)
            start_dt = end_dt - timedelta(minutes=max(self.start_offset_minutes, 0))
        else:
            start_base = self.start if self.start is not None else target_time
            end_base = target_time if self.end_at_target and target_time is not None else self.end
            if start_base is None or end_base is None:
                raise ValueError(f"{self.name} window requires start/end definitions")
            start_dt = datetime.combine(base_date, start_base, tzinfo=tz)
            end_dt = datetime.combine(base_date, end_base, tzinfo=tz)

        if end_dt <= start_dt:
            end_dt += timedelta(days=1)
        return start_dt, end_dt


@dataclass(frozen=True)
class IndexOpsConfig:
    """Aggregated operations configuration for index ladder scanners and microlive."""

    window_hourly: IndexOpsWindow
    window_close: IndexOpsWindow
    min_ev_usd: float
    max_bins_per_series: int

    @property
    def timezone(self) -> ZoneInfo:
        return _DEFAULT_TZ

    @property
    def window_noon(self) -> IndexOpsWindow:  # pragma: no cover - backward compatibility
        warnings.warn(
            "window_noon is deprecated; use window_hourly",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.window_hourly

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

    hourly_payload = payload.get("window_hourly") or payload.get("window_noon") or {}
    close_payload = payload.get("window_close", {})
    window_hourly = _parse_window(hourly_payload, label="window_hourly")
    window_close = _parse_window(close_payload, label="window_close")
    min_ev = float(payload.get("min_ev_usd", 0.05))
    max_bins = int(payload.get("max_bins_per_series", 2))

    return IndexOpsConfig(
        window_hourly=window_hourly,
        window_close=window_close,
        min_ev_usd=min_ev,
        max_bins_per_series=max(max_bins, 1),
    )


def _parse_window(payload: dict[str, object], *, label: str) -> IndexOpsWindow:
    if not payload:
        raise ValueError(f"{label} configuration cannot be empty")

    cancel_buffer = float(payload.get("cancel_buffer_seconds", 2.0))
    name = label.removeprefix("window_") if label.startswith("window_") else label

    start_offset_raw = payload.get("start_offset_min")
    start_offset_minutes: int | None = None
    if start_offset_raw is not None:
        try:
            start_offset_minutes = int(start_offset_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} start_offset_min must be an integer") from exc
        if start_offset_minutes < 0:
            raise ValueError(f"{label} start_offset_min must be non-negative")

    end_at_target = bool(payload.get("end_at_target", False))

    start_time: time | None = None
    end_time: time | None = None
    if "start" in payload:
        start_raw = str(payload.get("start", "")).strip()
        if not start_raw:
            raise ValueError(f"{label} start cannot be empty")
        start_time = _parse_time(start_raw)
    if "end" in payload:
        end_raw = str(payload.get("end", "")).strip()
        if not end_raw:
            raise ValueError(f"{label} end cannot be empty")
        end_time = _parse_time(end_raw)

    if start_time is None and start_offset_minutes is None:
        raise ValueError(f"{label} requires start or start_offset_min")
    if end_time is None and not end_at_target:
        raise ValueError(f"{label} requires end or end_at_target=true")

    return IndexOpsWindow(
        name=name,
        start=start_time,
        end=end_time,
        cancel_buffer_seconds=cancel_buffer,
        start_offset_minutes=start_offset_minutes,
        end_at_target=end_at_target,
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
