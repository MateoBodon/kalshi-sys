"""Windowed Polygon index websocket collector for hourly and close ladders."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from kalshi_alpha.datastore.paths import RAW_ROOT

from .client import PolygonIndicesClient

ET = ZoneInfo("America/New_York")
POLYGON_RAW_ROOT = (RAW_ROOT / "polygon").resolve()
TARGET_HOURS = (10, 11, 12, 13, 14, 15, 16)
CLOSE_START_MINUTE = (15, 45)
CLOSE_END_MINUTE = (16, 1)


@dataclass(frozen=True)
class SubscriptionWindow:
    """Inclusive subscription window for a specific aggregate timespan."""

    label: str
    start_utc: datetime
    end_utc: datetime
    timespan: str

    def contains(self, timestamp: datetime) -> bool:
        return self.start_utc <= timestamp <= self.end_utc


@dataclass(frozen=True)
class WindowLatency:
    count: int
    avg_ms: float | None
    p95_ms: float | None
    max_ms: float | None


@dataclass(frozen=True)
class WindowResult:
    window: SubscriptionWindow
    latencies: Mapping[str, WindowLatency]


def _target_datetime(trading_day: date, hour: int, minute: int) -> datetime:
    return datetime(trading_day.year, trading_day.month, trading_day.day, hour, minute, tzinfo=ET)


def hourly_windows(trading_day: date) -> list[SubscriptionWindow]:
    """Return per-hour subscription windows for the trading day."""

    windows: list[SubscriptionWindow] = []
    for hour in TARGET_HOURS:
        target = _target_datetime(trading_day, hour, 0)
        start = target - timedelta(minutes=20)
        end = target + timedelta(minutes=1)
        windows.append(
            SubscriptionWindow(
                label=f"hourly-{hour:02d}00",
                start_utc=start.astimezone(UTC),
                end_utc=end.astimezone(UTC),
                timespan="minute",
            )
        )
    return windows


def close_window(trading_day: date) -> SubscriptionWindow:
    """Return the close subscription window for the trading day."""

    start = _target_datetime(trading_day, CLOSE_START_MINUTE[0], CLOSE_START_MINUTE[1])
    end = _target_datetime(trading_day, CLOSE_END_MINUTE[0], CLOSE_END_MINUTE[1])
    return SubscriptionWindow(
        label="close-1600",
        start_utc=start.astimezone(UTC),
        end_utc=end.astimezone(UTC),
        timespan="minute",
    )


def close_second_window(trading_day: date) -> SubscriptionWindow:
    """Return the second-level close window."""

    start = _target_datetime(trading_day, CLOSE_START_MINUTE[0], CLOSE_START_MINUTE[1])
    end = _target_datetime(trading_day, CLOSE_END_MINUTE[0], CLOSE_END_MINUTE[1])
    return SubscriptionWindow(
        label="close-second",
        start_utc=start.astimezone(UTC),
        end_utc=end.astimezone(UTC),
        timespan="second",
    )


def _truncate_latencies(latencies: Sequence[float]) -> WindowLatency:
    if not latencies:
        return WindowLatency(count=0, avg_ms=None, p95_ms=None, max_ms=None)
    sorted_lat = sorted(latencies)
    index = max(0, math.ceil(0.95 * len(sorted_lat)) - 1)
    index = min(index, len(sorted_lat) - 1)
    return WindowLatency(
        count=len(latencies),
        avg_ms=mean(latencies),
        p95_ms=sorted_lat[index],
        max_ms=max(latencies),
    )


def _message_timestamp(message: Mapping[str, object]) -> datetime | None:
    candidates = ("s", "t", "start", "startEpochMs", "timestamp")
    for key in candidates:
        value = message.get(key)
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
            except (OverflowError, OSError, ValueError):
                continue
    iso_value = message.get("ts")
    if isinstance(iso_value, str):
        try:
            parsed = datetime.fromisoformat(iso_value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _aggregated_value(entry: Mapping[str, object], key: str) -> Any:
    candidates = (key, key[:1], key.capitalize())
    for name in candidates:
        if not name:
            continue
        if name in entry:
            return entry[name]
    return None


def _aggregate_price(entry: Mapping[str, object], key: str) -> float | None:
    raw = _aggregated_value(entry, key)
    return _safe_float(raw)


def _aggregate_trades(entry: Mapping[str, object]) -> int | None:
    raw = _aggregated_value(entry, "trades")
    if raw is None:
        raw = _aggregated_value(entry, "n")
    if raw is None:
        raw = entry.get("transactions")
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return None
    return None


class PolygonWindowedCollector:
    """Collect Polygon aggregates for hourly/close windows and persist parquet files."""

    def __init__(  # noqa: PLR0913
        self,
        symbols: Sequence[str],
        *,
        client: PolygonIndicesClient | None = None,
        output_root: Path | None = None,
        include_seconds: bool = False,
        clock: Callable[[], datetime] | None = None,
        reconnect_attempts: int | None = 5,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
    ) -> None:
        if not symbols:
            raise ValueError("at least one symbol must be provided")
        self._symbols = tuple(symbols)
        self._symbol_set = {symbol.upper() for symbol in symbols}
        self._client = client or PolygonIndicesClient()
        self._output_root = (output_root or POLYGON_RAW_ROOT).resolve()
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._include_seconds = include_seconds
        self._clock = clock or (lambda: datetime.now(tz=UTC))
        self._reconnect_attempts = reconnect_attempts
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff

    async def collect(self, trading_day: date) -> list[WindowResult]:
        """Collect aggregates for the trading day and return latency summaries."""

        results: list[WindowResult] = []
        minute_windows = hourly_windows(trading_day) + [close_window(trading_day)]
        results.extend(await self._collect_windows(trading_day, minute_windows))
        if self._include_seconds:
            second_window = close_second_window(trading_day)
            results.extend(await self._collect_windows(trading_day, [second_window]))
        return results

    async def _collect_windows(  # noqa: PLR0912
        self,
        trading_day: date,
        windows: Sequence[SubscriptionWindow],
    ) -> list[WindowResult]:
        if not windows:
            return []
        timespan = windows[0].timespan
        for window in windows:
            if window.timespan != timespan:
                raise ValueError("all subscription windows must share the same timespan")
        start_utc = min(window.start_utc for window in windows)
        end_utc = max(window.end_utc for window in windows)

        stream = self._client.stream_aggregates(
            self._symbols,
            timespan=timespan,
            reconnect_attempts=self._reconnect_attempts,
            initial_backoff=self._initial_backoff,
            max_backoff=self._max_backoff,
        )
        latencies: dict[str, dict[str, list[float]]] = {
            window.label: {symbol: [] for symbol in self._symbols} for window in windows
        }
        records: dict[str, list[dict[str, object]]] = {symbol: [] for symbol in self._symbols}
        try:
            async for message in stream:
                symbol = str(message.get("sym") or "").upper()
                if symbol not in self._symbol_set:
                    continue
                timestamp = _message_timestamp(message)
                if timestamp is None:
                    continue
                if timestamp < start_utc:
                    continue
                if timestamp > end_utc:
                    break

                now = self._clock()
                if now.tzinfo is None:
                    now = now.replace(tzinfo=UTC)
                latency_ms = max(0.0, (now - timestamp).total_seconds() * 1000.0)

                row = {
                    "timestamp": timestamp,
                    "ingested_at": now,
                    "symbol": symbol,
                    "timespan": timespan,
                    "open": _aggregate_price(message, "open"),
                    "high": _aggregate_price(message, "high"),
                    "low": _aggregate_price(message, "low"),
                    "close": _aggregate_price(message, "close"),
                    "volume": _aggregate_price(message, "volume"),
                    "vwap": _aggregate_price(message, "vwap"),
                    "trades": _aggregate_trades(message),
                    "latency_ms": latency_ms,
                }
                records[symbol].append(row)
                for window in windows:
                    if window.contains(timestamp):
                        latencies[window.label][symbol].append(latency_ms)
        finally:  # Ensure websocket cleanup
            await stream.aclose()

        self._persist_records(trading_day, records)
        results: list[WindowResult] = []
        for window in windows:
            per_symbol = {
                symbol: _truncate_latencies(values)
                for symbol, values in latencies[window.label].items()
                if values
            }
            results.append(WindowResult(window=window, latencies=per_symbol))
        return results

    def _persist_records(self, trading_day: date, records: Mapping[str, Sequence[dict[str, object]]]) -> None:
        for symbol, rows in records.items():
            if not rows:
                continue
            symbol_slug = symbol.replace(":", "_").upper()
            symbol_dir = self._output_root / symbol_slug
            symbol_dir.mkdir(parents=True, exist_ok=True)
            day = trading_day.strftime("%Y-%m-%d")
            path = symbol_dir / f"{day}.parquet"
            frame = pl.DataFrame(rows).unique(subset="timestamp").sort("timestamp")
            if path.exists():
                existing = pl.read_parquet(path)
                frame = (
                    pl.concat([existing, frame], how="vertical_relaxed")
                    .unique(subset="timestamp")
                    .sort("timestamp")
                )
            frame.write_parquet(path)


__all__ = [
    "SubscriptionWindow",
    "WindowLatency",
    "WindowResult",
    "PolygonWindowedCollector",
    "hourly_windows",
    "close_window",
    "close_second_window",
]
