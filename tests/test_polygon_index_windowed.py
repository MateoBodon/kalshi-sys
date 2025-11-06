from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from kalshi_alpha.drivers.polygon_index.windowed import (
    PolygonWindowedCollector,
    close_second_window,
    close_window,
    hourly_windows,
)

ET = ZoneInfo("America/New_York")


def test_hourly_windows_cover_expected_schedule() -> None:
    trading_day = date(2025, 11, 5)
    windows = hourly_windows(trading_day)
    assert len(windows) == 7
    first = windows[0]
    last = windows[-1]
    assert first.start_utc == datetime(2025, 11, 5, 14, 40, tzinfo=UTC)
    assert first.end_utc == datetime(2025, 11, 5, 15, 1, tzinfo=UTC)
    assert last.start_utc == datetime(2025, 11, 5, 20, 40, tzinfo=UTC)
    assert last.end_utc == datetime(2025, 11, 5, 21, 1, tzinfo=UTC)


class _StubStream:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = list(messages)
        self.closed = False

    def __aiter__(self) -> _StubStream:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)

    async def aclose(self) -> None:
        self.closed = True


class _StubClient:
    def __init__(self, streams: dict[str, list[dict[str, Any]]]) -> None:
        self._streams = {key: list(value) for key, value in streams.items()}
        self.calls: list[dict[str, Any]] = []
        self.open_streams: list[_StubStream] = []

    def stream_aggregates(
        self,
        symbols: Sequence[str],
        *,
        timespan: str = "minute",
        reconnect_attempts: int | None = None,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
    ) -> _StubStream:
        self.calls.append(
            {
                "symbols": tuple(symbols),
                "timespan": timespan,
                "reconnect_attempts": reconnect_attempts,
                "initial_backoff": initial_backoff,
                "max_backoff": max_backoff,
            }
        )
        stream = _StubStream(self._streams.get(timespan, []))
        self.open_streams.append(stream)
        return stream


class _Clock:
    def __init__(self, moments: list[datetime]) -> None:
        if not moments:
            raise ValueError("clock requires at least one moment")
        self._moments = [moment.astimezone(UTC) for moment in moments]
        self._last = self._moments[-1]

    def __call__(self) -> datetime:
        if self._moments:
            self._last = self._moments.pop(0)
        return self._last


def _aggregate_message(symbol: str, ts: datetime, channel: str) -> dict[str, Any]:
    timestamp_ms = int(ts.astimezone(UTC).timestamp() * 1000)
    return {
        "ev": "AM" if channel == "minute" else "AS",
        "sym": symbol,
        "s": timestamp_ms,
        "e": timestamp_ms + (60_000 if channel == "minute" else 1_000),
        "o": 5000.0,
        "h": 5001.0,
        "l": 4999.5,
        "c": 5000.5,
        "v": 1200.0,
        "vw": 5000.8,
        "n": 42,
    }


def test_windowed_collector_writes_and_reports(tmp_path: Path) -> None:
    trading_day = date(2025, 11, 5)
    minute_ts_spx = datetime(2025, 11, 5, 9, 45, tzinfo=ET)
    minute_ts_ndx = datetime(2025, 11, 5, 11, 5, tzinfo=ET)
    second_ts_spx = datetime(2025, 11, 5, 15, 45, 5, tzinfo=ET)

    streams = {
        "minute": [
            _aggregate_message("I:SPX", minute_ts_spx, "minute"),
            _aggregate_message("I:NDX", minute_ts_ndx, "minute"),
        ],
        "second": [
            _aggregate_message("I:SPX", second_ts_spx, "second"),
        ],
    }
    client = _StubClient(streams)
    clock = _Clock(
        [
            minute_ts_spx.astimezone(UTC) + timedelta(seconds=1),
            minute_ts_ndx.astimezone(UTC) + timedelta(seconds=2),
            second_ts_spx.astimezone(UTC) + timedelta(seconds=1),
        ]
    )

    collector = PolygonWindowedCollector(
        ("I:SPX", "I:NDX"),
        client=client,
        output_root=tmp_path,
        include_seconds=True,
        clock=clock,
        reconnect_attempts=1,
        initial_backoff=0.1,
        max_backoff=0.2,
    )

    results = asyncio.run(collector.collect(trading_day))

    assert len(results) == len(hourly_windows(trading_day)) + 1 + 1
    labels = [result.window.label for result in results]
    assert "hourly-1000" in labels
    assert close_window(trading_day).label in labels
    assert close_second_window(trading_day).label in labels

    spx_path = tmp_path / "I_SPX" / "2025-11-05.parquet"
    ndx_path = tmp_path / "I_NDX" / "2025-11-05.parquet"
    assert spx_path.exists()
    assert ndx_path.exists()

    spx_frame = pl.read_parquet(spx_path)
    assert spx_frame.height == 2  # minute + second rows
    assert set(spx_frame.get_column("timespan").to_list()) == {"minute", "second"}
    assert spx_frame.get_column("latency_ms").min() >= 0.0

    ndx_frame = pl.read_parquet(ndx_path)
    assert ndx_frame.height == 1
    assert ndx_frame.get_column("symbol")[0] == "I:NDX"

    minute_result = next(result for result in results if result.window.label == "hourly-1000")
    assert "I:SPX" in minute_result.latencies
    spx_latency = minute_result.latencies["I:SPX"]
    assert spx_latency.count == 1
    assert spx_latency.max_ms is not None and spx_latency.max_ms >= 0.0

    second_result = next(result for result in results if result.window.label == "close-second")
    assert "I:SPX" in second_result.latencies
    assert second_result.latencies["I:SPX"].count == 1

    assert all(stream.closed for stream in client.open_streams)
