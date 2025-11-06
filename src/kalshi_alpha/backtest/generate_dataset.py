"""Build minute-level backtest dataset for index ladders."""

from __future__ import annotations

import argparse
import math
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import polars as pl

from kalshi_alpha.backtest.index_calendar import TargetSpec, TargetType, targets_for_day, trading_days
from kalshi_alpha.datastore.paths import DATA_ROOT
from kalshi_alpha.drivers.polygon_index.client import MinuteBar, PolygonAPIError, PolygonIndicesClient

ET = ZoneInfo("America/New_York")
DEFAULT_OUTPUT = DATA_ROOT / "backtest" / "index_minutes.parquet"
DEFAULT_SYMBOLS: tuple[str, ...] = ("I:SPX", "I:NDX")
TRADING_START = time(9, 30)
TRADING_END = time(16, 0)
EWMA_SPAN = 30
EWMA_MIN_SAMPLES = 5
MICRO_DRIFT_WINDOW = 5


@dataclass(frozen=True)
class DatasetRow:
    symbol: str
    trading_day: date
    observation_timestamp: datetime
    observation_timestamp_et: datetime
    target_timestamp: datetime
    target_type: TargetType
    minutes_to_target: int
    price_close: float
    ewma_sigma_now: float
    micro_drift: float
    target_on_before: float | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate minute-level Polygon backtest dataset for index ladders.")
    parser.add_argument("--start", required=True, help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output parquet path")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SYMBOLS),
        help="Polygon index symbols to include (default: I:SPX I:NDX)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def build_dataset(
    *,
    start: date,
    end: date,
    symbols: Sequence[str],
    client: PolygonIndicesClient | None = None,
) -> pl.DataFrame:
    client = client or PolygonIndicesClient()
    rows: list[DatasetRow] = []
    for trading_day in trading_days(start, end):
        targets = targets_for_day(trading_day)
        if not targets:
            continue
        for symbol in symbols:
            bars = _fetch_day_bars(client, symbol, trading_day)
            day_rows = _rows_for_symbol(trading_day, symbol, bars, targets)
            rows.extend(day_rows)
    if not rows:
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "trading_day": pl.Date,
                "observation_timestamp": pl.Datetime("UTC"),
                "observation_timestamp_et": pl.Datetime("America/New_York"),
                "target_timestamp": pl.Datetime("America/New_York"),
                "target_type": pl.Utf8,
                "minutes_to_target": pl.Int32,
                "price_close": pl.Float64,
                "ewma_sigma_now": pl.Float64,
                "micro_drift": pl.Float64,
                "target_on_before": pl.Float64,
            }
        )
    frame = pl.DataFrame(
        {
            "symbol": [row.symbol for row in rows],
            "trading_day": [row.trading_day for row in rows],
            "observation_timestamp": [row.observation_timestamp.astimezone(UTC) for row in rows],
            "observation_timestamp_et": [row.observation_timestamp_et for row in rows],
            "target_timestamp": [row.target_timestamp for row in rows],
            "target_type": [row.target_type.value for row in rows],
            "minutes_to_target": [row.minutes_to_target for row in rows],
            "price_close": [row.price_close for row in rows],
            "ewma_sigma_now": [row.ewma_sigma_now for row in rows],
            "micro_drift": [row.micro_drift for row in rows],
            "target_on_before": [row.target_on_before for row in rows],
        }
    )
    return frame.sort(["symbol", "observation_timestamp"])


def _fetch_day_bars(client: PolygonIndicesClient, symbol: str, trading_day: date) -> list[MinuteBar]:
    start_et = datetime.combine(trading_day, time(9, 0), tzinfo=ET)
    end_et = datetime.combine(trading_day, time(17, 0), tzinfo=ET)
    return client.fetch_minute_bars(symbol, start_et.astimezone(UTC), end_et.astimezone(UTC))


def _rows_for_symbol(
    trading_day: date,
    symbol: str,
    bars: Sequence[MinuteBar],
    targets: Sequence[TargetSpec],
) -> list[DatasetRow]:
    filtered_bars = [
        bar
        for bar in sorted(bars, key=lambda item: item.timestamp)
        if _in_session(bar.timestamp, trading_day)
    ]
    if not filtered_bars:
        return []

    closes = [float(bar.close) for bar in filtered_bars]
    sigma_series = _ewma_sigma_series(filtered_bars, span=EWMA_SPAN, min_samples=EWMA_MIN_SAMPLES)
    drift_series = _micro_drift_series(filtered_bars, window=MICRO_DRIFT_WINDOW)

    target_times = [target.timestamp_et for target in targets]
    target_lookup = {(symbol, target.timestamp_et): target for target in targets}
    on_before_values = _on_before_map(filtered_bars, targets, symbol)

    rows: list[DatasetRow] = []
    target_index = 0
    for idx, bar in enumerate(filtered_bars):
        observation_et = bar.timestamp.astimezone(ET)
        while target_index < len(target_times) and target_times[target_index] < observation_et:
            target_index += 1
        if target_index >= len(target_times):
            break
        target_ts = target_times[target_index]
        minutes_to_target = int(
            (target_ts - observation_et).total_seconds() // 60,
        )
        if minutes_to_target < 0:
            continue
        target_spec = target_lookup[(symbol, target_ts)]
        on_before = on_before_values.get((symbol, target_ts))
        rows.append(
            DatasetRow(
                symbol=symbol,
                trading_day=trading_day,
                observation_timestamp=bar.timestamp,
                observation_timestamp_et=observation_et,
                target_timestamp=target_ts,
                target_type=target_spec.target_type,
                minutes_to_target=minutes_to_target,
                price_close=float(closes[idx]),
                ewma_sigma_now=float(sigma_series[idx]),
                micro_drift=float(drift_series[idx]),
                target_on_before=on_before,
            )
        )
    return rows


def _in_session(timestamp_utc: datetime, trading_day: date) -> bool:
    ts_et = timestamp_utc.astimezone(ET)
    if ts_et.date() != trading_day:
        return False
    if ts_et.time() < TRADING_START or ts_et.time() > TRADING_END:
        return False
    return True


def _ewma_sigma_series(
    bars: Sequence[MinuteBar],
    *,
    span: int,
    min_samples: int,
) -> list[float]:
    closes = [float(bar.close) for bar in bars]
    results: list[float] = []
    if not closes:
        return results
    alpha = 2.0 / (span + 1.0)
    prev_price = closes[0]
    mean = 0.0
    variance = 0.0
    returns_seen = 0
    results.append(0.0)
    for price in closes[1:]:
        if prev_price <= 0.0 or price <= 0.0:
            results.append(0.0)
            prev_price = price
            continue
        ret = math.log(price / prev_price)
        if returns_seen == 0:
            mean = ret
            variance = 0.0
        else:
            mean = alpha * ret + (1 - alpha) * mean
            variance = (1 - alpha) * variance + alpha * (ret - mean) ** 2
        returns_seen += 1
        if returns_seen + 1 < max(min_samples, 2):
            sigma_value = 0.0
        else:
            sigma_value = math.sqrt(max(variance, 0.0)) * price
        results.append(float(sigma_value))
        prev_price = price
    if len(results) < len(closes):
        results.extend([0.0] * (len(closes) - len(results)))
    return results


def _micro_drift_series(
    bars: Sequence[MinuteBar],
    *,
    window: int,
) -> list[float]:
    closes = [float(bar.close) for bar in bars]
    results: list[float] = []
    buffer: deque[float] = deque(maxlen=window + 1)
    for price in closes:
        buffer.append(price)
        if len(buffer) < 2:
            results.append(0.0)
            continue
        deltas = [curr - prev for prev, curr in zip(buffer, list(buffer)[1:], strict=False)]
        average = sum(deltas) / len(deltas)
        results.append(float(average))
    return results


def _on_before_map(
    bars: Sequence[MinuteBar],
    targets: Sequence[TargetSpec],
    symbol: str,
) -> dict[tuple[str, datetime], float | None]:
    bar_times_et = [bar.timestamp.astimezone(ET) for bar in bars]
    closes = [float(bar.close) for bar in bars]
    mapping: dict[tuple[str, datetime], float | None] = {}
    for target in targets:
        target_key = (symbol, target.timestamp_et)
        position = bisect_right(bar_times_et, target.timestamp_et)
        if position == 0:
            mapping[target_key] = None
            continue
        mapping[target_key] = closes[position - 1]
    return mapping


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    symbols = tuple(symbol.strip() for symbol in args.symbols if symbol.strip())
    if not symbols:
        raise SystemExit("No Polygon symbols supplied")
    try:
        frame = build_dataset(start=start, end=end, symbols=symbols)
    except PolygonAPIError as exc:
        raise SystemExit(f"BLOCKED: Polygon API error: {exc}") from exc
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path)


__all__ = ["build_dataset", "main"]


if __name__ == "__main__":  # pragma: no cover
    main()
