"""Build noon calibration curves for index ladders using Polygon minute bars."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from kalshi_alpha.drivers.polygon_index.client import MinuteBar, PolygonIndicesClient
from kalshi_alpha.drivers.polygon_index.snapshots import write_minute_bars
from kalshi_alpha.drivers.polygon_index.symbols import resolve_series
from kalshi_alpha.strategies.index.noon_above_below import NOON_CALIBRATION_PATH

from ._index_calibration import ET, build_sigma_curve

TARGET_TIME = time(12, 0)
RESIDUAL_WINDOW_MINUTES = 5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate noon index distribution from Polygon minute bars.")
    parser.add_argument("--start", type=_parse_date, help="Start date (YYYY-MM-DD). Defaults to 30 trading days ago.")
    parser.add_argument("--end", type=_parse_date, help="End date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Fallback trading day window when start/end not provided.",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=["INXU", "NASDAQ100U"],
        help="Kalshi series symbols to calibrate (resolved to Polygon indices).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=NOON_CALIBRATION_PATH,
        help="Output parquet path for calibration table.",
    )
    parser.add_argument(
        "--skip-snapshots",
        action="store_true",
        help="Skip writing raw datastore snapshots.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start_date, end_date = _resolve_window(args.start, args.end, args.days)
    tickers = _resolve_tickers(args.series)

    client = PolygonIndicesClient()
    bars_by_symbol: dict[str, list[MinuteBar]] = defaultdict(list)
    for ticker in tickers:
        start_ts, end_ts = _time_bounds(start_date, end_date)
        bars = client.fetch_minute_bars(ticker, start_ts, end_ts)
        if not args.skip_snapshots:
            write_minute_bars(ticker, bars)
        bars_by_symbol[ticker].extend(bars)

    if not bars_by_symbol:
        raise RuntimeError("No minute bars fetched for calibration")

    frame = build_sigma_curve(bars_by_symbol, target_time=TARGET_TIME, residual_window=RESIDUAL_WINDOW_MINUTES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(args.output)


def _resolve_tickers(series_list: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    for series in series_list:
        try:
            resolved.append(resolve_series(series).polygon_ticker)
        except KeyError:
            resolved.append(series)
    return sorted(set(resolved))


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _resolve_window(
    start: date | None,
    end: date | None,
    days: int,
) -> tuple[date, date]:
    if end is None:
        end = datetime.now(tz=ET).date()
    if start is None:
        start = end - timedelta(days=max(days, 1))
    if start > end:
        raise ValueError("start date must be on or before end date")
    return start, end


def _time_bounds(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    start_dt = datetime.combine(start_date, time(8, 0), tzinfo=ET).astimezone(UTC)
    end_dt = datetime.combine(end_date + timedelta(days=1), time(4, 0), tzinfo=ET).astimezone(UTC)
    return start_dt, end_dt


if __name__ == "__main__":
    main()
