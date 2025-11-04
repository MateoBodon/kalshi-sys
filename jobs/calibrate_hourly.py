"""Build hourly calibration curves for index ladders using Polygon minute bars."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import polars as pl

from kalshi_alpha.drivers.polygon_index.client import MinuteBar, PolygonIndicesClient
from kalshi_alpha.drivers.polygon_index.snapshots import write_minute_bars
from kalshi_alpha.drivers.polygon_index.symbols import resolve_series
from kalshi_alpha.strategies.index.hourly_above_below import HOURLY_CALIBRATION_PATH

from ._index_calibration import (
    ET,
    build_sigma_curve,
    event_tail_multiplier,
    extend_calibration_window,
)

TARGET_TIME = time(12, 0)
RESIDUAL_WINDOW_MINUTES = 5
EVENT_WINDOW_MINUTES = 30
EVENT_KAPPA_CLAMP = (1.0, 1.65)
EVENT_TAGS = ("CPI", "FOMC")
DEFAULT_DAYS = 35
HORIZON = "hourly"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate hourly index distribution from Polygon minute bars.")
    parser.add_argument("--start", type=_parse_date, help="Start date (YYYY-MM-DD). Defaults to trailing window.")
    parser.add_argument("--end", type=_parse_date, help="End date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Trailing trading-day window when start/end not provided.",
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
        default=HOURLY_CALIBRATION_PATH,
        help="Output directory root for calibration parameters (default: data/proc/calib/index).",
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

    frame, records = build_sigma_curve(
        bars_by_symbol,
        target_time=TARGET_TIME,
        residual_window=RESIDUAL_WINDOW_MINUTES,
        return_records=True,
    )
    extras = _derive_event_extras(records)
    _write_params(frame, args.output, extras=extras)


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
    start, end = extend_calibration_window(start, end, tags=EVENT_TAGS)
    return start, end


def _time_bounds(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    start_dt = datetime.combine(start_date, time(8, 0), tzinfo=ET).astimezone(UTC)
    end_dt = datetime.combine(end_date + timedelta(days=1), time(4, 0), tzinfo=ET).astimezone(UTC)
    return start_dt, end_dt


def _derive_event_extras(records: pl.DataFrame) -> dict[str, dict[str, object]]:
    if records.is_empty():
        return {}
    symbols = records.get_column("symbol").unique().to_list()
    extras: dict[str, dict[str, object]] = {}
    for symbol in symbols:
        kappa = event_tail_multiplier(
            records,
            symbol,
            window=EVENT_WINDOW_MINUTES,
            tags=EVENT_TAGS,
            clamp=EVENT_KAPPA_CLAMP,
        )
        extras[symbol] = {
            "event_tail": {
                "tags": list(EVENT_TAGS),
                "kappa": round(float(kappa), 4),
            }
        }
    return extras


def _write_params(
    frame: pl.DataFrame,
    output: Path,
    *,
    extras: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    unique_symbols = frame.get_column("symbol").unique().to_list()
    generated_at = datetime.now(tz=UTC).isoformat()
    for symbol in unique_symbols:
        subset = frame.filter(pl.col("symbol") == symbol).sort("minutes_to_target")
        slug = symbol.split(":")[-1].lower()
        target_dir = output / slug / HORIZON
        target_dir.mkdir(parents=True, exist_ok=True)
        minutes_map: dict[str, dict[str, float]] = {}
        for row in subset.iter_rows(named=True):
            minute = int(row["minutes_to_target"])
            minutes_map[str(minute)] = {
                "sigma": float(row.get("sigma", 0.0)),
                "drift": float(row.get("drift", 0.0)),
            }
        residual_column = subset.get_column("residual_std") if "residual_std" in subset.columns else None
        residual_values = residual_column.drop_nulls().to_list() if residual_column is not None else []
        residual_std = float(residual_values[0]) if residual_values else 0.0
        kappa_value = 1.0
        payload: dict[str, object] = {
            "symbol": symbol,
            "horizon": HORIZON,
            "generated_at": generated_at,
            "minutes_to_target": minutes_map,
            "residual_std": residual_std,
            "pit_bias": None,
        }
        extra_entry = (extras or {}).get(symbol)
        if extra_entry:
            event_tail = extra_entry.get("event_tail", {})
            tags = [str(tag).strip() for tag in event_tail.get("tags", EVENT_TAGS) if str(tag).strip()]
            if tags:
                kappa_value = float(event_tail.get("kappa", 1.0))
                payload["event_tail"] = {
                    "tags": [tag.upper() for tag in tags],
                    "kappa": kappa_value,
                }
        payload["kappa_event"] = kappa_value
        target_path = target_dir / "params.json"
        target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
