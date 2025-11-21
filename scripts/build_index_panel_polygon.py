"""Build a minute-level Polygon index panel with basic time-to-target features.

Inputs:
    - Raw Polygon minute bars stored under data/raw/polygon/index/ (or the
      legacy per-symbol layout data/raw/polygon/<symbol>/YYYY-MM-DD.parquet).
Outputs:
    - data/proc/index_panel_polygon.parquet containing:
        timestamp (UTC), timestamp_et, symbol, price (= close),
        minutes_to_noon, minutes_to_close, day_of_week, realized_vol_30m.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl

from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.drivers.index_polygon import (
    DEFAULT_SYMBOLS,
    POLYGON_INDEX_RAW_ROOT,
    load_minutes,
)

DEFAULT_OUTPUT = PROC_ROOT / "index_panel_polygon.parquet"


def _add_time_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Add ET calendar/time-to-target features required by the model."""

    if frame.is_empty():
        return frame

    et_ts = pl.col("timestamp_et")
    noon = pl.datetime(
        et_ts.dt.year(),
        et_ts.dt.month(),
        et_ts.dt.day(),
        12,
        0,
        0,
        time_zone="America/New_York",
    )
    close = pl.datetime(
        et_ts.dt.year(),
        et_ts.dt.month(),
        et_ts.dt.day(),
        16,
        0,
        0,
        time_zone="America/New_York",
    )
    return frame.with_columns(
        [
            et_ts.dt.date().alias("trading_day"),
            et_ts.dt.weekday().alias("day_of_week"),
            (noon - et_ts).dt.total_minutes().cast(pl.Int32).alias("minutes_to_noon"),
            (close - et_ts).dt.total_minutes().cast(pl.Int32).alias("minutes_to_close"),
            pl.col("close").alias("price"),
        ]
    )


def _add_realized_vol(frame: pl.DataFrame) -> pl.DataFrame:
    """Compute a simple rolling realized volatility per symbol and day."""

    if frame.is_empty():
        return frame

    ordered = frame.sort(["symbol", "trading_day", "timestamp_et"])
    ordered = ordered.with_columns(
        pl.col("close")
        .log()
        .diff()
        .over(["symbol", "trading_day"])
        .alias("log_ret")
    )

    def _rolling(expr: pl.Expr) -> pl.Expr:
        try:
            return expr.rolling_std(window_size=30, min_samples=5)
        except TypeError:
            return expr.rolling_std(window_size=30, min_periods=5)

    ordered = ordered.with_columns(
        _rolling(pl.col("log_ret"))
        .over(["symbol", "trading_day"])
        .alias("realized_vol_30m")
    )
    return ordered.drop("log_ret")


def build_panel(  # noqa: PLR0913
    *,
    symbols: list[str] | tuple[str, ...] = DEFAULT_SYMBOLS,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    input_root: Path | None = None,
    output_path: Path | None = DEFAULT_OUTPUT,
) -> pl.DataFrame:
    """Load raw Polygon minutes, add features, and persist a panel."""

    data = load_minutes(
        symbols,
        start_date=start_date,
        end_date=end_date,
        base_root=input_root,
        as_pandas=False,
    )
    if data.is_empty():
        raise RuntimeError("No Polygon index rows loaded for requested inputs")
    panel = _add_time_features(data)
    panel = _add_realized_vol(panel)
    panel = panel.sort(["symbol", "timestamp_et"])
    if output_path is not None:
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(output_path, compression="zstd")
    return panel


def _parse_date(value: str | None) -> date | None:
    return None if value is None else date.fromisoformat(value)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        help="Optional inclusive start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        help="Optional inclusive end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Polygon tickers to include (default: I:SPX I:NDX).",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=POLYGON_INDEX_RAW_ROOT,
        help="Root directory containing per-symbol parquet files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output parquet path (default: data/proc/index_panel_polygon.parquet).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    symbols = tuple(args.symbols) if args.symbols else DEFAULT_SYMBOLS
    panel = build_panel(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        input_root=args.input_root,
        output_path=args.output,
    )
    print(
        f"Wrote panel with {panel.height} rows, "
        f"{panel.width} columns to {Path(args.output).resolve()}"
    )


if __name__ == "__main__":
    main()
