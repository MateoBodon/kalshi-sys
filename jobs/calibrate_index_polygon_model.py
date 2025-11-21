"""Fit simple Polygon-only index distributions and write params.json artifacts."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl

from kalshi_alpha.drivers.polygon_index.symbols import resolve_series
from kalshi_alpha.strategies.index.model_polygon import PARAM_ROOT, fit_from_panel, params_path, save_params

DEFAULT_PANEL = Path("data/proc/index_panel_polygon.parquet")
ALL_SERIES = ("INX", "INXU", "NASDAQ100", "NASDAQ100U")


def _parse_date(value: str | None) -> date | None:
    return None if value is None else date.fromisoformat(value)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=DEFAULT_PANEL,
        help="Path to index_panel_polygon.parquet.",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        choices=ALL_SERIES + ("all",),
        default=("all",),
        help="Series to fit (default: all index series).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PARAM_ROOT,
        help="Root directory for params.json outputs.",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        help="Optional inclusive start date when trimming the panel.",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        help="Optional inclusive end date when trimming the panel.",
    )
    parser.add_argument(
        "--horizon",
        choices=("noon", "close"),
        help="Override horizon for all series (otherwise inferred: U->noon, others->close).",
    )
    return parser.parse_args(argv)


def _horizon_for_series(series: str, override: str | None) -> str:
    if override:
        return override
    return "noon" if series.upper().endswith("U") else "close"


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    series_list = tuple(ALL_SERIES) if "all" in args.series else tuple(args.series)
    panel_path = args.panel.resolve()
    panel = pl.read_parquet(panel_path)
    if panel.is_empty():
        raise SystemExit(f"No rows found in panel {panel_path}")
    if args.start_date:
        panel = panel.filter(pl.col("trading_day") >= args.start_date)
    if args.end_date:
        panel = panel.filter(pl.col("trading_day") <= args.end_date)

    for series in series_list:
        meta = resolve_series(series)
        symbol = meta.polygon_ticker.upper()
        horizon = _horizon_for_series(series, args.horizon)
        filtered = panel.filter(pl.col("symbol") == symbol)
        if filtered.is_empty():
            print(f"[skip] {series}: no rows for symbol {symbol} in panel slice")
            continue
        params = fit_from_panel(filtered, horizon=horizon, symbols=[symbol])
        output_path = params_path(series, horizon, root=args.output_root)
        save_params(params, output_path)
        print(f"[ok] {series} {horizon}: wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
