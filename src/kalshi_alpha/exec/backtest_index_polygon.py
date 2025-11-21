"""CLI entrypoint for Polygon-only index ladder backtests."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from kalshi_alpha.drivers.kalshi_index_history import DEFAULT_QUOTES_ROOT
from kalshi_alpha.strategies.index.backtest_polygon import (
    BacktestConfig,
    run_backtest,
    summarize,
    write_report,
    write_trades_csv,
)
from kalshi_alpha.strategies.index.model_polygon import PARAM_ROOT

DEFAULT_PANEL = Path("data/proc/index_panel_polygon.parquet")
DEFAULT_TRADES_DIR = Path("data/proc/backtest/index_polygon")
DEFAULT_REPORT_DIR = Path("reports/index_backtests")
ALL_SERIES = ("INX", "INXU", "NASDAQ100", "NASDAQ100U")


def _parse_date(value: str | None) -> date | None:
    return None if value is None else date.fromisoformat(value)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        nargs="+",
        choices=ALL_SERIES + ("all",),
        default=("all",),
        help="Series to backtest (default: all index series).",
    )
    parser.add_argument("--start-date", type=_parse_date, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=_parse_date, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--panel",
        type=Path,
        default=DEFAULT_PANEL,
        help="Path to index_panel_polygon.parquet.",
    )
    parser.add_argument(
        "--params-root",
        type=Path,
        default=PARAM_ROOT,
        help="Root directory containing fitted params.json files.",
    )
    parser.add_argument(
        "--ev-threshold-cents",
        type=float,
        default=2.0,
        help="Maker EV threshold in cents per contract (default: 2).",
    )
    parser.add_argument(
        "--maker-edge-cents",
        type=float,
        default=2.0,
        help="Edge subtracted from model probability when quoting as maker.",
    )
    parser.add_argument(
        "--max-bins-per-market",
        type=int,
        default=2,
        help="Maximum number of ladder bins to trade per market (default: 2).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_TRADES_DIR,
        help="Directory to write trades CSVs into.",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory to write Markdown summaries into.",
    )
    parser.add_argument(
        "--use-kalshi-quotes",
        action="store_true",
        help="Prefer recorded Kalshi ladder quotes for strike grids and spreads.",
    )
    parser.add_argument(
        "--quotes-dir",
        type=Path,
        default=DEFAULT_QUOTES_ROOT,
        help="Root directory containing historical Kalshi quotes (default: data/raw/kalshi/index_quotes).",
    )
    parser.add_argument(
        "--disable-fill-model",
        action="store_true",
        help="Skip probabilistic fills (assume deterministic maker fills).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    chosen_series = tuple(ALL_SERIES) if "all" in args.series else tuple(args.series)
    panel_path = args.panel.resolve()
    panel = pl.read_parquet(panel_path)
    if panel.is_empty():
        raise SystemExit(f"No rows found in panel {panel_path}")

    config = BacktestConfig(
        series=chosen_series,
        start_date=args.start_date,
        end_date=args.end_date,
        ev_threshold=float(args.ev_threshold_cents) / 100.0,
        max_bins=int(args.max_bins_per_market),
        maker_edge=float(args.maker_edge_cents) / 100.0,
        params_root=args.params_root,
        panel_path=panel_path,
        use_kalshi_quotes=bool(args.use_kalshi_quotes),
        quotes_root=args.quotes_dir,
        use_fill_model=not args.disable_fill_model,
    )
    trades = run_backtest(panel, config)
    summary = summarize(trades)

    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root.resolve()
    report_dir = args.report_root.resolve()
    trades_path = output_dir / f"trades_{stamp}.csv"
    report_path = report_dir / f"summary_{stamp}.md"
    write_trades_csv(trades, trades_path)
    write_report(summary, report_path)

    print(f"Wrote {len(trades)} trades to {trades_path}")
    print(f"Summary -> {report_path}")


if __name__ == "__main__":
    main()
