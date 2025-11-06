"""Evaluate close index calibrations against historical dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from kalshi_alpha.backtest.generate_dataset import DEFAULT_OUTPUT as DEFAULT_DATASET_PATH
from kalshi_alpha.backtest.scoring import evaluate_backtest

DEFAULT_OUTPUT_DIR = Path("reports/backtests/close")
POLYGON_TO_SERIES = {
    "I:SPX": "INX",
    "I:NDX": "NASDAQ100",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score close index calibrations.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Backtest dataset parquet path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics.md and ev_table.csv",
    )
    parser.add_argument(
        "--contracts",
        type=int,
        default=100,
        help="Contract size used when computing EV after fees",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    evaluate_backtest(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        horizon="close",
        polygon_to_series=POLYGON_TO_SERIES,
        contracts=int(args.contracts),
    )


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    main()
