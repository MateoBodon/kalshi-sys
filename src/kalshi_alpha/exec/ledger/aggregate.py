"""Aggregate paper ledger CSV outputs into a single Parquet dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from kalshi_alpha.exec.ledger.schema import LedgerRowV1

DEFAULT_REPORTS_DIR = Path("reports/_artifacts")
DEFAULT_OUTPUT_PATH = Path("data/proc/ledger_all.parquet")
COLUMN_DEFAULTS: dict[str, float | int | str] = {
    "t_fill_ms": 0.0,
    "size_partial": 0,
    "slippage_ticks": 0.0,
    "ev_expected_bps": 0.0,
    "ev_realized_bps": 0.0,
    "fees_bps": 0.0,
    "fill_ratio_observed": 0.0,
    "alpha_target": 0.0,
    "visible_depth": 0.0,
    "side_depth_total": 0.0,
    "depth_fraction": 0.0,
    "best_bid": 0.0,
    "best_ask": 0.0,
    "spread": 0.0,
    "seconds_to_event": 0.0,
    "minutes_to_event": 0.0,
}
SUPPORTED_VERSIONS = {1, 2}
FLOAT_COLUMNS = {
    "bin",
    "price",
    "model_p",
    "market_p",
    "delta_p",
    "fill_ratio",
    "fill_ratio_observed",
    "alpha_target",
    "visible_depth",
    "side_depth_total",
    "depth_fraction",
    "best_bid",
    "best_ask",
    "spread",
    "seconds_to_event",
    "minutes_to_event",
    "t_fill_ms",
    "slippage_ticks",
    "ev_expected_bps",
    "ev_realized_bps",
    "fees_bps",
    "impact_cap",
    "fees_maker",
    "ev_after_fees",
    "pnl_simulated",
}
INT_COLUMNS = {
    "size",
    "expected_contracts",
    "expected_fills",
    "size_partial",
    "ledger_schema_version",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ledger CSV files into Parquet.")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory containing *_ledger.csv files (default: reports/_artifacts).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination Parquet file (default: data/proc/ledger_all.parquet).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    csv_files = sorted(_discover_ledger_csv(args.reports_dir))
    if not csv_files:
        _write_empty_output(args.output)
        print(
            "[aggregate] No ledger CSV files found in "
            f"{args.reports_dir}; wrote empty dataset to {args.output}."
        )
        return

    frames: list[pl.DataFrame] = []
    expected_columns = list(LedgerRowV1.canonical_fields())
    for csv_path in csv_files:
        frame = pl.read_csv(csv_path, try_parse_dates=False)
        if "ledger_schema_version" in frame.columns:
            frame = frame.with_columns(pl.col("ledger_schema_version").cast(pl.Int64, strict=False))
        missing = [col for col in expected_columns if col not in frame.columns]
        if missing:
            for column in missing:
                if column not in COLUMN_DEFAULTS:
                    raise ValueError(f"Ledger file {csv_path} missing columns: {missing}")
                default_value = COLUMN_DEFAULTS[column]
                frame = frame.with_columns(pl.lit(default_value).alias(column))
        extra = [col for col in frame.columns if col not in expected_columns]
        if extra:
            raise ValueError(f"Ledger file {csv_path} contains unknown columns: {extra}")
        versions = {
            int(value)
            for value in frame["ledger_schema_version"].to_list()
            if value is not None
        }
        if not versions:
            raise ValueError(f"Ledger file {csv_path} has missing schema version")
        unsupported = versions - SUPPORTED_VERSIONS
        if unsupported:
            raise ValueError(
                f"Ledger file {csv_path} contains unsupported schema versions: "
                f"{sorted(unsupported)}"
            )
        if 1 in versions:
            frame = frame.with_columns(pl.lit(2).alias("ledger_schema_version"))
        float_casts = [pl.col(col).cast(pl.Float64, strict=False).alias(col) for col in FLOAT_COLUMNS if col in frame.columns]
        int_casts = [pl.col(col).cast(pl.Int64, strict=False).alias(col) for col in INT_COLUMNS if col in frame.columns]
        if float_casts or int_casts:
            frame = frame.with_columns([*float_casts, *int_casts])
        frames.append(frame.select(expected_columns))

    combined = pl.concat(frames, how="vertical")
    combined = combined.sort("timestamp_et") if "timestamp_et" in combined.columns else combined
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(args.output)
    print(f"[aggregate] Wrote {combined.height} rows to {args.output}")


def _discover_ledger_csv(reports_dir: Path) -> list[Path]:
    if not reports_dir.exists():
        return []
    return [
        path
        for path in reports_dir.rglob("*_ledger.csv")
        if path.is_file()
    ]


def _write_empty_output(output_path: Path) -> None:
    expected_columns = LedgerRowV1.canonical_fields()
    schema = {column: pl.Utf8 for column in expected_columns}
    # refine known numeric types
    for column in FLOAT_COLUMNS:
        schema[column] = pl.Float64
    for column in INT_COLUMNS:
        schema[column] = pl.Int64
    empty = pl.DataFrame(schema=schema)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    empty.write_parquet(output_path)


if __name__ == "__main__":
    main()
