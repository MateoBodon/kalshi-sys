"""AAA gasoline bootstrap ingestion."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from kalshi_alpha.datastore.paths import BOOTSTRAP_ROOT, PROC_ROOT


def bootstrap_from_csv(path: Path) -> dict[str, Path]:
    """Create daily and monthly Parquet files from historical AAA CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Bootstrap CSV not found: {path}")
    frame = pl.read_csv(path, try_parse_dates=True)
    columns = [col.lower() for col in frame.columns]
    frame = frame.rename(dict(zip(frame.columns, columns, strict=True)))
    if "price" not in frame.columns:
        if "regular_gas_price" in frame.columns:
            frame = frame.rename({"regular_gas_price": "price"})
        elif "regular" in frame.columns:
            frame = frame.rename({"regular": "price"})
    if "date" not in frame.columns or "price" not in frame.columns:
        raise ValueError("CSV must contain 'date' and 'price' columns")
    frame = frame.with_columns(pl.col("date").cast(pl.Date))
    frame = frame.with_columns(pl.col("price").cast(pl.Float64))
    frame = frame.sort("date")
    if frame["date"].is_duplicated().any():
        raise ValueError("Duplicate dates detected in AAA bootstrap CSV")
    if not frame["date"].is_sorted():
        raise ValueError("Dates must be monotonic ascending in AAA bootstrap CSV")

    daily_path = PROC_ROOT / "aaa_daily.parquet"
    monthly_path = PROC_ROOT / "aaa_monthly.parquet"

    frame.write_parquet(daily_path)
    monthly = (
        frame.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(pl.col("price").mean().alias("avg_price"))
        .sort("month")
    )
    monthly.write_parquet(monthly_path)
    return {"daily": daily_path, "monthly": monthly_path}


def _ensure_bootstrap_dir(csv_path: Path) -> Path:
    BOOTSTRAP_ROOT.mkdir(parents=True, exist_ok=True)
    target = BOOTSTRAP_ROOT / csv_path.name
    if not target.exists():
        target.write_bytes(csv_path.read_bytes())
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bootstrap AAA daily gas price parquet files.")
    parser.add_argument("--bootstrap", required=True, type=Path, help="Path to AAA bootstrap CSV.")
    args = parser.parse_args(argv)
    csv_path = _ensure_bootstrap_dir(args.bootstrap)
    paths = bootstrap_from_csv(csv_path)
    print(f"Wrote daily data to {paths['daily']}")
    print(f"Wrote monthly data to {paths['monthly']}")


if __name__ == "__main__":
    main()
