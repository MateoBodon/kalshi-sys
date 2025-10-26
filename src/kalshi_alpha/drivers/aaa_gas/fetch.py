"""AAA gasoline national average fetcher."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import requests

from kalshi_alpha.datastore import snapshots
from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.utils.http import fetch_with_cache

AAA_URL = "https://gasprices.aaa.com/"
DAILY_PATH = PROC_ROOT / "aaa_daily.parquet"
MONTHLY_PATH = PROC_ROOT / "aaa_monthly.parquet"


@dataclass(frozen=True)
class AAANationalPrice:
    as_of_date: date
    price: float
    fetched_ts: datetime


def fetch_latest(
    *,
    offline: bool = False,
    fixtures_dir: Path | None = None,
    force_refresh: bool = False,
    session: requests.Session | None = None,
) -> AAANationalPrice:
    if offline:
        if fixtures_dir is None:
            raise RuntimeError("fixtures_dir required for offline mode")
        html = (fixtures_dir / "aaa_national.html").read_text(encoding="utf-8")
    else:
        content = fetch_with_cache(
            AAA_URL,
            cache_path=DAILY_PATH.parent / "_cache_aaa.html",
            session=session,
            force_refresh=force_refresh,
        )
        html = content.decode("utf-8")
        snapshots.write_text_snapshot("aaa", "national_raw.html", html)
    as_of, price = _parse_price_html(html)
    fetched_ts = datetime.now(UTC)
    snapshots.write_json_snapshot(
        "aaa",
        "national_price",
        {"as_of_date": as_of.isoformat(), "price": price, "fetched_ts": fetched_ts.isoformat()},
    )
    _update_daily_parquet(as_of, price)
    return AAANationalPrice(as_of_date=as_of, price=price, fetched_ts=fetched_ts)


def _parse_price_html(html: str) -> tuple[date, float]:
    price_match = re.search(r"\$?(\d\.\d{2})", html)
    if not price_match:
        raise ValueError("Unable to locate AAA national price in HTML.")
    price = float(price_match.group(1))
    date_match = re.search(r"Price as of\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", html)
    if date_match:
        as_of = datetime.strptime(date_match.group(1), "%B %d, %Y").replace(tzinfo=UTC).date()
    else:
        as_of = datetime.now(UTC).date()
    return as_of, price


def _update_daily_parquet(as_of: date, price: float) -> None:
    new_row = pl.DataFrame({"date": [as_of], "price": [price]}).with_columns(
        pl.col("date").cast(pl.Date)
    )
    if DAILY_PATH.exists():
        existing = pl.read_parquet(DAILY_PATH)
        if as_of in existing["date"].to_list():
            return
        frame = pl.concat([existing, new_row]).sort("date")
    else:
        frame = new_row
    frame.write_parquet(DAILY_PATH)
    _update_monthly(frame)


def _update_monthly(frame: pl.DataFrame) -> None:
    monthly = (
        frame.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(pl.col("price").mean().alias("avg_price"))
        .sort("month")
    )
    monthly.write_parquet(MONTHLY_PATH)


def mtd_average(reference: date | None = None) -> float | None:
    if not DAILY_PATH.exists():
        return None
    frame = pl.read_parquet(DAILY_PATH)
    raw_dates = frame["date"].to_list()
    if not raw_dates:
        return None
    dates: list[date] = []
    for value in raw_dates:
        if isinstance(value, date):
            dates.append(value)
        elif isinstance(value, datetime):
            dates.append(value.date())
        else:
            continue
    if not dates:
        return None
    reference_date = reference or dates[-1]
    mask = frame["date"].dt.month() == reference_date.month
    mask &= frame["date"].dt.year() == reference_date.year
    subset = frame.filter(mask)
    if subset.is_empty():
        return None
    avg = subset.select(pl.col("price").mean().alias("avg"))["avg"][0]
    return float(avg) if avg is not None else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch AAA national gasoline price.")
    parser.add_argument("--offline", action="store_true", help="Use offline fixture.")
    parser.add_argument("--fixtures", type=Path, help="Path to fixtures directory when offline.")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached HTTP metadata.")
    args = parser.parse_args(argv)
    result = fetch_latest(
        offline=args.offline,
        fixtures_dir=args.fixtures,
        force_refresh=args.force_refresh,
    )
    print(json.dumps(result.__dict__, default=str))


if __name__ == "__main__":
    main()
