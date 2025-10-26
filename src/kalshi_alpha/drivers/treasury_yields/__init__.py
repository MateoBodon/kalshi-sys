"""Treasury par yield driver with offline support."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import requests

from kalshi_alpha.datastore import snapshots
from kalshi_alpha.datastore.paths import RAW_ROOT
from kalshi_alpha.utils.http import fetch_with_cache

TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv"
)
CACHE_PATH = RAW_ROOT / "_cache" / "treasury" / "daily_yields.csv"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ParYield:
    as_of: datetime
    maturity: str
    rate: float


def fetch_daily_yields(
    *,
    offline: bool = False,
    fixtures_dir: Path | None = None,
    force_refresh: bool = False,
    session: requests.Session | None = None,
) -> list[ParYield]:
    """Fetch daily par yields."""
    if offline:
        if fixtures_dir is None:
            raise RuntimeError("fixtures_dir required for offline mode")
        content = (fixtures_dir / "treasury_par_yields.csv").read_text(encoding="utf-8")
    else:
        content_bytes = fetch_with_cache(
            TREASURY_URL,
            cache_path=CACHE_PATH,
            session=session,
            force_refresh=force_refresh,
        )
        content = content_bytes.decode("utf-8")
        snapshots.write_text_snapshot("treasury_yields", "daily.csv", content)

    yields = _parse_treasury_csv(content)
    snapshots.write_json_snapshot(
        "treasury_yields",
        "latest",
        [
            {
                "as_of": entry.as_of.date().isoformat(),
                "maturity": entry.maturity,
                "rate": entry.rate,
            }
            for entry in yields
        ],
    )
    return yields


def _parse_treasury_csv(csv_text: str) -> list[ParYield]:
    reader = csv.DictReader(io.StringIO(csv_text))
    results: list[ParYield] = []
    for row in reader:
        if "Date" not in row:
            continue
        try:
            as_of = datetime.strptime(row["Date"], "%m/%d/%Y").replace(tzinfo=UTC)
        except ValueError:
            continue
        for maturity, value in row.items():
            if maturity == "Date" or not value:
                continue
            try:
                rate = float(value)
            except ValueError:
                continue
            results.append(ParYield(as_of=as_of, maturity=maturity, rate=rate))
    return results


def dgs10_latest_rate(yields: Iterable[ParYield]) -> float | None:
    for entry in yields:
        if entry.maturity.upper() in {"DGS10", "10 YR"}:
            return entry.rate
    return None


def yields_to_frame(yields: Iterable[ParYield]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "as_of": [entry.as_of.date() for entry in yields],
            "maturity": [entry.maturity for entry in yields],
            "rate": [entry.rate for entry in yields],
        }
    )
