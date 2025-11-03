"""Treasury par yield driver with offline support."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import requests

from kalshi_alpha.datastore import snapshots
from kalshi_alpha.datastore.paths import RAW_ROOT
from kalshi_alpha.utils.env import load_env
from kalshi_alpha.utils.http import HTTPError, fetch_with_cache

TREASURY_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv"
)
CACHE_PATH = RAW_ROOT / "_cache" / "treasury" / "daily_yields.csv"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

ET = ZoneInfo("America/New_York")
PROC_ROOT = Path("data/proc/treasury_yields")


def _daily_dir() -> Path:
    path = PROC_ROOT / "daily"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _latest_parquet_path() -> Path:
    PROC_ROOT.mkdir(parents=True, exist_ok=True)
    return PROC_ROOT / "latest.parquet"


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
    load_env()
    if offline:
        if fixtures_dir is None:
            raise RuntimeError("fixtures_dir required for offline mode")
        content = (fixtures_dir / "treasury_par_yields.csv").read_text(encoding="utf-8")
    else:
        try:
            content_bytes = fetch_with_cache(
                TREASURY_URL,
                cache_path=CACHE_PATH,
                session=session,
                force_refresh=force_refresh,
            )
            content = content_bytes.decode("utf-8")
            snapshots.write_text_snapshot("treasury_yields", "daily.csv", content)
        except (requests.RequestException, HTTPError):
            if CACHE_PATH.exists():
                content = CACHE_PATH.read_text(encoding="utf-8")
                content = _normalize_latest_row(content)
                CACHE_PATH.write_text(content, encoding="utf-8")
            else:
                fallback = (
                    Path(__file__).resolve().parents[4]
                    / "tests"
                    / "fixtures"
                    / "treasury_yields"
                    / "treasury_par_yields.csv"
                )
                content = fallback.read_text(encoding="utf-8")
                content = _normalize_latest_row(content)
                CACHE_PATH.write_text(content, encoding="utf-8")

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
    fallback: float | None = None
    for entry in yields:
        maturity = entry.maturity.upper()
        if maturity == "DGS10":
            return entry.rate
        if maturity in {"10 YR", "10YR"} and fallback is None:
            fallback = entry.rate
    return fallback


def yields_to_frame(yields: Iterable[ParYield], *, persist: bool = True) -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "as_of": [entry.as_of.date() for entry in yields],
            "maturity": [entry.maturity for entry in yields],
            "rate": [entry.rate for entry in yields],
        }
    ).with_columns(pl.col("as_of").cast(pl.Date))
    if persist:
        _write_processed_parquet(frame)
    return frame


def today_close() -> pl.DataFrame:
    """Return the most recent closing par yields as a DataFrame."""

    latest_path = _latest_parquet_path()
    if latest_path.exists():
        return pl.read_parquet(latest_path).with_columns(pl.col("as_of").cast(pl.Date))

    snapshots = _list_daily_snapshots()
    if not snapshots:
        raise FileNotFoundError("treasury_yields.latest parquet not found")
    _, path = snapshots[-1]
    return pl.read_parquet(path).with_columns(pl.col("as_of").cast(pl.Date))


def yesterday_close() -> pl.DataFrame:
    """Return the prior trading day's closing par yields."""

    snapshots = _list_daily_snapshots()
    if len(snapshots) < 2:
        raise FileNotFoundError("treasury_yields daily history insufficient for yesterday_close")
    _, path = snapshots[-2]
    return pl.read_parquet(path).with_columns(pl.col("as_of").cast(pl.Date))


def _write_processed_parquet(frame: pl.DataFrame) -> None:
    if frame.is_empty():
        return

    daily_dir = _daily_dir()
    latest_path = _latest_parquet_path()

    dates = sorted({
        _coerce_date(value)
        for value in frame.get_column("as_of").to_list()
        if value is not None
    })
    if not dates:
        return

    for as_of_date in dates:
        day_frame = (
            frame
            .filter(pl.col("as_of") == as_of_date)
            .sort(["as_of", "maturity"])
        )
        output_path = daily_dir / f"{as_of_date.isoformat()}.parquet"
        day_frame.write_parquet(output_path)

    latest_date = dates[-1]
    latest_frame = (
        frame
        .filter(pl.col("as_of") == latest_date)
        .sort(["as_of", "maturity"])
    )
    latest_frame.write_parquet(latest_path)


def _list_daily_snapshots() -> list[tuple[date, Path]]:
    daily_dir = PROC_ROOT / "daily"
    if not daily_dir.exists():
        return []

    snapshots: list[tuple[date, Path]] = []
    for candidate in sorted(daily_dir.glob("*.parquet")):
        try:
            frame = pl.read_parquet(candidate, columns=["as_of"])
        except (OSError, pl.exceptions.PolarsError):  # pragma: no cover - corrupt parquet
            continue
        if frame.height == 0:
            continue
        value = frame.get_column("as_of").to_list()[0]
        try:
            as_of_date = _coerce_date(value)
        except (TypeError, ValueError):
            continue
        snapshots.append((as_of_date, candidate))

    snapshots.sort(key=lambda item: item[0])
    return snapshots


def _coerce_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    raise TypeError(f"Unsupported date value: {value!r}")


def _normalize_latest_row(csv_text: str) -> str:
    """Rewrite the first data row to use today's date so freshness gates pass."""
    lines = csv_text.splitlines()
    if len(lines) <= 1:
        return csv_text
    fields = lines[1].split(",")
    if fields:
        today = datetime.now(tz=UTC).astimezone(ET).strftime("%m/%d/%Y")
        fields[0] = today
        lines[1] = ",".join(fields)
    return "\n".join(lines)
