"""Load historical Kalshi index ladder quotes from disk for offline backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import polars as pl

from kalshi_alpha.datastore.paths import RAW_ROOT

ET = ZoneInfo("America/New_York")
DEFAULT_QUOTES_ROOT = RAW_ROOT / "kalshi" / "index_quotes"
SUPPORTED_SUFFIXES: tuple[str, ...] = (".parquet", ".csv", ".json")


@dataclass(frozen=True)
class QuoteSnapshot:
    """Latest ladder snapshot for a given day/horizon."""

    series: str
    trading_day: date
    horizon: str
    as_of: datetime
    quotes: pl.DataFrame

    @property
    def strikes(self) -> list[float]:
        return sorted({float(v) for v in self.quotes.get_column("strike").to_list()})


def _candidate_paths(series: str, trading_day: date, root: Path | None) -> Iterable[Path]:
    base = (root or DEFAULT_QUOTES_ROOT) / series.upper()
    date_str = trading_day.isoformat()
    for suffix in SUPPORTED_SUFFIXES:
        yield base / f"{date_str}{suffix}"
    # Accept slightly more permissive file names for ad-hoc exports.
    for suffix in SUPPORTED_SUFFIXES:
        yield base / f"{series.lower()}_{date_str}{suffix}"


def _load_frame(path: Path) -> pl.DataFrame:
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.suffix == ".csv":
        return pl.read_csv(path)
    if path.suffix == ".json":
        return pl.read_json(path)
    raise ValueError(f"Unsupported extension: {path.suffix}")


def _ensure_et_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ET)
    return dt.astimezone(ET)


def _normalize_columns(frame: pl.DataFrame, series: str, trading_day: date) -> pl.DataFrame:
    columns = {col.lower(): col for col in frame.columns}
    rename_map: dict[str, str] = {}
    if "timestamp_et" not in columns and "timestamp" in columns:
        rename_map[columns["timestamp"]] = "timestamp_et"
    if "horizon" not in columns and "window" in columns:
        rename_map[columns["window"]] = "horizon"
    for canonical in ("strike", "bid", "ask", "timestamp_et", "horizon"):
        if canonical in columns and columns[canonical] != canonical:
            rename_map[columns[canonical]] = canonical
    normalized = frame.rename(rename_map) if rename_map else frame

    required = {"timestamp_et", "strike", "bid", "ask"}
    missing = required.difference({c.lower() for c in normalized.columns})
    if missing:
        raise ValueError(f"Missing required columns {missing} in {normalized.columns}")

    normalized = normalized.with_columns(
        [
            pl.lit(series.upper()).alias("series"),
            pl.lit(trading_day).alias("trading_day"),
        ]
    )
    if "horizon" not in normalized.columns:
        normalized = normalized.with_columns(pl.lit("close").alias("horizon"))

    normalized = normalized.select(
        "series",
        "trading_day",
        pl.col("horizon").str.to_lowercase().alias("horizon"),
        pl.col("timestamp_et"),
        pl.col("strike").cast(pl.Float64),
        pl.col("bid").cast(pl.Float64),
        pl.col("ask").cast(pl.Float64),
    )
    timestamps = [_ensure_et_timestamp(value) for value in normalized.get_column("timestamp_et").to_list()]
    normalized = normalized.with_columns(
        pl.Series("timestamp_et", timestamps, dtype=pl.Datetime(time_zone="UTC"))
    )
    normalized = normalized.with_columns(pl.col("timestamp_et").dt.convert_time_zone("America/New_York"))
    normalized = normalized.with_columns(
        [
            ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
            ((pl.col("ask") - pl.col("bid")) * 100).alias("spread_cents"),
        ]
    )
    normalized = normalized.sort(["timestamp_et", "strike"])
    return normalized


def load_quotes_for_day(series: str, trading_day: date, *, root: Path | None = None) -> pl.DataFrame:
    """Load all quotes for a trading day from disk, normalizing columns."""
    for candidate in _candidate_paths(series, trading_day, root):
        if candidate.exists():
            frame = _load_frame(candidate)
            return _normalize_columns(frame, series, trading_day)
    raise FileNotFoundError(f"No quote file found for {series} {trading_day} under {root or DEFAULT_QUOTES_ROOT}")


def latest_snapshot(
    series: str,
    trading_day: date,
    horizon: str,
    *,
    as_of: datetime | None = None,
    root: Path | None = None,
) -> QuoteSnapshot | None:
    """Return the last snapshot at or before `as_of` (or the day’s last) for the horizon."""
    horizon = horizon.lower()
    quotes = load_quotes_for_day(series, trading_day, root=root)
    quotes = quotes.filter(pl.col("horizon") == horizon)
    if quotes.is_empty():
        return None

    if as_of is not None:
        as_of_et = as_of.astimezone(ET)
        quotes = quotes.filter(pl.col("timestamp_et") <= as_of_et)
        if quotes.is_empty():
            return None

    latest_ts = quotes.select(pl.max("timestamp_et")).item()
    snapshot = quotes.filter(pl.col("timestamp_et") == latest_ts).sort("strike")
    return QuoteSnapshot(
        series=series.upper(),
        trading_day=trading_day,
        horizon=horizon,
        as_of=_ensure_et_timestamp(latest_ts),
        quotes=snapshot,
    )


__all__ = ["QuoteSnapshot", "load_quotes_for_day", "latest_snapshot", "DEFAULT_QUOTES_ROOT"]
