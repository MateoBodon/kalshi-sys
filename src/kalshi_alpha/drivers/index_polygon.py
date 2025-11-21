"""Polygon index history loader for offline modelling/backtests.

This module reads minute-level Polygon index aggregates that have already been
persisted to disk (typically under ``data/raw/polygon/index/``) and normalizes
timestamps to US/Eastern. The helper functions are intentionally light-weight
and avoid any live Polygon API calls; the expectation is that another ingestion
job fetches and stores the raw parquet files.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

import polars as pl

from kalshi_alpha.datastore.paths import RAW_ROOT

ET = ZoneInfo("America/New_York")

# Default on-disk layout for Polygon index aggregates.
POLYGON_INDEX_RAW_ROOT = (RAW_ROOT / "polygon" / "index").resolve()
# Legacy path (per-symbol directories directly under data/raw/polygon/).
POLYGON_LEGACY_RAW_ROOT = (RAW_ROOT / "polygon").resolve()

DEFAULT_SYMBOLS: tuple[str, ...] = ("I:SPX", "I:NDX")


def _sanitize_symbol(symbol: str) -> str:
    """Convert a Polygon ticker into a filesystem-friendly slug."""

    cleaned = symbol.strip().upper()
    return cleaned.replace(":", "_")


def _normalize_symbol_label(symbol: str) -> str:
    """Return a colon-delimited Polygon ticker for downstream use."""

    cleaned = symbol.strip().upper()
    if ":" in cleaned:
        return cleaned
    if cleaned.startswith("I_"):
        return cleaned.replace("_", ":", 1)
    return cleaned.replace("_", ":")


def _parse_date(value: date | str | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _resolve_symbol_dir(symbol: str, base_root: Path | None = None) -> Path:
    """Pick the first existing directory for the provided symbol."""

    sanitized = _sanitize_symbol(symbol)
    roots = (
        (Path(base_root) if base_root else POLYGON_INDEX_RAW_ROOT),
        POLYGON_LEGACY_RAW_ROOT,
    )
    candidates = []
    for root in roots:
        if not root:
            continue
        candidates.append(root / sanitized)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No Polygon index data directory found for {symbol!r}; "
        f"tried {', '.join(str(path) for path in candidates)}"
    )


def _iter_symbol_files(
    symbol_dir: Path,
    *,
    start_date: date | None,
    end_date: date | None,
) -> list[Path]:
    files: list[Path] = []
    for path in sorted(symbol_dir.glob("*.parquet")):
        stem = path.stem.split(".")[0]
        try:
            file_date = date.fromisoformat(stem)
        except ValueError:
            continue
        if start_date and file_date < start_date:
            continue
        if end_date and file_date > end_date:
            continue
        files.append(path)
    return files


def load_symbol_minutes(  # noqa: PLR0913
    symbol: str,
    *,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    base_root: Path | None = None,
    as_pandas: bool = False,
) -> pl.DataFrame:
    """Load minute-level Polygon aggregates for a single index symbol.

    Args:
        symbol: Polygon ticker (e.g., ``I:SPX`` or ``I_SPX``).
        start_date: Optional inclusive start date (YYYY-MM-DD).
        end_date: Optional inclusive end date (YYYY-MM-DD).
        base_root: Optional override for the raw datastore root.
        as_pandas: When True, return a pandas DataFrame instead of Polars.

    Returns:
        Polars (or pandas) DataFrame with UTC ``timestamp``, ``timestamp_et``,
        OHLCV columns, and a ``symbol`` column.
    """

    start = _parse_date(start_date)
    end = _parse_date(end_date)
    symbol_dir = _resolve_symbol_dir(symbol, base_root=base_root)
    files = _iter_symbol_files(symbol_dir, start_date=start, end_date=end)
    if not files:
        return pl.DataFrame()

    frames: list[pl.DataFrame] = []
    for path in files:
        frame = pl.read_parquet(path)
        if frame.is_empty():
            continue
        frames.append(frame)
    if not frames:
        return pl.DataFrame()

    data = pl.concat(frames, how="vertical").sort("timestamp")
    data = data.with_columns(
        [
            pl.col("timestamp").dt.convert_time_zone("UTC"),
            pl.col("timestamp").dt.convert_time_zone("America/New_York").alias("timestamp_et"),
            pl.lit(_normalize_symbol_label(symbol)).alias("symbol"),
        ]
    )
    if as_pandas:
        return data.to_pandas()  # type: ignore[return-value]
    return data


def load_minutes(  # noqa: PLR0913
    symbols: Iterable[str] | None = None,
    *,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    base_root: Path | None = None,
    as_pandas: bool = False,
) -> pl.DataFrame:
    """Load and concatenate Polygon minute bars for multiple symbols."""

    selected = tuple(symbols) if symbols is not None else DEFAULT_SYMBOLS
    frames: list[pl.DataFrame] = []
    for symbol in selected:
        frame = load_symbol_minutes(
            symbol,
            start_date=start_date,
            end_date=end_date,
            base_root=base_root,
            as_pandas=False,
        )
        if frame.is_empty():
            continue
        frames.append(frame)
    if not frames:
        return pl.DataFrame()
    combined = pl.concat(frames, how="vertical").sort(["symbol", "timestamp"])
    combined = combined.unique(subset=["symbol", "timestamp"], keep="first")
    if as_pandas:
        return combined.to_pandas()  # type: ignore[return-value]
    return combined


__all__ = [
    "DEFAULT_SYMBOLS",
    "POLYGON_INDEX_RAW_ROOT",
    "POLYGON_LEGACY_RAW_ROOT",
    "load_minutes",
    "load_symbol_minutes",
]
