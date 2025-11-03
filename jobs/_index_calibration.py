"""Shared helpers for index calibration jobs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import polars as pl

from kalshi_alpha.drivers.polygon_index.client import MinuteBar

ET = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


def build_sigma_curve(
    bars_by_symbol: Mapping[str, Sequence[MinuteBar]],
    *,
    target_time: time,
    residual_window: int,
) -> pl.DataFrame:
    records = []
    for symbol, bars in bars_by_symbol.items():
        records.extend(_collect_symbol_records(symbol, bars, target_time))
    if not records:
        raise ValueError("No minute bars available for calibration")

    frame = pl.DataFrame(records)
    grouped = (
        frame.group_by("symbol", "minutes_to_target")
        .agg(
            pl.col("delta").mean().alias("drift"),
            pl.col("delta").std().alias("sigma"),
        )
        .with_columns(pl.col("sigma").fill_null(0.0))
    )
    residuals = (
        frame.filter(pl.col("minutes_to_target") <= residual_window)
        .group_by("symbol")
        .agg(pl.col("delta").std().alias("residual_std"))
        .with_columns(pl.col("residual_std").fill_null(0.0))
    )
    joined = grouped.join(residuals, on="symbol", how="left")
    return (
        joined.fill_null(0.0)
        .with_columns(pl.col("minutes_to_target").cast(pl.Int64))
        .sort(["symbol", "minutes_to_target"])
    )


def _collect_symbol_records(
    symbol: str,
    bars: Sequence[MinuteBar],
    target_time: time,
) -> list[dict[str, object]]:
    per_day: defaultdict[date, list[tuple[datetime, MinuteBar]]] = defaultdict(list)
    for bar in bars:
        ts_et = bar.timestamp.astimezone(ET)
        if ts_et.time() < MARKET_OPEN or ts_et.time() > MARKET_CLOSE:
            continue
        per_day[ts_et.date()].append((ts_et, bar))

    records: list[dict[str, object]] = []
    for day, entries in per_day.items():
        entries.sort(key=lambda item: item[0])
        target_dt = datetime.combine(day, target_time, tzinfo=ET)
        target_price = _resolve_target_price(entries, target_dt)
        if target_price is None:
            continue
        for ts, bar in entries:
            if ts > target_dt:
                break
            delta_minutes = int((target_dt - ts).total_seconds() // 60)
            if delta_minutes < 0:
                continue
            records.append(
                {
                    "symbol": symbol,
                    "date": day.isoformat(),
                    "minutes_to_target": delta_minutes,
                    "delta": float(target_price - bar.close),
                }
            )
    return records


def _resolve_target_price(
    entries: Sequence[tuple[datetime, MinuteBar]],
    target_dt: datetime,
) -> float | None:
    target_price: float | None = None
    for ts, bar in entries:
        if ts <= target_dt:
            target_price = float(bar.close)
        else:
            break
    return target_price


__all__ = ["build_sigma_curve", "ET", "MARKET_OPEN", "MARKET_CLOSE"]
