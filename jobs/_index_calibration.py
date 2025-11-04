"""Shared helpers for index calibration jobs."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from kalshi_alpha.drivers.calendar.loader import calendar_tags_for
from kalshi_alpha.drivers.polygon_index.client import MinuteBar

ET = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
DEFAULT_EVENT_TAGS: tuple[str, ...] = ("CPI", "FOMC")
DEFAULT_LATE_DAY_CLAMP = 1.25


def build_sigma_curve(
    bars_by_symbol: Mapping[str, Sequence[MinuteBar]],
    *,
    target_time: time,
    residual_window: int,
    return_records: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, pl.DataFrame]:
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
    aggregated = (
        joined.fill_null(0.0)
        .with_columns(pl.col("minutes_to_target").cast(pl.Int64))
        .sort(["symbol", "minutes_to_target"])
    )
    if return_records:
        return aggregated, frame
    return aggregated


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


def _with_event_flag(records: pl.DataFrame, tags: Sequence[str] | None = None) -> pl.DataFrame:
    if "is_event" in records.columns:
        return records
    chosen_tags = tags if tags is not None else DEFAULT_EVENT_TAGS
    tag_set = {str(tag).strip().upper() for tag in chosen_tags if str(tag).strip()}

    def _has_tag(value: str | None) -> bool:
        if not value:
            return False
        try:
            day = date.fromisoformat(str(value))
        except ValueError:
            return False
        event_tags = calendar_tags_for(day)
        return any(tag in tag_set for tag in event_tags)

    return records.with_columns(
        pl.col("date")
        .map_elements(_has_tag, return_dtype=pl.Boolean)
        .alias("is_event")
    )


def _std(series: pl.DataFrame) -> float | None:
    if series.is_empty():
        return None
    value = series.select(pl.col("delta").var(ddof=0)).item()
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric) or numeric <= 0.0:
        return None
    return math.sqrt(numeric)


def _variance(series: pl.DataFrame) -> float | None:
    if series.is_empty():
        return None
    value = series.select(pl.col("delta").var(ddof=0)).item()
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric) or numeric < 0.0:
        return None
    return numeric


def _baseline_variance(frame: pl.DataFrame, symbol: str, window: int) -> float | None:
    subset = frame.filter(
        (pl.col("symbol") == symbol)
        & (pl.col("minutes_to_target") <= window)
    )
    if subset.is_empty():
        return None
    value = subset.select(pl.col("sigma").pow(2).mean()).item()
    if value is None:
        return None
    numeric = float(value)
    if math.isnan(numeric) or numeric <= 0.0:
        return None
    return numeric


def event_tail_multiplier(
    records: pl.DataFrame,
    symbol: str,
    *,
    window: int,
    tags: Sequence[str] | None = None,
    clamp: tuple[float, float] = (1.0, 1.8),
) -> float:
    subset = records.filter(pl.col("symbol") == symbol)
    if subset.is_empty():
        return 1.0
    tagged = _with_event_flag(subset, tags)
    mask = (pl.col("minutes_to_target") > 0) & (pl.col("minutes_to_target") <= window)
    event_std = _std(tagged.filter(mask & pl.col("is_event")))
    if event_std is None:
        return 1.0
    base_std = _std(tagged.filter(mask & ~pl.col("is_event")))
    if base_std is None or base_std <= 0.0:
        base_std = _std(tagged.filter(mask))
    if base_std is None or base_std <= 0.0:
        return 1.0
    ratio = event_std / base_std
    lower, upper = clamp
    lower = max(lower, 1.0)
    upper = max(upper, lower)
    return float(max(lower, min(ratio, upper)))


def late_day_lambda(
    records: pl.DataFrame,
    frame: pl.DataFrame,
    symbol: str,
    *,
    window: int,
    tags: Sequence[str] | None = None,
) -> dict[str, float] | None:
    subset = records.filter(pl.col("symbol") == symbol)
    if subset.is_empty():
        return None
    tagged = _with_event_flag(subset, tags)
    mask = (pl.col("minutes_to_target") > 0) & (pl.col("minutes_to_target") <= window)
    event_var = _variance(tagged.filter(mask & pl.col("is_event")))
    baseline_var = _variance(tagged.filter(mask & ~pl.col("is_event")))
    combined_var = _variance(tagged.filter(mask))
    if event_var is None:
        event_var = combined_var
    if baseline_var is None:
        baseline_var = combined_var
    if event_var is None or baseline_var is None or baseline_var <= 0.0:
        return None
    delta_var = event_var - baseline_var
    if delta_var <= 0.0:
        return None
    baseline = _baseline_variance(frame, symbol, window)
    reference = baseline if baseline is not None else baseline_var
    if reference <= 0.0:
        reference = baseline_var
    max_lambda = reference * DEFAULT_LATE_DAY_CLAMP
    lambda_value = min(delta_var, max_lambda)
    if lambda_value <= 1e-6:
        return None
    return {
        "minutes_threshold": int(window),
        "lambda": float(lambda_value),
    }


__all__ = [
    "build_sigma_curve",
    "event_tail_multiplier",
    "late_day_lambda",
    "extend_calibration_window",
    "ET",
    "MARKET_OPEN",
    "MARKET_CLOSE",
]


def extend_calibration_window(  # noqa: PLR0913 - explicit knobs for ops tuning
    start_date: date,
    end_date: date,
    *,
    tags: Sequence[str] | None = None,
    event_pad_days: int = 2,
    dst_pad_days: int = 3,
    lookback_days: int = 90,
) -> tuple[date, date]:
    """Expand the trailing calibration window to cover DST and macro event weeks."""

    normalized_tags = tuple(tags) if tags is not None else DEFAULT_EVENT_TAGS
    expanded_start = _extend_for_events(start_date, end_date, normalized_tags, event_pad_days, lookback_days)
    expanded_start = _extend_for_dst(expanded_start, end_date, dst_pad_days)
    return expanded_start, end_date


def _extend_for_events(
    start_date: date,
    end_date: date,
    tags: Sequence[str],
    pad_days: int,
    lookback_days: int,
) -> date:
    updated = start_date
    for tag in tags:
        event_date = _last_event_before(end_date, tag, lookback_days)
        if event_date is None:
            continue
        candidate_start = event_date - timedelta(days=max(pad_days, 0))
        updated = min(updated, candidate_start)
    return updated


def _last_event_before(end_date: date, tag: str, lookback_days: int) -> date | None:
    normalized = tag.strip().upper()
    if not normalized:
        return None
    for offset in range(max(lookback_days, 0) + 1):
        day = end_date - timedelta(days=offset)
        if normalized in calendar_tags_for(day):
            return day
    return None


def _extend_for_dst(start_date: date, end_date: date, pad_days: int) -> date:
    transition = _last_dst_transition(end_date)
    if transition is None:
        return start_date
    boundary = transition - timedelta(days=max(pad_days, 0))
    return boundary if boundary < start_date else start_date


def _last_dst_transition(end_date: date, *, lookback_days: int = 365) -> date | None:
    zone = ET
    previous_offset = None
    previous_day: date | None = None
    for offset in range(max(lookback_days, 0) + 1):
        day = end_date - timedelta(days=offset)
        noon = datetime.combine(day, time(12, 0), tzinfo=zone)
        current_offset = noon.utcoffset()
        if previous_offset is not None and current_offset != previous_offset:
            return previous_day or day
        previous_offset = current_offset
        previous_day = day
    return None
