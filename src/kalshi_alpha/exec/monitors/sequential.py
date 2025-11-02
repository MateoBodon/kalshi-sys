"""Sequential change-detection guardrails for EV deltas."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl

DEFAULT_SEQ_THRESHOLD = 10.0  # bps
DEFAULT_SEQ_DRIFT = 1.0  # bps
DEFAULT_SEQ_MIN_SAMPLE = 12


@dataclass(slots=True)
class SequentialGuardParams:
    threshold: float = DEFAULT_SEQ_THRESHOLD
    drift: float = DEFAULT_SEQ_DRIFT
    min_sample: int = DEFAULT_SEQ_MIN_SAMPLE


@dataclass(slots=True)
class SequentialGuardResult:
    alert: bool
    triggers: list[dict[str, Any]]
    max_stat: float
    series_stats: dict[str, dict[str, Any]]


def evaluate_sequential_guard(
    ledger: pl.DataFrame,
    *,
    params: SequentialGuardParams,
    window_start: datetime | None = None,
) -> SequentialGuardResult:
    """Evaluate a CuSum-style sequential guard on Δbps streams."""

    if ledger.is_empty():
        return SequentialGuardResult(False, [], 0.0, {})

    filtered = ledger
    if window_start is not None and "timestamp_et" in ledger.columns:
        filtered = ledger.filter(pl.col("timestamp_et") >= window_start)
    if filtered.is_empty():
        return SequentialGuardResult(False, [], 0.0, {})

    frame = filtered.with_columns(
        (pl.col("ev_realized_bps") - pl.col("ev_expected_bps")).alias("delta_bps")
    ).drop_nulls("delta_bps")
    if frame.is_empty():
        return SequentialGuardResult(False, [], 0.0, {})

    series_names = (
        frame.select(pl.col("series").str.to_uppercase().alias("series"))
        .unique()
        .to_series()
        .to_list()
    )

    triggers: list[dict[str, Any]] = []
    max_stat = 0.0
    series_stats: dict[str, dict[str, Any]] = {}

    for series in series_names:
        series_frame = frame.filter(pl.col("series").str.to_uppercase() == series)
        if "timestamp_et" in series_frame.columns:
            series_frame = series_frame.sort("timestamp_et")

        deltas = series_frame["delta_bps"].to_list()
        timestamps = (
            series_frame["timestamp_et"].to_list()
            if "timestamp_et" in series_frame.columns
            else [None] * len(deltas)
        )
        sample_size = len(deltas)
        stats: dict[str, Any] = {
            "sample_size": sample_size,
        }
        if sample_size == 0:
            series_stats[series] = stats
            continue

        pos_sum = 0.0
        neg_sum = 0.0
        max_pos = 0.0
        min_neg = 0.0
        max_pos_ts: datetime | None = None
        min_neg_ts: datetime | None = None
        max_pos_idx = -1
        min_neg_idx = -1

        for idx, (delta, ts) in enumerate(zip(deltas, timestamps, strict=False)):
            pos_sum = max(0.0, pos_sum + delta - params.drift)
            if pos_sum > max_pos:
                max_pos = pos_sum
                max_pos_ts = ts
                max_pos_idx = idx
            neg_sum = min(0.0, neg_sum + delta + params.drift)
            if neg_sum < min_neg:
                min_neg = neg_sum
                min_neg_ts = ts
                min_neg_idx = idx

        stats.update(
            {
                "max_positive": max_pos,
                "max_negative": min_neg,
            }
        )
        series_stats[series] = stats
        max_stat = max(max_stat, max_pos, abs(min_neg))

        if sample_size < params.min_sample:
            stats["insufficient"] = True
            continue

        def _serialize_timestamp(ts: datetime | None) -> str | None:
            if ts is None:
                return None
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    return ts.replace(tzinfo=UTC).isoformat()
                return ts.astimezone(UTC).isoformat()
            return str(ts)

        if max_pos > params.threshold:
            triggers.append(
                {
                    "series": series,
                    "direction": "positive",
                    "stat": max_pos,
                    "threshold": params.threshold,
                    "index": max_pos_idx,
                    "timestamp": _serialize_timestamp(max_pos_ts),
                }
            )
        if abs(min_neg) > params.threshold:
            triggers.append(
                {
                    "series": series,
                    "direction": "negative",
                    "stat": abs(min_neg),
                    "threshold": params.threshold,
                    "index": min_neg_idx,
                    "timestamp": _serialize_timestamp(min_neg_ts),
                }
            )

    alert = bool(triggers)
    return SequentialGuardResult(alert, triggers, max_stat, series_stats)
