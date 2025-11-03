"""Snapshot helpers for Polygon index data."""

from __future__ import annotations

import math
from collections.abc import Sequence

from kalshi_alpha.datastore import snapshots as datastore_snapshots

from .client import IndexSnapshot, MinuteBar

NAMESPACE = "polygon_index"


def write_minute_bars(symbol: str, bars: Sequence[MinuteBar]) -> None:
    payload = [
        {
            "symbol": symbol,
            "timestamp": bar.timestamp.isoformat(),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "vwap": bar.vwap,
            "trades": bar.trades,
        }
        for bar in bars
    ]
    datastore_snapshots.write_json_snapshot(NAMESPACE, f"{symbol}_minute.json", payload)


def write_snapshot(snapshot: IndexSnapshot) -> None:
    datastore_snapshots.write_json_snapshot(
        NAMESPACE,
        f"{snapshot.ticker}_snapshot.json",
        {
            "ticker": snapshot.ticker,
            "last_price": snapshot.last_price,
            "change": snapshot.change,
            "change_percent": snapshot.change_percent,
            "previous_close": snapshot.previous_close,
            "timestamp": snapshot.timestamp.isoformat() if snapshot.timestamp else None,
        },
    )


def distance_to_strikes(price: float | None, strikes: Sequence[float]) -> dict[float, float]:
    if price is None:
        return {float(strike): float("nan") for strike in strikes}
    return {float(strike): float(price) - float(strike) for strike in strikes}


def ewma_sigma_now(
    bars: Sequence[MinuteBar],
    *,
    span: int = 30,
    min_samples: int = 5,
) -> float:
    if span <= 0:
        raise ValueError("span must be positive")
    closes = [float(bar.close) for bar in bars if float(bar.close) > 0]
    if len(closes) < max(min_samples, 2):
        return 0.0
    returns: list[float] = []
    for previous, current in zip(closes, closes[1:], strict=False):
        if previous <= 0:
            continue
        returns.append(math.log(current / previous))
    if not returns:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    mean = returns[0]
    variance = 0.0
    for ret in returns[1:]:
        mean = alpha * ret + (1 - alpha) * mean
        variance = (1 - alpha) * variance + alpha * (ret - mean) ** 2
    sigma = math.sqrt(max(variance, 0.0))
    last_price = closes[-1]
    return float(sigma * last_price)


def micro_drift(bars: Sequence[MinuteBar], *, window: int = 5) -> float:
    if window <= 0:
        raise ValueError("window must be positive")
    if len(bars) < 2:
        return 0.0
    recent = list(bars)[- (window + 1) :]
    if len(recent) < 2:
        recent = list(bars)
    deltas = [float(cur.close) - float(prev.close) for prev, cur in zip(recent, recent[1:], strict=False)]
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


def build_snapshot_metrics(
    *,
    price: float | None,
    strikes: Sequence[float],
    bars: Sequence[MinuteBar],
    ewma_span: int = 30,
    drift_window: int = 5,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "sigma_now": ewma_sigma_now(bars, span=ewma_span),
        "micro_drift": micro_drift(bars, window=drift_window),
    }
    if strikes:
        metrics["distance_to_strike"] = distance_to_strikes(price, strikes)
    return metrics


__all__ = [
    "write_minute_bars",
    "write_snapshot",
    "distance_to_strikes",
    "ewma_sigma_now",
    "micro_drift",
    "build_snapshot_metrics",
]
