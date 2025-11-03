"""Snapshot helpers for Polygon index data."""

from __future__ import annotations

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


__all__ = ["write_minute_bars", "write_snapshot"]
