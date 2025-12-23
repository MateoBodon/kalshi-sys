"""Bounded top-of-book snapshot + quote-intent logger for index ladders."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from kalshi_alpha.core.kalshi_api import Orderbook
from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.exec.telemetry.sink import (
    DEFAULT_TELEMETRY_MAX_WINDOW_BYTES,
    TelemetryJsonlSink,
)

DEFAULT_TOB_DIR = PROC_ROOT / "telemetry"
DEFAULT_TOB_DEPTH = 3
MAX_TOB_DEPTH = 5
DEFAULT_TOB_MAX_BYTES = 10 * 1024
DEFAULT_INTENT_MAX_BYTES = 2 * 1024
DEFAULT_TOB_WINDOW_MAX_BYTES = DEFAULT_TELEMETRY_MAX_WINDOW_BYTES


@dataclass(slots=True)
class TobSnapshotLogger:
    run_id: str
    output_dir: Path
    depth: int = DEFAULT_TOB_DEPTH
    max_bytes: int = DEFAULT_TOB_MAX_BYTES
    max_intent_bytes: int = DEFAULT_INTENT_MAX_BYTES
    max_window_bytes: int = DEFAULT_TOB_WINDOW_MAX_BYTES
    max_snapshots: int | None = None
    compress: bool = True
    _snapshot_sink: TelemetryJsonlSink = field(init=False)
    _intent_sink: TelemetryJsonlSink = field(init=False)
    _count: int = field(init=False, default=0)
    _last_snapshot: dict[str, dict[str, Any]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        depth = int(self.depth) if self.depth is not None else DEFAULT_TOB_DEPTH
        self.depth = max(1, min(depth, MAX_TOB_DEPTH))
        max_bytes = int(self.max_bytes) if self.max_bytes is not None else DEFAULT_TOB_MAX_BYTES
        self.max_bytes = max(256, max_bytes)
        max_intent_bytes = (
            int(self.max_intent_bytes) if self.max_intent_bytes is not None else DEFAULT_INTENT_MAX_BYTES
        )
        self.max_intent_bytes = max(256, max_intent_bytes)
        max_window_bytes = (
            int(self.max_window_bytes) if self.max_window_bytes is not None else DEFAULT_TOB_WINDOW_MAX_BYTES
        )
        self.max_window_bytes = max(0, max_window_bytes)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_sink = TelemetryJsonlSink(
            run_id=self.run_id,
            stream="tob",
            base_dir=self.output_dir,
            compress=self.compress,
            max_bytes_per_window=self.max_window_bytes,
        )
        self._intent_sink = TelemetryJsonlSink(
            run_id=self.run_id,
            stream="quote_intents",
            base_dir=self.output_dir,
            compress=self.compress,
            max_bytes_per_window=self.max_window_bytes,
        )

    def log_snapshot(
        self,
        *,
        ts_utc: datetime,
        series: str,
        window_label: str | None,
        window_ts_utc: str | None,
        window_ts_et: str | None,
        market_ticker: str,
        market_id: str | None,
        orderbook: Orderbook | None,
    ) -> bool:
        if self.max_snapshots is not None and self._count >= self.max_snapshots:
            return False
        snapshot = build_tob_snapshot(
            run_id=self.run_id,
            ts_utc=ts_utc,
            series=series,
            window_label=window_label,
            window_ts_utc=window_ts_utc,
            window_ts_et=window_ts_et,
            market_ticker=market_ticker,
            market_id=market_id,
            orderbook=orderbook,
            depth=self.depth,
            max_bytes=self.max_bytes,
        )
        if snapshot is None:
            return False
        if not self._snapshot_sink.emit(snapshot):
            return False
        self._count += 1
        key = str(market_ticker).upper()
        self._last_snapshot[key] = {
            "ts": snapshot.get("ts"),
            "bid_price": snapshot.get("bid_price"),
            "ask_price": snapshot.get("ask_price"),
            "bid_size": snapshot.get("bid_size"),
            "ask_size": snapshot.get("ask_size"),
        }
        return True

    def log_quote_intent(
        self,
        *,
        ts_utc: datetime,
        series: str,
        window_label: str | None,
        window_ts_utc: str | None,
        window_ts_et: str | None,
        market_ticker: str,
        market_id: str | None,
        side: str,
        price: float,
        size: int,
    ) -> bool:
        key = str(market_ticker).upper()
        reference = self._last_snapshot.get(key, {})
        record = build_quote_intent(
            run_id=self.run_id,
            ts_utc=ts_utc,
            series=series,
            window_label=window_label,
            window_ts_utc=window_ts_utc,
            window_ts_et=window_ts_et,
            market_ticker=market_ticker,
            market_id=market_id,
            side=side,
            price=price,
            size=size,
            tob_ts=reference.get("ts"),
            max_bytes=self.max_intent_bytes,
        )
        if record is None:
            return False
        return self._intent_sink.emit(record)


def build_tob_snapshot(
    *,
    run_id: str,
    ts_utc: datetime,
    series: str,
    window_label: str | None,
    window_ts_utc: str | None,
    window_ts_et: str | None,
    market_ticker: str,
    market_id: str | None,
    orderbook: Orderbook | None,
    depth: int = DEFAULT_TOB_DEPTH,
    max_bytes: int = DEFAULT_TOB_MAX_BYTES,
) -> dict[str, Any] | None:
    levels, best_bid, best_ask = _top_levels(orderbook, depth=depth)
    bid_price, bid_size = best_bid
    ask_price, ask_size = best_ask
    ts_iso = _as_iso(ts_utc)
    window_id = _window_id(series, window_label, window_ts_utc, ts_iso)
    snapshot: dict[str, Any] = {
        "record_type": "tob_snapshot",
        "run_id": run_id,
        "window_id": window_id,
        "ts": ts_iso,
        "ts_utc": ts_iso,
        "series": series,
        "window_label": window_label,
        "window_ts_utc": window_ts_utc,
        "window_ts_et": window_ts_et,
        "market_ticker": market_ticker,
        "market_id": market_id,
        "bid_price": bid_price,
        "bid_size": bid_size,
        "ask_price": ask_price,
        "ask_size": ask_size,
        "best_bid_price": bid_price,
        "best_bid_size": bid_size,
        "best_ask_price": ask_price,
        "best_ask_size": ask_size,
    }
    if levels:
        snapshot["top_levels"] = levels
    bounded = enforce_snapshot_bounds(snapshot, max_bytes=max_bytes)
    return bounded


def build_quote_intent(
    *,
    run_id: str,
    ts_utc: datetime,
    series: str,
    window_label: str | None,
    window_ts_utc: str | None,
    window_ts_et: str | None,
    market_ticker: str,
    market_id: str | None,
    side: str,
    price: float,
    size: int,
    tob_ts: str | None = None,
    max_bytes: int = DEFAULT_INTENT_MAX_BYTES,
) -> dict[str, Any] | None:
    ts_iso = _as_iso(ts_utc)
    window_id = _window_id(series, window_label, window_ts_utc, ts_iso)
    record = {
        "record_type": "quote_intent",
        "run_id": run_id,
        "window_id": window_id,
        "ts": ts_iso,
        "ts_utc": ts_iso,
        "series": series,
        "window_label": window_label,
        "window_ts_utc": window_ts_utc,
        "window_ts_et": window_ts_et,
        "market_ticker": market_ticker,
        "market_id": market_id,
        "quote_side": side,
        "quote_price": float(price),
        "quote_size": int(size),
        "placement_type": "maker_only",
        "tob_ts": tob_ts,
    }
    max_bytes = max(256, int(max_bytes))
    if _estimate_size(record) <= max_bytes:
        return record
    return None


def enforce_snapshot_bounds(
    snapshot: Mapping[str, Any],
    *,
    max_bytes: int = DEFAULT_TOB_MAX_BYTES,
) -> dict[str, Any] | None:
    max_bytes = max(256, int(max_bytes))
    record = dict(snapshot)
    if _estimate_size(record) <= max_bytes:
        return record

    levels = record.get("top_levels")
    if isinstance(levels, Sequence):
        trimmed = _trim_levels(levels)
        if trimmed is not None:
            record["top_levels"] = trimmed
            record["truncated"] = True
            if _estimate_size(record) <= max_bytes:
                return record
        record.pop("top_levels", None)
        record["truncated"] = True
        if _estimate_size(record) <= max_bytes:
            return record

    # If still too large, fail closed by dropping the snapshot.
    return None


def _trim_levels(levels: Sequence[object]) -> list[dict[str, Any]] | None:
    bids: list[dict[str, Any]] = []
    asks: list[dict[str, Any]] = []
    for entry in levels:
        if not isinstance(entry, Mapping):
            continue
        side = str(entry.get("side") or "").lower()
        if side == "bid":
            bids.append(dict(entry))
        elif side == "ask":
            asks.append(dict(entry))
    if not bids and not asks:
        return None
    depth = min(MAX_TOB_DEPTH, max(len(bids), len(asks), 1))
    for new_depth in range(depth, 0, -1):
        trimmed = bids[:new_depth] + asks[:new_depth]
        if trimmed:
            return trimmed
    return None


def _window_id(
    series: str,
    window_label: str | None,
    window_ts_utc: str | None,
    fallback_ts: str,
) -> str:
    series_tag = str(series or "").upper() or "UNKNOWN"
    label = str(window_label or "window").replace(" ", "_")
    ts_value = window_ts_utc or fallback_ts or "unknown"
    return f"{series_tag}:{label}@{ts_value}"


def _top_levels(
    orderbook: Orderbook | None,
    *,
    depth: int,
) -> tuple[list[dict[str, Any]], tuple[float | None, float | None], tuple[float | None, float | None]]:
    if orderbook is None:
        return [], (None, None), (None, None)
    depth = max(1, min(int(depth), MAX_TOB_DEPTH))
    bid_levels = _sorted_levels(orderbook.bids, descending=True)
    ask_levels = _sorted_levels(orderbook.asks, descending=False)
    levels: list[dict[str, Any]] = []
    for price, size in bid_levels[:depth]:
        levels.append({"side": "bid", "price": price, "size": size})
    for price, size in ask_levels[:depth]:
        levels.append({"side": "ask", "price": price, "size": size})
    best_bid = bid_levels[0] if bid_levels else (None, None)
    best_ask = ask_levels[0] if ask_levels else (None, None)
    return levels, best_bid, best_ask


def _sorted_levels(entries: Sequence[object], *, descending: bool) -> list[tuple[float, float]]:
    levels: list[tuple[float, float]] = []
    for entry in entries:
        price: float | None
        size: float | None
        if isinstance(entry, Mapping):
            price = _safe_float(entry.get("price"))
            size = _safe_float(entry.get("size"))
        elif isinstance(entry, Sequence) and len(entry) >= 2 and not isinstance(entry, (str, bytes, bytearray)):
            price = _safe_float(entry[0])
            size = _safe_float(entry[1])
        else:
            price = _safe_float(entry)
            size = None
        if price is None:
            continue
        levels.append((price, float(size or 0.0)))
    return sorted(levels, key=lambda item: item[0], reverse=descending)


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, str) and value.strip():
            return float(value)
    except (TypeError, ValueError):
        return None
    return None


def _estimate_size(payload: Mapping[str, Any]) -> int:
    return len(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _as_iso(moment: datetime) -> str:
    value = moment
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


__all__ = [
    "DEFAULT_TOB_DEPTH",
    "DEFAULT_TOB_DIR",
    "DEFAULT_TOB_MAX_BYTES",
    "DEFAULT_INTENT_MAX_BYTES",
    "DEFAULT_TOB_WINDOW_MAX_BYTES",
    "MAX_TOB_DEPTH",
    "TobSnapshotLogger",
    "build_tob_snapshot",
    "build_quote_intent",
    "enforce_snapshot_bounds",
]
