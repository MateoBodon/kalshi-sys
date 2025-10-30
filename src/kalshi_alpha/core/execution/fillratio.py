"""Estimate expected fills using simple visible-depth heuristics."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

import polars as pl

from kalshi_alpha.core.kalshi_api import Orderbook

LEDGER_PATH = Path("data/proc/ledger_all.parquet")
STATE_PATH = Path("data/proc/state/fill_alpha.json")



def alpha_row(depth: float, size: int | float, alpha_base: float) -> float:
    """Row-level fill alpha adjusted by visible depth."""

    if size <= 0 or alpha_base <= 0:
        return 0.0
    depth = max(float(depth), 0.0)
    alpha_base = max(0.0, min(1.0, float(alpha_base)))
    ratio = depth / float(size)
    ratio = max(0.0, min(ratio, 1.0))
    return alpha_base * ratio


def expected_fills(size: int | float, visible_depth: float, alpha: float) -> tuple[int, float]:
    """Return expected filled contracts and fill ratio for requested size."""
    if size <= 0 or visible_depth <= 0 or alpha <= 0:
        return 0, 0.0
    size = float(size)
    visible = max(0.0, float(visible_depth))
    alpha = max(0.0, min(1.0, float(alpha)))
    estimate = min(size, alpha * visible)
    filled = max(0, min(int(math.floor(estimate)), int(size)))
    ratio = (filled / size) if size > 0 else 0.0
    return filled, ratio


def tune_alpha(
    series: str,
    archives_dir: Path | str,
    *,
    lookback_days: int = 14,
) -> float | None:
    """Estimate an alpha multiplier from archived orderbooks and ledger fills."""

    series = series.upper()
    ledger = _load_series_ledger(series)
    if ledger.is_empty():
        return None

    cutoff = datetime.now(tz=UTC) - timedelta(days=max(lookback_days, 1))
    if "timestamp_et" in ledger.columns:
        ledger = ledger.filter(pl.col("timestamp_et") >= cutoff)
    if ledger.is_empty():
        return None

    archives_base = Path(archives_dir)
    samples: list[float] = []
    orderbook_cache: dict[tuple[Path, str], Orderbook] = {}
    manifest_cache: dict[Path, dict] = {}

    for row in ledger.to_dicts():
        manifest_raw = row.get("manifest_path")
        market_id = row.get("market")
        price = float(row.get("price") or 0.0)
        side = str(row.get("side") or "YES")
        realized = float(row.get("expected_fills") or row.get("size") or 0.0)
        if not manifest_raw or not market_id or realized <= 0 or price <= 0:
            continue
        manifest_path = _resolve_manifest_path(str(manifest_raw), archives_base)
        if manifest_path is None:
            continue
        orderbook = _load_orderbook_from_manifest(
            manifest_path,
            str(market_id),
            manifest_cache,
            orderbook_cache,
        )
        if orderbook is None:
            continue
        visible = _visible_depth(side, price, orderbook)
        if visible <= 0:
            continue
        samples.append(min(1.0, realized / visible))

    if not samples:
        return None

    mean_alpha = statistics.mean(samples)
    tuned = max(0.4, min(0.8, mean_alpha))
    _persist_alpha(series, tuned)
    return tuned


@dataclass(frozen=True)
class FillRatioEstimator:
    alpha: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be within [0, 1]")

    def estimate(
        self,
        *,
        side: str,
        price: float,
        contracts: int,
        orderbook: Orderbook,
    ) -> tuple[int, float]:
        if contracts <= 0:
            return 0, 0.0
        visible = _visible_depth(side, price, orderbook)
        if visible <= 0:
            return 0, 0.0
        alpha_effective = alpha_row(visible, contracts, self.alpha)
        if alpha_effective <= 0:
            return 0, 0.0
        expected = min(contracts, int(math.floor(contracts * alpha_effective)))
        ratio = (expected / contracts) if contracts > 0 else 0.0
        return expected, ratio

    def expected_contracts(
        self,
        *,
        side: str,
        price: float,
        contracts: int,
        orderbook: Orderbook,
    ) -> int:
        expected, _ = self.estimate(
            side=side,
            price=price,
            contracts=contracts,
            orderbook=orderbook,
        )
        return expected


def _visible_depth(side: str, price: float, orderbook: Orderbook) -> float:
    entries = orderbook.asks if side.upper() == "YES" else orderbook.bids
    depth = 0.0
    for entry in entries:
        try:
            entry_price = float(entry.get("price", 0.0))
            entry_size = float(entry.get("size", 0.0))
        except (TypeError, ValueError):
            continue
        if abs(entry_price - price) <= 1e-6:
            depth += max(entry_size, 0.0)
    return depth


def _load_series_ledger(series: str) -> pl.DataFrame:
    if not LEDGER_PATH.exists():
        return pl.DataFrame()
    frame = pl.read_parquet(LEDGER_PATH)
    if "timestamp_et" in frame.columns and frame["timestamp_et"].dtype == pl.Utf8:
        frame = frame.with_columns(pl.col("timestamp_et").str.strptime(pl.Datetime, strict=False))
    return frame.filter(pl.col("series") == series)


def _resolve_manifest_path(raw: str, archives_base: Path) -> Path | None:
    path = Path(raw)
    if not path.is_absolute():
        path = archives_base / path
    if path.exists():
        return path
    return None


def _load_orderbook_from_manifest(
    manifest_path: Path,
    market_id: str,
    manifest_cache: dict[Path, dict],
    orderbook_cache: dict[tuple[Path, str], Orderbook],
) -> Orderbook | None:
    key = (manifest_path, market_id)
    if key in orderbook_cache:
        return orderbook_cache[key]
    manifest = manifest_cache.get(manifest_path)
    if manifest is None:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        manifest_cache[manifest_path] = manifest
    entries: Iterable[str] = manifest.get("paths", {}).get("orderbooks", [])
    target_entry = None
    for entry in entries:
        if Path(entry).stem == market_id:
            target_entry = entry
            break
    if target_entry is None:
        return None
    orderbook_path = manifest_path.parent / target_entry
    try:
        payload = json.loads(orderbook_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    orderbook = Orderbook.from_payload(payload)
    orderbook_cache[key] = orderbook
    return orderbook


def _persist_alpha(series: str, alpha: float) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if STATE_PATH.exists():
        try:
            payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    series_map = payload.setdefault("series", {})
    series_map[series] = {
        "alpha": round(alpha, 4),
        "ts": datetime.now(tz=UTC).isoformat(),
    }
    payload["updated"] = datetime.now(tz=UTC).isoformat()
    STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
