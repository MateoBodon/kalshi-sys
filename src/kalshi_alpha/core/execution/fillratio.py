"""Estimate expected fills using simple visible-depth heuristics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from kalshi_alpha.core.execution.series_utils import canonical_series_family
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
    archives_dir: Path | str | None = None,
    *,
    lookback_days: int = 14,
    min_observations: int = 30,
    persist: bool = True,
) -> float | None:
    """Estimate a fill-allocation alpha from aggregated ledger outcomes."""

    _ = archives_dir  # retained for backwards compatibility with legacy callers
    family = canonical_series_family(series)
    ledger = _load_ledger_frame()
    if ledger.is_empty() or "size" not in ledger.columns:
        return None

    cutoff = datetime.now(tz=UTC) - timedelta(days=max(lookback_days, 1))
    frame = ledger
    if "timestamp_et" in frame.columns:
        timestamp_expr = pl.col("timestamp_et")
        if frame["timestamp_et"].dtype == pl.Utf8:
            frame = frame.with_columns(timestamp_expr.str.strptime(pl.Datetime, strict=False))
            timestamp_expr = pl.col("timestamp_et")
        frame = frame.filter(timestamp_expr >= cutoff)

    frame = frame.with_columns(
        pl.col("series")
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.to_uppercase()
        .map_elements(canonical_series_family)
        .alias("series_family"),
        pl.col("size").cast(pl.Float64, strict=False).alias("size_float"),
        pl.col("size_partial")
        .fill_null(0)
        .cast(pl.Float64, strict=False)
        .alias("size_partial_float"),
    )
    filtered = frame.filter(
        (pl.col("series_family") == family)
        & (pl.col("size_float") > 0)
    )
    if filtered.is_empty():
        return None

    filtered = filtered.with_columns(
        (pl.col("size_float") - pl.col("size_partial_float"))
        .clip(lower_bound=0.0)
        .alias("filled_contracts"),
    )
    filtered = filtered.filter(pl.col("filled_contracts") > 0)
    if filtered.is_empty():
        return None

    sample_size = filtered.height
    if sample_size < max(min_observations, 1):
        return None

    total_requested = float(filtered["size_float"].sum())
    if total_requested <= 0:
        return None

    total_filled = float(filtered["filled_contracts"].sum())
    alpha = max(0.0, min(total_filled / total_requested, 1.0))
    alpha = max(0.2, min(alpha, 0.98))
    alpha = round(alpha, 4)
    if persist:
        _persist_alpha(family, alpha, sample_size)
    return alpha


def load_alpha(series: str) -> float | None:
    """Return the persisted alpha value for the given series family."""

    family = canonical_series_family(series)
    if not STATE_PATH.exists():
        return None
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    series_map = payload.get("series", {})
    if not isinstance(series_map, dict):
        return None
    entry = series_map.get(family)
    if entry is None:
        return None
    if isinstance(entry, dict):
        value = entry.get("alpha")
    else:
        value = entry
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _load_ledger_frame() -> pl.DataFrame:
    if not LEDGER_PATH.exists():
        return pl.DataFrame()
    frame = pl.read_parquet(LEDGER_PATH)
    if "timestamp_et" in frame.columns and frame["timestamp_et"].dtype == pl.Utf8:
        frame = frame.with_columns(pl.col("timestamp_et").str.strptime(pl.Datetime, strict=False))
    return frame


def _persist_alpha(series: str, alpha: float, sample_size: int | None = None) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if STATE_PATH.exists():
        try:
            payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    series_map = payload.setdefault("series", {})
    series_payload: dict[str, object] = {
        "alpha": round(alpha, 4),
        "ts": datetime.now(tz=UTC).isoformat(),
    }
    if sample_size is not None:
        series_payload["sample_size"] = int(sample_size)
    series_map[series] = series_payload
    payload["updated"] = datetime.now(tz=UTC).isoformat()
    STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
