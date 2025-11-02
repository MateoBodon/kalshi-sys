"""Slippage modelling utilities for paper execution."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from kalshi_alpha.core.execution.series_utils import canonical_series_family
from kalshi_alpha.core.kalshi_api import Orderbook

LEDGER_PATH = Path("data/proc/ledger_all.parquet")
STATE_PATH = Path("data/proc/state/slippage.json")
TICK_SIZE = 0.01


@dataclass(frozen=True)
class SlippageModel:
    """Represents a piecewise linear slippage curve."""

    mode: str = "top"
    impact_cap: float = 0.02
    depth_curve: Sequence[tuple[float, float]] = field(
        default_factory=lambda: ((0.0, 0.0), (0.5, 0.0025), (1.0, 0.005))
    )

    def __post_init__(self) -> None:
        if self.mode not in {"top", "depth"}:
            raise ValueError("mode must be 'top' or 'depth'")
        if self.impact_cap <= 0:
            raise ValueError("impact_cap must be positive")
        points = sorted(self.depth_curve)
        object.__setattr__(self, "depth_curve", tuple(points))
        if not points or points[0][0] != 0.0:
            raise ValueError("depth_curve must start at depth 0.0")

    def depth_impact(self, depth_fraction: float) -> float:
        depth_fraction = max(0.0, min(depth_fraction, 1.0))
        points: Iterable[tuple[float, float]] = self.depth_curve
        last_depth, last_value = 0.0, 0.0
        for depth, value in points:
            if depth_fraction == depth:
                return min(value, self.impact_cap)
            if depth_fraction < depth:
                span = depth - last_depth
                if span <= 0:
                    return min(value, self.impact_cap)
                proportion = (depth_fraction - last_depth) / span
                interpolated = last_value + proportion * (value - last_value)
                return min(interpolated, self.impact_cap)
            last_depth, last_value = depth, value
        return min(last_value, self.impact_cap)


@dataclass(frozen=True)
class SlippageCalibration:
    """Persisted calibration describing a fitted depth-impact curve."""

    family: str
    impact_cap: float
    depth_curve: tuple[tuple[float, float], ...]
    sample_size: int
    updated: datetime

    def as_model(self, *, mode: str = "depth") -> SlippageModel:
        return SlippageModel(mode=mode, impact_cap=self.impact_cap, depth_curve=self.depth_curve)


def price_with_slippage(
    *,
    side: str,
    contracts: int,
    proposal_price: float,
    orderbook: Orderbook,
    model: SlippageModel,
) -> tuple[float, float]:
    """Return adjusted fill price and slippage (signed) given an orderbook."""
    top_price = _top_of_book(side, proposal_price, orderbook)
    if model.mode == "top":
        return top_price, 0.0

    book_levels = _levels_for_side(side, orderbook)
    total_depth = sum(float(level.get("size", 0.0)) for level in book_levels)
    if total_depth <= 0:
        return top_price, 0.0

    remaining = float(max(contracts, 1))
    consumed = 0.0
    weighted_price = 0.0
    for level in book_levels:
        level_size = float(level.get("size", 0.0))
        if level_size <= 0:
            continue
        level_price = float(level.get("price", top_price))
        take = min(level_size, remaining)
        weighted_price += level_price * take
        consumed += take
        remaining -= take
        if remaining <= 0:
            break

    if consumed <= 0:
        return top_price, 0.0

    depth_fraction = min(float(contracts) / total_depth, 1.0)
    curve_impact = model.depth_impact(depth_fraction)
    weighted_avg = weighted_price / consumed

    if side.upper() == "YES":
        raw_impact = max(weighted_avg - top_price, 0.0)
        impact = min(model.impact_cap, raw_impact + curve_impact)
        return min(1.0, top_price + impact), impact

    raw_impact = max(top_price - weighted_avg, 0.0)
    impact = min(model.impact_cap, raw_impact + curve_impact)
    return max(0.0, top_price - impact), -impact


def _top_of_book(side: str, proposal_price: float, orderbook: Orderbook) -> float:
    if side.upper() == "YES":
        if orderbook.asks:
            return float(orderbook.asks[0].get("price", proposal_price))
        return proposal_price
    if orderbook.bids:
        return float(orderbook.bids[0].get("price", proposal_price))
    return proposal_price


def _levels_for_side(side: str, orderbook: Orderbook) -> Sequence[dict[str, float]]:
    return orderbook.asks if side.upper() == "YES" else orderbook.bids


def fit_slippage(
    series: str,
    *,
    lookback_days: int = 14,
    min_observations: int = 30,
    persist: bool = True,
) -> SlippageCalibration | None:
    """Fit a simple depth-impact curve from realized ledger slippage."""

    family = canonical_series_family(series)
    ledger = _load_ledger_frame()
    if ledger.is_empty() or "slippage_ticks" not in ledger.columns:
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
        pl.col("slippage_ticks").cast(pl.Float64, strict=False).abs().alias("slippage_ticks_abs"),
        pl.col("size").cast(pl.Float64, strict=False).alias("size_float"),
        pl.col("delta_p").cast(pl.Float64, strict=False).abs().alias("imbalance"),
    )
    filtered = frame.filter(
        (pl.col("series_family") == family)
        & (pl.col("size_float") > 0)
        & pl.col("slippage_ticks_abs").is_not_null()
    )
    filtered = filtered.filter(pl.col("slippage_ticks_abs") > 0)
    if filtered.is_empty():
        return None

    sample_size = filtered.height
    if sample_size < max(min_observations, 1):
        return None

    max_size = float(filtered["size_float"].max())
    if max_size <= 0:
        return None

    filtered = filtered.with_columns(
        (pl.col("slippage_ticks_abs") * TICK_SIZE).alias("impact"),
        (pl.col("size_float") / max_size).clip(0.0, 1.0).alias("size_fraction"),
    )

    global_impact = _quantile(filtered, 0.85, fallback=0.001)
    low_subset = filtered.filter(
        (pl.col("size_fraction") <= 0.33)
        & (pl.col("imbalance") <= 0.05)
    )
    mid_subset = filtered.filter(
        (pl.col("size_fraction") > 0.33)
        & (pl.col("size_fraction") <= 0.66)
        & (pl.col("imbalance") <= 0.15)
    )
    high_subset = filtered.filter(
        (pl.col("size_fraction") > 0.66)
        | (pl.col("imbalance") > 0.15)
    )

    small_impact = max(0.0, _quantile(low_subset, 0.50, fallback=global_impact * 0.5))
    medium_impact = max(
        small_impact,
        _quantile(mid_subset, 0.75, fallback=max(global_impact, small_impact)),
    )
    large_impact = max(
        medium_impact,
        _quantile(high_subset, 0.90, fallback=max(global_impact, medium_impact)),
    )
    impact_cap = max(0.001, min(0.05, large_impact))

    raw_curve = (
        (0.0, 0.0),
        (0.35, small_impact),
        (0.65, medium_impact),
        (1.0, impact_cap),
    )
    monotonic_curve = _ensure_monotonic(raw_curve)
    depth_curve = tuple((round(depth, 3), round(value, 4)) for depth, value in monotonic_curve)

    calibration = SlippageCalibration(
        family=family,
        impact_cap=round(impact_cap, 4),
        depth_curve=depth_curve,
        sample_size=sample_size,
        updated=datetime.now(tz=UTC),
    )
    if persist:
        _persist_slippage(calibration)
    return calibration


def load_slippage_calibration(series: str) -> SlippageCalibration | None:
    """Load a persisted slippage calibration for the given series family."""

    family = canonical_series_family(series)
    entry = _load_calibration_entry(family)
    if entry is None:
        return None
    try:
        impact_cap = float(entry.get("impact_cap", 0.0))
    except (TypeError, ValueError):
        return None
    if impact_cap <= 0:
        return None
    curve_payload = entry.get("depth_curve") or []
    curve: list[tuple[float, float]] = []
    for point in curve_payload:
        try:
            depth = float(point[0])
            value = float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        curve.append((depth, value))
    if not curve or curve[0][0] != 0.0:
        curve.insert(0, (0.0, 0.0))
    curve = list(_ensure_monotonic(tuple(curve)))
    if curve[-1][0] < 1.0:
        curve.append((1.0, curve[-1][1]))
    depth_curve = tuple((round(depth, 3), round(value, 4)) for depth, value in curve)
    sample_size = int(entry.get("sample_size", 0) or 0)
    updated = _parse_timestamp(entry.get("updated"))
    return SlippageCalibration(
        family=family,
        impact_cap=impact_cap,
        depth_curve=depth_curve,
        sample_size=sample_size,
        updated=updated,
    )


def load_slippage_model(series: str, *, mode: str = "depth") -> SlippageModel | None:
    """Instantiate a `SlippageModel` from persisted calibration parameters."""

    calibration = load_slippage_calibration(series)
    if calibration is None:
        return None
    return calibration.as_model(mode=mode)


def _load_ledger_frame() -> pl.DataFrame:
    if not LEDGER_PATH.exists():
        return pl.DataFrame()
    frame = pl.read_parquet(LEDGER_PATH)
    if "timestamp_et" in frame.columns and frame["timestamp_et"].dtype == pl.Utf8:
        frame = frame.with_columns(pl.col("timestamp_et").str.strptime(pl.Datetime, strict=False))
    return frame


def _persist_slippage(calibration: SlippageCalibration) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    if STATE_PATH.exists():
        try:
            payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    series_map = payload.setdefault("series", {})
    series_map[calibration.family] = {
        "impact_cap": round(calibration.impact_cap, 6),
        "depth_curve": [[depth, value] for depth, value in calibration.depth_curve],
        "sample_size": calibration.sample_size,
        "updated": calibration.updated.isoformat(),
    }
    payload["updated"] = calibration.updated.isoformat()
    STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_calibration_entry(family: str) -> dict[str, object] | None:
    if not STATE_PATH.exists():
        return None
    try:
        payload = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    series_map = payload.get("series")
    if not isinstance(series_map, dict):
        return None
    entry = series_map.get(family)
    if isinstance(entry, dict):
        return entry
    return None


def _quantile(frame: pl.DataFrame, quantile: float, *, fallback: float) -> float:
    if frame.is_empty():
        return fallback
    series = frame.select(pl.col("impact").quantile(quantile, interpolation="nearest")).to_series()
    value = series.item()
    if value is None:
        return fallback
    return float(value)


def _ensure_monotonic(curve: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    ordered = sorted(curve, key=lambda item: item[0])
    result: list[tuple[float, float]] = []
    current = 0.0
    for depth, value in ordered:
        clamped_depth = max(0.0, min(1.0, float(depth)))
        current = max(current, float(value))
        result.append((clamped_depth, current))
    if result[0][0] != 0.0:
        result.insert(0, (0.0, 0.0))
    if result[-1][0] != 1.0:
        result.append((1.0, result[-1][1]))
    return tuple(result)


def _parse_timestamp(value: object) -> datetime:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except ValueError:
            pass
    return datetime.now(tz=UTC)
