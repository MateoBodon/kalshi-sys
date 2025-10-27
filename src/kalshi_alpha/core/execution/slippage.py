"""Slippage modelling utilities for paper execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from kalshi_alpha.core.kalshi_api import Orderbook


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
