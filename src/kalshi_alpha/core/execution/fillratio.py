"""Estimate expected fills using simple visible-depth heuristics."""

from __future__ import annotations

import math
from dataclasses import dataclass

from kalshi_alpha.core.kalshi_api import Orderbook


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
        return expected_fills(contracts, visible, self.alpha)

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
