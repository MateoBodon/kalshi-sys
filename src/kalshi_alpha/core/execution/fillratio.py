"""Estimate expected fills using simple visible-depth heuristics."""

from __future__ import annotations

import math
from dataclasses import dataclass

from kalshi_alpha.core.kalshi_api import Orderbook


@dataclass(frozen=True)
class FillRatioEstimator:
    alpha: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be within [0, 1]")

    def expected_contracts(
        self,
        *,
        side: str,
        price: float,
        contracts: int,
        orderbook: Orderbook,
    ) -> int:
        if contracts <= 0:
            return 0
        visible = _visible_depth(side, price, orderbook)
        estimate = min(float(contracts), self.alpha * visible)
        return max(0, min(contracts, math.floor(estimate)))


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
