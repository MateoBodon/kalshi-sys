"""Quote optimization utilities: PMF skew gating, microprice bias, freshness widening."""

from __future__ import annotations

from dataclasses import dataclass

from kalshi_alpha.core.kalshi_api import Orderbook
from kalshi_alpha.core.pricing import OrderSide


@dataclass(frozen=True)
class QuoteContext:
    """Per-proposal signal bundle used to calculate EV penalties."""

    market_id: str
    strike: float
    side: OrderSide
    pmf_probability: float
    market_probability: float
    microprice: float | None
    best_bid: float | None
    best_ask: float | None
    freshness_ms: float | None
    maker_ev_per_contract: float


class QuoteOptimizer:
    """Heuristic EV penalty engine with replacement throttling."""

    def __init__(
        self,
        *,
        skew_floor: float = 0.015,
        skew_weight: float = 8.0,
        microprice_weight: float = 4.0,
        freshness_soft_ms: float = 1200.0,
        freshness_slope: float = 0.0001,
    ) -> None:
        self.skew_floor = max(float(skew_floor), 0.0)
        self.skew_weight = max(float(skew_weight), 0.0)
        self.microprice_weight = max(float(microprice_weight), 0.0)
        self.freshness_soft_ms = max(float(freshness_soft_ms), 0.0)
        self.freshness_slope = max(float(freshness_slope), 0.0)

    def penalty(self, context: QuoteContext) -> float:
        """Return the EV penalty (USD per contract) for the provided context."""

        penalty = 0.0
        skew_gap = abs(context.pmf_probability - context.market_probability)
        if skew_gap < self.skew_floor:
            penalty += (self.skew_floor - skew_gap) * self.skew_weight

        if (
            context.microprice is not None
            and context.best_bid is not None
            and context.best_ask is not None
            and context.best_ask > context.best_bid
        ):
            mid = 0.5 * (context.best_bid + context.best_ask)
            bias = abs(context.microprice - mid)
            penalty += bias * self.microprice_weight

        if context.freshness_ms is not None and context.freshness_ms > self.freshness_soft_ms:
            penalty += (context.freshness_ms - self.freshness_soft_ms) * self.freshness_slope

        return penalty

    @staticmethod
    def key_for_order(market_id: str, strike: float, side: OrderSide) -> str:
        return f"{market_id}:{strike:.2f}:{side.name}"


__all__ = ["QuoteContext", "QuoteOptimizer"]
