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
        max_replacements_per_bin: int = 3,
    ) -> None:
        self.skew_floor = max(float(skew_floor), 0.0)
        self.skew_weight = max(float(skew_weight), 0.0)
        self.microprice_weight = max(float(microprice_weight), 0.0)
        self.freshness_soft_ms = max(float(freshness_soft_ms), 0.0)
        self.freshness_slope = max(float(freshness_slope), 0.0)
        self.max_replacements_per_bin = max(int(max_replacements_per_bin), 1)
        self._replacement_counts: dict[str, int] = {}

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

    def key_for_order(self, market_id: str, strike: float, side: OrderSide) -> str:
        return f"{market_id}:{strike:.2f}:{side.name}"

    def should_throttle(self, key: str) -> bool:
        return self._replacement_counts.get(key, 0) >= self.max_replacements_per_bin

    def record_submission(self, key: str) -> None:
        self._replacement_counts[key] = self._replacement_counts.get(key, 0) + 1


def microprice_from_orderbook(orderbook: Orderbook | None) -> tuple[float | None, float | None, float | None]:
    """Return (microprice, best_bid, best_ask) derived from the first level of an orderbook."""

    if orderbook is None or not orderbook.bids or not orderbook.asks:
        return None, None, None
    try:
        best_bid_entry = orderbook.bids[0]
        best_ask_entry = orderbook.asks[0]
        best_bid = float(best_bid_entry.get("price"))
        best_ask = float(best_ask_entry.get("price"))
        bid_size = max(float(best_bid_entry.get("size", 0.0)), 0.0)
        ask_size = max(float(best_ask_entry.get("size", 0.0)), 0.0)
    except (TypeError, ValueError, KeyError):
        return None, None, None
    denom = bid_size + ask_size
    if denom <= 0.0:
        return None, best_bid, best_ask
    microprice = (best_ask * bid_size + best_bid * ask_size) / denom
    return microprice, best_bid, best_ask


__all__ = ["QuoteContext", "QuoteOptimizer", "microprice_from_orderbook"]
