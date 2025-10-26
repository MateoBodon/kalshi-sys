"""Risk utilities including PAL (Position and Loss) policy enforcement."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from kalshi_alpha.core.fees import DEFAULT_FEE_SCHEDULE, FeeSchedule
from kalshi_alpha.core.pricing import Liquidity, OrderSide


@dataclass(frozen=True)
class PALPolicy:
    """Position and loss guardrails for a single Kalshi series."""

    series: str
    default_max_loss: float
    per_strike: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> PALPolicy:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        series = str(payload["series"])
        default_max_loss = float(payload["default_max_loss"])
        per_strike = {
            str(key): float(value) for key, value in (payload.get("per_strike") or {}).items()
        }
        return cls(series=series, default_max_loss=default_max_loss, per_strike=per_strike)

    def limit_for_strike(self, strike_id: str) -> float:
        return self.per_strike.get(strike_id, self.default_max_loss)


@dataclass(frozen=True)
class OrderProposal:
    """Dry-run order proposal evaluated by the PAL guard."""

    strike_id: str
    yes_price: float
    contracts: int
    side: OrderSide
    liquidity: Liquidity
    market_name: str | None = None


def max_loss_for_order(
    order: OrderProposal,
    *,
    schedule: FeeSchedule = DEFAULT_FEE_SCHEDULE,
) -> float:
    """Return the maximum loss in USD for the order, including trading fees."""
    contracts = order.contracts
    price = order.yes_price
    if contracts <= 0:
        raise ValueError("contracts must be positive")

    if order.side is OrderSide.YES:
        fee = float(schedule.taker_fee(contracts, price, market_name=order.market_name))
        loss = contracts * price
    elif order.side is OrderSide.NO:
        no_price = 1.0 - price
        fee = float(schedule.taker_fee(contracts, no_price, market_name=order.market_name))
        loss = contracts * no_price
    else:
        raise ValueError(f"Unsupported order side: {order.side}")

    return loss + fee


class PALGuard:
    """Tracks rolling exposure versus PAL limits."""

    def __init__(self, policy: PALPolicy, *, schedule: FeeSchedule = DEFAULT_FEE_SCHEDULE) -> None:
        self.policy = policy
        self.schedule = schedule
        self._exposure: MutableMapping[str, float] = defaultdict(float)

    def can_accept(self, order: OrderProposal) -> bool:
        limit = self.policy.limit_for_strike(order.strike_id)
        projected = self._exposure[order.strike_id] + max_loss_for_order(
            order, schedule=self.schedule
        )
        return projected <= limit

    def register(self, order: OrderProposal) -> None:
        if not self.can_accept(order):
            raise ValueError(f"Order exceeds PAL limit for strike {order.strike_id}")
        self._exposure[order.strike_id] += max_loss_for_order(order, schedule=self.schedule)

    def exposure_for(self, strike_id: str) -> float:
        return self._exposure.get(strike_id, 0.0)

    def exposure_snapshot(self) -> dict[str, float]:
        return dict(self._exposure)
