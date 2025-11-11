"""Construct hedged Range↔AB structures from adjacent strikes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from kalshi_alpha.core.pricing import (
    LadderRung,
    Liquidity,
    OrderSide,
    expected_value_after_fees,
)

__all__ = [
    "RangeABStructure",
    "StructureLeg",
    "build_range_structures",
]


@dataclass(frozen=True)
class StructureLeg:
    strike: float
    side: OrderSide
    yes_price: float
    probability: float
    contracts: int
    direction: int = 1  # +1 long, -1 short

    @property
    def max_loss(self) -> float:
        if self.contracts <= 0:
            return 0.0
        quantity = float(self.contracts)
        if self.side is OrderSide.YES:
            if self.direction >= 0:
                return quantity * float(self.yes_price)
            return quantity * float(1.0 - self.yes_price)
        if self.side is OrderSide.NO:
            if self.direction >= 0:
                return quantity * float(1.0 - self.yes_price)
            return quantity * float(self.yes_price)
        raise ValueError(f"Unsupported side {self.side}")


@dataclass(frozen=True)
class RangeABStructure:
    structure_id: str
    series: str
    market_id: str
    market_ticker: str
    lower_strike: float
    upper_strike: float
    contracts: int
    range_probability: float
    synthetic_price: float
    maker_ev: float
    max_loss: float
    sigma: float
    legs: tuple[StructureLeg, StructureLeg]

    def to_summary(self) -> dict[str, object]:
        return {
            "id": self.structure_id,
            "series": self.series,
            "market_id": self.market_id,
            "market_ticker": self.market_ticker,
            "lower": self.lower_strike,
            "upper": self.upper_strike,
            "contracts": self.contracts,
            "probability": self.range_probability,
            "price": self.synthetic_price,
            "maker_ev": self.maker_ev,
            "max_loss": self.max_loss,
            "sigma": self.sigma,
        }


def _structure_id(market_id: str, lower: float, upper: float) -> str:
    return f"{market_id}:{lower:.2f}-{upper:.2f}"


def build_range_structures(
    *,
    series: str,
    market_id: str,
    market_ticker: str,
    rungs: Sequence[LadderRung],
    strategy_survival: Sequence[float],
    contracts: int,
    schedule_series: str | None = None,
) -> list[RangeABStructure]:
    """Return Range↔AB structures formed by adjacent strikes.

    Each structure replicates a close-range payoff by going long YES on the lower
    strike and short YES on the next strike. The result is for monitoring and
    VaR accounting; execution still occurs leg-by-leg in the scanner.
    """

    if len(rungs) < 2 or len(rungs) != len(strategy_survival):
        return []
    ordered = sorted(zip(rungs, strategy_survival, strict=True), key=lambda item: item[0].strike)
    structures: list[RangeABStructure] = []
    lot_size = max(int(contracts), 1)
    for idx in range(len(ordered) - 1):
        lower_rung, prob_lower = ordered[idx]
        upper_rung, prob_upper = ordered[idx + 1]
        lower_prob = float(prob_lower)
        upper_prob = float(prob_upper)
        range_probability = max(lower_prob - upper_prob, 0.0)
        synthetic_price = max(float(lower_rung.yes_price) - float(upper_rung.yes_price), 0.0)
        if range_probability <= 0.0 and synthetic_price <= 0.0:
            continue
        leg_yes = StructureLeg(
            strike=float(lower_rung.strike),
            side=OrderSide.YES,
            yes_price=float(lower_rung.yes_price),
            probability=lower_prob,
            contracts=lot_size,
            direction=1,
        )
        leg_short_yes = StructureLeg(
            strike=float(upper_rung.strike),
            side=OrderSide.YES,
            yes_price=float(upper_rung.yes_price),
            probability=upper_prob,
            contracts=lot_size,
            direction=-1,
        )
        ev_long = expected_value_after_fees(
            contracts=lot_size,
            yes_price=leg_yes.yes_price,
            event_probability=leg_yes.probability,
            side=OrderSide.YES,
            liquidity=Liquidity.MAKER,
            series=schedule_series or series,
            market_name=market_ticker,
        )
        ev_short = expected_value_after_fees(
            contracts=lot_size,
            yes_price=leg_short_yes.yes_price,
            event_probability=leg_short_yes.probability,
            side=OrderSide.YES,
            liquidity=Liquidity.MAKER,
            series=schedule_series or series,
            market_name=market_ticker,
        )
        maker_ev = ev_long - ev_short
        max_loss = leg_yes.max_loss + leg_short_yes.max_loss
        sigma = math.sqrt(max(range_probability * (1.0 - range_probability), 0.0)) * float(lot_size)
        structure = RangeABStructure(
            structure_id=_structure_id(market_id, leg_yes.strike, leg_short_yes.strike),
            series=series.upper(),
            market_id=market_id,
            market_ticker=market_ticker,
            lower_strike=leg_yes.strike,
            upper_strike=leg_short_yes.strike,
            contracts=lot_size,
            range_probability=range_probability,
            synthetic_price=synthetic_price,
            maker_ev=maker_ev,
            max_loss=max_loss,
            sigma=sigma,
            legs=(leg_yes, leg_short_yes),
        )
        structures.append(structure)
    return structures
