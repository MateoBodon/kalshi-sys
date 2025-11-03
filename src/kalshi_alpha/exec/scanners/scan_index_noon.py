"""Scanner helpers for intraday noon index ladders."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from kalshi_alpha.core.pricing import LadderBinProbability, OrderSide
from kalshi_alpha.exec.scanners.utils import expected_value_summary
from kalshi_alpha.strategies.index import NoonInputs, noon_pmf
from kalshi_alpha.strategies.index import cdf as index_cdf


@dataclass(frozen=True)
class QuoteOpportunity:
    strike: float
    yes_price: float
    model_probability: float
    maker_ev: float
    contracts: int
    range_mass: float
    side: OrderSide = OrderSide.YES


@dataclass(frozen=True)
class IndexScanResult:
    pmf: list[LadderBinProbability]
    survival: dict[float, float]
    opportunities: list[QuoteOpportunity]
    below_first_mass: float
    tail_mass: float


def evaluate_noon(  # noqa: PLR0913
    strikes: Sequence[float],
    yes_prices: Sequence[float],
    inputs: NoonInputs,
    *,
    contracts: int = 1,
    min_ev: float = 0.05,
) -> IndexScanResult:
    if len(strikes) != len(yes_prices):
        raise ValueError("strikes and prices must have equal length")
    pmf = noon_pmf(strikes, inputs)
    survival = index_cdf.survival_map(strikes, pmf)
    tail_lower = float(pmf[0].probability) if pmf else 0.0
    tail_upper = float(pmf[-1].probability) if pmf else 0.0

    opportunities: list[QuoteOpportunity] = []
    for idx, (strike, yes_price) in enumerate(zip(strikes, yes_prices, strict=True)):
        model_prob = float(survival[float(strike)])
        ev_summary = expected_value_summary(
            contracts=contracts,
            yes_price=float(yes_price),
            event_probability=model_prob,
            series=inputs.series,
        )
        maker_ev = float(ev_summary["maker_yes"])
        if maker_ev < min_ev:
            continue
        range_mass = float(pmf[idx + 1].probability) if (idx + 1) < len(pmf) else tail_upper
        opportunities.append(
            QuoteOpportunity(
                strike=float(strike),
                yes_price=float(yes_price),
                model_probability=model_prob,
                maker_ev=maker_ev,
                contracts=contracts,
                range_mass=range_mass,
            )
        )

    return IndexScanResult(
        pmf=pmf,
        survival=survival,
        opportunities=opportunities,
        below_first_mass=tail_lower,
        tail_mass=tail_upper,
    )


__all__ = ["QuoteOpportunity", "IndexScanResult", "evaluate_noon"]
