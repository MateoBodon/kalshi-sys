"""Utilities shared by ladder scanners."""

from __future__ import annotations

from collections.abc import Sequence

from kalshi_alpha.core.fees import DEFAULT_FEE_SCHEDULE, FeeSchedule
from kalshi_alpha.core.pricing import (
    LadderBinProbability,
    Liquidity,
    OrderSide,
    expected_value_after_fees,
)


def pmf_to_survival(pmf: Sequence[LadderBinProbability], strikes: Sequence[float]) -> list[float]:
    if len(pmf) != len(strikes) + 1:
        raise ValueError("PMF length must equal strikes length + 1")
    survival: list[float] = []
    tail_sum = 0.0
    for index in reversed(range(len(pmf))):
        tail_sum += pmf[index].probability
        if index > 0:
            survival.append(tail_sum)
    return list(reversed(survival))


def expected_value_summary(  # noqa: PLR0913
    *,
    contracts: int,
    yes_price: float,
    event_probability: float,
    schedule: FeeSchedule = DEFAULT_FEE_SCHEDULE,
    series: str | None = None,
    market_name: str | None = None,
) -> dict[str, float]:
    maker_yes = expected_value_after_fees(
        contracts=contracts,
        yes_price=yes_price,
        event_probability=event_probability,
        side=OrderSide.YES,
        liquidity=Liquidity.MAKER,
        schedule=schedule,
        series=series,
        market_name=market_name,
    )
    taker_yes = expected_value_after_fees(
        contracts=contracts,
        yes_price=yes_price,
        event_probability=event_probability,
        side=OrderSide.YES,
        liquidity=Liquidity.TAKER,
        schedule=schedule,
        series=series,
        market_name=market_name,
    )
    maker_no = expected_value_after_fees(
        contracts=contracts,
        yes_price=yes_price,
        event_probability=event_probability,
        side=OrderSide.NO,
        liquidity=Liquidity.MAKER,
        schedule=schedule,
        series=series,
        market_name=market_name,
    )
    taker_no = expected_value_after_fees(
        contracts=contracts,
        yes_price=yes_price,
        event_probability=event_probability,
        side=OrderSide.NO,
        liquidity=Liquidity.TAKER,
        schedule=schedule,
        series=series,
        market_name=market_name,
    )
    return {
        "maker_yes": maker_yes,
        "taker_yes": taker_yes,
        "maker_no": maker_no,
        "taker_no": taker_no,
    }
