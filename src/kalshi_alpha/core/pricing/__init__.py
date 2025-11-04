"""Ladder pricing utilities: survival curve projection, PMF extraction, and EV analysis.

Based on Kalshi greater-than/bin ladders and fee schedule (Oct 1, 2025).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, auto

from kalshi_alpha.core.fees import (
    DEFAULT_FEE_SCHEDULE,
    FeeSchedule,
    get_index_fee_curve,
    load_index_fee_curves,
    round_up_to_cent,
)

from .mispricing import (
    KinkMetrics as KinkMetrics,
)
from .mispricing import (
    implied_cdf_kinks as implied_cdf_kinks,
)
from .mispricing import (
    kink_spreads as kink_spreads,
)
from .mispricing import (
    prob_sum_gap as prob_sum_gap,
)

MIN_PROB = 0.0
MAX_PROB = 1.0
_INDEX_FEE_CURVES = load_index_fee_curves()
_INDEX_SERIES = frozenset(_INDEX_FEE_CURVES.keys())
_DEFAULT_INDEX_COEFFICIENT = Decimal("0.035")


class Liquidity(Enum):
    """Maker/taker liquidity flag."""

    MAKER = auto()
    TAKER = auto()


class OrderSide(Enum):
    """YES or NO order direction."""

    YES = auto()
    NO = auto()


@dataclass(frozen=True)
class LadderRung:
    """Represents a single ladder strike with YES price quoted in probability space."""

    strike: float
    yes_price: float
    label: str | None = None

    def __post_init__(self) -> None:
        if not (MIN_PROB <= self.yes_price <= MAX_PROB):
            raise ValueError("yes_price must lie in [0, 1]")


@dataclass(frozen=True)
class LadderBinProbability:
    """Probability mass associated with a bin."""

    lower: float | None
    upper: float | None
    probability: float


def _index_fee(contracts: int, price: float, series_key: str) -> float:
    if price < MIN_PROB or price > MAX_PROB:
        raise ValueError("price must lie in [0, 1]")
    if contracts <= 0:
        raise ValueError("contracts must be positive")
    price_dec = Decimal(str(price))
    contracts_dec = Decimal(str(contracts))
    curve = get_index_fee_curve(series_key)
    coefficient = curve.coefficient if curve else _DEFAULT_INDEX_COEFFICIENT
    raw_fee = coefficient * contracts_dec * price_dec * (Decimal("1") - price_dec)
    rounded = round_up_to_cent(raw_fee)
    return float(rounded)


def _clamp_probability(value: float) -> float:
    if value < MIN_PROB:
        return MIN_PROB
    if value > MAX_PROB:
        return MAX_PROB
    return value


def project_survival(values: Sequence[float]) -> list[float]:
    """Project values onto the monotone non-increasing cone via PAV."""
    # Clamp into [0, 1] first.
    clamped = [_clamp_probability(value) for value in values]
    blocks: list[tuple[float, int]] = [(value, 1) for value in clamped]
    index = 0
    while index < len(blocks) - 1:
        value_current, weight_current = blocks[index]
        value_next, weight_next = blocks[index + 1]
        if value_current < value_next:
            total_weight = weight_current + weight_next
            averaged = (value_current * weight_current + value_next * weight_next) / total_weight
            blocks[index] = (averaged, total_weight)
            del blocks[index + 1]
            index = max(index - 1, 0)
        else:
            index += 1

    projected: list[float] = []
    for value, weight in blocks:
        projected.extend([value] * weight)
    return projected[: len(values)]


def survival_from_quotes(
    rungs: Sequence[LadderRung],
    *,
    enforce_monotonic: bool = True,
) -> list[float]:
    """Extract the implied survival curve from ladder quotes."""
    yes_probs = [rung.yes_price for rung in sorted(rungs, key=lambda r: r.strike)]
    if enforce_monotonic:
        return project_survival(yes_probs)
    return list(map(_clamp_probability, yes_probs))


def pmf_from_survival(
    strikes: Sequence[float],
    survival: Sequence[float],
    *,
    tolerance: float = 1e-8,
) -> list[LadderBinProbability]:
    """Convert a survival curve into discrete bin probabilities."""
    if len(strikes) != len(survival):
        raise ValueError("strikes and survival must have the same length")
    if not strikes:
        return []

    ordered = sorted(zip(strikes, survival, strict=True), key=lambda item: item[0])
    bins: list[LadderBinProbability] = []

    prev_survival = MAX_PROB
    prev_strike = None
    total = 0.0

    for strike, survival_prob_raw in ordered:
        survival_prob = _clamp_probability(survival_prob_raw)
        if survival_prob > prev_survival + tolerance:
            raise ValueError("survival curve is not non-increasing")
        probability_mass = max(prev_survival - survival_prob, 0.0)
        bins.append(
            LadderBinProbability(
                lower=prev_strike,
                upper=strike,
                probability=probability_mass,
            )
        )
        total += probability_mass
        prev_survival = survival_prob
        prev_strike = strike

    # Tail bin (>= last strike)
    tail_mass = max(prev_survival, 0.0)
    bins.append(
        LadderBinProbability(
            lower=prev_strike,
            upper=None,
            probability=tail_mass,
        )
    )
    total += tail_mass

    deficit = 1.0 - total
    if abs(deficit) > tolerance:
        # Adjust final bin to absorb numerical drift; keep non-negativity.
        adjusted_tail = max(bins[-1].probability + deficit, 0.0)
        correction = adjusted_tail - bins[-1].probability
        bins[-1] = LadderBinProbability(
            lower=bins[-1].lower,
            upper=bins[-1].upper,
            probability=adjusted_tail,
        )
        total += correction

    probabilities = [max(bin_prob.probability, 0.0) for bin_prob in bins]
    probabilities = project_simplex(probabilities)
    return [
        LadderBinProbability(lower=bin_prob.lower, upper=bin_prob.upper, probability=prob)
        for bin_prob, prob in zip(bins, probabilities, strict=True)
    ]


def pmf_from_quotes(
    rungs: Sequence[LadderRung],
    *,
    tolerance: float = 1e-8,
) -> list[LadderBinProbability]:
    """Convenience wrapper that goes from ladder quotes → survival → PMF."""
    ordered = sorted(rungs, key=lambda r: r.strike)
    strikes = [r.strike for r in ordered]
    survival = survival_from_quotes(ordered)
    return pmf_from_survival(strikes, survival, tolerance=tolerance)


def has_probability_arbitrage(
    pmf: Sequence[LadderBinProbability],
    *,
    tolerance: float = 1e-6,
) -> bool:
    """Return True when probabilities violate basic add-up constraints."""
    total = sum(bin_prob.probability for bin_prob in pmf)
    if any(bin_prob.probability < -tolerance for bin_prob in pmf):
        return True
    return abs(total - 1.0) > tolerance


def _validate_inputs(
    probability: float,
    yes_price: float,
    *,
    tolerance: float = 1e-9,
) -> tuple[float, float]:
    for label, value in (("probability", probability), ("yes_price", yes_price)):
        if value < MIN_PROB - tolerance or value > MAX_PROB + tolerance:
            raise ValueError(f"{label} must lie in [0, 1], received {value}")
    prob = _clamp_probability(probability)
    price = _clamp_probability(yes_price)
    return prob, price


def expected_value_after_fees(  # noqa: PLR0913
    *,
    contracts: int,
    yes_price: float,
    event_probability: float,
    side: OrderSide,
    liquidity: Liquidity,
    schedule: FeeSchedule = DEFAULT_FEE_SCHEDULE,
    series: str | None = None,
    market_name: str | None = None,
) -> float:
    """Compute the expected value of a trade after trading fees."""
    probability, price = _validate_inputs(event_probability, yes_price)
    contracts_f = float(contracts)
    if contracts_f <= 0:
        raise ValueError("contracts must be positive")

    series_key = series.upper() if isinstance(series, str) else None

    if side is OrderSide.YES:
        if series_key in _INDEX_SERIES:
            fee_price = price
        else:
            fee_price = price
        payoff_if_win = (1.0 - price) * contracts_f
        payoff_if_lose = -price * contracts_f
        expected = probability * payoff_if_win + (1.0 - probability) * payoff_if_lose
    elif side is OrderSide.NO:
        no_price = 1.0 - price
        if series_key in _INDEX_SERIES:
            fee_price = no_price
        else:
            fee_price = no_price
        payoff_if_win = (1.0 - no_price) * contracts_f  # event fails
        payoff_if_lose = -no_price * contracts_f
        expected = (1.0 - probability) * payoff_if_win + probability * payoff_if_lose
    else:
        raise ValueError(f"Unsupported order side: {side}")

    if series_key in _INDEX_SERIES:
        if liquidity is Liquidity.MAKER:
            fee = 0.0
        else:
            fee = _index_fee(contracts, fee_price, series_key)
    else:
        fee_fn = schedule.maker_fee if liquidity is Liquidity.MAKER else schedule.taker_fee
        fee = float(
            fee_fn(
                contracts,
                fee_price,
                series=series,
                market_name=market_name,
            )
        )

    return expected - fee


def yes_no_expected_values(  # noqa: PLR0913
    *,
    contracts: int,
    yes_price: float,
    event_probability: float,
    liquidity: Liquidity,
    schedule: FeeSchedule = DEFAULT_FEE_SCHEDULE,
    series: str | None = None,
    market_name: str | None = None,
) -> dict[OrderSide, float]:
    """Return both YES and NO EVs after fees."""
    return {
        OrderSide.YES: expected_value_after_fees(
            contracts=contracts,
            yes_price=yes_price,
            event_probability=event_probability,
            side=OrderSide.YES,
            liquidity=liquidity,
            schedule=schedule,
            series=series,
            market_name=market_name,
        ),
        OrderSide.NO: expected_value_after_fees(
            contracts=contracts,
            yes_price=yes_price,
            event_probability=event_probability,
            side=OrderSide.NO,
            liquidity=liquidity,
            schedule=schedule,
            series=series,
            market_name=market_name,
        ),
    }


def project_simplex(values: Sequence[float]) -> list[float]:
    """Project raw scores onto the probability simplex."""
    if not values:
        return []
    shifted = [value for value in values]
    sorted_values = sorted(shifted, reverse=True)
    cumulative = 0.0
    rho = -1
    theta = 0.0
    for index, value in enumerate(sorted_values, start=1):
        cumulative += value
        threshold = (cumulative - 1.0) / index
        if value - threshold > 0:
            rho = index
            theta = threshold
    if rho == -1:
        return [1.0 / len(values)] * len(values)
    theta = (sum(sorted_values[:rho]) - 1.0) / rho
    projected = [max(value - theta, 0.0) for value in shifted]
    total = sum(projected)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in projected]
