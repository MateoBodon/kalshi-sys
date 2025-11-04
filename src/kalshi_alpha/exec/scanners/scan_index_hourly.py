"""Scanner helpers for intraday hourly index ladders."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from kalshi_alpha.config import IndexRule, lookup_index_rule
from kalshi_alpha.core.pricing import LadderBinProbability, OrderSide
from kalshi_alpha.drivers.polygon_index.symbols import resolve_series as resolve_index_series
from kalshi_alpha.exec.scanners.utils import expected_value_summary
from kalshi_alpha.strategies.index import (
    HOURLY_CALIBRATION_PATH as _HOURLY_CALIBRATION_PATH,
)
from kalshi_alpha.strategies.index import HourlyInputs, hourly_pmf
from kalshi_alpha.strategies.index import cdf as index_cdf

HOURLY_CALIBRATION_PATH = _HOURLY_CALIBRATION_PATH


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
    rule: IndexRule | None = None


def evaluate_hourly(  # noqa: PLR0913
    strikes: Sequence[float],
    yes_prices: Sequence[float],
    inputs: HourlyInputs,
    *,
    contracts: int = 1,
    min_ev: float = 0.05,
) -> IndexScanResult:
    if len(strikes) != len(yes_prices):
        raise ValueError("strikes and prices must have equal length")
    pmf = hourly_pmf(strikes, inputs)
    survival = index_cdf.survival_map(strikes, pmf)
    tail_lower = float(pmf[0].probability) if pmf else 0.0
    tail_upper = float(pmf[-1].probability) if pmf else 0.0
    calibration = None
    try:
        meta = resolve_index_series(inputs.series)
        for horizon in ("hourly", "noon"):
            try:
                calibration = index_cdf.load_calibration(
                    HOURLY_CALIBRATION_PATH,
                    meta.polygon_ticker,
                    horizon=horizon,
                )
                break
            except FileNotFoundError:
                continue
    except Exception:  # pragma: no cover - defensive fallback
        calibration = None

    opportunities: list[QuoteOpportunity] = []
    for idx, (strike, yes_price) in enumerate(zip(strikes, yes_prices, strict=True)):
        model_prob = float(survival[float(strike)])
        if calibration is not None:
            model_prob = calibration.apply_pit(model_prob)
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

    try:
        rule = lookup_index_rule(inputs.series)
    except KeyError:
        rule = None

    return IndexScanResult(
        pmf=pmf,
        survival=survival,
        opportunities=opportunities,
        below_first_mass=tail_lower,
        tail_mass=tail_upper,
        rule=rule,
    )


__all__ = ["HOURLY_CALIBRATION_PATH", "QuoteOpportunity", "IndexScanResult", "evaluate_hourly"]
