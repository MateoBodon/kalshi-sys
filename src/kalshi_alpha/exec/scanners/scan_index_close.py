"""Scanner helpers for daily close index ladders."""

from __future__ import annotations

from collections.abc import Sequence

from kalshi_alpha.config import lookup_index_rule
from kalshi_alpha.drivers.polygon_index.symbols import resolve_series as resolve_index_series
from kalshi_alpha.exec.scanners.utils import expected_value_summary
from kalshi_alpha.strategies.index import CLOSE_CALIBRATION_PATH, CloseInputs, close_pmf
from kalshi_alpha.strategies.index import cdf as index_cdf

from .scan_index_noon import IndexScanResult, QuoteOpportunity


def evaluate_close(  # noqa: PLR0913
    strikes: Sequence[float],
    yes_prices: Sequence[float],
    inputs: CloseInputs,
    *,
    contracts: int = 1,
    min_ev: float = 0.05,
) -> IndexScanResult:
    if len(strikes) != len(yes_prices):
        raise ValueError("strikes and prices must have equal length")
    pmf = close_pmf(strikes, inputs)
    survival = index_cdf.survival_map(strikes, pmf)
    tail_lower = float(pmf[0].probability) if pmf else 0.0
    tail_upper = float(pmf[-1].probability) if pmf else 0.0
    calibration = None
    try:
        meta = resolve_index_series(inputs.series)
        calibration = index_cdf.load_calibration(
            CLOSE_CALIBRATION_PATH,
            meta.polygon_ticker,
            horizon="close",
        )
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


__all__ = ["evaluate_close"]
