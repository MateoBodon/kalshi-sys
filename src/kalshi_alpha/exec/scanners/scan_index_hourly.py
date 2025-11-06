"""Scanner helpers and CLI for intraday hourly index ladders."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from kalshi_alpha.config import IndexRule, lookup_index_rule
from kalshi_alpha.core.pricing import LadderBinProbability, OrderSide
from kalshi_alpha.drivers.polygon_index.symbols import resolve_series as resolve_index_series
from kalshi_alpha.exec.scanners.index_scan_common import (
    ScannerConfig,
    build_parser,
    parse_timestamp,
    run_index_scan,
)
from kalshi_alpha.exec.scanners.utils import expected_value_summary
from kalshi_alpha.strategies.index import (
    HOURLY_CALIBRATION_PATH as _HOURLY_CALIBRATION_PATH,
)
from kalshi_alpha.strategies.index import HourlyInputs, hourly_pmf
from kalshi_alpha.strategies.index import cdf as index_cdf

HOURLY_CALIBRATION_PATH = _HOURLY_CALIBRATION_PATH
DEFAULT_SERIES: tuple[str, ...] = ("INXU", "NASDAQ100U")


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
    variant = None
    if getattr(inputs, "target_hour_et", None) is not None:
        variant = f"{int(inputs.target_hour_et) % 24:02d}00"
    calibration = None
    meta = resolve_index_series(inputs.series)
    try:
        calibration = index_cdf.load_calibration(
            HOURLY_CALIBRATION_PATH,
            meta.polygon_ticker,
            horizon="hourly",
            variant=variant,
        )
    except FileNotFoundError:
        try:
            calibration = index_cdf.load_calibration(
                HOURLY_CALIBRATION_PATH,
                meta.polygon_ticker,
                horizon="hourly",
                variant=None,
            )
        except Exception:  # pragma: no cover - defensive fallback
            try:
                calibration = index_cdf.load_calibration(
                    HOURLY_CALIBRATION_PATH,
                    meta.polygon_ticker,
                    horizon="noon",
                    variant=None,
                )
            except Exception:
                calibration = None
    except Exception:  # pragma: no cover - defensive fallback
        calibration = None
    pmf = hourly_pmf(strikes, inputs, calibration=calibration)
    survival = index_cdf.survival_map(strikes, pmf)
    tail_lower = float(pmf[0].probability) if pmf else 0.0
    tail_upper = float(pmf[-1].probability) if pmf else 0.0
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser(DEFAULT_SERIES)
    args = parser.parse_args(argv)
    timestamp = parse_timestamp(args.now)
    emit_report = True
    if getattr(args, "no_report", False):
        emit_report = False
    elif getattr(args, "report", False):
        emit_report = True
    paper_ledger = True
    if getattr(args, "no_paper_ledger", False):
        paper_ledger = False
    elif getattr(args, "paper_ledger", False):
        paper_ledger = True
    maker_only = True
    if getattr(args, "no_maker_only", False):
        maker_only = False
    elif getattr(args, "maker_only", False):
        maker_only = True
    config = ScannerConfig(
        series=tuple(s.upper() for s in args.series),
        min_ev=float(args.min_ev),
        max_bins=int(args.max_bins),
        contracts=int(args.contracts),
        kelly_cap=float(args.kelly_cap),
        offline=bool(args.offline),
        fixtures_root=Path(args.fixtures_root),
        output_root=Path(args.output_root),
        run_label="index_hourly",
        timestamp=timestamp,
        paper_ledger=paper_ledger,
        maker_only=maker_only,
        emit_report=emit_report,
    )
    run_index_scan(config)


__all__ = [
    "DEFAULT_SERIES",
    "HOURLY_CALIBRATION_PATH",
    "IndexScanResult",
    "QuoteOpportunity",
    "evaluate_hourly",
    "main",
]
