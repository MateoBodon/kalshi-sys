"""Scanner helpers and CLI for intraday hourly index ladders."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from kalshi_alpha.config import IndexRule, load_index_ops_config, lookup_index_rule
from kalshi_alpha.core.pricing import LadderBinProbability, OrderSide
from kalshi_alpha.datastore.paths import PROC_ROOT

if TYPE_CHECKING:
    from kalshi_alpha.strategies.index import HourlyInputs

HOURLY_CALIBRATION_PATH = PROC_ROOT / "calib" / "index"
INDEX_OPS_CONFIG = load_index_ops_config()
DEFAULT_SERIES: tuple[str, ...] = ("INXU", "NASDAQ100U")
ET = ZoneInfo("America/New_York")
DEFAULT_TARGET_HOURS = (10, 11, 12, 13, 14, 15, 16)


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


def _preparse_fast_flags(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fast-fixtures", action="store_true", dest="fast_fixtures")
    parser.add_argument("--fixtures-root", default="tests/data_fixtures")
    parser.add_argument("--output-root", default="reports/index_ladders")
    parser.add_argument("--series", nargs="+")
    parser.add_argument("--target-hour", dest="target_hours", type=int, action="append")
    parser.add_argument("--now")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--min-ev", type=float, default=float(INDEX_OPS_CONFIG.min_ev_usd))
    return parser.parse_known_args(list(argv) if argv is not None else None)


def evaluate_hourly(  # noqa: PLR0913
    strikes: Sequence[float],
    yes_prices: Sequence[float],
    inputs: HourlyInputs,
    *,
    contracts: int = 1,
    min_ev: float = 0.05,
) -> IndexScanResult:
    from kalshi_alpha.drivers.polygon_index.symbols import resolve_series as resolve_index_series
    from kalshi_alpha.exec.scanners.utils import expected_value_summary
    from kalshi_alpha.strategies.index import hourly_pmf
    from kalshi_alpha.strategies.index import cdf as index_cdf

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
    fast_args, _remaining = _preparse_fast_flags(argv)
    if fast_args.fast_fixtures:
        from kalshi_alpha.exec.scanners.fast_index import run_fast_hourly

        run_fast_hourly(fast_args)
        return

    from kalshi_alpha.exec.scanners.index_scan_common import (
        ScannerConfig,
        build_parser,
        parse_timestamp,
        run_index_scan,
    )

    parser = build_parser(DEFAULT_SERIES)
    parser.add_argument(
        "--target-hour",
        dest="target_hours",
        type=int,
        action="append",
        help="Specific ET target hour to scan (repeat for multiple). Default scans 10:00-16:00 ET.",
    )
    parser.add_argument(
        "--fast-fixtures",
        action="store_true",
        dest="fast_fixtures",
        help="Use trimmed Polygon index fixtures for quick offline runs.",
    )
    args = parser.parse_args(argv)
    base_timestamp = parse_timestamp(args.now) or datetime.now(tz=UTC)
    base_date_et = base_timestamp.astimezone(ET).date()
    target_hours = (
        sorted({hour % 24 for hour in args.target_hours})
        if args.target_hours
        else list(DEFAULT_TARGET_HOURS)
    )
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
    fixtures_root = Path(args.fixtures_root)
    base_output_root = Path(args.output_root)
    for hour in target_hours:
        target_time_et = time(hour % 24, 0)
        target_dt_et = datetime.combine(base_date_et, target_time_et, tzinfo=ET)
        now_override = target_dt_et - timedelta(minutes=5)
        config = ScannerConfig(
            series=tuple(s.upper() for s in args.series),
            min_ev=float(args.min_ev),
            max_bins=int(args.max_bins),
            contracts=int(args.contracts),
            kelly_cap=float(args.kelly_cap),
            offline=bool(args.offline),
            fixtures_root=fixtures_root,
            output_root=base_output_root / f"{hour % 24:02d}00",
            run_label=f"index_hourly_{hour % 24:02d}00",
            timestamp=target_dt_et.astimezone(UTC),
            now_override=now_override.astimezone(UTC),
            target_time_et=target_time_et,
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
