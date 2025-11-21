"""Scanner helpers for daily close index ladders."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from kalshi_alpha.config import lookup_index_rule
from kalshi_alpha.datastore.paths import PROC_ROOT

from .scan_index_hourly import IndexScanResult, QuoteOpportunity

if TYPE_CHECKING:
    from kalshi_alpha.strategies.index import CloseInputs

CLOSE_CALIBRATION_PATH = PROC_ROOT / "calib" / "index"


def _preparse_fast_flags(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fast-fixtures", action="store_true", dest="fast_fixtures")
    parser.add_argument("--fixtures-root", default="tests/data_fixtures")
    parser.add_argument("--output-root", default="reports/index_ladders")
    parser.add_argument("--series", nargs="+")
    parser.add_argument("--now")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--min-ev", type=float, default=0.05)
    return parser.parse_known_args(list(argv) if argv is not None else None)


def evaluate_close(  # noqa: PLR0913
    strikes: Sequence[float],
    yes_prices: Sequence[float],
    inputs: CloseInputs,
    *,
    contracts: int = 1,
    min_ev: float = 0.05,
) -> IndexScanResult:
    from kalshi_alpha.drivers.polygon_index.symbols import resolve_series as resolve_index_series
    from kalshi_alpha.exec.scanners.utils import expected_value_summary
    from kalshi_alpha.strategies.index import close_pmf
    from kalshi_alpha.strategies.index import cdf as index_cdf

    if len(strikes) != len(yes_prices):
        raise ValueError("strikes and prices must have equal length")
    meta = resolve_index_series(inputs.series)
    calibration = None
    try:
        calibration = index_cdf.load_calibration(
            CLOSE_CALIBRATION_PATH,
            meta.polygon_ticker,
            horizon="close",
            variant=None,
        )
    except Exception:  # pragma: no cover - defensive fallback
        calibration = None
    pmf = close_pmf(strikes, inputs, calibration=calibration)
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


DEFAULT_SERIES: tuple[str, ...] = ("INX", "NASDAQ100")
ET = ZoneInfo("America/New_York")
TARGET_CLOSE_TIME = time(16, 0)


def main(argv: Sequence[str] | None = None) -> None:
    fast_args, _remaining = _preparse_fast_flags(argv)
    if fast_args.fast_fixtures:
        from kalshi_alpha.exec.scanners.fast_index import run_fast_close

        run_fast_close(fast_args)
        return

    from .index_scan_common import ScannerConfig, build_parser, parse_timestamp, run_index_scan

    parser = build_parser(DEFAULT_SERIES)
    parser.add_argument(
        "--fast-fixtures",
        action="store_true",
        dest="fast_fixtures",
        help="Use trimmed Polygon index fixtures for quick offline runs.",
    )
    args = parser.parse_args(argv)
    base_timestamp = parse_timestamp(args.now) or datetime.now(tz=UTC)
    target_dt_et = datetime.combine(base_timestamp.astimezone(ET).date(), TARGET_CLOSE_TIME, tzinfo=ET)
    now_override = target_dt_et - timedelta(minutes=10)
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
        run_label="index_close",
        timestamp=target_dt_et.astimezone(UTC),
        now_override=now_override.astimezone(UTC),
        target_time_et=TARGET_CLOSE_TIME,
        paper_ledger=paper_ledger,
        maker_only=maker_only,
        emit_report=emit_report,
    )
    run_index_scan(config)


__all__ = ["evaluate_close", "main"]


if __name__ == "__main__":  # pragma: no cover
    main()
