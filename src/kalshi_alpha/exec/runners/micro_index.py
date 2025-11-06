"""Microlive runner for index ladders: one window, 1-lot maker quotes."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta, time
from pathlib import Path

from kalshi_alpha.config import load_index_ops_config
from kalshi_alpha.core.execution.fillratio import tune_alpha
from kalshi_alpha.core.execution.series_utils import canonical_series_family
from kalshi_alpha.core.execution.slippage import fit_slippage
from kalshi_alpha.datastore.paths import RAW_ROOT
from kalshi_alpha.exec import scoreboard
from kalshi_alpha.exec.runners import scan_ladders

INDEX_OPS_CONFIG = load_index_ops_config()


def _default_hourly_target(reference: datetime) -> time:
    local = reference.astimezone(INDEX_OPS_CONFIG.timezone)
    hour = local.hour
    if local.minute >= 40:
        hour = (hour + 1) % 24
    return time(hour, 0)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run microlive index scan for a single window.")
    parser.add_argument("--series", required=True, help="Kalshi series ticker (INX/INXU/NASDAQ100/NASDAQ100U)")
    parser.add_argument("--fixtures-root", default="tests/data_fixtures", help="Offline fixtures root directory")
    parser.add_argument(
        "--min-ev",
        type=float,
        default=float(INDEX_OPS_CONFIG.min_ev_usd),
        help="Minimum EV_after_fees per contract (USD)",
    )
    parser.add_argument("--offline", action="store_true", help="Use offline fixtures for driver data")
    parser.add_argument("--broker", default="dry", choices=["dry", "live"], help="Execution broker adapter")
    parser.add_argument("--kill-switch-file", help="Override kill-switch sentinel path")
    parser.add_argument("--contracts", type=int, default=1, help="Contracts per quote (default 1)")
    parser.add_argument("--regenerate-scoreboard", action="store_true", help="Regenerate index scoreboard after run")
    parser.add_argument("--quiet", action="store_true", help="Suppress scan_ladders stdout summary")
    parser.add_argument(
        "--quality-gates-config",
        type=Path,
        help="Override quality gates configuration file passed to scan_ladders.",
    )
    parser.add_argument("--now", type=str, help="Override timestamp for logging (ISO-8601)")
    return parser.parse_args(list(argv) if argv is not None else None)


def _build_scan_args(args: argparse.Namespace) -> list[str]:
    scan_args: list[str] = [
        "--series",
        args.series,
        "--min-ev",
        f"{float(args.min_ev):.4f}",
        "--contracts",
        str(max(int(args.contracts), 1)),
        "--maker-only",
        "--pilot",
        "--paper-ledger",
        "--report",
        "--fixtures-root",
        str(Path(args.fixtures_root)),
        "--broker",
        args.broker,
    ]
    if args.offline:
        scan_args.append("--offline")
    else:
        scan_args.append("--online")
    if args.kill_switch_file:
        scan_args.extend(["--kill-switch-file", args.kill_switch_file])
    if args.quality_gates_config:
        scan_args.extend(["--quality-gates-config", str(Path(args.quality_gates_config))])
    if args.broker == "live":
        scan_args.append("--i-understand-the-risks")
    if args.quiet:
        scan_args.append("--quiet")
    return scan_args


def _log_ops_window(series: str, *, reference: datetime | None, quiet: bool) -> None:
    if quiet:
        return
    try:
        window = INDEX_OPS_CONFIG.window_for_series(series)
    except KeyError:
        return
    base = reference if reference is not None else datetime.now(tz=UTC)
    local_reference = base.astimezone(INDEX_OPS_CONFIG.timezone)
    target_time = None
    if window.start_offset_minutes is not None:
        target_time = _default_hourly_target(base)
    start_local, end_local = window.bounds_for(
        reference=local_reference,
        target_time=target_time,
        timezone=INDEX_OPS_CONFIG.timezone,
    )
    cancel_buffer = float(window.cancel_buffer_seconds)
    cancel_by = end_local - timedelta(seconds=cancel_buffer)
    print(
        "[microlive] ops window "
        f"{window.name}: {start_local.isoformat()} -> {end_local.isoformat()} "
        f"(cancel by {cancel_by.isoformat()}, buffer={cancel_buffer:.0f}s)"
    )


def _refit_execution_curves(series: str) -> None:
    family = canonical_series_family(series)
    tuned_alpha = tune_alpha(family, RAW_ROOT / "kalshi")
    if tuned_alpha is not None:
        print(f"[microlive] tuned fill alpha for {family}: {tuned_alpha:.3f}")
    calibration = fit_slippage(family)
    if calibration is not None:
        print(
            f"[microlive] fitted slippage curve for {family}: impact_cap={calibration.impact_cap:.4f}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    scan_args = _build_scan_args(args)
    timestamp = None
    if args.now:
        try:
            parsed = datetime.fromisoformat(args.now)
            timestamp = parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            raise SystemExit("--now must be ISO-8601 format, e.g. 2025-11-04T15:20:00+00:00") from None
    print(f"[microlive] starting scan_ladders with args: {' '.join(scan_args)}")
    _log_ops_window(args.series.upper(), reference=timestamp, quiet=args.quiet)
    scan_ladders.main(scan_args)
    _refit_execution_curves(args.series)
    if args.regenerate_scoreboard:
        print("[microlive] regenerating scoreboard and pilot readiness reports")
        scoreboard.main([])
    print("[microlive] complete")


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    main()
