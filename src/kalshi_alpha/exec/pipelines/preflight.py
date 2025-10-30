"""Lightweight preflight helper for daily pipeline windows."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from types import SimpleNamespace

from kalshi_alpha.exec.pipelines.calendar import ET, resolve_run_window
from kalshi_alpha.exec.pipelines.daily import (
    _compute_next_window,
    _evaluate_quality_gates,
    _format_window_line,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect upcoming scan window and gate status.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["pre_cpi", "pre_claims", "teny_close", "weather_cycle"],
        help="Pipeline mode to query.",
    )
    parser.add_argument(
        "--broker",
        choices=["dry", "live"],
        default="dry",
        help="Broker context for gate evaluation (affects kill-switch warnings).",
    )
    parser.add_argument(
        "--daily-loss-cap",
        type=float,
        help="Optional daily loss cap used in gate evaluation.",
    )
    parser.add_argument(
        "--weekly-loss-cap",
        type=float,
        help="Optional weekly loss cap used in gate evaluation.",
    )
    parser.add_argument(
        "--kill-switch-file",
        help="Optional kill-switch path to inspect.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    now_utc = datetime.now(tz=UTC)
    now_et = now_utc.astimezone(ET)

    target_date = now_et.date()
    run_window = resolve_run_window(mode=args.mode, target_date=target_date, now=now_utc)
    scan_allowed = run_window.scan_allowed(now_utc)

    next_date, next_window = _compute_next_window(mode=args.mode, start_date=target_date, now_utc=now_utc)

    print(f"[preflight] Mode: {args.mode}")
    print(f"[preflight] Now (ET): {now_et.isoformat()}")
    print(f"[preflight] Scan allowed now: {scan_allowed}")
    print(_format_window_line("[preflight] Current window (ET)", run_window, include_notes=True))
    print(_format_window_line("[preflight] Next window  (ET)", next_window, include_notes=True))

    gate_args = SimpleNamespace(
        daily_loss_cap=args.daily_loss_cap,
        weekly_loss_cap=args.weekly_loss_cap,
        kill_switch_file=args.kill_switch_file,
        broker=args.broker,
    )
    result, _ = _evaluate_quality_gates(
        gate_args,
        now_utc,
        monitors={"preflight": True},
        apply_side_effects=False,
    )

    if result.go:
        print("[preflight] Quality gates: GO")
    else:
        print("[preflight] Quality gates: NO-GO")
    if result.reasons:
        joined = ", ".join(result.reasons)
        print(f"[preflight] Reasons: {joined}")
    if result.details:
        print(f"[preflight] Details: {result.details}")


if __name__ == "__main__":
    main()
