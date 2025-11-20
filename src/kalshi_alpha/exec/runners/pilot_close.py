"""Entry point for maker-only close index pilot sessions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from kalshi_alpha.exec.runners import scan_ladders
from kalshi_alpha.exec.window_guard import guard_series_window, parse_now_override

DEFAULT_SERIES = ("INX", "NASDAQ100")
DEFAULT_MIN_EV = 0.05
DEFAULT_KILL_SWITCH = Path("data/proc/state/pilot_close.kill")
DEFAULT_FIXTURES_ROOT = Path("tests/data_fixtures")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a constrained pilot session for close index ladders.",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=list(DEFAULT_SERIES),
        help="Index close series to pilot (default: INX NASDAQ100).",
    )
    parser.add_argument(
        "--broker",
        choices=["live", "dry"],
        default="dry",
        help="Execution broker (default dry for paper verification).",
    )
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=DEFAULT_FIXTURES_ROOT,
        help="Offline fixtures root for supplemental data.",
    )
    parser.add_argument(
        "--min-ev",
        type=float,
        default=DEFAULT_MIN_EV,
        help="Minimum maker EV per contract in USD (default 0.05).",
    )
    parser.add_argument(
        "--kill-switch-file",
        type=Path,
        default=DEFAULT_KILL_SWITCH,
        help="Kill-switch sentinel path (default data/proc/state/pilot_close.kill).",
    )
    parser.add_argument(
        "--pilot-config",
        type=Path,
        help="Override pilot configuration YAML (defaults to configs/pilot.yaml).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write markdown report for each series scan.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output from scan.ladders.",
    )
    parser.add_argument(
        "--ack",
        action="store_true",
        help="Acknowledge live trading risks (required when --broker live).",
    )
    parser.add_argument(
        "--now",
        help="Override current timestamp (ISO-8601) for window gating (default: now).",
    )
    return parser.parse_args(argv)


def _forward_args(series: str, config: argparse.Namespace) -> list[str]:
    forwarded: list[str] = [
        "--series",
        series.upper(),
        "--pilot",
        "--min-ev",
        f"{config.min_ev:.2f}",
        "--contracts",
        "1",
        "--maker-only",
        "--max-legs",
        "2",
        "--fixtures-root",
        str(config.fixtures_root),
        "--kill-switch-file",
        str(config.kill_switch_file),
        "--broker",
        config.broker,
    ]
    if config.pilot_config:
        forwarded.extend(["--pilot-config", str(config.pilot_config)])
    if config.broker == "dry":
        forwarded.append("--paper-ledger")
    if config.report:
        forwarded.append("--report")
    if config.quiet:
        forwarded.append("--quiet")
    if config.broker == "live":
        if not config.ack:
            raise ValueError("Live pilot mode requires --ack acknowledgement flag.")
        forwarded.append("--i-understand-the-risks")
    return forwarded


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    now_override = parse_now_override(args.now)
    series_list = [value.strip().upper() for value in args.series if value.strip()]
    if not series_list:
        raise ValueError("No series specified for pilot_close.")
    ran = False
    for series in series_list:
        allowed, _, next_window = guard_series_window(series, now=now_override, quiet=args.quiet)
        if not allowed:
            if not args.quiet:
                suffix = (
                    f"; next={next_window.start_et.isoformat()}->{next_window.target_et.isoformat()}"
                    if next_window
                    else ""
                )
                print(f"[pilot_close] window closed for {series}{suffix}")
            continue
        forwarded = _forward_args(series, args)
        scan_ladders.main(forwarded)
        ran = True
    if not ran:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
