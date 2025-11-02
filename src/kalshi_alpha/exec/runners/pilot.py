"""Single-entry CLI wrapper for pilot ladder sessions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from kalshi_alpha.exec.runners import scan_ladders


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a constrained live pilot session for a single ladder family.",
    )
    parser.add_argument(
        "--series",
        required=True,
        help="Kalshi series ticker (e.g. CPI).",
    )
    parser.add_argument(
        "--pilot-config",
        type=Path,
        help="Optional pilot configuration override (default: configs/pilot.yaml).",
    )
    parser.add_argument(
        "--fixtures-root",
        default="tests/data_fixtures",
        help="Offline fixtures root (used for supplemental data only).",
    )
    parser.add_argument(
        "--broker",
        default="live",
        choices=["live", "dry"],
        help="Broker adaptor for execution (default live).",
    )
    parser.add_argument(
        "--kill-switch-file",
        help="Override kill-switch sentinel path.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Write pilot markdown report for the scan.",
    )
    parser.add_argument(
        "--paper-ledger",
        action="store_true",
        help="Simulate fills via paper ledger for diagnostics.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout summary from the underlying scanner.",
    )
    return parser.parse_args(argv)


def _build_forward_args(config: argparse.Namespace) -> list[str]:
    forwarded: list[str] = [
        "--series",
        config.series,
        "--pilot",
        "--fixtures-root",
        str(config.fixtures_root),
        "--broker",
        config.broker,
    ]
    if config.pilot_config:
        forwarded.extend(["--pilot-config", str(config.pilot_config)])
    if config.kill_switch_file:
        forwarded.extend(["--kill-switch-file", str(config.kill_switch_file)])
    if config.report:
        forwarded.append("--report")
    if config.paper_ledger:
        forwarded.append("--paper-ledger")
    if config.quiet:
        forwarded.append("--quiet")
    if config.broker == "live":
        forwarded.append("--i-understand-the-risks")
    return forwarded


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    forwarded = _build_forward_args(args)
    scan_ladders.main(forwarded)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution
    raise SystemExit(main())
