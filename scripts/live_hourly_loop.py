#!/usr/bin/env python3
"""Continuous runner for INX/NDX hourly scans with multiple parameter sets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import UTC, datetime
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo


ET_ZONE = ZoneInfo("America/New_York")


def _next_target_hour(now_utc: datetime) -> int:
    now_et = now_utc.astimezone(ET_ZONE)
    hour = now_et.hour
    if now_et.minute >= 55:
        hour = (hour + 1) % 24
    return hour


def _run_commands(commands: Sequence[Sequence[str]], target_hour: int, dry: bool = False) -> None:
    env = dict(os.environ)
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    existing = env.get("PYTHONPATH", "")
    parts = [str(src_path)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)

    for base_cmd in commands:
        cmd = [*base_cmd, "--target-hour", str(target_hour)]
        print(f"[live-loop] running: {' '.join(cmd)}", flush=True)
        if dry:
            continue
        try:
            subprocess.run(cmd, check=False, env=env)
        except KeyboardInterrupt:  # pragma: no cover - handled by outer loop
            raise
        except Exception as exc:  # pragma: no cover - best-effort logging
            print(f"[live-loop] ERROR executing {' '.join(cmd)} -> {exc}", file=sys.stderr, flush=True)


def _base_command(
    *,
    series: str,
    contracts: int,
    impact_cap: float,
    pilot_config: str | None = None,
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        "-m",
        "kalshi_alpha.exec.runners.scan_ladders",
        "--series",
        series,
        "--online",
        "--pilot",
        "--contracts",
        str(max(1, contracts)),
        "--min-ev",
        "0.0",
        "--maker-only",
        "--impact-cap",
        f"{impact_cap:.4f}",
        "--report",
        "--paper-ledger",
        "--broker",
        "live",
        "--daily-loss-cap",
        "50",
        "--weekly-loss-cap",
        "250",
        "--i-understand-the-risks",
        "--force-gate-pass",
    ]
    if pilot_config:
        cmd.extend(["--pilot-config", pilot_config])
    return cmd


def build_hourly_command_sets() -> list[list[str]]:
    """Return the default list of scanner invocations per loop."""

    return [
        _base_command(series="INXU", contracts=1, impact_cap=0.05, pilot_config="configs/pilot_bins4.yaml"),
        _base_command(series="INXU", contracts=2, impact_cap=0.02),
        _base_command(series="NASDAQ100U", contracts=1, impact_cap=0.05, pilot_config="configs/pilot_bins4.yaml"),
        _base_command(series="NASDAQ100U", contracts=2, impact_cap=0.02),
    ]


def build_close_command_sets() -> list[list[str]]:
    """Return commands specifically for the 16:00 ET close window."""

    base_args: list[list[str]] = [
        _base_command(series="INX", contracts=2, impact_cap=0.02),
        _base_command(series="NASDAQ100", contracts=2, impact_cap=0.02),
    ]
    return base_args


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Continuously run hourly INX/NDX live scans.")
    parser.add_argument("--interval", type=float, default=180.0, help="Sleep seconds between loops (default: 180).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--once", action="store_true", help="Run a single loop (useful for testing) instead of continuous service."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    hourly_commands = build_hourly_command_sets()
    close_commands = build_close_command_sets()

    try:
        while True:
            now_utc = datetime.now(tz=UTC)
            target_hour = _next_target_hour(now_utc)
            _run_commands(hourly_commands, target_hour, dry=args.dry_run)

            if target_hour == 16:
                _run_commands(close_commands, target_hour, dry=args.dry_run)

            if args.once:
                break
            sleep_seconds = max(5.0, float(args.interval))
            time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        print("[live-loop] received interrupt, exiting.", flush=True)


if __name__ == "__main__":
    main()
