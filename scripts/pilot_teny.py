"""Guarded microlive harness for TENY pilot operations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_steps(python: str) -> list[list[str]]:
    return [
        [python, "-m", "kalshi_alpha.datastore.ingest", "--all", "--online"],
        [python, "-m", "kalshi_alpha.drivers.macro_calendar.cli", "--quiet"],
        [python, "-m", "kalshi_alpha.dev.imbalance_snap", "--tickers", "TNEY", "--quiet"],
        [
            python,
            "-m",
            "kalshi_alpha.exec.runners.scan_ladders",
            "--series",
            "TNEY",
            "--online",
            "--report",
            "--paper-ledger",
            "--maker-only",
            "--contracts",
            "1",
            "--max-legs",
            "2",
            "--min-ev",
            "0.05",
        ],
        [python, "-m", "kalshi_alpha.exec.scoreboard"],
        [python, "-m", "kalshi_alpha.exec.reports.ramp"],
    ]


def run_pipeline(python: str, *, root: Path = ROOT) -> Path:
    commands = build_steps(python)
    for command in commands:
        print(f"[pilot] running: {' '.join(command)}")
        subprocess.run(command, check=True, cwd=root)  # noqa: S603
    summary_path = root / "reports" / "pilot_ready.json"
    print(f"[pilot] Pilot summary available at {summary_path}")
    return summary_path


def main(argv: list[str] | None = None) -> Path:
    return run_pipeline(sys.executable)


if __name__ == "__main__":  # pragma: no cover
    main()
