"""Guarded microlive harness for TENY pilot operations."""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ET_ZONE = ZoneInfo("America/New_York")


def resolve_tney_ticker(now: datetime | None = None) -> str:
    moment = now or datetime.now(tz=ET_ZONE)
    localized = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    return f"TNEY-{localized.astimezone(ET_ZONE).date():%Y%m%d}"


def build_steps(python: str, *, ticker: str | None = None) -> list[list[str]]:
    target_ticker = ticker or resolve_tney_ticker()
    return [
        [python, "-m", "kalshi_alpha.datastore.ingest", "--all", "--online"],
        [python, "-m", "kalshi_alpha.drivers.macro_calendar.cli", "--days", "30"],
        [python, "-m", "kalshi_alpha.dev.ws_smoke", "--tickers", target_ticker, "--run-seconds", "600"],
        [
            python,
            "-m",
            "kalshi_alpha.exec.pipelines.daily",
            "--mode",
            "teny_close",
            "--online",
            "--report",
            "--paper-ledger",
            "--ev-honesty-shrink",
            "0.9",
        ],
        [
            python,
            "-m",
            "kalshi_alpha.exec.scoreboard",
            "--window",
            "7",
            "--window",
            "30",
        ],
    ]


def _latest_report(report_dir: Path) -> Path | None:
    if not report_dir.exists():
        return None
    candidates = sorted(report_dir.glob("*.md"))
    return candidates[-1] if candidates else None


def run_pipeline(python: str, *, root: Path = ROOT) -> Path | None:
    commands = build_steps(python)
    for command in commands:
        print(f"[pilot] running: {' '.join(command)}")
        subprocess.run(command, check=True, cwd=root)  # noqa: S603

    reports_dir = root / "reports"
    teny_report = _latest_report(reports_dir / "TNEY")
    artifact_path = reports_dir / "_artifacts" / "go_no_go.json"
    scoreboard_paths = [
        reports_dir / "scoreboard_7d.md",
        reports_dir / "scoreboard_30d.md",
        reports_dir / "pilot_readiness.md",
    ]

    if teny_report and teny_report.exists():
        print(f"[pilot] TENY report: {teny_report}")
    if artifact_path.exists():
        print(f"[pilot] GO/NO-GO artifact: {artifact_path}")
    for path in scoreboard_paths:
        if path.exists():
            print(f"[pilot] Scoreboard artifact: {path}")

    return teny_report


def main(argv: list[str] | None = None) -> Path | None:
    return run_pipeline(sys.executable)


if __name__ == "__main__":  # pragma: no cover
    main()
