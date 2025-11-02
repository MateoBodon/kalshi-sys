"""Utility to bundle telemetry JSONL into artifacts for shipping."""

from __future__ import annotations

import argparse
import gzip
import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

TELEMETRY_ROOT = Path("data/raw/kalshi")
ARTIFACTS_ROOT = Path("reports/_artifacts/telemetry")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package telemetry JSONL files for shipping.")
    parser.add_argument(
        "--source",
        type=Path,
        default=TELEMETRY_ROOT,
        help="Telemetry root directory. Defaults to data/raw/kalshi.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=ARTIFACTS_ROOT,
        help="Output directory for packaged telemetry. Defaults to reports/_artifacts/telemetry.",
    )
    parser.add_argument(
        "--day",
        type=str,
        default=None,
        help="ISO date (YYYY-MM-DD) to export or 'yesterday'. Defaults to today.",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Write gzip compressed archives (default: off).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    now = datetime.now(tz=UTC)
    target_day = _resolve_day(args.day, now)
    source_file = _telemetry_path(args.source, target_day)
    if not source_file.exists():
        print(f"[telemetry] no telemetry found for {target_day.isoformat()} at {source_file}")
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    if args.compress:
        output = args.dest / f"{target_day.isoformat()}.exec.jsonl.gz"
        with source_file.open("rb") as src, gzip.open(output, "wb") as dst:
            shutil.copyfileobj(src, dst)
        print(f"[telemetry] wrote {output}")
    else:
        output = args.dest / f"{target_day.isoformat()}.exec.jsonl"
        shutil.copy2(source_file, output)
        print(f"[telemetry] copied {output}")
    return 0


def _resolve_day(day: str | None, now: datetime) -> date:
    if day is None:
        return now.date()
    lowered = day.strip().lower()
    if lowered == "yesterday":
        return (now - timedelta(days=1)).date()
    return date.fromisoformat(day)


def _telemetry_path(root: Path, day: date) -> Path:
    return root / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}" / "exec.jsonl"


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
