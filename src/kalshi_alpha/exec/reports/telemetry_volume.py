"""Generate ops telemetry volume report for bounded TOB + quote-intent streams."""

from __future__ import annotations

import argparse
import gzip
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from kalshi_alpha.exec.collectors.tob_logger import (
    DEFAULT_INTENT_MAX_BYTES,
    DEFAULT_TOB_MAX_BYTES,
    DEFAULT_TOB_WINDOW_MAX_BYTES,
)
from kalshi_alpha.exec.housekeep import DEFAULT_KEEP_DAYS
from kalshi_alpha.exec.telemetry.sink import REQUIRED_TELEMETRY_FIELDS


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize telemetry file volumes for a run_id.")
    parser.add_argument("--run-id", required=True, help="Telemetry run identifier.")
    parser.add_argument(
        "--telemetry-root",
        type=Path,
        default=Path("data/proc/telemetry"),
        help="Root directory for telemetry streams (default: data/proc/telemetry).",
    )
    parser.add_argument(
        "--report-date",
        type=_parse_date,
        default=_utc_today(),
        help="Report date (YYYY-MM-DD, default: today).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/ops"),
        help="Directory for ops reports (default: reports/ops).",
    )
    return parser.parse_args(argv)


def _parse_date(value: str) -> date:
    return datetime.fromisoformat(value).date()


def _utc_today() -> date:
    return datetime.now(tz=UTC).date()


def _stream_path(root: Path, stream: str, run_id: str) -> Path:
    return root / stream / f"{run_id}.jsonl.gz"


def _count_lines(path: Path) -> tuple[int, int, int, int]:
    lines = 0
    max_line_bytes = 0
    missing_required = 0
    invalid_json = 0
    required = tuple(REQUIRED_TELEMETRY_FIELDS)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            lines += 1
            line_bytes = len(line.encode("utf-8"))
            if line_bytes > max_line_bytes:
                max_line_bytes = line_bytes
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            if not isinstance(payload, dict):
                invalid_json += 1
                continue
            if any(not payload.get(field) for field in required):
                missing_required += 1
    return lines, max_line_bytes, missing_required, invalid_json


def _format_stream(
    *,
    name: str,
    path: Path,
    stats: tuple[int, int, int, int] | None,
) -> list[str]:
    if not path.exists():
        return [f"- {name}: MISSING ({path.as_posix()})"]
    size_bytes = path.stat().st_size
    lines, max_line_bytes, missing_required, invalid_json = stats or (0, 0, 0, 0)
    return [
        f"- {name}:",
        f"  - path: {path.as_posix()}",
        f"  - gzip_bytes: {size_bytes}",
        f"  - jsonl_lines: {lines}",
        f"  - max_line_bytes: {max_line_bytes}",
        f"  - missing_required_fields: {missing_required}",
        f"  - invalid_json_lines: {invalid_json}",
    ]


def main(argv: list[str] | None = None) -> Path:
    args = _parse_args(argv)
    run_id = str(args.run_id)
    telemetry_root = Path(args.telemetry_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_date = args.report_date

    streams = {
        "tob": _stream_path(telemetry_root, "tob", run_id),
        "quote_intents": _stream_path(telemetry_root, "quote_intents", run_id),
    }

    stats: dict[str, tuple[int, int, int, int] | None] = {}
    for name, path in streams.items():
        if path.exists():
            stats[name] = _count_lines(path)
        else:
            stats[name] = None

    report_path = output_dir / f"telemetry_volume_{report_date.isoformat()}.md"
    lines: list[str] = [
        f"# Telemetry Volume Report — {report_date.isoformat()}",
        "",
        f"- Generated at (UTC): {datetime.now(tz=UTC).isoformat()}",
        f"- Run ID: {run_id}",
        f"- Telemetry root: {telemetry_root.as_posix()}",
        "",
        "## Files",
    ]
    for name, path in streams.items():
        lines.extend(_format_stream(name=name, path=path, stats=stats.get(name)))
    lines.extend(
        [
            "",
            "## Caps + Retention",
            f"- Max bytes per window (per stream): {int(DEFAULT_TOB_WINDOW_MAX_BYTES)}",
            f"- Per-record cap (TOB): {int(DEFAULT_TOB_MAX_BYTES)}",
            f"- Per-record cap (quote intent): {int(DEFAULT_INTENT_MAX_BYTES)}",
            f"- Retention days: {int(DEFAULT_KEEP_DAYS)}",
            f"- Prune command: python -m kalshi_alpha.exec.housekeep --keep-days {int(DEFAULT_KEEP_DAYS)}",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[telemetry_volume] wrote {report_path}")
    return report_path


if __name__ == "__main__":  # pragma: no cover
    main()
