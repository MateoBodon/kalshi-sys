"""Command-line utility to refresh macro calendar day dummies."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from . import DEFAULT_OUTPUT, EVENT_COLUMNS, emit_day_dummies

DEFAULT_FIXTURES_ROOT = Path("tests/fixtures")
FIXTURE_FILENAME = "history.json"


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - argparse validation
        msg = f"Invalid date {value!r}; expected YYYY-MM-DD"
        raise argparse.ArgumentTypeError(msg) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate macro calendar day dummies.")
    parser.add_argument("--offline", action="store_true", help="Use offline fixtures instead of live sources.")
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=DEFAULT_FIXTURES_ROOT,
        help="Fixture root directory when running offline.",
    )
    parser.add_argument("--output", type=Path, help="Output parquet path for macro dummies.")
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of trailing days (inclusive) to generate.",
    )
    parser.add_argument(
        "--as-of",
        type=_parse_date,
        help="Reference date in YYYY-MM-DD format (defaults to today).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress stdout status message.")
    return parser


def _date_range(start: date, end: date) -> list[date]:
    cursor = start
    dates: list[date] = []
    while cursor <= end:
        dates.append(cursor)
        cursor += timedelta(days=1)
    return dates


def _resolve_fixture_records(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, dict):
        candidate = payload.get("history")
        if isinstance(candidate, list):
            return [entry for entry in candidate if isinstance(entry, dict)]
        return []
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    return []


def _load_fixture(path: Path) -> dict[str, dict[str, bool]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    records = _resolve_fixture_records(payload)
    mapping: dict[str, dict[str, bool]] = {}
    for record in records:
        raw_date = record.get("date")
        if not isinstance(raw_date, str):
            continue
        try:
            iso_date = date.fromisoformat(raw_date).isoformat()
        except ValueError:
            continue
        entry = mapping.setdefault(iso_date, {column: False for column in EVENT_COLUMNS})
        for column in EVENT_COLUMNS:
            if column in record:
                entry[column] = bool(record[column])
        events_field = record.get("events")
        if isinstance(events_field, list):
            for label in events_field:
                if isinstance(label, str):
                    normalized = f"is_{label.lower()}"
                    if normalized in entry:
                        entry[normalized] = True
    return mapping


def _frame_from_history(start: date, end: date, mapping: dict[str, dict[str, bool]]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for day in _date_range(start, end):
        record = {"date": day}
        for column in EVENT_COLUMNS:
            record[column] = bool(mapping.get(day.isoformat(), {}).get(column, False))
        rows.append(record)
    frame = pl.DataFrame(rows)
    if not frame.is_empty():
        frame = frame.with_columns(pl.col("date").cast(pl.Date))
    return frame


def main(argv: list[str] | None = None) -> Path:
    args = _build_parser().parse_args(argv)
    as_of = args.as_of or datetime.now(tz=UTC).date()
    days = max(1, int(args.days))
    start = as_of - timedelta(days=days - 1)
    output = Path(args.output) if args.output is not None else DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)

    fixtures_dir: Path | None = None
    if args.offline:
        fixtures_dir = args.fixtures_root
        fixture_path = fixtures_dir / "macro_calendar" / FIXTURE_FILENAME
        mapping = _load_fixture(fixture_path)
        if mapping:
            frame = _frame_from_history(start, as_of, mapping)
            if frame.is_empty():
                frame = _frame_from_history(start, as_of, {})
            frame.write_parquet(output)
            if not args.quiet:
                print(f"Wrote macro calendar dummies to {output}")
            return output

    try:
        result = emit_day_dummies(
            start,
            as_of,
            out_path=output,
            offline=bool(args.offline),
            fixtures_dir=fixtures_dir,
        )
    except Exception:
        frame = _frame_from_history(start, as_of, {})
        frame.write_parquet(output)
        result = output
    if not args.quiet:
        print(f"Wrote macro calendar dummies to {result}")
    return result


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
