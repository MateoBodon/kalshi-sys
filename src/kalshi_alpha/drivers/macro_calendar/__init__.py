"""Macro calendar driver producing release-day dummy variables."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from kalshi_alpha.drivers import bls_cpi

PROC_ROOT = Path("data/proc/macro_calendar")
DEFAULT_OUTPUT = PROC_ROOT / "macro_day_dummies.parquet"

FOMC_SCHEDULE: dict[int, Sequence[date]] = {
    2024: (
        date(2024, 1, 31),
        date(2024, 3, 20),
        date(2024, 5, 1),
        date(2024, 6, 12),
        date(2024, 7, 31),
        date(2024, 9, 18),
        date(2024, 11, 7),
        date(2024, 12, 18),
    ),
    2025: (
        date(2025, 1, 29),
        date(2025, 3, 19),
        date(2025, 5, 7),
        date(2025, 6, 18),
        date(2025, 7, 30),
        date(2025, 9, 17),
        date(2025, 10, 29),
        date(2025, 12, 10),
    ),
    2026: (
        date(2026, 1, 28),
        date(2026, 3, 18),
        date(2026, 4, 29),
        date(2026, 6, 17),
        date(2026, 7, 29),
        date(2026, 9, 16),
        date(2026, 10, 28),
        date(2026, 12, 9),
    ),
}


def emit_day_dummies(
    start: date | datetime,
    end: date | datetime,
    *,
    offline: bool = False,
    fixtures_dir: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    start_date = _coerce_date(start)
    end_date = _coerce_date(end)
    if start_date > end_date:
        raise ValueError("macro calendar start must be on or before end")

    all_dates = list(_date_range(start_date, end_date))
    cpi_dates = _cpi_release_dates(start_date, end_date, offline=offline, fixtures_dir=fixtures_dir)
    fomc_dates = _fomc_release_dates(start_date, end_date, fixtures_dir=fixtures_dir)
    jobs_dates = _jobs_release_dates(start_date, end_date)
    claims_dates = _claims_release_dates(start_date, end_date)

    frame = pl.DataFrame({"date": all_dates}).with_columns(pl.col("date").cast(pl.Date))
    frame = frame.with_columns(
        pl.col("date").is_in(sorted(fomc_dates)).alias("is_fomc"),
        pl.col("date").is_in(sorted(cpi_dates)).alias("is_cpi"),
        pl.col("date").is_in(sorted(jobs_dates)).alias("is_jobs"),
        pl.col("date").is_in(sorted(claims_dates)).alias("is_claims"),
    )

    target = output_path or DEFAULT_OUTPUT
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target)
    return target


def _coerce_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"Unsupported date value: {value!r}")


def _date_range(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _cpi_release_dates(
    start: date,
    end: date,
    *,
    offline: bool,
    fixtures_dir: Path | None,
) -> set[date]:
    fixtures = fixtures_dir.joinpath("bls_cpi") if fixtures_dir else None
    try:
        calendar = bls_cpi.fetch_release_calendar(
            offline=offline,
            fixtures_dir=fixtures,
        )
    except Exception:
        return set()
    results: set[date] = set()
    for release_dt in calendar:
        release_date = release_dt.date()
        if start <= release_date <= end:
            results.add(release_date)
    return results


def _fomc_release_dates(start: date, end: date, fixtures_dir: Path | None) -> set[date]:
    schedule: set[date] = set()
    if fixtures_dir is not None:
        fixture_path = fixtures_dir / "macro_calendar" / "fomc_dates.json"
        if fixture_path.exists():
            try:
                payload = json.loads(fixture_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = None
            items: Sequence[object] = []
            if isinstance(payload, dict):
                candidate = payload.get("dates")
                if isinstance(candidate, list):
                    items = candidate
            elif isinstance(payload, list):
                items = payload
            for item in items:
                parsed = _parse_date(item)
                if parsed is not None:
                    schedule.add(parsed)
    if not schedule:
        for year, dates in FOMC_SCHEDULE.items():
            if year < start.year - 1 or year > end.year + 1:
                continue
            for item in dates:
                schedule.add(item)
    return {day for day in schedule if start <= day <= end}


def _jobs_release_dates(start: date, end: date) -> set[date]:
    cursor = date(start.year, start.month, 1)
    results: set[date] = set()
    while cursor <= end:
        first_friday = _first_weekday(cursor.year, cursor.month, weekday=4)
        if start <= first_friday <= end:
            results.add(first_friday)
        cursor = _first_of_next_month(cursor)
    return results


def _claims_release_dates(start: date, end: date) -> set[date]:
    return {
        day
        for day in _date_range(start, end)
        if day.weekday() == 3  # Thursday
    }


def _parse_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def _first_weekday(year: int, month: int, *, weekday: int) -> date:
    candidate = date(year, month, 1)
    while candidate.weekday() != weekday:
        candidate += timedelta(days=1)
    return candidate


def _first_of_next_month(moment: date) -> date:
    if moment.month == 12:
        return date(moment.year + 1, 1, 1)
    return date(moment.year, moment.month + 1, 1)
