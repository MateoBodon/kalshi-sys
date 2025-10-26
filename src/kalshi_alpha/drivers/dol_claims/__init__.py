"""DOL ETA-539 weekly jobless claims driver."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dateutil import parser as date_parser


@dataclass(frozen=True)
class ClaimsReport:
    week_ending: date
    initial_claims: int
    insured_unemployment: int | None = None


def load_latest_report(*, offline_path: Path | None = None) -> ClaimsReport:
    if offline_path is None:
        raise RuntimeError("Offline mode requires a CSV or JSON fixture")
    if offline_path.suffix.lower() == ".csv":
        return _load_csv(offline_path)
    raise ValueError(f"Unsupported claims fixture type: {offline_path.suffix}")


def _load_csv(path: Path) -> ClaimsReport:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        latest_row = next(reader)
        week = _to_date(latest_row["Week Ending"])
        initial = int(latest_row["Initial Claims"].replace(",", ""))
        insured = latest_row.get("Insured Unemployment")
        insured_val = int(insured.replace(",", "")) if insured else None
    return ClaimsReport(week_ending=week, initial_claims=initial, insured_unemployment=insured_val)


def _to_date(value: str) -> date:
    parsed = date_parser.parse(value)
    return date(parsed.year, parsed.month, parsed.day)
