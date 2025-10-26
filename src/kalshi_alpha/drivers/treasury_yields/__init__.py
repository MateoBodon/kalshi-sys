"""Treasury par yield driver."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dateutil import parser as date_parser


@dataclass(frozen=True)
class ParYield:
    as_of: date
    maturity: str
    rate: float


def load_daily_yields(*, offline_path: Path | None = None) -> list[ParYield]:
    if offline_path is None:
        raise RuntimeError("Offline mode requires a CSV fixture")
    yields: list[ParYield] = []
    with offline_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            as_of = _to_date(row["Date"])
            for maturity, value in row.items():
                if maturity == "Date" or not value:
                    continue
                yields.append(ParYield(as_of=as_of, maturity=maturity, rate=float(value)))
    return yields


def mirror_dgs10(yields: list[ParYield]) -> ParYield | None:
    for entry in yields:
        if entry.maturity.upper() in {"10 YR", "DGS10"}:
            return entry
    return None


def _to_date(value: str) -> date:
    parsed = date_parser.parse(value)
    return date(parsed.year, parsed.month, parsed.day)
