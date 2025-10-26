"""BLS CPI driver with fixture-friendly loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

from dateutil import parser as date_parser


@dataclass(frozen=True)
class CPIRelease:
    release_date: date
    period: str  # e.g., "2025-09"
    seasonally_adjusted_mom: float


def load_latest_release(*, offline_path: Path | None = None) -> CPIRelease:
    """Load the latest CPI release from fixture or API."""
    if offline_path is None:
        raise RuntimeError("Offline mode requires a path to a fixture JSON file")
    payload = _read_json(offline_path)
    return CPIRelease(
        release_date=_to_date(payload["release_date"]),
        period=str(payload["period"]),
        seasonally_adjusted_mom=float(payload["seasonally_adjusted_mom"]),
    )


def load_release_calendar(*, offline_path: Path | None = None) -> list[date]:
    if offline_path is None:
        raise RuntimeError("Offline mode requires a calendar fixture")
    payload = _read_json(offline_path)
    return [_to_date(entry) for entry in payload["calendar"]]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return cast(dict[str, Any], json.load(handle))


def _to_date(value: str) -> date:
    parsed = date_parser.parse(value)
    return date(parsed.year, parsed.month, parsed.day)
