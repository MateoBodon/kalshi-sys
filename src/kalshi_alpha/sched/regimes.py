"""Trading-day regime flags for macro events (FOMC, CPI) with SLO overrides."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from kalshi_alpha.drivers import macro_calendar

DEFAULT_CALENDAR_PATH = macro_calendar.DEFAULT_OUTPUT
DEFAULT_LOOKBACK_DAYS = 90
SLO_MULTIPLIERS = {
    "fomc": 0.6,
    "cpi": 0.75,
}
SIZE_MULTIPLIERS = {
    "fomc": 0.5,
    "cpi": 0.8,
}


@dataclass(frozen=True)
class RegimeFlags:
    date: date
    is_fomc: bool
    is_cpi: bool
    slo_multiplier: float
    size_multiplier: float
    label: str


def regime_for(moment: datetime | None = None, *, calendar_path: Path | None = None) -> RegimeFlags:
    reference_dt = moment or datetime.now(tz=UTC)
    reference_date = reference_dt.date()
    frame = _load_calendar(calendar_path)
    if "date" not in frame.columns:
        return RegimeFlags(reference_date, False, False, 1.0, 1.0, "baseline")
    row = frame.filter(pl.col("date") == reference_date)
    if row.is_empty():
        return RegimeFlags(reference_date, False, False, 1.0, 1.0, "baseline")
    is_fomc = bool(row[0, "is_fomc"]) if "is_fomc" in row.columns else False
    is_cpi = bool(row[0, "is_cpi"]) if "is_cpi" in row.columns else False
    slo_multiplier = 1.0
    size_multiplier = 1.0
    labels: list[str] = []
    if is_fomc:
        slo_multiplier = min(slo_multiplier, SLO_MULTIPLIERS["fomc"])
        size_multiplier = min(size_multiplier, SIZE_MULTIPLIERS["fomc"])
        labels.append("fomc")
    if is_cpi:
        slo_multiplier = min(slo_multiplier, SLO_MULTIPLIERS["cpi"])
        size_multiplier = min(size_multiplier, SIZE_MULTIPLIERS["cpi"])
        labels.append("cpi")
    label = ",".join(labels) if labels else "baseline"
    return RegimeFlags(
        date=reference_date,
        is_fomc=is_fomc,
        is_cpi=is_cpi,
        slo_multiplier=slo_multiplier,
        size_multiplier=size_multiplier,
        label=label,
    )


def _load_calendar(path: Path | None) -> pl.DataFrame:
    target = Path(path) if path is not None else DEFAULT_CALENDAR_PATH
    if not target.exists():
        today = date.today()
        start = today - timedelta(days=DEFAULT_LOOKBACK_DAYS)
        end = today + timedelta(days=DEFAULT_LOOKBACK_DAYS)
        macro_calendar.emit_day_dummies(start, end, out_path=target)
    return pl.read_parquet(target)


__all__ = ["RegimeFlags", "regime_for"]
