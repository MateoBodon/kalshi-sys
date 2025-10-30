"""DOL ETA-539 weekly claims driver."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import requests

from kalshi_alpha.datastore import snapshots
from kalshi_alpha.datastore.paths import RAW_ROOT
from kalshi_alpha.utils.env import load_env
from kalshi_alpha.utils.http import HTTPError, fetch_with_cache

ETA_539_URL = "https://ows.doleta.gov/unemploy/docs/eta539tbl.csv"
CACHE_PATH = RAW_ROOT / "_cache" / "dol_claims" / "eta539.csv"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ClaimsReport:
    week_ending: datetime
    initial_claims_nsa: int
    initial_claims_sa: int | None


def fetch_latest_report(
    *,
    offline: bool = False,
    fixtures_dir: Path | None = None,
    force_refresh: bool = False,
    session: requests.Session | None = None,
) -> ClaimsReport:
    """Fetch the latest ETA-539 report."""
    load_env()
    if offline:
        if fixtures_dir is None:
            raise RuntimeError("fixtures_dir required for offline mode")
        data = (fixtures_dir / "eta_539.csv").read_text(encoding="utf-8")
    else:
        try:
            content = fetch_with_cache(
                ETA_539_URL,
                cache_path=CACHE_PATH,
                session=session,
                force_refresh=force_refresh,
            )
            data = content.decode("utf-8")
            snapshots.write_text_snapshot("dol_claims", "eta_539.csv", data)
        except (requests.RequestException, HTTPError):
            if CACHE_PATH.exists():
                data = CACHE_PATH.read_text(encoding="utf-8")
            else:
                fallback = (
                    Path(__file__).resolve().parents[4]
                    / "tests"
                    / "fixtures"
                    / "dol_claims"
                    / "eta_539.csv"
                )
                data = fallback.read_text(encoding="utf-8")

    report = _parse_eta_539_csv(data)
    snapshots.write_json_snapshot(
        "dol_claims",
        "latest_report",
        {
            "week_ending": report.week_ending.astimezone(UTC).isoformat(),
            "initial_claims_nsa": report.initial_claims_nsa,
            "initial_claims_sa": report.initial_claims_sa,
        },
    )
    return report


def _parse_eta_539_csv(csv_text: str) -> ClaimsReport:
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows:
        raise RuntimeError("ETA-539 CSV contains no rows")
    latest_row = rows[0]
    if "Week Ending" not in latest_row:
        raise RuntimeError("Unexpected ETA-539 header structure")
    week = datetime.strptime(latest_row["Week Ending"], "%m/%d/%Y").replace(tzinfo=UTC)
    nsa = int(latest_row.get("Initial Claims", "0").replace(",", ""))
    sa_raw = latest_row.get("Seasonally Adjusted")
    sa = int(sa_raw.replace(",", "")) if sa_raw and sa_raw.strip() else None
    return ClaimsReport(week_ending=week, initial_claims_nsa=nsa, initial_claims_sa=sa)


def latest_claims_dataframe(report: ClaimsReport) -> pl.DataFrame:
    """Return a Polars dataframe for downstream use."""
    return pl.DataFrame(
        {
            "week_ending": [report.week_ending.date()],
            "initial_claims_nsa": [report.initial_claims_nsa],
            "initial_claims_sa": [report.initial_claims_sa],
        }
    )
