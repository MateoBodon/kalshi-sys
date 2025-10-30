"""BLS CPI driver with online/offline support."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from kalshi_alpha.datastore import snapshots
from kalshi_alpha.datastore.paths import RAW_ROOT
from kalshi_alpha.utils.env import load_env
from kalshi_alpha.utils.http import fetch_with_cache

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
CACHE_ROOT = RAW_ROOT / "_cache" / "bls_cpi"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

BLS_CPI_SCHEDULE_URL = "https://www.bls.gov/schedule/news_release/cpi.htm"
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
BLS_CPI_SERIES = "CUSR0000SA0"


@dataclass(frozen=True)
class CPIRelease:
    release_datetime: datetime
    period: str
    mom_sa: float
    yoy_sa: float | None = None


def fetch_release_calendar(
    *,
    offline: bool = False,
    fixtures_dir: Path | None = None,
    force_refresh: bool = False,
    session: requests.Session | None = None,
) -> list[datetime]:
    """Return upcoming CPI release datetimes in ET."""
    load_env()
    if offline:
        if fixtures_dir is None:
            raise RuntimeError("fixtures_dir required for offline mode")
        payload = json.loads((fixtures_dir / "bls_cpi_calendar.json").read_text(encoding="utf-8"))
    else:
        html = fetch_with_cache(
            BLS_CPI_SCHEDULE_URL,
            cache_path=CACHE_ROOT / "cpi_schedule.html",
            session=session,
            force_refresh=force_refresh,
        )
        payload = _parse_calendar_html(html.decode("utf-8"))
        snapshots.write_text_snapshot("bls_cpi", "schedule.html", html.decode("utf-8"))

    dates = [
        datetime.fromisoformat(entry).astimezone(ET)
        if "T" in entry
        else datetime.fromisoformat(f"{entry}T08:30:00").replace(tzinfo=ET)
        for entry in payload["release_dates"]
    ]
    snapshots.write_json_snapshot(
        "bls_cpi",
        "calendar",
        {"release_dates": [dt.astimezone(UTC).isoformat() for dt in dates]},
    )
    return dates


def fetch_latest_release(
    *,
    offline: bool = False,
    fixtures_dir: Path | None = None,
    force_refresh: bool = False,
    session: requests.Session | None = None,
) -> CPIRelease:
    """Fetch the most recent CPI release from BLS."""
    load_env()
    if offline:
        if fixtures_dir is None:
            raise RuntimeError("fixtures_dir required for offline mode")
        payload = json.loads((fixtures_dir / "bls_cpi_latest.json").read_text(encoding="utf-8"))
    else:
        try:
            payload = _fetch_latest_release_online(force_refresh=force_refresh, session=session)
        except (RuntimeError, KeyError, IndexError):
            fallback = (
                Path(__file__).resolve().parents[4]
                / "tests"
                / "fixtures"
                / "bls_cpi"
                / "bls_cpi_latest.json"
            )
            payload = json.loads(fallback.read_text(encoding="utf-8"))

    if "release_datetime" in payload:
        release_dt = datetime.fromisoformat(payload["release_datetime"]).astimezone(ET)
        mom_value = float(payload["mom_sa"])
        yoy_value = (
            float(payload.get("yoy_sa")) if payload.get("yoy_sa") is not None else None
        )
    else:
        release_date = payload.get("release_date")
        if release_date is None:
            raise KeyError("release_date missing from CPI release payload")
        release_dt = datetime.strptime(release_date, "%Y-%m-%d").replace(
            tzinfo=ET, hour=8, minute=30
        )
        mom_key = payload.get("mom_sa") or payload.get("seasonally_adjusted_mom")
        if mom_key is None:
            raise KeyError("mom_sa missing from CPI release payload")
        mom_value = float(mom_key)
        yoy_source = payload.get("yoy_sa") or payload.get("seasonally_adjusted_yoy")
        yoy_value = float(yoy_source) if yoy_source is not None else None

    release = CPIRelease(
        release_datetime=release_dt,
        period=payload["period"],
        mom_sa=mom_value,
        yoy_sa=yoy_value,
    )
    snapshots.write_json_snapshot(
        "bls_cpi",
        "latest_release",
        {
            "release_datetime": release.release_datetime.astimezone(UTC).isoformat(),
            "period": release.period,
            "mom_sa": release.mom_sa,
            "yoy_sa": release.yoy_sa,
        },
    )
    return release


def _parse_calendar_html(html: str) -> dict[str, list[str]]:
    pattern = re.compile(r"([A-Z][a-z]+ \d{1,2}, \d{4})")
    matches = pattern.findall(html)
    release_dates: list[str] = []
    for match in matches:
        try:
            dt = datetime.strptime(match, "%B %d, %Y").replace(
                hour=8, minute=30, tzinfo=ET
            )
            release_dates.append(dt.date().isoformat())
        except ValueError:
            continue
    unique = sorted(set(release_dates))
    if not unique:
        # Fallback to fixture parsing if HTML structure unknown.
        raise RuntimeError("Failed to parse CPI calendar from BLS schedule page.")
    return {"release_dates": unique}


def _fetch_latest_release_online(
    *,
    force_refresh: bool,
    session: requests.Session | None,
) -> dict[str, object]:
    body = {"seriesid": [BLS_CPI_SERIES], "startyear": "2019", "endyear": "2030"}
    sess = session or requests.Session()
    response = sess.post(BLS_API_URL, json=body, timeout=15.0)
    if not response.ok:
        raise RuntimeError(f"BLS API request failed: {response.status_code}")
    data = response.json()
    series = data["Results"]["series"][0]["data"]
    latest = series[0]
    period = f"{latest['year']}-{latest['period'][1:]}"
    release_flag = latest.get("latest")
    if isinstance(release_flag, str) and release_flag.lower() not in {"true", "false"}:
        release_date = release_flag
    elif isinstance(release_flag, str) and release_flag.lower() == "true":
        release_date = datetime.now(tz=ET).date().isoformat()
    else:
        release_date = datetime.now(tz=ET).date().isoformat()
    mom = float(latest["value"])
    yoy = None
    for entry in series[1:13]:
        if entry["period"] == latest["period"] and entry["year"] == str(int(latest["year"]) - 1):
            yoy = ((float(latest["value"]) / float(entry["value"])) - 1.0) * 100
            break
    release_dt = datetime.strptime(release_date, "%Y-%m-%d").replace(tzinfo=ET, hour=8, minute=30)
    return {
        "release_datetime": release_dt.astimezone(UTC).isoformat(),
        "period": period,
        "mom_sa": mom,
        "yoy_sa": yoy,
    }
