"""Cleveland Fed inflation nowcast driver."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from kalshi_alpha.datastore import snapshots
from kalshi_alpha.datastore.paths import RAW_ROOT
from kalshi_alpha.utils.env import load_env
from kalshi_alpha.utils.http import fetch_with_cache

CLEVELAND_URL = "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"
MONTHLY_JSON_URL = (
    "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json"
    "?sc_lang=en"
)
CACHE_DIR = RAW_ROOT / "_cache" / "cleveland_nowcast"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MONTHLY_CACHE_PATH = CACHE_DIR / "nowcast_month.json"
PAGE_CACHE_PATH = CACHE_DIR / "nowcast.html"


@dataclass(frozen=True)
class NowcastSeries:
    label: str
    as_of: datetime
    value: float


def fetch_nowcast(
    *,
    offline: bool = False,
    fixtures_dir: Path | None = None,
    force_refresh: bool = False,
    session: requests.Session | None = None,
) -> dict[str, NowcastSeries]:
    """Return headline/core nowcasts."""
    load_env()
    if offline:
        if fixtures_dir is None:
            raise RuntimeError("fixtures_dir required for offline mode")
        headline = json.loads((fixtures_dir / "headline.json").read_text(encoding="utf-8"))
        core = json.loads((fixtures_dir / "core.json").read_text(encoding="utf-8"))
    else:
        # Cache the landing page for auditing while sourcing data from the JSON endpoint.
        page_bytes = fetch_with_cache(
            CLEVELAND_URL,
            cache_path=PAGE_CACHE_PATH,
            session=session,
            force_refresh=force_refresh,
        )
        page_text = page_bytes.decode("utf-8")
        snapshots.write_text_snapshot("cleveland_nowcast", "page.html", page_text)

        json_url = _extract_monthly_json_url(page_text) or MONTHLY_JSON_URL

        json_bytes = fetch_with_cache(
            json_url,
            cache_path=MONTHLY_CACHE_PATH,
            session=session,
            force_refresh=force_refresh,
        )
        json_text = json_bytes.decode("utf-8")
        snapshots.write_text_snapshot("cleveland_nowcast", "nowcast_month.json", json_text)
        headline, core = _parse_monthly_nowcast(json_text)

    series = {
        "headline": _to_series(headline),
        "core": _to_series(core),
    }
    snapshots.write_json_snapshot(
        "cleveland_nowcast",
        "latest",
        {
            name: {
                "label": item.label,
                "as_of": item.as_of.isoformat(),
                "value": item.value,
            }
            for name, item in series.items()
        },
    )
    return series


def _to_series(payload: Mapping[str, object]) -> NowcastSeries:
    as_of_raw = payload["as_of"]
    value_raw = payload["value"]
    return NowcastSeries(
        label=str(payload.get("label", "")),
        as_of=datetime.fromisoformat(str(as_of_raw)),
        value=float(value_raw),
    )


def _parse_monthly_nowcast(json_text: str) -> tuple[dict[str, object], dict[str, object]]:
    """Parse the Cleveland Fed chart JSON into headline and core payloads."""
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:  # pragma: no cover - log via caller
        raise RuntimeError("Invalid Cleveland Fed nowcast JSON payload") from exc

    latest_entry = _select_latest_entry(payload)
    chart_meta = latest_entry.get("chart", {})
    comment = chart_meta.get("_comment")
    if comment is None:
        raise RuntimeError("Missing metadata comment in latest nowcast entry")
    try:
        as_of = datetime.strptime(comment, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
    except ValueError as exc:
        raise RuntimeError(f"Unexpected comment datetime format: {comment}") from exc

    dataset = latest_entry.get("dataset", [])
    headline_value, headline_date = _extract_series(dataset, "CPI Inflation", as_of)
    core_value, core_date = _extract_series(dataset, "Core CPI Inflation", as_of)
    return (
        {
            "label": "Headline CPI m/m",
            "as_of": headline_date.isoformat(),
            "value": headline_value,
        },
        {
            "label": "Core CPI m/m",
            "as_of": core_date.isoformat(),
            "value": core_value,
        },
    )


def _select_latest_entry(payload: object) -> dict[str, object]:
    if not isinstance(payload, list):
        raise RuntimeError("Cleveland nowcast payload must be a list of chart entries")
    dated_entries: list[tuple[datetime, dict[str, object]]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        chart = entry.get("chart")
        if not isinstance(chart, dict):
            continue
        comment = chart.get("_comment")
        if not isinstance(comment, str):
            continue
        try:
            ts = datetime.strptime(comment, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        except ValueError:
            continue
        dated_entries.append((ts, entry))
    if not dated_entries:
        raise RuntimeError("No dated entries found in Cleveland nowcast payload")
    dated_entries.sort(key=lambda pair: pair[0])
    return dated_entries[-1][1]


def _extract_series(
    dataset: object,
    target: str,
    as_of: datetime,
) -> tuple[float, datetime]:
    best_value: float | None = None
    best_date: datetime | None = None
    series_list = dataset if isinstance(dataset, list) else []
    for series in series_list:
        if not isinstance(series, dict):
            continue
        name = str(series.get("seriesname", "")).strip()
        if name.lower() != target.lower():
            continue
        data_points = series.get("data")
        if not isinstance(data_points, list):
            continue
        for point in reversed(data_points):
            value_raw = point.get("value")
            if value_raw in {"", None}:
                continue
            try:
                best_value = float(value_raw)
            except (TypeError, ValueError):
                continue
            tooltext = point.get("tooltext")
            best_date = _parse_tooltext_date(tooltext, fallback=as_of)
            break
    if best_value is None or best_date is None:
        raise RuntimeError(f"Series {target} missing from Cleveland nowcast dataset")
    return best_value, best_date


def _parse_tooltext_date(tooltext: object, fallback: datetime) -> datetime:
    if isinstance(tooltext, str):
        parts = [part for part in tooltext.split("{br}") if part]
        if len(parts) >= 2:
            date_part = parts[1]
            try:
                parsed = datetime.strptime(date_part, "%m/%d").replace(
                    tzinfo=fallback.tzinfo, year=fallback.year
                )
                # Handle year rollover if the parsed month is ahead of the fallback month by >6
                if parsed.month > fallback.month + 6:
                    parsed = parsed.replace(year=fallback.year - 1)
                return parsed
            except ValueError:
                pass
    return fallback


def _extract_monthly_json_url(page_text: str) -> str | None:
    pattern = re.compile(r"data-data-config=\"(?P<path>[^\"]+nowcast_month[^\"]*)\"")
    match = pattern.search(page_text)
    if not match:
        return None
    href = match.group("path")
    if href.startswith("http"):
        return href
    return "https://www.clevelandfed.org" + href
