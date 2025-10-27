"""NOAA/NWS Daily Climate Report (DCR) driver.

Settlement reminder: only the NWS Daily Climate Report is authoritative for weather ladders.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import requests
from dateutil import parser as date_parser

from kalshi_alpha.datastore import snapshots
from kalshi_alpha.datastore.paths import RAW_ROOT
from kalshi_alpha.utils.env import load_env
from kalshi_alpha.utils.http import fetch_with_cache

SETTLEMENT_SOURCE = "NWS Daily Climate Report"
NWS_STATION_URL = "https://w1.weather.gov/xml/current_obs/index.xml"
RAW_CACHE = RAW_ROOT / "_cache" / "nws_cli"
RAW_CACHE.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class StationConfig:
    station_id: str
    name: str
    wban: str | None = None


@dataclass(frozen=True)
class DailyClimateRecord:
    station_id: str
    record_date: date
    high_temp_f: float | None
    low_temp_f: float | None


def load_station_config(path: Path) -> dict[str, StationConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        entry["station_id"]: StationConfig(
            station_id=entry["station_id"],
            name=entry["name"],
            wban=entry.get("wban"),
        )
        for entry in payload["stations"]
    }


def parse_daily_climate_report(path: Path) -> DailyClimateRecord:
    """Parse a minimal DCR text fixture."""
    text = path.read_text(encoding="utf-8")
    station_match = re.search(r"STATION:\s*(?P<station>\w+)", text)
    date_match = re.search(r"DATE:\s*(?P<date>[\d-]+)", text)
    high_match = re.search(r"HIGH:\s*(?P<high>-?\d+)", text)
    low_match = re.search(r"LOW:\s*(?P<low>-?\d+)", text)

    if not station_match or not date_match:
        raise ValueError("invalid DCR format")

    return DailyClimateRecord(
        station_id=station_match.group("station"),
        record_date=_to_date(date_match.group("date")),
        high_temp_f=float(high_match.group("high")) if high_match else None,
        low_temp_f=float(low_match.group("low")) if low_match else None,
    )


def parse_multi_station_report(path: Path) -> list[DailyClimateRecord]:
    text = path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    records: list[DailyClimateRecord] = []
    for block in blocks:
        station_match = re.search(r"STATION:\s*(?P<station>\w+)", block)
        date_match = re.search(r"DATE:\s*(?P<date>[\d-]+)", block)
        high_match = re.search(r"HIGH:\s*(?P<high>-?\d+)", block)
        low_match = re.search(r"LOW:\s*(?P<low>-?\d+)", block)
        if not station_match or not date_match:
            continue
        records.append(
            DailyClimateRecord(
                station_id=station_match.group("station"),
                record_date=_to_date(date_match.group("date")),
                high_temp_f=float(high_match.group("high")) if high_match else None,
                low_temp_f=float(low_match.group("low")) if low_match else None,
            )
        )
    return records


def settlement_assertion() -> str:
    return f"All weather settlements must use the {SETTLEMENT_SOURCE}."


def _to_date(value: str) -> date:
    parsed = date_parser.parse(value)
    return date(parsed.year, parsed.month, parsed.day)


def fetch_station_metadata(
    *,
    offline: bool = False,
    fixtures_dir: Path | None = None,
    force_refresh: bool = False,
    session: requests.Session | None = None,
) -> dict[str, StationConfig]:
    load_env()
    if offline:
        if fixtures_dir is None:
            raise RuntimeError("fixtures_dir required for offline mode")
        payload = json.loads((fixtures_dir / "stations.json").read_text(encoding="utf-8"))
    else:
        response = fetch_with_cache(
            "https://www.weather.gov/cli/climate-stations",
            cache_path=RAW_CACHE / "stations.html",
            session=session,
            force_refresh=force_refresh,
        )
        html = response.decode("utf-8")
        snapshots.write_text_snapshot("nws_cli", "stations.html", html)
        if fixtures_dir:
            payload = json.loads((fixtures_dir / "stations.json").read_text(encoding="utf-8"))
        else:
            payload = {"stations": []}
    snapshots.write_json_snapshot("nws_cli", "stations", payload)
    config_path = (
        (fixtures_dir / "stations.json")
        if offline and fixtures_dir
        else _write_temp_station(payload)
    )
    return load_station_config(config_path)


def fetch_daily_climate_report(
    station_id: str,
    *,
    offline: bool = False,
    fixtures_dir: Path | None = None,
    force_refresh: bool = False,
    session: requests.Session | None = None,
) -> DailyClimateRecord:
    load_env()
    if offline:
        if fixtures_dir is None:
            raise RuntimeError("fixtures_dir required for offline mode")
        path = fixtures_dir / f"{station_id.lower()}_dcr.txt"
        text = path.read_text(encoding="utf-8")
    else:
        url = f"https://w1.weather.gov/climate/nwscd.php?wfo=BOX&climate={station_id}&recent=&wmoid="
        content = fetch_with_cache(
            url,
            cache_path=RAW_CACHE / f"{station_id}_dcr.html",
            session=session,
            force_refresh=force_refresh,
        )
        text = content.decode("utf-8")
    snapshots.write_text_snapshot("nws_cli", f"dcr_{station_id}.txt", text)
    return _parse_report_text(text)


def _parse_report_text(text: str) -> DailyClimateRecord:
    station_match = re.search(r"STATION:\s*(?P<station>\w+)", text)
    date_match = re.search(r"DATE:\s*(?P<date>[\d-]+)", text)
    high_match = re.search(r"HIGH:\s*(?P<high>-?\d+)", text)
    low_match = re.search(r"LOW:\s*(?P<low>-?\d+)", text)
    if not station_match or not date_match:
        raise ValueError("invalid DCR format")
    return DailyClimateRecord(
        station_id=station_match.group("station"),
        record_date=_to_date(date_match.group("date")),
        high_temp_f=float(high_match.group("high")) if high_match else None,
        low_temp_f=float(low_match.group("low")) if low_match else None,
    )


def _write_temp_station(payload: dict) -> Path:
    """Persist payload temporarily for load_station_config when online fetch unsupported."""
    temp_path = RAW_CACHE / "stations_offline.json"
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return temp_path
