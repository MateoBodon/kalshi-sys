"""Cleveland Fed inflation nowcast driver."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from kalshi_alpha.datastore import snapshots
from kalshi_alpha.datastore.paths import RAW_ROOT
from kalshi_alpha.utils.http import fetch_with_cache

CLEVELAND_URL = "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"
CACHE_PATH = RAW_ROOT / "_cache" / "cleveland_nowcast" / "nowcast.html"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


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
    if offline:
        if fixtures_dir is None:
            raise RuntimeError("fixtures_dir required for offline mode")
        headline = json.loads((fixtures_dir / "headline.json").read_text(encoding="utf-8"))
        core = json.loads((fixtures_dir / "core.json").read_text(encoding="utf-8"))
    else:
        html = fetch_with_cache(
            CLEVELAND_URL,
            cache_path=CACHE_PATH,
            session=session,
            force_refresh=force_refresh,
        )
        snapshots.write_text_snapshot("cleveland_nowcast", "page.html", html.decode("utf-8"))
        if fixtures_dir and (fixtures_dir / "headline.json").exists():
            headline = json.loads((fixtures_dir / "headline.json").read_text(encoding="utf-8"))
            core = json.loads((fixtures_dir / "core.json").read_text(encoding="utf-8"))
        else:
            raise RuntimeError("Online parsing for Cleveland Fed nowcast is not yet implemented.")

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


def _to_series(payload: Mapping[str, Any]) -> NowcastSeries:
    as_of_raw = payload["as_of"]
    value_raw = payload["value"]
    return NowcastSeries(
        label=str(payload.get("label", "")),
        as_of=datetime.fromisoformat(str(as_of_raw)),
        value=float(value_raw),
    )
