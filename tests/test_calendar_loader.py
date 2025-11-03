from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from kalshi_alpha.drivers.calendar import loader

UTC = ZoneInfo("UTC")


def _write_events(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "events.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_load_calendar_returns_normalized_tags(tmp_path: Path) -> None:
    path = _write_events(
        tmp_path,
        [
            {"date": "2025-03-19", "tags": ["FOMC", "cpi", "FOMC", ""]},
            {"date": "2025-03-20", "tags": []},
            {"date": "invalid-date", "tags": ["FOMC"]},
        ],
    )
    calendar = loader.load_calendar(path)
    assert calendar.tags_for(date(2025, 3, 19)) == ("CPI", "FOMC")
    assert calendar.tags_for(date(2025, 3, 20)) == ()


def test_calendar_lookup_handles_dst_start(tmp_path: Path) -> None:
    path = _write_events(tmp_path, [{"date": "2025-03-09", "tags": ["FOMC"]}])
    calendar = loader.load_calendar(path)
    moment = datetime(2025, 3, 9, 16, 0, tzinfo=UTC)
    assert calendar.tags_for(moment) == ("FOMC",)


def test_calendar_lookup_handles_dst_end(tmp_path: Path) -> None:
    path = _write_events(tmp_path, [{"date": "2025-11-02", "tags": ["CPI"]}])
    calendar = loader.load_calendar(path)
    moment = datetime(2025, 11, 2, 5, 0, tzinfo=UTC)
    assert calendar.tags_for(moment) == ("CPI",)
    second_moment = datetime(2025, 11, 2, 9, 0, tzinfo=UTC)
    assert calendar.tags_for(second_moment) == ("CPI",)


def test_has_tag_case_insensitive(tmp_path: Path) -> None:
    path = _write_events(tmp_path, [{"date": "2025-03-19", "tags": ["FOMC"]}])
    calendar = loader.load_calendar(path)
    moment = datetime(2025, 3, 19, 12, 0, tzinfo=UTC)
    assert calendar.has_tag(moment, "fomc")
