"""Load and query minimal macro event calendar metadata."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo
 
ET = ZoneInfo("America/New_York")
EVENTS_PATH = Path(__file__).resolve().with_name("events.json")


def _normalize_tags(tags: Iterable[str]) -> tuple[str, ...]:
    normalized = {str(tag).strip().upper() for tag in tags if str(tag).strip()}
    return tuple(sorted(normalized))


def _to_date(moment: date | datetime) -> date:
    if isinstance(moment, datetime):
        if moment.tzinfo is None:
            raise ValueError("datetime must be timezone-aware for event lookup")
        return moment.astimezone(ET).date()
    return moment


@dataclass(frozen=True)
class EventCalendar:
    """In-memory lookup helper for macro event tags keyed by ET date."""

    events: Mapping[date, tuple[str, ...]]

    def tags_for(self, moment: date | datetime) -> tuple[str, ...]:
        return self.events.get(_to_date(moment), ())

    def has_tag(self, moment: date | datetime, tag: str) -> bool:
        target = tag.strip().upper()
        return any(candidate == target for candidate in self.tags_for(moment))

    @classmethod
    def empty(cls) -> EventCalendar:
        return cls(events={})


def load_calendar(path: Path | None = None) -> EventCalendar:
    resolved = (path or EVENTS_PATH).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Event calendar file not found: {resolved}")
    return _load_calendar_cached(str(resolved))


@lru_cache(maxsize=4)
def _load_calendar_cached(path_str: str) -> EventCalendar:
    path = Path(path_str)
    data = json.loads(path.read_text(encoding="utf-8"))
    events: dict[date, tuple[str, ...]] = {}
    for entry in data:
        iso_date = entry.get("date")
        if not iso_date:
            continue
        try:
            event_date = date.fromisoformat(str(iso_date))
        except ValueError:
            continue
        tags = _normalize_tags(entry.get("tags", ()))
        if not tags:
            continue
        events[event_date] = tags
    return EventCalendar(events=events)


def calendar_tags_for(moment: date | datetime, *, path: Path | None = None) -> tuple[str, ...]:
    try:
        calendar = load_calendar(path)
    except FileNotFoundError:
        return ()
    return calendar.tags_for(moment)


__all__: Sequence[str] = ["EventCalendar", "load_calendar", "calendar_tags_for"]
