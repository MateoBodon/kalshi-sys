"""Market discovery utilities for INX/NDX ladders."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
import re
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

from kalshi_alpha.core.kalshi_api import KalshiPublicClient, Market
from kalshi_alpha.sched import TradingWindow, windows_for_day
from kalshi_alpha.utils.series import (
    INDEX_CANONICAL_SERIES,
    index_series_query_candidates,
    normalize_index_series,
    normalize_index_series_list,
)

ET = ZoneInfo("America/New_York")
_MATCH_TOLERANCE = timedelta(seconds=90)
_TICKER_PATTERN = re.compile(r"-(?P<year>\d{2})(?P<month>[A-Z]{3})(?P<day>\d{2})H(?P<hour>\d{2})(?P<minute>\d{2})")
_MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


@dataclass(frozen=True)
class DiscoveredMarket:
    series: str
    event_id: str
    event_ticker: str | None
    close_time: datetime
    market_ids: tuple[str, ...]
    market_tickers: tuple[str, ...]
    market_count: int
    status: str | None = None

    @property
    def close_time_et(self) -> datetime:
        return self.close_time.astimezone(ET)


@dataclass(frozen=True)
class WindowDiscovery:
    label: str
    target_et: datetime
    target_type: str
    window: TradingWindow | None
    markets: tuple[DiscoveredMarket, ...]
    expected_series: tuple[str, ...]
    missing_series: tuple[str, ...]


def discover_markets_for_day(
    client: KalshiPublicClient,
    *,
    trading_day: date,
    series: Sequence[str] | None = None,
    status: str = "open",
) -> list[WindowDiscovery]:
    """Return discovered markets for the provided trading day grouped by scheduler windows."""

    series_list = _normalize_series(series)
    markets_by_series = _collect_markets(client, trading_day=trading_day, series_list=series_list, status=status)
    windows = windows_for_day(trading_day)
    used_events: set[str] = set()
    discoveries: list[WindowDiscovery] = []

    for window in windows:
        matched: list[DiscoveredMarket] = []
        for series_code in window.series:
            match = _match_for_window(markets_by_series.get(series_code, ()), window, used_events)
            if match is not None:
                matched.append(match)
                used_events.add(match.event_id)
        missing = tuple(sorted(set(window.series) - {market.series for market in matched}))
        discoveries.append(
            WindowDiscovery(
                label=window.label,
                target_et=window.target_et,
                target_type=window.target_type,
                window=window,
                markets=tuple(sorted(matched, key=lambda market: market.series)),
                expected_series=tuple(window.series),
                missing_series=missing,
            )
        )

    orphans = _orphan_markets(markets_by_series.values(), used_events)
    if orphans:
        discoveries.extend(_derived_windows(orphans))

    discoveries.sort(key=lambda item: item.target_et)
    return discoveries


def _normalize_series(series: Sequence[str] | None) -> tuple[str, ...]:
    return normalize_index_series_list(series)


def _canonical_series(label: str) -> str:
    return normalize_index_series(label)


def _collect_markets(
    client: KalshiPublicClient,
    *,
    trading_day: date,
    series_list: Sequence[str],
    status: str,
) -> dict[str, list[DiscoveredMarket]]:
    per_series: dict[str, list[DiscoveredMarket]] = {}
    for series_code in series_list:
        markets: list[Market] = []
        for candidate in index_series_query_candidates(series_code):
            candidate_markets = client.search_markets(series_ticker=candidate, status=status)
            if candidate_markets:
                markets = candidate_markets
                break
        grouped = _group_by_event(markets, trading_day)
        if grouped:
            per_series[series_code] = grouped
    return per_series


def _group_by_event(markets: Sequence[Market], trading_day: date) -> list[DiscoveredMarket]:
    grouped: dict[str, list[Market]] = defaultdict(list)
    for market in markets:
        close_dt = market.close_time or _infer_close_from_ticker(market.ticker)
        if close_dt is None:
            continue
        close_utc = close_dt.astimezone(UTC)
        if close_utc.astimezone(ET).date() != trading_day:
            continue
        grouped[market.event_id].append(market)

    discovered: list[DiscoveredMarket] = []
    for event_id, entries in grouped.items():
        entries = sorted(entries, key=lambda market: market.ticker)
        series = _canonical_series(entries[0].series_ticker or "")
        if not series:
            continue
        close_dt = entries[0].close_time or _infer_close_from_ticker(entries[0].ticker)
        if close_dt is None:
            continue
        discovered.append(
            DiscoveredMarket(
                series=series,
                event_id=event_id,
                event_ticker=entries[0].event_ticker,
                close_time=close_dt.astimezone(UTC),
                market_ids=tuple(market.id for market in entries),
                market_tickers=tuple(market.ticker for market in entries),
                market_count=len(entries),
                status=entries[0].status,
            )
        )
    return sorted(discovered, key=lambda item: item.close_time)


def _match_for_window(
    candidates: Sequence[DiscoveredMarket],
    window: TradingWindow,
    used_events: set[str],
) -> DiscoveredMarket | None:
    matched: list[tuple[float, DiscoveredMarket]] = []
    for market in candidates:
        if market.event_id in used_events:
            continue
        delta = abs((market.close_time_et - window.target_et).total_seconds())
        if delta <= _MATCH_TOLERANCE.total_seconds():
            matched.append((delta, market))
    if not matched:
        return None
    matched.sort(key=lambda item: item[0])
    return matched[0][1]


def _orphan_markets(
    all_series_markets: Iterable[Sequence[DiscoveredMarket]],
    used_events: Iterable[str],
) -> list[DiscoveredMarket]:
    used = set(used_events)
    orphans: list[DiscoveredMarket] = []
    for entries in all_series_markets:
        for market in entries:
            if market.event_id in used:
                continue
            orphans.append(market)
    return orphans


def _derived_windows(markets: Sequence[DiscoveredMarket]) -> list[WindowDiscovery]:
    buckets: dict[tuple[date, int, int], list[DiscoveredMarket]] = defaultdict(list)
    for market in markets:
        close_et = market.close_time_et
        buckets[(close_et.date(), close_et.hour, close_et.minute)].append(market)

    derived: list[WindowDiscovery] = []
    for key in sorted(buckets):
        _, hour, minute = key
        entries = sorted(buckets[key], key=lambda market: market.series)
        close_et = entries[0].close_time_et
        target_dt = datetime.combine(close_et.date(), time(hour, minute), tzinfo=ET)
        label = f"derived-{hour:02d}{minute:02d}"
        target_type = "close" if hour >= 16 else "hourly"
        expected = tuple(sorted({market.series for market in entries if market.series}))
        derived.append(
            WindowDiscovery(
                label=label,
                target_et=target_dt,
                target_type=target_type,
                window=None,
                markets=tuple(entries),
                expected_series=expected,
                missing_series=(),
            )
        )
    return derived


def _infer_close_from_ticker(ticker: str) -> datetime | None:
    match = _TICKER_PATTERN.search(ticker.upper())
    if not match:
        return None
    month = _MONTHS.get(match.group("month"))
    if month is None:
        return None
    year = 2000 + int(match.group("year"))
    day = int(match.group("day"))
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    try:
        et_value = datetime(year, month, day, hour % 24, minute % 60, tzinfo=ET)
    except ValueError:
        return None
    return et_value.astimezone(UTC)


__all__ = [
    "DiscoveredMarket",
    "WindowDiscovery",
    "discover_markets_for_day",
]
