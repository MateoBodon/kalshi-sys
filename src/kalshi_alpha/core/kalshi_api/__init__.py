"""Read-only Kalshi public market-data client.

Falls back to offline fixtures when network access is unavailable.

References:
- Kalshi public market-data quick start (series, events, markets, orderbooks).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import requests
from cachetools import LRUCache

DEFAULT_BASE_URL = os.getenv("KALSHI_PUBLIC_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")
MARKETS_SEARCH_OFFLINE_STUB = "markets_search.json"
CacheKey = tuple[str, tuple[tuple[str, Any], ...]]
Payload = dict[str, Any] | list[Any]
_STATUS_EQUIVALENTS: dict[str, set[str]] = {
    "open": {"open", "active", "initialized"},
    "unopened": {"unopened", "scheduled"},
    "closed": {"closed"},
    "settled": {"settled", "resolved"},
}
_CANONICAL_SERIES = ("INXU", "NASDAQ100U", "INX", "NASDAQ100")
_SERIES_CANONICAL_MAP = {
    **{label: label for label in _CANONICAL_SERIES},
    **{f"KX{label}": label for label in _CANONICAL_SERIES},
}
_SERIES_QUERY_CACHE: dict[str, tuple[str, ...]] = {}
_TICKER_STRIKE_PATTERN = re.compile(r"[-:]([0-9]+(?:\.[0-9]+)?)$")


class KalshiPublicClientError(RuntimeError):
    """Raised when the public Kalshi API client encounters unrecoverable errors."""


@dataclass(frozen=True)
class Series:
    id: str
    ticker: str
    name: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Series:
        ticker = str(payload.get("ticker") or payload.get("series_ticker"))
        identifier = payload.get("id") or payload.get("series_id") or ticker
        return cls(
            id=str(identifier),
            ticker=ticker,
            name=str(payload.get("name") or payload.get("title") or ticker),
        )


@dataclass(frozen=True)
class Event:
    id: str
    series_id: str
    ticker: str
    title: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Event:
        event_ticker = payload.get("event_ticker") or payload.get("ticker")
        return cls(
            id=str(payload.get("id") or payload.get("event_id") or event_ticker),
            series_id=str(payload.get("series_id") or payload.get("series_ticker") or ""),
            ticker=str(event_ticker),
            title=str(payload.get("title") or payload.get("name") or event_ticker),
        )


@dataclass(frozen=True)
class Market:
    id: str
    event_id: str
    ticker: str
    title: str
    ladder_strikes: list[float]
    ladder_yes_prices: list[float]
    event_ticker: str | None = None
    series_ticker: str | None = None
    status: str | None = None
    close_time: datetime | None = None
    rung_tickers: list[str] | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Market:
        strikes = payload.get("ladder_strikes") or payload.get("strike_prices") or []
        yes_prices = payload.get("ladder_yes_prices") or payload.get("yes_prices") or []
        close_time = _parse_timestamp(payload.get("close_ts") or payload.get("close_time"))
        event_ticker = payload.get("event_ticker") or payload.get("event")
        series_ticker = payload.get("series_ticker") or payload.get("series")
        if not series_ticker and isinstance(event_ticker, str) and "-" in event_ticker:
            series_ticker = event_ticker.split("-", 1)[0]
        status = payload.get("status")
        market_id = payload.get("id") or payload.get("market_id") or payload.get("ticker")
        event_id = payload.get("event_id") or event_ticker
        rung_tickers = payload.get("rung_tickers") or None
        if isinstance(rung_tickers, list):
            rung_tickers = [str(item) for item in rung_tickers]
        else:
            rung_tickers = None
        return cls(
            id=str(market_id) if market_id is not None else "",
            event_id=str(event_id) if event_id is not None else "",
            ticker=str(payload.get("ticker")),
            title=str(payload.get("title") or payload.get("name") or payload.get("ticker")),
            ladder_strikes=[float(value) for value in strikes],
            ladder_yes_prices=[float(value) for value in yes_prices],
            event_ticker=str(event_ticker) if event_ticker else None,
            series_ticker=str(series_ticker).upper() if series_ticker else None,
            status=str(status).lower() if isinstance(status, str) else None,
            close_time=close_time,
            rung_tickers=rung_tickers,
        )


@dataclass(frozen=True)
class Orderbook:
    market_id: str
    bids: list[dict[str, Any]]
    asks: list[dict[str, Any]]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Orderbook:
        if "orderbook" in payload:
            payload = payload["orderbook"]
        bids = list(payload.get("bids", []))
        asks = list(payload.get("asks", []))
        if not bids and "yes" in payload:
            bids = _orderbook_side_from_levels(payload.get("yes"), payload.get("yes_dollars"))
        if not asks and "no" in payload:
            asks = _orderbook_side_from_levels(payload.get("no"), payload.get("no_dollars"))
        market_id = str(payload.get("market_id") or payload.get("ticker") or "")
        return cls(
            market_id=market_id,
            bids=bids,
            asks=asks,
        )

    def depth_weighted_mid(self) -> float:
        if not self.bids or not self.asks:
            raise ValueError("Orderbook requires both bids and asks for mid calculation")
        bid_total = sum(float(entry.get("size", 0.0)) for entry in self.bids)
        ask_total = sum(float(entry.get("size", 0.0)) for entry in self.asks)
        if bid_total <= 0 or ask_total <= 0:
            raise ValueError("Orderbook depth must be positive on both sides")
        weighted_bid = sum(float(entry["price"]) * float(entry["size"]) for entry in self.bids)
        weighted_ask = sum(float(entry["price"]) * float(entry["size"]) for entry in self.asks)
        combined_size = bid_total + ask_total
        return (weighted_bid + weighted_ask) / combined_size


class KalshiPublicClient:
    """Simple public API client with offline fixture support."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 5.0,
        offline_dir: Path | None = None,
        use_offline: bool = False,
        cache_size: int = 32,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.offline_dir = offline_dir
        self.use_offline = use_offline
        self.session = session or requests.Session()
        self._cache: LRUCache[CacheKey, Payload] = LRUCache(maxsize=cache_size)

    def get_series(self, *, force_refresh: bool = False) -> list[Series]:
        payload = self._get(
            "/series",
            cache_key=("series",),
            offline_stub="series.json",
            force_refresh=force_refresh,
        )
        data = payload.get("series") if isinstance(payload, dict) else payload
        return [Series.from_payload(item) for item in data or []]

    def get_events(self, series_id: str, *, force_refresh: bool = False) -> list[Event]:
        candidates = _series_query_candidates(series_id) or (series_id,)
        for candidate in candidates:
            payload = self._get(
                "/events",
                params={"series_ticker": candidate},
                cache_key=("events", candidate),
                offline_stub=f"events_{candidate}.json",
                force_refresh=force_refresh,
            )
            data = payload.get("events") if isinstance(payload, dict) else payload
            if not data:
                continue
            events = [Event.from_payload(item) for item in data]
            for event in events:
                if not event.series_id:
                    event.series_id = candidate  # type: ignore[attr-defined]
            return events
        return []

    def get_markets(self, event_id: str, *, force_refresh: bool = False) -> list[Market]:
        if self.use_offline:
            payload = self._get(
                f"/events/{event_id}/markets",
                cache_key=("markets", event_id),
                offline_stub=f"markets_{event_id}.json",
                force_refresh=force_refresh,
            )
            data = payload.get("markets") if isinstance(payload, dict) else payload
            return [Market.from_payload(item) for item in data or []]
        payload = self._get(
            "/markets",
            params={"event_ticker": event_id, "limit": 1000},
            cache_key=("markets_event", event_id),
            offline_stub=f"markets_{event_id}.json",
            force_refresh=force_refresh,
        )
        data = payload.get("markets") if isinstance(payload, dict) else payload
        aggregated = _aggregate_markets(event_id, data or [])
        return [Market.from_payload(aggregated)] if aggregated else []

    def get_orderbook(self, market_id: str, *, force_refresh: bool = False) -> Orderbook:
        if self.use_offline:
            payload = self._get(
                f"/markets/{market_id}/orderbook",
                cache_key=("orderbook", market_id),
                offline_stub=f"orderbook_{market_id}.json",
                force_refresh=force_refresh,
            )
            return Orderbook.from_payload(payload if isinstance(payload, dict) else payload[0])
        # Aggregated orderbooks are no longer available via the public API; return empty structure.
        return Orderbook(market_id=market_id, bids=[], asks=[])

    def search_markets(  # noqa: PLR0913 - filter flexibility
        self,
        *,
        series_ticker: str | None = None,
        status: str | None = None,
        event_ticker: str | None = None,
        limit: int | None = None,
        force_refresh: bool = False,
    ) -> list[Market]:
        status_value = status.lower() if isinstance(status, str) else None
        event_value = str(event_ticker) if event_ticker else None
        limit_value = int(limit) if limit is not None else None
        series_candidates: tuple[str | None, ...]
        if series_ticker:
            series_candidates = _series_query_candidates(series_ticker)
            if not series_candidates:
                series_candidates = (_canonical_series_label(series_ticker),)
        else:
            series_candidates = (None,)

        raw_entries: list[Mapping[str, Any]] = []
        for candidate in series_candidates:
            params: dict[str, Any] = {}
            if candidate:
                params["series_ticker"] = candidate
            if status_value:
                params["status"] = status_value
            if event_value:
                params["event_ticker"] = event_value
            if limit_value is not None:
                params["limit"] = limit_value
            cache_key = (
                "markets_search",
                candidate or "ALL",
                tuple(sorted((key, str(value)) for key, value in params.items())),
            )
            payload = self._get(
                "/markets",
                params=params or None,
                cache_key=cache_key,
                offline_stub=MARKETS_SEARCH_OFFLINE_STUB,
                force_refresh=force_refresh,
            )
            data = payload.get("markets") if isinstance(payload, dict) else payload
            if data:
                raw_entries.extend(data)

        markets: list[Market] = []
        if raw_entries and isinstance(raw_entries, list) and raw_entries and "ladder_strikes" not in raw_entries[0]:
            grouped: dict[str, list[Mapping[str, Any]]] = {}
            for entry in raw_entries:
                event_key = str(entry.get("event_ticker") or entry.get("event_id") or entry.get("ticker") or "")
                grouped.setdefault(event_key, []).append(entry)
            for event_id, entries in grouped.items():
                aggregated = _aggregate_markets(event_id, entries)
                if aggregated:
                    markets.append(Market.from_payload(aggregated))
        else:
            markets = [Market.from_payload(item) for item in raw_entries or []]
        return [
            market
            for market in markets
            if _matches_filter(market, series_ticker, status_value, event_ticker)
        ]

    # Internal helpers -------------------------------------------------------------------------

    def _get(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        cache_key: Iterable[Any],
        offline_stub: str | None,
        force_refresh: bool,
    ) -> Payload:
        params_tuple = tuple(sorted((params or {}).items()))
        key = tuple(cache_key) + params_tuple
        if not force_refresh and key in self._cache:
            return cast(Payload, self._cache[key])

        if self.use_offline:
            payload: Payload = self._load_offline(offline_stub)
        else:
            url = f"{self.base_url}{endpoint}"
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                payload = cast(Payload, response.json())
            except (requests.RequestException, json.JSONDecodeError) as exc:
                if offline_stub and self.use_offline:
                    payload = self._load_offline(offline_stub)
                else:
                    raise KalshiPublicClientError(f"Kalshi request failed: {exc}") from exc

        self._cache[key] = payload
        return payload

    def _load_offline(self, offline_stub: str | None) -> Payload:
        if self.offline_dir is None or offline_stub is None:
            raise RuntimeError("Offline data unavailable for this request")
        offline_path = self.offline_dir / offline_stub
        if not offline_path.exists():
            raise FileNotFoundError(f"Offline fixture missing: {offline_path}")
        with offline_path.open("r", encoding="utf-8") as handle:
            return cast(Payload, json.load(handle))


def _matches_filter(
    market: Market,
    series_ticker: str | None,
    status: str | None,
    event_ticker: str | None,
) -> bool:
    if series_ticker:
        target_series = _canonical_series_label(series_ticker)
        market_series = _canonical_series_label(market.series_ticker)
        if target_series and market_series and market_series != target_series:
            return False
    if status:
        normalized = (market.status or "").lower()
        accepted = _STATUS_EQUIVALENTS.get(status, {status})
        if normalized not in accepted:
            return False
    if event_ticker and (market.event_ticker or "").upper() != event_ticker.upper():
        return False
    return True


def _canonical_series_label(label: str | None) -> str:
    if not label:
        return ""
    normalized = label.strip().upper()
    return _SERIES_CANONICAL_MAP.get(normalized, normalized)


def _series_query_candidates(label: str | None) -> tuple[str, ...]:
    key = (label or "").strip().upper()
    if not key:
        return ()
    cached = _SERIES_QUERY_CACHE.get(key)
    if cached:
        return cached
    candidates: list[str] = [key]
    if key.startswith("KX") and len(key) > 2:
        stripped = key[2:]
        candidates.append(stripped)
    else:
        candidates.append(f"KX{key}")
    deduped: list[str] = []
    for entry in candidates:
        if entry and entry not in deduped:
            deduped.append(entry)
    result = tuple(deduped)
    _SERIES_QUERY_CACHE[key] = result
    return result


def _aggregate_markets(event_id: str, entries: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    preferred_entries = list(entries)
    has_between = any(
        isinstance(entry.get("strike_type"), str)
        and entry["strike_type"].lower() == "between"
        for entry in preferred_entries
    )
    if has_between:
        preferred_entries = [
            entry for entry in preferred_entries if str(entry.get("strike_type", "")).lower() == "between"
        ]
    rung_details: list[tuple[float, float, str]] = []
    seen: set[float] = set()
    for entry in preferred_entries:
        strike = _resolve_strike(entry)
        price = _resolve_yes_price(entry)
        if strike is None or price is None:
            continue
        if strike in seen:
            continue
        seen.add(strike)
        rung_details.append((strike, max(0.0, min(1.0, price)), str(entry.get("ticker") or "")))
    if not rung_details:
        return None
    rung_details.sort(key=lambda item: item[0])
    rung_strikes = [strike for strike, _, _ in rung_details]
    rung_prices = [price for _, price, _ in rung_details]
    rung_tickers = [ticker for _, _, ticker in rung_details]
    first = entries[0]
    market_id = str(first.get("market_id") or first.get("event_ticker") or event_id)
    ticker = str(first.get("event_ticker") or first.get("ticker") or market_id)
    title = str(first.get("title") or first.get("event_ticker") or event_id)
    status = first.get("status")
    series_label = first.get("series_ticker") or first.get("series")
    if not series_label:
        event_label = first.get("event_ticker") or ticker
        if isinstance(event_label, str) and "-" in event_label:
            series_label = event_label.split("-", 1)[0]
    close_time = first.get("close_time") or first.get("close_ts")
    return {
        "id": market_id,
        "event_id": str(first.get("event_ticker") or event_id),
        "ticker": ticker,
        "title": title,
        "ladder_strikes": rung_strikes,
        "ladder_yes_prices": rung_prices,
        "rung_tickers": rung_tickers,
        "status": status,
        "series_ticker": series_label,
        "close_time": close_time,
    }


def _resolve_strike(entry: Mapping[str, Any]) -> float | None:
    for key in ("floor_strike", "functional_strike", "cap_strike"):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    ticker = entry.get("ticker")
    if isinstance(ticker, str):
        match = _TICKER_STRIKE_PATTERN.search(ticker.replace("-", ":"))
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                return None
    return None


def _resolve_yes_price(entry: Mapping[str, Any]) -> float | None:
    yes_values: list[float] = []
    for key in ("yes_bid_dollars", "yes_ask_dollars", "last_price_dollars"):
        value = entry.get(key)
        parsed = _parse_price_value(value)
        if parsed is not None:
            yes_values.append(parsed)
    for key in ("yes_bid", "yes_ask", "last_price"):
        value = entry.get(key)
        parsed = _parse_price_value(value, assume_cents=True)
        if parsed is not None:
            yes_values.append(parsed)
    if yes_values:
        if len(yes_values) >= 2:
            return sum(yes_values[:2]) / 2.0
        return yes_values[0]
    no_values: list[float] = []
    for key in ("no_bid_dollars", "no_ask_dollars"):
        value = entry.get(key)
        parsed = _parse_price_value(value)
        if parsed is not None:
            no_values.append(1.0 - parsed)
    for key in ("no_bid", "no_ask"):
        value = entry.get(key)
        parsed = _parse_price_value(value, assume_cents=True)
        if parsed is not None:
            no_values.append(1.0 - parsed)
    if no_values:
        if len(no_values) >= 2:
            return sum(no_values[:2]) / 2.0
        return no_values[0]
    return None


def _parse_price_value(value: Any, *, assume_cents: bool = False) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        price = float(value)
        if assume_cents and price > 1.0:
            return price / 100.0
        return price
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            price = float(stripped)
        except ValueError:
            return None
        return price
    return None


def _orderbook_side_from_levels(
    levels_raw: Any,
    dollars_raw: Any,
) -> list[dict[str, Any]]:
    levels: list[list[float]] = []
    if isinstance(dollars_raw, Sequence):
        for entry in dollars_raw:
            if isinstance(entry, Sequence) and len(entry) >= 2:
                try:
                    price = float(entry[0])
                    size = float(entry[1])
                except (TypeError, ValueError):
                    continue
                levels.append([price, size])
    if not levels and isinstance(levels_raw, Sequence):
        for entry in levels_raw:
            if isinstance(entry, Sequence) and len(entry) >= 2:
                try:
                    price = float(entry[0]) / (100.0 if float(entry[0]) > 1.0 else 1.0)
                    size = float(entry[1])
                except (TypeError, ValueError):
                    continue
                levels.append([price, size])
    result: list[dict[str, Any]] = []
    for price, size in levels:
        result.append({"price": price, "size": size})
    return result


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1_000_000_000_000:
            seconds /= 1000.0
        return datetime.fromtimestamp(seconds, tz=UTC)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.isdigit():
            return datetime.fromtimestamp(float(raw), tz=UTC)
        normalized = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None
