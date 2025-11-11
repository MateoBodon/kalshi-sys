"""Read-only Kalshi public market-data client.

Falls back to offline fixtures when network access is unavailable.

References:
- Kalshi public market-data quick start (series, events, markets, orderbooks).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import requests
from cachetools import LRUCache

DEFAULT_BASE_URL = "https://trading-api.kalshi.com/v1"
MARKETS_SEARCH_OFFLINE_STUB = "markets_search.json"
CacheKey = tuple[str, tuple[tuple[str, Any], ...]]
Payload = dict[str, Any] | list[Any]


@dataclass(frozen=True)
class Series:
    id: str
    ticker: str
    name: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Series:
        return cls(
            id=str(payload.get("id") or payload.get("series_id")),
            ticker=str(payload.get("ticker")),
            name=str(payload.get("name") or payload.get("title") or payload.get("ticker")),
        )


@dataclass(frozen=True)
class Event:
    id: str
    series_id: str
    ticker: str
    title: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Event:
        return cls(
            id=str(payload.get("id") or payload.get("event_id")),
            series_id=str(payload.get("series_id")),
            ticker=str(payload.get("ticker")),
            title=str(payload.get("title") or payload.get("name") or payload.get("ticker")),
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

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Market:
        strikes = payload.get("ladder_strikes") or payload.get("strike_prices") or []
        yes_prices = payload.get("ladder_yes_prices") or payload.get("yes_prices") or []
        close_time = _parse_timestamp(payload.get("close_ts"))
        event_ticker = payload.get("event_ticker") or payload.get("event")
        series_ticker = payload.get("series_ticker") or payload.get("series")
        status = payload.get("status")
        return cls(
            id=str(payload.get("id") or payload.get("market_id")),
            event_id=str(payload.get("event_id")),
            ticker=str(payload.get("ticker")),
            title=str(payload.get("title") or payload.get("name") or payload.get("ticker")),
            ladder_strikes=[float(value) for value in strikes],
            ladder_yes_prices=[float(value) for value in yes_prices],
            event_ticker=str(event_ticker) if event_ticker else None,
            series_ticker=str(series_ticker).upper() if series_ticker else None,
            status=str(status).lower() if isinstance(status, str) else None,
            close_time=close_time,
        )


@dataclass(frozen=True)
class Orderbook:
    market_id: str
    bids: list[dict[str, Any]]
    asks: list[dict[str, Any]]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Orderbook:
        return cls(
            market_id=str(payload.get("market_id")),
            bids=list(payload.get("bids", [])),
            asks=list(payload.get("asks", [])),
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
        payload = self._get(
            f"/series/{series_id}/events",
            cache_key=("events", series_id),
            offline_stub=f"events_{series_id}.json",
            force_refresh=force_refresh,
        )
        data = payload.get("events") if isinstance(payload, dict) else payload
        return [Event.from_payload(item) for item in data or []]

    def get_markets(self, event_id: str, *, force_refresh: bool = False) -> list[Market]:
        payload = self._get(
            f"/events/{event_id}/markets",
            cache_key=("markets", event_id),
            offline_stub=f"markets_{event_id}.json",
            force_refresh=force_refresh,
        )
        data = payload.get("markets") if isinstance(payload, dict) else payload
        return [Market.from_payload(item) for item in data or []]

    def get_orderbook(self, market_id: str, *, force_refresh: bool = False) -> Orderbook:
        payload = self._get(
            f"/markets/{market_id}/orderbook",
            cache_key=("orderbook", market_id),
            offline_stub=f"orderbook_{market_id}.json",
            force_refresh=force_refresh,
        )
        return Orderbook.from_payload(payload if isinstance(payload, dict) else payload[0])

    def search_markets(  # noqa: PLR0913 - filter flexibility
        self,
        *,
        series_ticker: str | None = None,
        status: str | None = None,
        event_ticker: str | None = None,
        limit: int | None = None,
        force_refresh: bool = False,
    ) -> list[Market]:
        params: dict[str, Any] = {}
        series_value = series_ticker.upper() if isinstance(series_ticker, str) else None
        status_value = status.lower() if isinstance(status, str) else None
        if series_value:
            params["series_ticker"] = series_value
        if status_value:
            params["status"] = status_value
        if event_ticker:
            params["event_ticker"] = str(event_ticker)
        if limit is not None:
            params["limit"] = int(limit)

        cache_key = (
            "markets_search",
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
        markets = [Market.from_payload(item) for item in data or []]
        return [
            market
            for market in markets
            if _matches_filter(market, series_value, status_value, event_ticker)
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
            except (requests.RequestException, json.JSONDecodeError):
                payload = self._load_offline(offline_stub)

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
    if series_ticker and (market.series_ticker or "").upper() != series_ticker:
        return False
    if status and (market.status or "").lower() != status:
        return False
    if event_ticker and (market.event_ticker or "").upper() != event_ticker.upper():
        return False
    return True


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
