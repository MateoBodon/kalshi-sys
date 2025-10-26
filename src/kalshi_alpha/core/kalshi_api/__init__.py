"""Read-only Kalshi public market-data client.

Falls back to offline fixtures when network access is unavailable.

References:
- Kalshi public market-data quick start (series, events, markets, orderbooks).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import requests
from cachetools import LRUCache

DEFAULT_BASE_URL = "https://trading-api.kalshi.com/v1"
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

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Market:
        strikes = payload.get("ladder_strikes") or payload.get("strike_prices") or []
        yes_prices = payload.get("ladder_yes_prices") or payload.get("yes_prices") or []
        return cls(
            id=str(payload.get("id") or payload.get("market_id")),
            event_id=str(payload.get("event_id")),
            ticker=str(payload.get("ticker")),
            title=str(payload.get("title") or payload.get("name") or payload.get("ticker")),
            ladder_strikes=[float(value) for value in strikes],
            ladder_yes_prices=[float(value) for value in yes_prices],
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
