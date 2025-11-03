"""Polygon indices client supporting REST and websocket access."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
import websockets
from requests import Response
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from kalshi_alpha.utils.keys import load_polygon_api_key

DEFAULT_REST_URL = "https://api.polygon.io"
DEFAULT_WS_URL = "wss://socket.polygon.io/stocks"
INDEX_MINUTE_CHANNEL = "XA"


class PolygonAPIError(RuntimeError):
    """Raised when the Polygon API returns an error."""


@dataclass(frozen=True)
class MinuteBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None
    trades: int | None = None


@dataclass(frozen=True)
class IndexSnapshot:
    ticker: str
    last_price: float | None
    change: float | None
    change_percent: float | None
    previous_close: float | None
    timestamp: datetime | None


class PolygonIndicesClient:
    """REST + websocket client for Polygon index data."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        api_key: str | None = None,
        rest_base_url: str = DEFAULT_REST_URL,
        ws_url: str = DEFAULT_WS_URL,
        session: requests.Session | None = None,
        timeout: float = 10.0,
        ws_ping_interval: float = 20.0,
        ws_auth_timeout: float = 5.0,
    ) -> None:
        self._api_key = api_key
        self._rest_base_url = rest_base_url.rstrip("/")
        self._ws_url = ws_url
        self._session = session or requests.Session()
        self._timeout = float(timeout)
        self._ws_ping_interval = max(5.0, float(ws_ping_interval))
        self._ws_auth_timeout = max(1.0, float(ws_auth_timeout))

    # REST ------------------------------------------------------------------

    def _resolved_api_key(self) -> str:
        api_key = (self._api_key or load_polygon_api_key() or "").strip()
        if not api_key:
            raise PolygonAPIError(
                "Polygon API key not configured; store it in Keychain "
                '("kalshi-sys:POLYGON_API_KEY") or set POLYGON_API_KEY.'
            )
        return api_key

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        api_key = self._resolved_api_key()
        if path.startswith("http://") or path.startswith("https://"):
            url = path
        else:
            url = f"{self._rest_base_url}{path}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "kalshi-alpha/polygon-client",
        }
        try:
            response: Response = self._session.request(
                method,
                url,
                params=params,
                timeout=self._timeout,
                headers=headers,
            )
        except requests.RequestException as exc:  # pragma: no cover - network errors
            raise PolygonAPIError(f"Polygon request failed: {exc}") from exc
        if response.status_code >= 400:
            raise PolygonAPIError(
                f"Polygon request failed ({response.status_code}): {response.text[:200]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise PolygonAPIError("Polygon response was not valid JSON") from exc
        status = payload.get("status")
        if status and status.upper() not in {"OK", "SUCCESS"}:
            message = payload.get("error") or payload.get("message") or str(payload)
            raise PolygonAPIError(f"Polygon API returned error ({status}): {message}")
        return payload

    def fetch_minute_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        adjusted: bool = True,
        limit: int = 50_000,
    ) -> list[MinuteBar]:
        """Fetch aggregated minute bars for the given index."""

        return self._fetch_aggregate_bars(
            symbol=symbol,
            start=start,
            end=end,
            multiplier=1,
            timespan="minute",
            adjusted=adjusted,
            limit=limit,
        )

    def fetch_second_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        adjusted: bool = True,
        limit: int = 50_000,
    ) -> list[MinuteBar]:
        """Fetch aggregated second bars for the given index."""

        return self._fetch_aggregate_bars(
            symbol=symbol,
            start=start,
            end=end,
            multiplier=1,
            timespan="second",
            adjusted=adjusted,
            limit=limit,
        )

    def _fetch_aggregate_bars(  # noqa: PLR0913
        self,
        *,
        symbol: str,
        start: datetime,
        end: datetime,
        multiplier: int,
        timespan: str,
        adjusted: bool,
        limit: int,
    ) -> list[MinuteBar]:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware")
        if end <= start:
            raise ValueError("end must be after start")

        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        path = f"/v2/aggs/ticker/{symbol}/range/{int(multiplier)}/{timespan}/{start_ms}/{end_ms}"
        params = {
            "adjusted": str(bool(adjusted)).lower(),
            "sort": "asc",
            "limit": int(limit),
        }
        bars: list[MinuteBar] = []
        for payload in self._paginate(path, params):
            bars.extend(self._parse_aggregate_payload(payload))
        return bars

    def _paginate(
        self,
        path: str,
        params: dict[str, Any] | None,
    ) -> Iterable[dict[str, Any]]:
        next_path = path
        next_params = dict(params) if params else None
        while True:
            payload = self._request("GET", next_path, params=next_params)
            yield payload
            next_url = payload.get("next_url")
            if not next_url:
                break
            next_path, next_params = self._parse_next_url(str(next_url))

    def _parse_next_url(self, url: str) -> tuple[str, dict[str, Any]]:
        parsed = urlparse(url)
        path = parsed.path or url
        if not path.startswith("http") and not path.startswith("/"):
            path = f"/{path}"
        params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
        return path, params

    def _parse_aggregate_payload(self, payload: Mapping[str, Any]) -> list[MinuteBar]:
        results = payload.get("results") or []
        bars: list[MinuteBar] = []
        for entry in results:
            try:
                timestamp = datetime.fromtimestamp(float(entry["t"]) / 1000.0, tz=UTC)
            except (KeyError, TypeError, ValueError) as exc:
                raise PolygonAPIError(f"invalid aggregate payload: {entry}") from exc
            bars.append(
                MinuteBar(
                    timestamp=timestamp,
                    open=float(entry.get("o", entry.get("open")) or 0.0),
                    high=float(entry.get("h", entry.get("high")) or 0.0),
                    low=float(entry.get("l", entry.get("low")) or 0.0),
                    close=float(entry.get("c", entry.get("close")) or 0.0),
                    volume=float(entry.get("v", entry.get("volume")) or 0.0),
                    vwap=float(entry.get("vw") or entry.get("vwap") or 0.0) or None,
                    trades=int(entry.get("n") or entry.get("transactions") or 0) or None,
                )
            )
        return bars

    def fetch_snapshot(self, symbol: str) -> IndexSnapshot:
        """Fetch the latest snapshot for an index ticker."""

        path = f"/v2/snapshot/locale/us/market/indices/tickers/{symbol}"
        payload = self._request("GET", path)
        results: dict[str, Any] = payload.get("results") or {}
        last_quote = results.get("lastQuote") or {}
        return IndexSnapshot(
            ticker=str(results.get("ticker") or symbol),
            last_price=float(last_quote.get("p") or last_quote.get("P") or 0.0) or None,
            change=float(results.get("todaysChange") or 0.0) or None,
            change_percent=float(results.get("todaysChangePerc") or 0.0) or None,
            previous_close=float(
                (results.get("prevDay") or {}).get("c")
                or (results.get("prevDay") or {}).get("close")
                or 0.0
            )
            or None,
            timestamp=(
                datetime.fromtimestamp(float(results.get("lastUpdated")) / 1000.0, tz=UTC)
                if results.get("lastUpdated")
                else None
            ),
        )

    # Websocket --------------------------------------------------------------

    @asynccontextmanager
    async def websocket(self) -> AsyncIterator[ClientConnection]:
        """Yield an authenticated websocket connection."""

        api_key = self._resolved_api_key()
        connection = await websockets.connect(
            self._ws_url,
            ping_interval=self._ws_ping_interval,
            open_timeout=self._ws_auth_timeout,
        )
        try:
            await self._authenticate(connection, api_key)
            yield connection
        finally:
            await connection.close()

    async def _authenticate(self, connection: ClientConnection, api_key: str) -> None:
        await connection.send(json.dumps({"action": "auth", "params": api_key}))
        try:
            raw = await asyncio.wait_for(connection.recv(), timeout=self._ws_auth_timeout)
        except TimeoutError as exc:  # pragma: no cover - network timeout
            raise PolygonAPIError("timeout while authenticating Polygon websocket") from exc
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:  # pragma: no cover - unexpected payload
            return
        status = str(message.get("status") or "").lower()
        if status and "success" not in status and "authenticated" not in status:
            detail = message.get("message") or message.get("error") or raw
            raise PolygonAPIError(f"Polygon websocket auth failed: {detail}")

    async def subscribe_minute_channels(
        self,
        connection: ClientConnection,
        symbols: Iterable[str],
    ) -> None:
        """Subscribe to minute bar updates for the provided symbols."""

        tickers = [symbol.strip() for symbol in symbols if symbol.strip()]
        if not tickers:
            return
        params = ",".join(f"{INDEX_MINUTE_CHANNEL}.{ticker}" for ticker in tickers)
        message = json.dumps({"action": "subscribe", "params": params})
        await connection.send(message)

    async def stream_minute_aggregates(
        self,
        symbols: Iterable[str],
        *,
        reconnect_attempts: int | None = 5,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield websocket aggregate payloads with automatic reconnect/backoff."""

        base_backoff = max(0.1, float(initial_backoff))
        ceiling = max(base_backoff, float(max_backoff))
        attempts = 0
        tickers = [symbol.strip() for symbol in symbols if symbol.strip()]

        while True:
            try:
                async with self.websocket() as connection:
                    if tickers:
                        await self.subscribe_minute_channels(connection, tickers)
                    attempts = 0
                    backoff = base_backoff
                    while True:
                        raw = await connection.recv()
                        try:
                            message = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        event_type = str(message.get("ev") or "").upper()
                        if event_type != INDEX_MINUTE_CHANNEL:
                            continue
                        yield message
            except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
                raise
            except (
                ConnectionClosedError,
                ConnectionClosedOK,
                PolygonAPIError,
                TimeoutError,
                OSError,
            ):
                attempts += 1
                if reconnect_attempts is not None and attempts > reconnect_attempts:
                    raise
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, ceiling)
                continue


__all__ = [
    "PolygonAPIError",
    "PolygonIndicesClient",
    "MinuteBar",
    "IndexSnapshot",
]
