"""Massive indices client supporting REST, websocket, and historical ingestion."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import polars as pl
import requests
import websockets
from requests import Response
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from kalshi_alpha.datastore.paths import RAW_ROOT
from kalshi_alpha.utils.keys import load_polygon_api_key

DEFAULT_REST_URL = "https://api.polygon.io"
DEFAULT_WS_URL = "wss://socket.massive.com/indices"
INDEX_MINUTE_CHANNEL = "AM"
INDEX_SECOND_CHANNEL = "AS"
CHANNEL_BY_TIMESPAN = {
    "minute": INDEX_MINUTE_CHANNEL,
    "second": INDEX_SECOND_CHANNEL,
}
CHANNEL_EVENT_ALIASES: dict[str, set[str]] = {
    INDEX_MINUTE_CHANNEL: {INDEX_MINUTE_CHANNEL, "XA"},
    INDEX_SECOND_CHANNEL: {INDEX_SECOND_CHANNEL, "XS"},
}
ET = ZoneInfo("America/New_York")
POLYGON_RAW_ROOT = (RAW_ROOT / "polygon").resolve()


class PolygonAPIError(RuntimeError):
    """Raised when the Polygon API returns an error."""


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


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
    """REST + websocket client for Massive (Polygon) index data."""

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
            payload = cast("dict[str, Any]", response.json())
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

    def download_minute_history(  # noqa: PLR0913
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        output_root: Path | None = None,
        chunk_limit: int = 50_000,
        adjusted: bool = True,
    ) -> list[Path]:
        """Download minute history into per-day parquet files under the raw datastore.

        The Polygon REST API enforces a 50k row ceiling per aggregation request.  This
        method walks the requested window in chunks (controlled by ``chunk_limit``),
        accumulates bars grouped by the underlying trading day (Eastern Time), and
        writes one parquet per day to ``data/raw/polygon/{symbol_slug}/YYYY-MM-DD.parquet``.
        """

        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be timezone-aware")
        if end <= start:
            raise ValueError("end must be after start")
        if chunk_limit <= 0:
            raise ValueError("chunk_limit must be positive")

        base_root = (output_root or POLYGON_RAW_ROOT).resolve()
        base_root.mkdir(parents=True, exist_ok=True)
        symbol_slug = symbol.replace(":", "_").upper()
        symbol_dir = base_root / symbol_slug
        symbol_dir.mkdir(parents=True, exist_ok=True)

        records_by_day: dict[tuple[int, int, int], list[dict[str, object]]] = defaultdict(list)
        chunk_minutes = max(chunk_limit - 1, 1)
        chunk_delta = timedelta(minutes=chunk_minutes)
        current_start = start
        while current_start < end:
            chunk_end = min(end, current_start + chunk_delta)
            bars = self.fetch_minute_bars(
                symbol,
                current_start,
                chunk_end,
                adjusted=adjusted,
                limit=chunk_limit,
            )
            if bars:
                for bar in bars:
                    day = bar.timestamp.astimezone(ET).date()
                    key = (day.year, day.month, day.day)
                    records_by_day[key].append(self._bar_to_row(bar))
                last_timestamp = bars[-1].timestamp + timedelta(minutes=1)
                if last_timestamp <= current_start:
                    current_start = chunk_end + timedelta(minutes=1)
                else:
                    current_start = min(end, last_timestamp)
            else:
                current_start = chunk_end + timedelta(minutes=1)

        written: list[Path] = []
        for key in sorted(records_by_day):
            rows = records_by_day[key]
            if not rows:
                continue
            frame = pl.DataFrame(rows).unique(subset="timestamp").sort("timestamp")
            day_label = f"{key[0]:04d}-{key[1]:02d}-{key[2]:02d}"
            path = symbol_dir / f"{day_label}.parquet"
            frame.write_parquet(path)
            written.append(path)
        return written

    def fetch_snapshot(self, symbol: str) -> IndexSnapshot:
        """Fetch the latest snapshot for an index ticker."""

        path = f"/v2/snapshot/locale/us/market/indices/tickers/{symbol}"
        payload = self._request("GET", path)
        results = cast("dict[str, Any]", payload.get("results") or {})
        last_quote = cast("dict[str, Any]", results.get("lastQuote") or {})
        prev_day = cast("dict[str, Any]", results.get("prevDay") or {})
        last_updated = results.get("lastUpdated")
        last_price = _safe_float(last_quote.get("p")) or _safe_float(last_quote.get("P"))
        change = _safe_float(results.get("todaysChange"))
        change_percent = _safe_float(results.get("todaysChangePerc"))
        previous_close = _safe_float(prev_day.get("c")) or _safe_float(prev_day.get("close"))
        return IndexSnapshot(
            ticker=str(results.get("ticker") or symbol),
            last_price=last_price,
            change=change,
            change_percent=change_percent,
            previous_close=previous_close,
            timestamp=(
                datetime.fromtimestamp(float(last_updated) / 1000.0, tz=UTC)
                if isinstance(last_updated, (int, float))
                else None
            ),
        )

    @staticmethod
    def _bar_to_row(bar: MinuteBar) -> dict[str, object]:
        trades = None if bar.trades is None else int(bar.trades)
        vwap = None if bar.vwap is None else float(bar.vwap)
        return {
            "timestamp": bar.timestamp,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": float(bar.volume),
            "vwap": vwap,
            "trades": trades,
        }

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

    async def subscribe_aggregate_channel(
        self,
        connection: ClientConnection,
        symbols: Iterable[str],
        *,
        timespan: str = "minute",
    ) -> None:
        """Subscribe to aggregate updates for the provided symbols."""

        tickers = [symbol.strip() for symbol in symbols if symbol.strip()]
        if not tickers:
            return
        try:
            channel = CHANNEL_BY_TIMESPAN[timespan]
        except KeyError as exc:
            raise ValueError(f"Unsupported aggregate timespan '{timespan}'") from exc
        params = ",".join(f"{channel}.{ticker}" for ticker in tickers)
        message = json.dumps({"action": "subscribe", "params": params})
        await connection.send(message)

    async def stream_aggregates(
        self,
        symbols: Iterable[str],
        *,
        timespan: str = "minute",
        reconnect_attempts: int | None = 5,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Yield websocket aggregate payloads with automatic reconnect/backoff."""

        try:
            channel = CHANNEL_BY_TIMESPAN[timespan]
        except KeyError as exc:
            raise ValueError(f"Unsupported aggregate timespan '{timespan}'") from exc
        expected_events = CHANNEL_EVENT_ALIASES.get(channel, {channel})

        base_backoff = max(0.1, float(initial_backoff))
        ceiling = max(base_backoff, float(max_backoff))
        attempts = 0
        tickers = [symbol.strip() for symbol in symbols if symbol.strip()]

        while True:
            try:
                async with self.websocket() as connection:
                    if tickers:
                        await self.subscribe_aggregate_channel(connection, tickers, timespan=timespan)
                    attempts = 0
                    backoff = base_backoff
                    while True:
                        raw = await connection.recv()
                        try:
                            message = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        event_type = str(message.get("ev") or "").upper()
                        if event_type not in expected_events:
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

    async def stream_minute_aggregates(
        self,
        symbols: Iterable[str],
        *,
        reconnect_attempts: int | None = 5,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Backward-compatible minute aggregate stream wrapper."""

        async for message in self.stream_aggregates(
            symbols,
            timespan="minute",
            reconnect_attempts=reconnect_attempts,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff,
        ):
            yield message


__all__ = [
    "PolygonAPIError",
    "PolygonIndicesClient",
    "MinuteBar",
    "IndexSnapshot",
]
