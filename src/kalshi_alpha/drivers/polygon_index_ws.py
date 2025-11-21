"""Shared Polygon index websocket helper with singleton lifecycle and metrics."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from kalshi_alpha.drivers.polygon_index.client import INDICES_WS_URL, PolygonIndicesClient

DEFAULT_SYMBOLS: tuple[str, ...] = ("I:SPX", "I:NDX")


@dataclass(slots=True)
class PolygonIndexWSConfig:
    """Configuration for the shared Polygon index websocket connection."""

    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    ws_url: str = INDICES_WS_URL
    timespan: str = "second"
    reconnect_attempts: int | None = None
    client_factory: Callable[[], PolygonIndicesClient] | None = None
    connection_factory: Callable[[PolygonIndexWSConfig], "PolygonWSClient"] | None = None


class PolygonWSClient:
    """Lightweight wrapper around PolygonIndicesClient.stream_aggregates."""

    def __init__(
        self,
        *,
        client: PolygonIndicesClient,
        symbols: Sequence[str],
        timespan: str = "second",
        reconnect_attempts: int | None = None,
    ) -> None:
        self._client = client
        self._symbols = tuple(symbols)
        self._timespan = timespan
        self._reconnect_attempts = reconnect_attempts
        self._stream: AsyncGenerator[dict[str, Any], None] | None = None
        self._consuming = False
        self._closed = False
        self._started = False

    def __aiter__(self) -> AsyncGenerator[dict[str, Any], None]:
        return self.messages()

    async def __aenter__(self) -> "PolygonWSClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001 - context protocol
        await self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    async def start(self) -> None:
        """Start the websocket stream if not already started."""

        if self._stream is not None:
            return
        self._stream = self._client.stream_aggregates(
            self._symbols,
            timespan=self._timespan,
            reconnect_attempts=self._reconnect_attempts,
        )
        _increment_active_connections()
        self._started = True
        self._closed = False

    async def messages(self) -> AsyncGenerator[dict[str, Any], None]:
        """Yield websocket messages, updating the shared freshness marker."""

        await self.start()
        if self._stream is None:
            return
        if self._consuming:
            raise RuntimeError("PolygonWSClient is already being consumed.")
        self._consuming = True
        try:
            async for message in self._stream:
                self.mark_message()
                yield message
        except asyncio.CancelledError:
            raise
        finally:
            self._consuming = False

    async def close(self) -> None:
        """Terminate the stream and release the shared connection slot."""

        if self._stream is not None:
            try:
                await self._stream.aclose()
            finally:
                self._stream = None
        if self._started:
            _decrement_active_connections()
        self._started = False
        self._closed = True

    def mark_message(self, timestamp: datetime | None = None) -> None:
        """Record the arrival time of a websocket message."""

        _record_message_timestamp(timestamp)


_shared_connection: PolygonWSClient | None = None
_active_connections = 0
_last_message_at: datetime | None = None


def active_connection_count() -> int:
    """Return the number of active Polygon index websocket connections."""

    return _active_connections


def last_message_at() -> datetime | None:
    """Return the timestamp of the most recent websocket message, if any."""

    return _last_message_at


def last_message_age_seconds(now: datetime | None = None) -> float | None:
    """Return the age in seconds of the most recent websocket message."""

    if _last_message_at is None:
        return None
    reference = now or datetime.now(tz=UTC)
    reference = reference if reference.tzinfo else reference.replace(tzinfo=UTC)
    delta = (reference.astimezone(UTC) - _last_message_at).total_seconds()
    return max(delta, 0.0)


def get_shared_connection(config: PolygonIndexWSConfig | None = None) -> PolygonWSClient:
    """Return the singleton Polygon websocket client for this process."""

    global _shared_connection

    cfg = config or PolygonIndexWSConfig()
    if _shared_connection is not None and not _shared_connection.closed:
        return _shared_connection

    if cfg.connection_factory is not None:
        _shared_connection = cfg.connection_factory(cfg)
        return _shared_connection

    factory = cfg.client_factory or (lambda: PolygonIndicesClient(ws_url=cfg.ws_url))
    client = factory()
    _shared_connection = PolygonWSClient(
        client=client,
        symbols=cfg.symbols,
        timespan=cfg.timespan,
        reconnect_attempts=cfg.reconnect_attempts,
    )
    return _shared_connection


async def close_shared_connection() -> None:
    """Close and reset the shared websocket connection, if active."""

    global _shared_connection

    if _shared_connection is None:
        return
    try:
        await _shared_connection.close()
    finally:
        _shared_connection = None


@asynccontextmanager
async def polygon_index_ws(config: PolygonIndexWSConfig | None = None) -> AsyncGenerator[PolygonWSClient, None]:
    """Context manager that yields the shared connection and cleans up on exit."""

    connection = get_shared_connection(config)
    await connection.start()
    try:
        yield connection
    finally:
        await close_shared_connection()


def _record_message_timestamp(timestamp: datetime | None = None) -> None:
    global _last_message_at

    moment = timestamp or datetime.now(tz=UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    _last_message_at = moment.astimezone(UTC)


def _increment_active_connections() -> None:
    global _active_connections
    _active_connections += 1


def _decrement_active_connections() -> None:
    global _active_connections
    _active_connections = max(_active_connections - 1, 0)


__all__ = [
    "PolygonIndexWSConfig",
    "PolygonWSClient",
    "active_connection_count",
    "close_shared_connection",
    "get_shared_connection",
    "last_message_age_seconds",
    "last_message_at",
    "polygon_index_ws",
]
