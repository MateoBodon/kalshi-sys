"""Authenticated Kalshi WebSocket client with reconnect/backoff."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.connection import State as WebsocketState
from websockets.exceptions import ConnectionClosedError

from kalshi_alpha.brokers.kalshi.http_client import KalshiHttpClient

LOGGER = logging.getLogger(__name__)


class KalshiWebsocketError(RuntimeError):
    """Raised when the Kalshi websocket client exhausts reconnect attempts."""


class KalshiWebsocketClient:
    """Small helper wrapping an authenticated Kalshi websocket connection."""

    def __init__(  # noqa: PLR0913 - websocket wiring requires multiple knobs
        self,
        *,
        base_url: str,
        http_client: KalshiHttpClient,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        connect_timeout: float = 10.0,
        ping_interval: float | None = 20.0,
        ping_timeout: float | None = 20.0,
    ) -> None:
        if not base_url.startswith("ws"):
            raise ValueError("Kalshi websocket base URL must start with ws:// or wss://")
        self._base_url = base_url
        self._http_client = http_client
        self._max_retries = max(1, max_retries)
        self._retry_backoff = max(0.0, retry_backoff)
        self._timeout = connect_timeout
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        parsed = urlparse(base_url)
        self._path = parsed.path or "/"
        self._connection: ClientConnection | None = None

    async def connect(self) -> ClientConnection:
        if self._is_open():
            return self._connection  # type: ignore[return-value]
        self._connection = None

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                headers = self._http_client.build_auth_headers("GET", self._path)
                connect_kwargs: dict[str, Any] = {
                    "open_timeout": self._timeout,
                    "ping_interval": self._ping_interval,
                    "ping_timeout": self._ping_timeout,
                }
                header_key = "additional_headers" if _HAS_ADDITIONAL_HEADERS else "extra_headers"
                connect_kwargs[header_key] = headers
                self._connection = await websockets.connect(self._base_url, **connect_kwargs)
            except Exception as exc:  # pragma: no cover - exercised in tests via reconnect
                last_error = exc
                LOGGER.warning(
                    "Kalshi websocket connection attempt failed",
                    extra={
                        "kalshi": {
                            "path": self._path,
                            "attempt": attempt,
                            "error": str(exc),
                        }
                    },
                )
                if attempt >= self._max_retries:
                    break
                await asyncio.sleep(self._retry_backoff * (2 ** (attempt - 1)))
            else:
                LOGGER.info(
                    "Kalshi websocket connected",
                    extra={
                        "kalshi": {
                            "path": self._path,
                            "attempt": attempt,
                        }
                    },
                )
                return self._connection

        raise KalshiWebsocketError(
            "Failed to establish Kalshi websocket connection"
        ) from last_error

    async def ensure_connection(self) -> ClientConnection:
        connection = await self.connect()
        return connection

    async def subscribe(self, payload: Mapping[str, Any]) -> None:
        websocket = await self.ensure_connection()
        message = json.dumps(payload)
        await websocket.send(message)

    async def receive(self) -> str:
        websocket = await self.ensure_connection()
        try:
            message = await websocket.recv()
        except ConnectionClosedError as exc:  # pragma: no cover - reconnect triggered in tests
            LOGGER.warning("Kalshi websocket closed while receiving: %s", exc)
            self._connection = None
            raise
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        return message

    async def close(self) -> None:
        if self._connection is not None and self._connection.state is not WebsocketState.CLOSED:
            await self._connection.close()
        self._connection = None

    def _is_open(self) -> bool:
        return self._connection is not None and self._connection.state is WebsocketState.OPEN
_HAS_ADDITIONAL_HEADERS = "additional_headers" in inspect.signature(websockets.connect).parameters
