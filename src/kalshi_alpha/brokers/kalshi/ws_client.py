"""Authenticated Kalshi WebSocket client with reconnect/backoff."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.connection import State as WebsocketState
from websockets.exceptions import ConnectionClosedError

from kalshi_alpha.brokers.kalshi.http_client import KalshiHttpClient
from kalshi_alpha.exec.telemetry.sink import TelemetrySink

LOGGER = logging.getLogger(__name__)


class KalshiWebsocketError(RuntimeError):
    """Raised when the Kalshi websocket client exhausts reconnect attempts."""


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


_STATUS_EVENT_MAP = {
    "acknowledged": "ack",
    "accepted": "ack",
    "partial_fill": "partial_fill",
    "filled": "fill",
    "canceled": "cancel",
    "cancelled": "cancel",
    "rejected": "reject",
}


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
        telemetry_sink: TelemetrySink | None = None,
        clock: Callable[[], datetime] | None = None,
        heartbeat_timeout: float | None = None,
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
        self._telemetry = telemetry_sink
        self._clock = clock or _utc_now
        self._subscriptions: list[str] = []
        self._heartbeat_timeout = heartbeat_timeout
        if self._heartbeat_timeout is None and ping_interval is not None:
            self._heartbeat_timeout = max(ping_interval * 2.0, 30.0)
        self._last_heartbeat: datetime | None = None
        self._needs_resubscribe = False

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
                self._emit_ws_event(
                    "ws_reconnect",
                    {"status": "retry", "attempt": attempt, "error": str(exc)},
                )
                if attempt >= self._max_retries:
                    break
                await asyncio.sleep(self._retry_backoff * (2 ** (attempt - 1)))
            else:
                self._last_heartbeat = self._clock()
                should_resubscribe = self._needs_resubscribe or attempt > 1
                if should_resubscribe and self._subscriptions:
                    await self._send_pending_subscriptions(force=True)
                self._needs_resubscribe = False
                self._emit_ws_event(
                    "ws_reconnect",
                    {"status": "connected", "attempt": attempt},
                )
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
        if self._is_open():
            await self._check_watchdog()
        connection: ClientConnection | None
        if not self._is_open():
            connection = await self.connect()
        else:
            connection = self._connection
        if connection is None:
            connection = await self.connect()
        return connection

    async def subscribe(self, payload: Mapping[str, Any]) -> None:
        message = json.dumps(payload)
        if message not in self._subscriptions:
            self._subscriptions.append(message)
        websocket = await self.ensure_connection()
        await websocket.send(message)

    async def receive(self) -> Mapping[str, Any] | str:
        websocket = await self.ensure_connection()
        try:
            message = await websocket.recv()
        except ConnectionClosedError as exc:  # pragma: no cover - reconnect triggered in tests
            LOGGER.warning("Kalshi websocket closed while receiving: %s", exc)
            self._emit_ws_event("ws_disconnect", {"error": str(exc)})
            self._connection = None
            self._needs_resubscribe = True
            raise
        decoded = self._decode_message(message)
        if isinstance(decoded, Mapping):
            self._process_message(decoded)
            return decoded
        return decoded

    async def close(self) -> None:
        if self._connection is not None and self._connection.state is not WebsocketState.CLOSED:
            await self._connection.close()
        self._connection = None
        self._needs_resubscribe = True

    async def _check_watchdog(self) -> None:
        if self._heartbeat_timeout is None or not self._is_open():
            return
        if self._last_heartbeat is None:
            return
        now = self._clock()
        delta = (now - self._last_heartbeat).total_seconds()
        if delta <= self._heartbeat_timeout:
            return
        self._emit_ws_event(
            "ws_heartbeat_timeout",
            {"elapsed": delta, "timeout": self._heartbeat_timeout},
        )
        if self._connection is not None:
            await self._connection.close()
        self._connection = None
        self._needs_resubscribe = True

    def _is_open(self) -> bool:
        return self._connection is not None and self._connection.state is WebsocketState.OPEN

    async def _send_pending_subscriptions(self, *, force: bool = False) -> None:
        if not self._subscriptions or not self._is_open():
            return
        if not force:
            return
        websocket = self._connection
        if websocket is None:
            return
        for payload in self._subscriptions:
            await websocket.send(payload)

    def _emit_ws_event(self, event_type: str, data: Mapping[str, Any]) -> None:
        if self._telemetry is None:
            return
        self._telemetry.emit(event_type, source="ws", data=data)

    def _decode_message(self, message: object) -> Mapping[str, Any] | str:
        if isinstance(message, bytes):
            message = message.decode("utf-8", errors="replace")
        if isinstance(message, str):
            try:
                parsed = json.loads(message)
            except json.JSONDecodeError:
                self._emit_ws_event("ws_malformed", {"reason": "json_decode", "raw": message[:256]})
                return message
            if isinstance(parsed, Mapping):
                return parsed
            return message
        if isinstance(message, Mapping):
            return message
        self._emit_ws_event(
            "ws_malformed",
            {"reason": "unsupported_type", "type": type(message).__name__},
        )
        return str(message)

    def _process_message(self, message: Mapping[str, Any]) -> None:
        msg_type = str(message.get("type") or "").lower()
        if msg_type == "heartbeat":
            self._last_heartbeat = self._clock()
            self._emit_ws_event("heartbeat", {"ts": message.get("ts")})
            return
        if msg_type == "order_update":
            status = str(message.get("status") or "").lower()
            event = _STATUS_EVENT_MAP.get(status)
            if event is None:
                self._emit_ws_event(
                    "ws_malformed",
                    {"reason": "unknown_status", "status": status, "message": message},
                )
                return
            payload = {
                "order_id": message.get("order_id"),
                "idempotency_key": message.get("idempotency_key"),
                "status": status,
                "filled_size": message.get("filled_size"),
                "remaining_size": message.get("remaining_size"),
                "fill_price": message.get("fill_price"),
                "book_snapshot": message.get("book_snapshot"),
                "ts_ms": message.get("ts_ms"),
            }
            self._emit_ws_event(event, payload)
            return
        if msg_type == "error":
            self._emit_ws_event("reject", {"message": message})
            return
        if msg_type:
            self._emit_ws_event(
                "ws_malformed",
                {"reason": "unknown_type", "type": msg_type, "message": message},
            )
_HAS_ADDITIONAL_HEADERS = "additional_headers" in inspect.signature(websockets.connect).parameters
