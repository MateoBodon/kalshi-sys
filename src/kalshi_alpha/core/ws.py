"""Lightweight Kalshi websocket client with RSA-PSS authentication."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import websockets
from websockets.asyncio.client import ClientConnection

from kalshi_alpha.brokers.kalshi.http_client import KalshiHttpClient

DEFAULT_WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
DEFAULT_USER_AGENT = "kalshi-alpha/ws-client"


class KalshiWebsocketClient:
    """Minimal Kalshi websocket client that signs the handshake with RSA-PSS."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_WS_URL,
        http_client: KalshiHttpClient | None = None,
        ping_interval: float = 30.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"ws", "wss"}:
            raise ValueError("Kalshi websocket URL must start with ws:// or wss://")
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        self._base_url = base_url
        self._path = path
        self._http_client = http_client or KalshiHttpClient()
        self._ping_interval = max(5.0, float(ping_interval))
        self._user_agent = user_agent

    def _auth_headers(self) -> dict[str, str]:
        headers = self._http_client.build_auth_headers("GET", self._path, absolute=True)
        headers.setdefault("User-Agent", self._user_agent)
        return headers

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ClientConnection]:
        """Yield an authenticated websocket connection."""

        headers = self._auth_headers()
        connection = await websockets.connect(
            self._base_url,
            additional_headers=headers,
            ping_interval=self._ping_interval,
        )
        try:
            yield connection
        finally:
            await connection.close()


__all__ = ["KalshiWebsocketClient", "DEFAULT_WS_URL"]
