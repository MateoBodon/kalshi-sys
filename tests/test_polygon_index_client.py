from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, AsyncIterator

import pytest

from kalshi_alpha.drivers.polygon_index.client import IndexSnapshot, MinuteBar, PolygonIndicesClient


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = "stub"

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.last_request: dict[str, Any] | None = None

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str],
        timeout: float,
    ) -> _FakeResponse:
        self.last_request = {
            "method": method,
            "url": url,
            "params": params,
            "headers": headers,
            "timeout": timeout,
        }
        return self._response


class _MultiResponseSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str],
        timeout: float,
    ) -> _FakeResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )
        if not self._responses:
            raise AssertionError("unexpected extra request")
        return self._responses.pop(0)


def test_fetch_minute_bars_parses_results(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "status": "OK",
        "results": [
            {"t": 1700000000000, "o": 5000.0, "h": 5002.0, "l": 4999.0, "c": 5001.0, "v": 1200, "n": 45},
            {"t": 1700000060000, "o": 5001.0, "h": 5003.0, "l": 5000.5, "c": 5002.5, "v": 900, "n": 30},
        ],
    }
    response = _FakeResponse(status_code=200, payload=payload)
    session = _FakeSession(response)
    client = PolygonIndicesClient(api_key="stub", session=session)
    start = datetime(2023, 11, 1, 15, 30, tzinfo=UTC)
    end = start + timedelta(minutes=2)
    bars = client.fetch_minute_bars("I:SPX", start, end)
    assert len(bars) == 2
    assert isinstance(bars[0], MinuteBar)
    assert bars[0].close == 5001.0


def test_fetch_minute_bars_handles_pagination() -> None:
    first = {
        "status": "OK",
        "results": [
            {"t": 1700000000000, "o": 5000.0, "h": 5001.0, "l": 4999.5, "c": 5000.5, "v": 1000, "n": 25}
        ],
        "next_url": "https://api.polygon.io/v2/aggs/ticker/I:SPX/range/1/minute/0/1?cursor=abc",
    }
    second = {
        "status": "OK",
        "results": [
            {"t": 1700000060000, "o": 5000.5, "h": 5002.5, "l": 5000.0, "c": 5002.0, "v": 800, "n": 20}
        ],
    }
    session = _MultiResponseSession([
        _FakeResponse(status_code=200, payload=first),
        _FakeResponse(status_code=200, payload=second),
    ])
    client = PolygonIndicesClient(api_key="stub", session=session)
    start = datetime(2023, 11, 1, 15, 30, tzinfo=UTC)
    end = start + timedelta(minutes=3)
    bars = client.fetch_minute_bars("I:SPX", start, end)
    assert len(bars) == 2
    assert session.calls and session.calls[1]["url"].startswith("https://api.polygon.io/v2/aggs/ticker/I:SPX")


def test_fetch_snapshot_parses_payload() -> None:
    payload = {
        "status": "OK",
        "results": {
            "ticker": "I:SPX",
            "lastQuote": {"p": 5030.25},
            "todaysChange": 10.0,
            "todaysChangePerc": 0.2,
            "prevDay": {"c": 5020.0},
            "lastUpdated": 1700000000000,
        },
    }
    session = _FakeSession(_FakeResponse(status_code=200, payload=payload))
    client = PolygonIndicesClient(api_key="stub", session=session)
    snapshot = client.fetch_snapshot("I:SPX")
    assert isinstance(snapshot, IndexSnapshot)
    assert snapshot.last_price == pytest.approx(5030.25)
    assert snapshot.previous_close == pytest.approx(5020.0)


@pytest.mark.asyncio
async def test_websocket_authenticates(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: list[str] = []

    class _FakeConnection:
        def __init__(self) -> None:
            self.closed = False

        async def send(self, message: str) -> None:
            messages.append(message)

        async def recv(self) -> str:
            return "{\"status\": \"auth_success\"}"

        async def close(self) -> None:
            self.closed = True

    async def _fake_connect(*args: Any, **kwargs: Any) -> _FakeConnection:
        return _FakeConnection()

    monkeypatch.setattr("websockets.connect", _fake_connect)
    client = PolygonIndicesClient(api_key="stub")
    async with client.websocket() as connection:
        assert connection is not None
    assert any("auth" in msg for msg in messages)


@pytest.mark.asyncio
async def test_stream_minute_aggregates_reconnects(monkeypatch: pytest.MonkeyPatch) -> None:
    client = PolygonIndicesClient(api_key="stub")

    class _FakeConnection:
        def __init__(self, payloads: list[str]) -> None:
            self.payloads = payloads
            self.sent: list[str] = []
            self.closed = False

        async def send(self, message: str) -> None:
            self.sent.append(message)

        async def recv(self) -> str:
            if not self.payloads:
                raise OSError("connection closed")
            return self.payloads.pop(0)

        async def close(self) -> None:
            self.closed = True

    connections = [
        _FakeConnection(['{"ev": "XA", "sym": "I:SPX", "p": 5000.0}']),
        _FakeConnection(['{"ev": "XA", "sym": "I:SPX", "p": 5001.0}']),
    ]
    iterator = iter(connections)

    @asynccontextmanager
    async def _fake_websocket() -> AsyncIterator[_FakeConnection]:
        try:
            connection = next(iterator)
        except StopIteration as exc:  # pragma: no cover - defensive
            raise AssertionError("websocket opened too many times") from exc
        try:
            yield connection
        finally:
            await connection.close()

    async def _sleep_stub(_delay: float) -> None:
        return None

    monkeypatch.setattr(client, "websocket", _fake_websocket)
    monkeypatch.setattr(asyncio, "sleep", _sleep_stub)

    stream = client.stream_minute_aggregates(["I:SPX"], reconnect_attempts=2, initial_backoff=0.0, max_backoff=0.0)
    first = await anext(stream)
    second = await anext(stream)
    await stream.aclose()

    assert first["ev"] == "XA"
    assert second["ev"] == "XA"
    assert any("XA.I:SPX" in msg for msg in connections[0].sent)
