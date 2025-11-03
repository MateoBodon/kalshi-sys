from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

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
