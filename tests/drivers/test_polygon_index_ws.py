from __future__ import annotations

from datetime import UTC, datetime

import pytest

from kalshi_alpha.drivers.polygon_index_ws import (
    PolygonIndexWSConfig,
    active_connection_count,
    close_shared_connection,
    get_shared_connection,
    last_message_age_seconds,
    last_message_at,
    polygon_index_ws,
)


class _FakeIndicesClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)
        self.stream_calls = 0
        self.closed = False

    def stream_aggregates(self, *_args, **_kwargs):
        self.stream_calls += 1

        async def _gen():
            try:
                for payload in list(self.payloads):
                    yield payload
            finally:
                self.closed = True

        return _gen()


@pytest.mark.asyncio
async def test_get_shared_connection_returns_singleton_and_resets() -> None:
    client = _FakeIndicesClient([{"sym": "I:SPX"}])
    config = PolygonIndexWSConfig(symbols=("I:SPX",), client_factory=lambda: client)

    first = get_shared_connection(config)
    second = get_shared_connection(config)
    assert first is second

    messages: list[dict[str, object]] = []
    async with polygon_index_ws(config) as ws:
        async for message in ws:
            messages.append(message)
    assert messages, "expected at least one streamed message"
    assert client.closed
    assert active_connection_count() == 0
    assert last_message_at() is not None
    assert last_message_age_seconds(datetime.now(tz=UTC)) is not None

    replacement = get_shared_connection(config)
    assert replacement is not first
    await close_shared_connection()


@pytest.mark.asyncio
async def test_close_shared_connection_closes_stream() -> None:
    client = _FakeIndicesClient([{"sym": "I:NDX"}])
    config = PolygonIndexWSConfig(symbols=("I:NDX",), client_factory=lambda: client)

    connection = get_shared_connection(config)
    await connection.start()
    assert active_connection_count() == 1

    await close_shared_connection()
    assert active_connection_count() == 0
    replacement = get_shared_connection(config)
    assert replacement is not connection
    await close_shared_connection()
