from __future__ import annotations

import pytest

from kalshi_alpha.drivers.polygon_index_ws import (
    PolygonIndexWSConfig,
    active_connection_count,
    close_shared_connection,
    get_shared_connection,
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
async def test_runner_opens_once_and_closes_after_run() -> None:
    client = _FakeIndicesClient([{"sym": "I:SPX"}])
    config = PolygonIndexWSConfig(symbols=("I:SPX",), client_factory=lambda: client)

    async def _fake_runner() -> None:
        async with polygon_index_ws(config) as ws:
            # get_shared_connection should reuse the same instance
            assert get_shared_connection(config) is ws
            counts_seen: list[int] = []
            counts_seen.append(active_connection_count())
            async for _message in ws:
                counts_seen.append(active_connection_count())
                break
            assert all(count == 1 for count in counts_seen)

    await _fake_runner()

    assert client.stream_calls == 1
    assert client.closed
    assert active_connection_count() == 0
    await close_shared_connection()
