from __future__ import annotations

from pathlib import Path

import pytest

from kalshi_alpha.core.execution.order_queue import OrderQueue
from kalshi_alpha.brokers.kalshi.base import BrokerOrder


def _sample_order(key: str) -> BrokerOrder:
    return BrokerOrder(
        idempotency_key=key,
        market_id="MKT-1",
        strike=250.0,
        side="YES",
        price=0.45,
        contracts=2,
        probability=0.6,
        metadata={"order_id": f"O-{key}"},
    )


def test_queue_processes_fifo() -> None:
    executed: list[tuple[str, str]] = []
    queue = OrderQueue(capacity=10, sleep=lambda _: None)

    queue.enqueue_cancel("A", lambda order_id: executed.append(("cancel", order_id)))
    queue.enqueue_cancel("B", lambda order_id: executed.append(("cancel", order_id)))

    assert executed == [("cancel", "A"), ("cancel", "B")]


def test_replace_cancels_before_place() -> None:
    events: list[str] = []
    queue = OrderQueue(capacity=5, sleep=lambda _: None)
    order = _sample_order("123")

    queue.enqueue_replace(
        order_id="O-OLD",
        new_order=order,
        cancel_fn=lambda order_id: events.append(f"cancel:{order_id}"),
        place_fn=lambda order_id, new_order: events.append(f"place:{order_id}:{new_order.idempotency_key}"),
    )

    assert events == [
        "cancel:O-OLD",
        "place:O-OLD:123",
    ]


def test_queue_retries_and_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    attempts = {"count": 0}

    queue = OrderQueue(capacity=3, max_retries=3, backoff=0.1, sleep=lambda delay: sleeps.append(delay))

    def flaky(order_id: str) -> None:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("network error")
        events.append(order_id)

    events: list[str] = []
    queue.enqueue_cancel("X", flaky)

    assert events == ["X"]
    assert sleeps == [0.1, 0.2]


def test_queue_drops_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    audits: list[dict[str, object]] = []

    def audit(action: str, payload: dict[str, object]) -> None:
        audits.append({"action": action, **payload})

    queue = OrderQueue(capacity=3, max_retries=2, backoff=0.0, sleep=lambda _: None, audit_callback=audit)

    def always_fail(order_id: str) -> None:
        raise RuntimeError("permanent failure")

    queue.enqueue_cancel("FAIL", always_fail)

    assert not queue.depth()
    assert audits
    record = audits[0]
    assert record["action"] == "queue_drop"
    assert record["order_id"] == "FAIL"
