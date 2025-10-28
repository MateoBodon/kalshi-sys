"""FIFO order queue for coordinating cancel/replace operations."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Tuple, TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover
    from kalshi_alpha.brokers.kalshi.base import BrokerOrder


AuditCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class _QueueItem:
    action: str
    execute: Callable[[], None]
    metadata: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    last_error: str | None = None


class OrderQueue:
    """Synchronous queue that retries cancel/replace operations with backoff."""

    def __init__(
        self,
        *,
        capacity: int = 32,
        max_retries: int = 3,
        backoff: float = 0.5,
        sleep: Callable[[float], None] | None = None,
        audit_callback: AuditCallback | None = None,
    ) -> None:
        self._capacity = max(1, capacity)
        self._max_retries = max(1, max_retries)
        self._backoff = max(0.0, backoff)
        self._sleep = sleep or time.sleep
        self._audit = audit_callback
        self._queue: Deque[_QueueItem] = deque()

    # Public API ------------------------------------------------------------------------------

    def enqueue_cancel(self, order_id: str, cancel_fn: Callable[[str], None]) -> None:
        self._add(
            "cancel",
            execute=lambda: cancel_fn(order_id),
            metadata={"order_id": order_id},
        )

    def enqueue_replace(
        self,
        *,
        order_id: str,
        new_order: "BrokerOrder",
        cancel_fn: Callable[[str], None],
        place_fn: Callable[[str, "BrokerOrder"], None],
    ) -> None:
        self._add(
            "replace",
            execute=lambda: self._execute_replace(order_id, new_order, cancel_fn, place_fn),
            metadata={
                "order_id": order_id,
                "idempotency_key": new_order.idempotency_key,
                "market_id": new_order.market_id,
            },
        )

    def enqueue_custom(self, action: str, func: Callable[[], None], metadata: dict[str, Any] | None = None) -> None:
        self._add(action, execute=func, metadata=metadata or {})

    def depth(self) -> int:
        return len(self._queue)

    # Internal helpers ------------------------------------------------------------------------

    def _add(self, action: str, *, execute: Callable[[], None], metadata: dict[str, Any]) -> None:
        if len(self._queue) >= self._capacity:
            raise RuntimeError("Order queue capacity exceeded")
        self._queue.append(_QueueItem(action=action, execute=execute, metadata=dict(metadata)))
        self._process()

    def _process(self) -> None:
        while self._queue:
            item = self._queue.popleft()
            try:
                item.execute()
            except Exception as exc:  # pragma: no cover - exercised in tests
                item.attempts += 1
                item.last_error = str(exc)
                if item.attempts > self._max_retries:
                    self._audit_failure(item)
                    continue
                delay = self._backoff * (2 ** (item.attempts - 1))
                if delay > 0:
                    self._sleep(delay)
                self._queue.appendleft(item)
            else:
                item.last_error = None

    def _audit_failure(self, item: _QueueItem) -> None:
        if self._audit is None:
            return
        payload = dict(item.metadata)
        payload.update({
            "attempts": item.attempts,
            "error": item.last_error,
            "operation": item.action,
        })
        self._audit("queue_drop", payload)

    @staticmethod
    def _execute_replace(
        order_id: str,
        new_order: "BrokerOrder",
        cancel_fn: Callable[[str], None],
        place_fn: Callable[[str, "BrokerOrder"], None],
    ) -> None:
        cancel_fn(order_id)
        place_fn(order_id, new_order)
