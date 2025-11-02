"""Live Kalshi broker adapter with rate limiting, backoff, and auditing."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from kalshi_alpha.brokers.kalshi.base import Broker, BrokerOrder, ensure_directory
from kalshi_alpha.brokers.kalshi.http_client import (
    KalshiClockSkewError,
    KalshiHttpClient,
    KalshiHttpError,
)
from kalshi_alpha.core.execution.order_queue import OrderQueue
from kalshi_alpha.utils.env import load_env

LOGGER = logging.getLogger(__name__)


class _RateLimiter:
    """Token bucket rate limiter used to throttle Trading API calls."""

    def __init__(self, max_calls: int, per_seconds: float) -> None:
        self._max_calls = max(1, max_calls)
        self._per_seconds = max(per_seconds, 0.1)
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until a slot becomes available."""
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self._per_seconds:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return
                earliest = self._timestamps[0]
                sleep_for = max(0.0, self._per_seconds - (now - earliest))
            if sleep_for <= 0:
                sleep_for = self._per_seconds / self._max_calls
            time.sleep(sleep_for)


class LiveBroker(Broker):
    """Networked broker adapter for the Kalshi trading API."""

    mode = "live"

    def __init__(  # noqa: PLR0913 - broker wiring requires multiple knobs
        self,
        *,
        artifacts_dir: Path,
        audit_dir: Path,
        session: requests.Session | None = None,
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
        rate_limit_per_second: int = 5,
        queue_capacity: int = 64,
        max_retries: int = 3,
        timeout: float = 10.0,
        retry_backoff: float = 0.5,
        http_client: KalshiHttpClient | None = None,
        order_queue: OrderQueue | None = None,
    ) -> None:
        if os.environ.get("CI"):
            raise RuntimeError("Live broker is disabled while running under CI.")

        load_env()
        self._artifacts_dir = ensure_directory(artifacts_dir)
        self._audit_dir = ensure_directory(audit_dir)
        self._http = http_client or KalshiHttpClient(
            base_url=base_url,
            session=session,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        self._rate_limiter = _RateLimiter(rate_limit_per_second, 1.0)
        self._seen_idempotency: set[str] = set()
        self._lock = threading.Lock()
        self._order_queue = order_queue or OrderQueue(
            capacity=queue_capacity,
            max_retries=max_retries,
            audit_callback=self._queue_audit,
        )

        LOGGER.info("Live broker initialized; submissions remain feature-gated.")

    # Broker interface ----------------------------------------------------------------------------

    def place(self, orders: Sequence[BrokerOrder]) -> None:
        accepted: list[BrokerOrder] = []
        for order in orders:
            with self._lock:
                if order.idempotency_key in self._seen_idempotency:
                    LOGGER.debug(
                        "Skipping duplicate order with idempotency %s",
                        order.idempotency_key,
                    )
                    continue
                self._seen_idempotency.add(order.idempotency_key)
            accepted.append(order)
        if not accepted:
            return

        for order in accepted:
            payload = self._order_payload(order)
            self._request(
                "POST",
                "/orders",
                json_body=payload,
                idempotency_key=order.idempotency_key,
            )
            self._write_audit("place_intent", order)

    def cancel(self, order_ids: Sequence[str]) -> None:
        for order_id in order_ids:
            self._order_queue.enqueue_cancel(order_id, self._submit_cancel)

    def replace(self, orders: Sequence[BrokerOrder]) -> None:
        for order in orders:
            existing_order_id = None
            if order.metadata:
                existing_order_id = str(order.metadata.get("order_id") or "")
            if not existing_order_id:
                LOGGER.warning("Replace called without existing order id; queuing as place only.")
                self.place([order])
                continue
            self._order_queue.enqueue_replace(
                order_id=existing_order_id,
                new_order=order,
                cancel_fn=self._submit_cancel,
                place_fn=self._submit_replace,
            )

    def status(self) -> dict[str, Any]:
        queue_depth = self._order_queue.depth()
        return {
            "mode": self.mode,
            "queue_depth": queue_depth,
            "orders_recorded": len(self._seen_idempotency),
            "message": "Live broker armed; submissions gated by CLI flags.",
        }

    # Internal helpers ---------------------------------------------------------------------------

    def _queue_audit(self, action: str, metadata: dict[str, Any]) -> None:
        payload = dict(metadata)
        payload.setdefault("queue_action", action)
        self._write_audit("queue_drop", extra=payload)

    def _submit_cancel(self, order_id: str) -> None:
        endpoint = f"/orders/{order_id}/cancel"
        self._request("POST", endpoint, json_body={})
        self._write_audit("cancel_intent", extra={"order_id": order_id})

    def _submit_replace(self, order_id: str, order: BrokerOrder) -> None:
        payload = self._order_payload(order)
        endpoint = f"/orders/{order_id}/replace"
        self._request(
            "POST",
            endpoint,
            json_body=payload,
            idempotency_key=order.idempotency_key,
        )
        self._write_audit(
            "replace_intent",
            extra={"order_id": order_id, "idempotency_key": order.idempotency_key},
        )

    def _order_payload(self, order: BrokerOrder) -> dict[str, Any]:
        return {
            "market_id": order.market_id,
            "side": order.side.upper(),
            "price": order.price,
            "contracts": order.contracts,
            "probability": order.probability,
            "metadata": order.metadata or {},
            "strike": order.strike,
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        json_body: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        self._rate_limiter.wait()
        try:
            self._http.request(
                method,
                endpoint,
                json_body=json_body,
                idempotency_key=idempotency_key,
            )
        except KalshiClockSkewError:
            raise
        except KalshiHttpError as exc:
            raise RuntimeError("Failed to execute Kalshi trading API request") from exc

    def _write_audit(
        self,
        action: str,
        order: BrokerOrder | None = None,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "mode": self.mode,
            "action": action,
        }
        if order is not None:
            entry.update(
                {
                    "idempotency_key": order.idempotency_key,
                    "market_id": order.market_id,
                    "side": order.side,
                    "price": order.price,
                    "contracts": order.contracts,
                    "probability": order.probability,
                    "strike": order.strike,
                }
            )
        if extra:
            entry.update(extra)

        filename = self._audit_dir / f"live_orders_{datetime.now(tz=UTC).date().isoformat()}.jsonl"
        with filename.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True))
            handle.write("\n")
