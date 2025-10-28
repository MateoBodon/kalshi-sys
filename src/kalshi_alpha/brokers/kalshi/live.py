"""Live Kalshi broker adapter with rate limiting, backoff, and auditing."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import requests
from requests import Response
from requests.auth import HTTPBasicAuth
from requests.exceptions import RequestException

from kalshi_alpha.core.execution.order_queue import OrderQueue
from kalshi_alpha.brokers.kalshi.base import Broker, BrokerOrder, ensure_directory
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

    def __init__(
        self,
        *,
        artifacts_dir: Path,
        audit_dir: Path,
        session: requests.Session | None = None,
        base_url: str = "https://trading-api.kalshi.com/v1",
        rate_limit_per_second: int = 5,
        queue_capacity: int = 64,
        max_retries: int = 3,
        timeout: float = 10.0,
        order_queue: OrderQueue | None = None,
    ) -> None:
        if os.environ.get("CI"):
            raise RuntimeError("Live broker is disabled while running under CI.")

        load_env()
        api_key = os.getenv("KALSHI_API_KEY")
        api_secret = os.getenv("KALSHI_API_SECRET")
        if not api_key or not api_secret:
            raise RuntimeError(
                "Missing Kalshi live credentials. Set KALSHI_API_KEY and KALSHI_API_SECRET in .env.local."
            )

        self._artifacts_dir = ensure_directory(artifacts_dir)
        self._audit_dir = ensure_directory(audit_dir)
        self._session = session or requests.Session()
        self._session.headers.setdefault("Content-Type", "application/json")
        self._session.auth = HTTPBasicAuth(api_key, api_secret)

        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._rate_limiter = _RateLimiter(rate_limit_per_second, 1.0)
        self._max_retries = max(1, max_retries)
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
            if order.idempotency_key in self._seen_idempotency:
                LOGGER.debug("Skipping duplicate order with idempotency %s", order.idempotency_key)
                continue
            self._seen_idempotency.add(order.idempotency_key)
            accepted.append(order)
        if not accepted:
            return

        for order in accepted:
            payload = self._order_payload(order)
            headers = {"Idempotency-Key": order.idempotency_key}
            self._request("POST", "/orders", json_body=payload, extra_headers=headers)
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
        headers = {"Idempotency-Key": order.idempotency_key} if order.idempotency_key else None
        endpoint = f"/orders/{order_id}/replace"
        self._request("POST", endpoint, json_body=payload, extra_headers=headers)
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
        extra_headers: dict[str, str] | None = None,
    ) -> Response:
        url = f"{self._base_url}{endpoint}"
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            self._rate_limiter.wait()
            try:
                response = self._session.request(
                    method,
                    url,
                    json=json_body,
                    headers=extra_headers,
                    timeout=self._timeout,
                )
            except RequestException as exc:
                last_error = exc
                self._sleep_backoff(attempt)
                continue

            if 200 <= response.status_code < 300:
                return response
            last_error = RuntimeError(
                f"Kalshi trading API returned {response.status_code} for {endpoint}"
            )
            if response.status_code in {429, 500, 502, 503, 504} and attempt < self._max_retries:
                self._sleep_backoff(attempt)
                continue
            break
        raise RuntimeError("Failed to execute Kalshi trading API request") from last_error

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(5.0, 0.5 * (2 ** (attempt - 1)))
        time.sleep(delay)

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
