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
from requests import Response

from kalshi_alpha.brokers.kalshi.base import Broker, BrokerOrder, ensure_directory
from kalshi_alpha.brokers.kalshi.http_client import (
    KalshiClockSkewError,
    KalshiHttpClient,
    KalshiHttpError,
)
from kalshi_alpha.core.execution.order_queue import OrderQueue
from kalshi_alpha.exec.telemetry import TelemetrySink, sanitize_book_snapshot
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
        base_url: str = os.getenv("KALSHI_TRADE_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2"),
        rate_limit_per_second: int = 5,
        queue_capacity: int = 64,
        max_retries: int = 3,
        timeout: float = 10.0,
        retry_backoff: float = 0.5,
        http_client: KalshiHttpClient | None = None,
        order_queue: OrderQueue | None = None,
        telemetry_sink: TelemetrySink | None = None,
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
        self._telemetry = telemetry_sink

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
            event_payload = self._telemetry_payload(order, payload)
            self._emit_telemetry("sent", event_payload)
            start_ns = time.perf_counter_ns()
            try:
                response = self._request(
                    "POST",
                    "/portfolio/orders",
                    json_body=payload,
                    idempotency_key=order.idempotency_key,
                )
            except Exception as exc:  # pragma: no cover - verified via tests
                error_payload = dict(event_payload)
                error_payload["error"] = str(exc)
                if exc.__cause__ is not None:
                    error_payload["error_cause"] = str(exc.__cause__)
                error_payload["latency_ms"] = self._elapsed_ms(start_ns)
                self._emit_telemetry("reject", error_payload)
                raise
            ack_payload = dict(event_payload)
            ack_payload["status_code"] = response.status_code
            ack_payload["latency_ms"] = self._elapsed_ms(start_ns)
            self._emit_telemetry("ack", ack_payload)
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
        if metadata:
            telem_payload = {
                "queue_action": action,
                "idempotency_key": metadata.get("idempotency_key"),
                "order_id": metadata.get("order_id"),
            }
            self._emit_telemetry("reject", telem_payload)

    def _submit_cancel(self, order_id: str) -> None:
        endpoint = f"/portfolio/orders/{order_id}"
        start_ns = time.perf_counter_ns()
        try:
            response = self._request("DELETE", endpoint, json_body={})
        except Exception as exc:  # pragma: no cover - network errors surfaced in tests
            self._emit_telemetry(
                "reject",
                {
                    "order_id": order_id,
                    "action": "cancel",
                    "error": str(exc),
                    "error_cause": str(exc.__cause__) if exc.__cause__ is not None else None,
                    "latency_ms": self._elapsed_ms(start_ns),
                },
            )
            raise
        self._emit_telemetry(
            "cancel",
            {
                "order_id": order_id,
                "status_code": response.status_code,
                "latency_ms": self._elapsed_ms(start_ns),
            },
        )
        self._write_audit("cancel_intent", extra={"order_id": order_id})

    def _submit_replace(self, order_id: str, order: BrokerOrder) -> None:
        payload = self._order_payload(order)
        endpoint = f"/portfolio/orders/{order_id}/replace"
        event_payload = self._telemetry_payload(order, payload)
        event_payload["replace_of"] = order_id
        self._emit_telemetry("sent", event_payload)
        start_ns = time.perf_counter_ns()
        try:
            response = self._request(
                "POST",
                endpoint,
                json_body=payload,
                idempotency_key=order.idempotency_key,
            )
        except Exception as exc:  # pragma: no cover
            error_payload = dict(event_payload)
            error_payload["error"] = str(exc)
            if exc.__cause__ is not None:
                error_payload["error_cause"] = str(exc.__cause__)
            error_payload["latency_ms"] = self._elapsed_ms(start_ns)
            self._emit_telemetry("reject", error_payload)
            raise
        ack_payload = dict(event_payload)
        ack_payload["status_code"] = response.status_code
        ack_payload["latency_ms"] = self._elapsed_ms(start_ns)
        self._emit_telemetry("ack", ack_payload)
        self._write_audit(
            "replace_intent",
            extra={"order_id": order_id, "idempotency_key": order.idempotency_key},
        )

    def _order_payload(self, order: BrokerOrder) -> dict[str, Any]:
        metadata = dict(order.metadata or {})
        ticker = metadata.get("market_ticker") or metadata.get("ticker")
        if not ticker:
            raise RuntimeError("Missing market ticker for Kalshi order payload")
        liquidity = str(metadata.get("liquidity") or "maker").lower()
        action = metadata.get("action") or ("sell" if liquidity == "maker" else "buy")
        side = order.side.lower()
        price_dollars = float(order.price)
        payload: dict[str, Any] = {
            "ticker": ticker,
            "action": action.lower(),
            "side": side,
            "type": "limit",
            "count": int(order.contracts),
            "client_order_id": order.idempotency_key,
        }
        price_str = f"{price_dollars:.2f}"
        if side == "yes":
            payload["yes_price_dollars"] = price_str
        else:
            payload["no_price_dollars"] = price_str
        return payload

    def _telemetry_payload(self, order: BrokerOrder, payload: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(order.metadata or {})
        book_snapshot = metadata.get("book_snapshot") or metadata.get("orderbook")
        sanitized_snapshot = sanitize_book_snapshot(book_snapshot)
        metadata.pop("book_snapshot", None)
        metadata.pop("orderbook", None)
        event_payload: dict[str, Any] = {
            "idempotency_key": order.idempotency_key,
            "market_id": order.market_id,
            "side": order.side,
            "contracts": order.contracts,
            "price": order.price,
            "size": order.contracts,
            "probability": order.probability,
        }
        if "order_id" in metadata:
            event_payload["order_id"] = metadata.get("order_id")
        if sanitized_snapshot is not None:
            event_payload["book_snapshot"] = sanitized_snapshot
        if metadata:
            event_payload["metadata"] = metadata
        return event_payload

    def _emit_telemetry(self, event_type: str, data: dict[str, Any]) -> None:
        if self._telemetry is None:
            return
        self._telemetry.emit(event_type, source="rest", data=data)

    @staticmethod
    def _elapsed_ms(start_ns: int) -> float:
        return max(0.0, (time.perf_counter_ns() - start_ns) / 1_000_000.0)

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        json_body: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> Response:
        self._rate_limiter.wait()
        try:
            response = self._http.request(
                method,
                endpoint,
                json_body=json_body,
                idempotency_key=idempotency_key,
            )
        except KalshiClockSkewError:
            raise
        except KalshiHttpError as exc:
            raise RuntimeError("Failed to execute Kalshi trading API request") from exc
        return response

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
