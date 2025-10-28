"""Dry-run broker adapter that records intended Kalshi orders."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from kalshi_alpha.core.execution.order_queue import OrderQueue

from .base import Broker, BrokerOrder, ensure_directory


class DryBroker(Broker):
    """Broker implementation that serializes orders without hitting the network."""

    mode = "dry"

    def __init__(
        self,
        *,
        artifacts_dir: Path,
        audit_dir: Path,
        order_queue: OrderQueue | None = None,
    ) -> None:
        self._artifacts_dir = ensure_directory(artifacts_dir)
        self._audit_dir = ensure_directory(audit_dir)
        self._seen: set[str] = set()
        self._last_orders_path: Path | None = None
        self._order_queue = order_queue

    @property
    def last_orders_path(self) -> Path | None:
        return self._last_orders_path

    def place(self, orders: Sequence[BrokerOrder]) -> None:
        accepted: list[BrokerOrder] = []
        for order in orders:
            if order.idempotency_key in self._seen:
                self._write_audit("duplicate", order)
                continue
            self._seen.add(order.idempotency_key)
            accepted.append(order)
        if not accepted:
            return
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ")
        output_path = self._artifacts_dir / f"orders_{timestamp}.json"
        payload = [
            {
                "idempotency_key": order.idempotency_key,
                "market_id": order.market_id,
                "strike": order.strike,
                "side": order.side,
                "price": order.price,
                "contracts": order.contracts,
                "probability": order.probability,
                "metadata": order.metadata or {},
            }
            for order in accepted
        ]
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self._last_orders_path = output_path
        for order in accepted:
            self._write_audit("place", order, artifact_path=output_path)

    def cancel(self, order_ids: Sequence[str]) -> None:
        if self._order_queue:
            for order_id in order_ids:
                self._order_queue.enqueue_cancel(order_id, self._perform_cancel)
        else:
            for order_id in order_ids:
                self._perform_cancel(order_id)

    def replace(self, orders: Sequence[BrokerOrder]) -> None:
        if self._order_queue:
            for order in orders:
                order_id = str((order.metadata or {}).get("order_id") or order.idempotency_key)
                self._order_queue.enqueue_replace(
                    order_id=order_id,
                    new_order=order,
                    cancel_fn=self._perform_cancel,
                    place_fn=lambda _, new_order=order: self._perform_replace(new_order),
                )
        else:
            for order in orders:
                self._perform_replace(order)

    def status(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "orders_recorded": len(self._seen),
            "last_orders_path": self._last_orders_path.as_posix() if self._last_orders_path else None,
        }

    def _write_audit(
        self,
        action: str,
        order: BrokerOrder | None = None,
        *,
        order_id: str | None = None,
        artifact_path: Path | None = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "action": action,
            "mode": self.mode,
        }
        if order is not None:
            entry.update(
                {
                    "idempotency_key": order.idempotency_key,
                    "market_id": order.market_id,
                    "strike": order.strike,
                    "side": order.side,
                    "price": order.price,
                    "contracts": order.contracts,
                    "probability": order.probability,
                }
            )
        if order_id is not None:
            entry["order_id"] = order_id
        if artifact_path is not None:
            entry["artifact"] = artifact_path.as_posix()

        filename = self._audit_dir / f"orders_{datetime.now(tz=UTC).date().isoformat()}.jsonl"
        with filename.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True))
            handle.write("\n")

    def _perform_cancel(self, order_id: str) -> None:
        self._write_audit("cancel", order_id=order_id)

    def _perform_replace(self, order: BrokerOrder) -> None:
        self._write_audit("replace", order)
