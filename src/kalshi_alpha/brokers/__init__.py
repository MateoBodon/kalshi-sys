"""Factory helpers for broker adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kalshi_alpha.brokers.kalshi.base import Broker
from kalshi_alpha.brokers.kalshi.dry import DryBroker
from kalshi_alpha.core.execution.order_queue import OrderQueue


def create_broker(
    mode: str,
    *,
    artifacts_dir: Path,
    audit_dir: Path,
    acknowledge_risks: bool = False,
    live_kwargs: dict[str, Any] | None = None,
) -> Broker:
    normalized = (mode or "dry").strip().lower()
    if normalized in {"dry", "paper"}:
        return DryBroker(
            artifacts_dir=artifacts_dir,
            audit_dir=audit_dir,
            order_queue=OrderQueue(),
        )
    if normalized == "live":
        if not acknowledge_risks:
            raise RuntimeError(
                "Live broker requires explicit acknowledgement via --i-understand-the-risks."
            )
        from kalshi_alpha.brokers.kalshi.live import LiveBroker  # noqa: PLC0415

        options = dict(live_kwargs or {})
        options.setdefault("artifacts_dir", artifacts_dir)
        options.setdefault("audit_dir", audit_dir)
        queue_capacity = int(options.get("queue_capacity", 64))
        queue_retries = int(options.get("max_retries", 3))
        order_queue = options.get("order_queue") or OrderQueue(
            capacity=queue_capacity,
            max_retries=queue_retries,
        )
        options.setdefault("order_queue", order_queue)
        return LiveBroker(**options)
    raise RuntimeError(f"Broker mode '{mode}' is not supported in this environment")
