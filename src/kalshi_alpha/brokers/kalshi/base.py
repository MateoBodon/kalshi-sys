"""Shared interfaces for Kalshi broker adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class BrokerOrder:
    """Canonical representation of an order submission."""

    idempotency_key: str
    market_id: str
    strike: float
    side: str
    price: float
    contracts: int
    probability: float
    metadata: dict[str, object] | None = None


class Broker(Protocol):
    """Protocol describing the expected broker adapter surface area."""

    mode: str

    def place(self, orders: Sequence[BrokerOrder]) -> None:
        """Submit a batch of orders."""

    def cancel(self, order_ids: Sequence[str]) -> None:
        """Cancel a batch of orders by identifier."""

    def replace(self, orders: Sequence[BrokerOrder]) -> None:
        """Replace a batch of orders."""

    def status(self) -> dict[str, object]:
        """Return adapter status metadata for diagnostic logging."""


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
