"""Persistence utilities for tracking outstanding broker orders."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kalshi_alpha.brokers.kalshi.base import BrokerOrder
from kalshi_alpha.datastore import paths as datastore_paths


def _default_state_path() -> Path:
    state_dir = datastore_paths.PROC_ROOT / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "orders.json"


@dataclass(frozen=True)
class OutstandingOrderRecord:
    """Serializable record of an outstanding order submission."""

    idempotency_key: str
    market_id: str
    strike: float
    side: str
    price: float
    contracts: int
    probability: float
    metadata: dict[str, Any]
    mode: str
    submitted_at: str
    status: str = "pending"


class OutstandingOrdersState:
    """Persisted state container for outstanding broker orders."""

    def __init__(self, path: Path, payload: dict[str, Any] | None = None) -> None:
        self._path = path
        self._payload: dict[str, Any] = payload if isinstance(payload, dict) else {}
        self._ensure_defaults()

    # Factories ---------------------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> OutstandingOrdersState:
        if path is None:
            target = _default_state_path()
        else:
            target = path
            target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                payload = json.loads(target.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}
        state = cls(target, payload)
        if not target.exists():
            state._write()
        return state

    # Public API --------------------------------------------------------------------------------

    def record_submission(self, mode: str, orders: Sequence[BrokerOrder]) -> None:
        normalized = self._normalize_mode(mode)
        if not orders:
            return
        bucket: dict[str, dict[str, Any]] = self._payload["outstanding"][normalized]
        timestamp = datetime.now(tz=UTC).isoformat()
        for order in orders:
            record = OutstandingOrderRecord(
                idempotency_key=order.idempotency_key,
                market_id=order.market_id,
                strike=float(order.strike),
                side=order.side,
                price=float(order.price),
                contracts=int(order.contracts),
                probability=float(order.probability),
                metadata=dict(order.metadata or {}),
                mode=normalized,
                submitted_at=timestamp,
            )
            bucket[order.idempotency_key] = asdict(record)
        self._payload["updated_at"] = timestamp
        if self._payload.get("cancel_all"):
            self._payload["cancel_all"] = None
        self._write()

    def remove(self, mode: str, idempotency_keys: Iterable[str]) -> list[str]:
        normalized = self._normalize_mode(mode)
        bucket: dict[str, dict[str, Any]] = self._payload["outstanding"][normalized]
        removed: list[str] = []
        for key in idempotency_keys:
            if key in bucket:
                bucket.pop(key, None)
                removed.append(key)
        if removed:
            self._payload["updated_at"] = datetime.now(tz=UTC).isoformat()
            self._write()
        return removed

    def mark_status(
        self,
        mode: str,
        idempotency_keys: Iterable[str],
        status: str,
    ) -> list[str]:
        normalized = self._normalize_mode(mode)
        bucket: dict[str, dict[str, Any]] = self._payload["outstanding"][normalized]
        updated: list[str] = []
        for key in idempotency_keys:
            order = bucket.get(key)
            if order is None:
                continue
            if order.get("status") == status:
                updated.append(key)
                continue
            order["status"] = status
            updated.append(key)
        if updated:
            self._payload["updated_at"] = datetime.now(tz=UTC).isoformat()
            self._write()
        return updated

    def reconcile(self, mode: str, active_idempotency_keys: Iterable[str]) -> list[str]:
        normalized = self._normalize_mode(mode)
        bucket: dict[str, dict[str, Any]] = self._payload["outstanding"][normalized]
        active = {str(key) for key in active_idempotency_keys}
        to_remove = [key for key in bucket if key not in active]
        return self.remove(normalized, to_remove)

    def mark_cancel_all(self, reason: str, *, modes: Iterable[str] | None = None) -> None:
        targets = (
            {self._normalize_mode(mode) for mode in modes}
            if modes
            else set(self._payload["outstanding"].keys())
        )
        timestamp = datetime.now(tz=UTC).isoformat()
        self._payload["cancel_all"] = {
            "reason": reason,
            "requested_at": timestamp,
            "modes": sorted(targets),
        }
        self._payload["updated_at"] = timestamp
        self._write()

    def cancel_all_request(self) -> dict[str, Any] | None:
        return self._payload.get("cancel_all")

    def clear_cancel_all(self) -> None:
        if self._payload.get("cancel_all") is None:
            return
        self._payload["cancel_all"] = None
        self._payload["updated_at"] = datetime.now(tz=UTC).isoformat()
        self._write()

    def outstanding_for(self, mode: str) -> dict[str, dict[str, Any]]:
        normalized = self._normalize_mode(mode)
        return dict(self._payload["outstanding"][normalized])

    def summary(self) -> dict[str, int]:
        return {
            mode: len(entries)
            for mode, entries in self._payload.get("outstanding", {}).items()
        }

    def total(self) -> int:
        return sum(self.summary().values())

    def save(self) -> None:
        self._write()

    # Internal helpers --------------------------------------------------------------------------

    def _ensure_defaults(self) -> None:
        outstanding = self._payload.setdefault("outstanding", {})
        outstanding.setdefault("dry", {})
        outstanding.setdefault("live", {})
        self._payload.setdefault("cancel_all", None)
        self._payload.setdefault("updated_at", datetime.now(tz=UTC).isoformat())

    def _normalize_mode(self, mode: str) -> str:
        normalized = (mode or "dry").strip().lower()
        if normalized in {"paper"}:
            normalized = "dry"
        if normalized not in {"dry", "live"}:
            normalized = "dry"
        return normalized

    def _write(self) -> None:
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self._payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self._path)
