"""Hot-restart snapshot utilities for maker ops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence

from kalshi_alpha.datastore import paths as datastore_paths
from kalshi_alpha.sched.windows import TradingWindow, next_windows

DEFAULT_STATE_PATH = (datastore_paths.PROC_ROOT / "state" / "hot_restart.json").resolve()


@dataclass(frozen=True)
class HotRestartSnapshot:
    captured_at: datetime
    active_window: dict[str, object] | None
    upcoming_windows: list[dict[str, object]]
    outstanding: dict[str, int]

    def age_seconds(self, *, now: datetime | None = None) -> float:
        reference = now or datetime.now(tz=UTC)
        return max((reference - self.captured_at).total_seconds(), 0.0)


class HotRestartManager:
    """Persist/restore scheduler state to enable <5s crash recovery."""

    def __init__(self, path: Path | None = None, *, max_age_seconds: float = 5.0) -> None:
        self._path = path or DEFAULT_STATE_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self.max_age_seconds = max(float(max_age_seconds), 0.5)

    def capture(
        self,
        *,
        window: TradingWindow | None,
        outstanding_summary: Mapping[str, int],
        now: datetime | None = None,
        upcoming: Sequence[TradingWindow] | None = None,
    ) -> HotRestartSnapshot:
        reference = now or datetime.now(tz=UTC)
        payload = {
            "captured_at": reference.isoformat(),
            "active_window": _encode_window(window),
            "upcoming_windows": [_encode_window(entry) for entry in (upcoming or next_windows(reference))],
            "outstanding": dict(outstanding_summary),
        }
        self._write(payload)
        return HotRestartSnapshot(
            captured_at=reference,
            active_window=payload["active_window"],
            upcoming_windows=payload["upcoming_windows"],
            outstanding=dict(outstanding_summary),
        )

    def restore(self) -> HotRestartSnapshot | None:
        if not self._path.exists():
            return None
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        captured_at_raw = payload.get("captured_at")
        if not isinstance(captured_at_raw, str):
            return None
        try:
            captured_at = datetime.fromisoformat(captured_at_raw)
        except ValueError:
            return None
        snapshot = HotRestartSnapshot(
            captured_at=captured_at,
            active_window=payload.get("active_window"),
            upcoming_windows=list(payload.get("upcoming_windows") or []),
            outstanding=dict(payload.get("outstanding") or {}),
        )
        if snapshot.age_seconds() > self.max_age_seconds:
            return None
        return snapshot

    def _write(self, payload: Mapping[str, object]) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self._path)


def summarize_orders_state(state: object | None) -> dict[str, int]:
    """Convert OutstandingOrdersState into a compact summary."""

    summary: dict[str, int] = {"dry": 0, "live": 0, "dry_contracts": 0, "live_contracts": 0}
    if state is None:
        return summary
    try:
        outstanding_for = state.outstanding_for  # type: ignore[attr-defined]
    except AttributeError:
        return summary
    for mode in ("dry", "live"):
        orders = outstanding_for(mode)
        summary[mode] = len(orders)
        summary[f"{mode}_contracts"] = sum(int(entry.get("contracts", 0)) for entry in orders.values())
    summary["total"] = summary["dry"] + summary["live"]
    summary["total_contracts"] = summary["dry_contracts"] + summary["live_contracts"]
    return summary


def _encode_window(window: TradingWindow | None) -> dict[str, object] | None:
    if window is None:
        return None
    return {
        "label": window.label,
        "target_type": window.target_type,
        "target_et": window.target_et.isoformat(),
        "start_et": window.start_et.isoformat(),
        "freeze_et": window.freeze_et.isoformat(),
        "series": list(window.series),
    }


__all__ = ["HotRestartManager", "HotRestartSnapshot", "summarize_orders_state"]
