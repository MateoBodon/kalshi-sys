"""Heartbeat and kill-switch utilities for execution pipelines."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from kalshi_alpha.datastore import paths as datastore_paths

ET = ZoneInfo("America/New_York")


def _state_dir() -> Path:
    state_dir = datastore_paths.PROC_ROOT / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def default_heartbeat_path() -> Path:
    return _state_dir() / "heartbeat.json"


def default_kill_switch_path() -> Path:
    return _state_dir() / "kill_switch"


def write_heartbeat(
    *,
    mode: str,
    monitors: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    now: datetime | None = None,
    path: Path | None = None,
) -> Path:
    """Persist the latest execution heartbeat.

    A heartbeat records the last successful pipeline activity with an ET timestamp.
    """

    timestamp = now or datetime.now(tz=UTC)
    heartbeat_path = Path(path) if path is not None else default_heartbeat_path()
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "mode": mode,
        "timestamp_utc": timestamp.isoformat(),
        "timestamp_et": timestamp.astimezone(ET).isoformat(),
        "monitors": monitors or {},
    }
    if extra:
        payload.update(extra)

    tmp_path = heartbeat_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(heartbeat_path)
    return heartbeat_path


def load_heartbeat(path: Path | None = None) -> dict[str, Any] | None:
    heartbeat_path = Path(path) if path is not None else default_heartbeat_path()
    if not heartbeat_path.exists():
        return None
    try:
        return json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def heartbeat_stale(
    *,
    threshold: timedelta,
    now: datetime | None = None,
    path: Path | None = None,
) -> tuple[bool, dict[str, Any] | None]:
    payload = load_heartbeat(path)
    if payload is None:
        return False, None
    timestamp_str = payload.get("timestamp_utc") or payload.get("timestamp")
    if not isinstance(timestamp_str, str):
        return True, payload
    try:
        recorded = datetime.fromisoformat(timestamp_str)
    except ValueError:
        return True, payload
    if recorded.tzinfo is None:
        recorded = recorded.replace(tzinfo=UTC)
    current = now or datetime.now(tz=UTC)
    return current - recorded > threshold, payload


def resolve_kill_switch_path(path: str | Path | None = None) -> Path:
    if path is None or path == "":
        return default_kill_switch_path()
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def kill_switch_engaged(path: str | Path | None = None) -> bool:
    return resolve_kill_switch_path(path).exists()
