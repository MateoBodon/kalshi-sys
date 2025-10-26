"""Helpers for writing raw datastore snapshots."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from kalshi_alpha.datastore.paths import RAW_ROOT


def _timestamp() -> datetime:
    return datetime.now(tz=UTC)


def _snapshot_dir(namespace: str, timestamp: datetime | None = None) -> Path:
    ts = timestamp or _timestamp()
    directory = RAW_ROOT / namespace / f"{ts.year:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


Serializable = Mapping[str, object] | Sequence[object] | object


def write_json_snapshot(
    namespace: str,
    name: str,
    payload: Serializable,
    *,
    timestamp: datetime | None = None,
) -> Path:
    ts = timestamp or _timestamp()
    directory = _snapshot_dir(namespace, ts)
    path = directory / f"{ts.strftime('%Y%m%dT%H%M%S')}_{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_text_snapshot(
    namespace: str,
    name: str,
    content: str,
    *,
    timestamp: datetime | None = None,
) -> Path:
    ts = timestamp or _timestamp()
    directory = _snapshot_dir(namespace, ts)
    path = directory / f"{ts.strftime('%Y%m%dT%H%M%S')}_{name}"
    path.write_text(content, encoding="utf-8")
    return path
