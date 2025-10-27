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
    directory = RAW_ROOT / f"{ts.year:04d}" / f"{ts.month:02d}" / f"{ts.day:02d}" / namespace
    directory.mkdir(parents=True, exist_ok=True)
    return directory


Serializable = Mapping[str, object] | Sequence[object] | object


def _unique_path(path: Path) -> Path:
    """Return a unique path by appending a numeric suffix if necessary."""
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


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
    path = _unique_path(path)
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
    path = _unique_path(path)
    path.write_text(content, encoding="utf-8")
    return path
