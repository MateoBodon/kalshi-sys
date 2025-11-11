"""Helpers for aggregating persisted monitor artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

MONITOR_ARTIFACTS_DIR = Path("reports/_artifacts/monitors")
DEFAULT_MONITOR_MAX_AGE_MINUTES = 1440
DEFAULT_PANIC_ALERT_THRESHOLD = 3
DEFAULT_PANIC_ALERT_WINDOW_MINUTES = 30


@dataclass(slots=True)
class MonitorArtifactsSummary:
    max_age_minutes: float | None
    latest_generated_at: datetime | None
    file_count: int
    statuses: dict[str, str]
    metrics: dict[str, dict[str, Any]]
    alerts_recent: set[str]


def summarize_monitor_artifacts(
    artifacts_dir: Path,
    *,
    now: datetime,
    window: timedelta,
) -> MonitorArtifactsSummary:
    """Aggregate monitor JSON artifacts into a summary snapshot."""

    if window.total_seconds() < 0:
        window = timedelta(0)

    if not artifacts_dir.exists():
        return MonitorArtifactsSummary(None, None, 0, {}, {}, set())

    statuses: dict[str, str] = {}
    metrics: dict[str, dict[str, Any]] = {}
    alerts_recent: set[str] = set()
    max_age_minutes: float | None = None
    latest_generated: datetime | None = None
    file_count = 0

    for path in sorted(artifacts_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        file_count += 1

        name = str(payload.get("name") or path.stem).strip() or path.stem
        status = _normalize_status(payload.get("status"))
        statuses[name] = status

        metrics_payload = payload.get("metrics")
        if isinstance(metrics_payload, dict):
            metrics[name] = metrics_payload

        generated = _parse_timestamp(payload.get("generated_at"))
        if generated is None:
            try:
                generated = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            except OSError:
                generated = None
        if generated is None:
            continue

        generated = min(generated, now)
        age_minutes = (now - generated).total_seconds() / 60.0
        if max_age_minutes is None or age_minutes > max_age_minutes:
            max_age_minutes = age_minutes
        if latest_generated is None or generated > latest_generated:
            latest_generated = generated
        if status == "ALERT" and now - generated <= window:
            alerts_recent.add(name)

    return MonitorArtifactsSummary(
        max_age_minutes,
        latest_generated,
        file_count,
        statuses,
        metrics,
        alerts_recent,
    )


def _normalize_status(raw: object) -> str:
    if isinstance(raw, str):
        candidate = raw.strip().upper()
        if candidate in {"OK", "ALERT", "NO_DATA"}:
            return candidate
        if not candidate:
            return "NO_DATA"
        return candidate
    if raw is None:
        return "NO_DATA"
    return str(raw).upper()


def _parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None
