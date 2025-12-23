"""Append-only telemetry sink for live execution events."""

from __future__ import annotations

import gzip
import json
import threading
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from kalshi_alpha.datastore.paths import PROC_ROOT

EVENT_TYPES = {
    "sent",
    "ack",
    "partial_fill",
    "fill",
    "cancel",
    "reject",
    "heartbeat",
    "ws_reconnect",
    "ws_malformed",
    "ws_disconnect",
    "ws_heartbeat_timeout",
}

DEFAULT_BOOK_DEPTH = 5


def sanitize_book_snapshot(
    snapshot: object,
    *,
    depth: int = DEFAULT_BOOK_DEPTH,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Return a depth-limited, JSON-serialisable book snapshot."""

    if snapshot is None:
        return None

    if isinstance(snapshot, Mapping):
        sanitized: dict[str, Any] = {}
        for key, value in snapshot.items():
            lowered = key.lower()
            if lowered in {"bids", "asks"}:
                sanitized[key] = _sanitize_book_levels(value, depth=depth)
                continue
            sanitized[key] = _coerce(value)
        return sanitized

    if isinstance(snapshot, Sequence) and not isinstance(snapshot, (str, bytes, bytearray)):
        return _sanitize_book_levels(snapshot, depth=depth)

    return {"value": _coerce(snapshot)}


def _sanitize_book_levels(
    levels: Sequence[object] | Mapping[str, object],
    *,
    depth: int,
) -> list[dict[str, Any]]:
    items: list[object]
    if isinstance(levels, Mapping):
        items = [dict(price=key, size=value) for key, value in levels.items()]
    else:
        items = list(levels)

    limited = items[: max(depth, 0)]
    result: list[dict[str, Any]] = []
    for entry in limited:
        if isinstance(entry, Mapping):
            price = _safe_float(entry.get("price"))
            size = _safe_float(entry.get("size"))
        elif isinstance(entry, Sequence) and len(entry) >= 2:
            price = _safe_float(entry[0])
            size = _safe_float(entry[1])
        else:
            price = _safe_float(entry)
            size = None
        payload: dict[str, Any] = {}
        if price is not None:
            payload["price"] = price
        if size is not None:
            payload["size"] = size
        if payload:
            result.append(payload)
    return result


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, str):
            return float(value)
    except (TypeError, ValueError):
        return None
    return None

DEFAULT_BASE_DIR = Path("data/raw/kalshi")
DEFAULT_PROC_TELEMETRY_DIR = PROC_ROOT / "telemetry"
DEFAULT_TELEMETRY_MAX_WINDOW_BYTES = 256 * 1024
REQUIRED_TELEMETRY_FIELDS = ("run_id", "window_id", "series", "market_ticker", "ts")


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _ensure_path(base: Path, moment: datetime) -> Path:
    dated = moment.astimezone(UTC)
    target = base / f"{dated.year:04d}" / f"{dated.month:02d}" / f"{dated.day:02d}"
    target.mkdir(parents=True, exist_ok=True)
    return target / "exec.jsonl"


def _mask(value: str | None) -> str | None:
    if not value:
        return value
    if len(value) <= 4:
        return "*" * len(value)
    return f"{'*' * (len(value) - 4)}{value[-4:]}"


def _sanitize(mapping: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in mapping.items():
        lowered = key.lower()
        if isinstance(value, str) and (
            "signature" in lowered
            or "access-key" in lowered
            or "private" in lowered
            or "idempotency" in lowered
        ):
            sanitized[key] = _mask(value)
            continue
        if isinstance(value, Mapping):
            sanitized[key] = _sanitize(value)
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            sanitized[key] = [
                _sanitize(item) if isinstance(item, Mapping) else _coerce(item) for item in value
            ]
            continue
        sanitized[key] = _coerce(value)
    return sanitized


def _coerce(value: object) -> object:
    if isinstance(value, datetime):
        moment = value
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment.astimezone(UTC).isoformat()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


@dataclass
class TelemetryEvent:
    """Serializable telemetry event payload."""

    event_type: str
    source: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime | None = None

    def asdict(self, *, clock: Callable[[], datetime]) -> dict[str, Any]:
        moment = self.timestamp or clock()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        payload = {
            "timestamp": moment.astimezone(UTC).isoformat(),
            "event_type": self.event_type,
            "source": self.source,
            "data": _sanitize(self.data),
        }
        return payload


class TelemetrySink:
    """Durable JSONL telemetry sink with daily rotation."""

    def __init__(
        self,
        *,
        base_dir: Path | str = DEFAULT_BASE_DIR,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._base_dir = Path(base_dir)
        self._clock = clock
        self._lock = threading.Lock()
        self._current_date: date | None = None
        self._current_path: Path | None = None

    def emit(self, event_type: str, *, source: str, data: Mapping[str, Any] | None = None) -> None:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported telemetry event type: {event_type}")
        mapping: dict[str, Any] = {}
        if data:
            mapping.update(dict(data))
        event = TelemetryEvent(event_type=event_type, source=source, data=mapping)
        payload = event.asdict(clock=self._clock)
        self._append(payload)

    def _append(self, payload: MutableMapping[str, Any]) -> None:
        moment = datetime.fromisoformat(str(payload["timestamp"]))
        target_path = self._ensure_current_path(moment)
        line = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        with self._lock:
            with target_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")

    def _ensure_current_path(self, moment: datetime) -> Path:
        moment_date = moment.date()
        path = self._current_path
        if path is not None and self._current_date == moment_date:
            return path
        target = _ensure_path(self._base_dir, moment)
        with self._lock:
            self._current_date = moment_date
            self._current_path = target
        return target


@dataclass(slots=True)
class TelemetryJsonlSink:
    """Append-only JSONL telemetry sink with run-based rotation + bounds."""

    run_id: str
    stream: str
    base_dir: Path | str = DEFAULT_PROC_TELEMETRY_DIR
    compress: bool = True
    max_bytes_per_window: int = DEFAULT_TELEMETRY_MAX_WINDOW_BYTES
    max_bytes_total: int | None = None
    required_fields: Sequence[str] = REQUIRED_TELEMETRY_FIELDS
    _path: Path = field(init=False)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _window_bytes: dict[str, int] = field(init=False, default_factory=dict)
    _total_bytes: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        stream = str(self.stream or "").strip()
        if not stream:
            raise ValueError("stream is required")
        self.stream = stream
        base_dir = Path(self.base_dir)
        (base_dir / stream).mkdir(parents=True, exist_ok=True)
        suffix = ".jsonl.gz" if self.compress else ".jsonl"
        self._path = base_dir / stream / f"{self.run_id}{suffix}"
        max_window = int(self.max_bytes_per_window) if self.max_bytes_per_window is not None else 0
        self.max_bytes_per_window = max(0, max_window)
        if self.max_bytes_total is not None:
            self.max_bytes_total = max(0, int(self.max_bytes_total))
        self.required_fields = tuple(self.required_fields)

    def emit(self, payload: Mapping[str, Any]) -> bool:
        record = dict(payload)
        self._validate(record)
        line = json.dumps(record, separators=(",", ":"), sort_keys=True)
        line_bytes = len(line.encode("utf-8")) + 1
        window_id = str(record.get("window_id", "")).strip()
        with self._lock:
            if self.max_bytes_per_window and window_id:
                used = self._window_bytes.get(window_id, 0)
                if used + line_bytes > self.max_bytes_per_window:
                    return False
                self._window_bytes[window_id] = used + line_bytes
            if self.max_bytes_total is not None:
                if self._total_bytes + line_bytes > self.max_bytes_total:
                    return False
                self._total_bytes += line_bytes
            self._append_line(line)
        return True

    def _append_line(self, line: str) -> None:
        if self.compress:
            with gzip.open(self._path, "at", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
            return
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")

    def _validate(self, record: Mapping[str, Any]) -> None:
        missing = [field for field in self.required_fields if not record.get(field)]
        if missing:
            raise ValueError(f"Telemetry record missing required fields: {', '.join(missing)}")
