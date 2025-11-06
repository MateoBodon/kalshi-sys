"""Replay recorded Polygon websocket aggregates into the index pipeline artifacts."""

from __future__ import annotations

import argparse
import json
import time as time_module
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

from kalshi_alpha.exec.collectors.polygon_ws import (
    DEFAULT_CHANNEL,
    DEFAULT_FRESHNESS_OUTPUT,
    DEFAULT_PROC_PARQUET,
    DEFAULT_SYMBOLS,
    _normalize_entries,
    _process_entries,
    _resolved_aliases,
)
from kalshi_alpha.exec.monitors import freshness

ET = ZoneInfo("America/New_York")
DEFAULT_FRESHNESS_CONFIG = Path("configs/freshness.yaml")
DEFAULT_SUMMARY_PATH = Path("reports/_artifacts/replay/polygon_index_replay.json")


@dataclass(frozen=True)
class ReplayRecord:
    timestamp: datetime
    payload: dict[str, object]


@dataclass
class ReplayConfig:
    data_file: Path
    speed: float
    start_time: time | None
    end_time: time | None
    freshness_config: Path
    freshness_output: Path
    proc_parquet: Path
    alias_map: dict[str, tuple[str, ...]]
    channel_prefix: str
    summary_path: Path
    data_root: Path | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay recorded Polygon index websocket data.")
    parser.add_argument("--file", type=Path, required=True, help="Recorded websocket JSON file")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier (default: 1.0)")
    parser.add_argument("--start", help='Start time in ET (HH:MM)', default=None)
    parser.add_argument("--end", help='End time in ET (HH:MM)', default=None)
    parser.add_argument(
        "--freshness-config",
        type=Path,
        default=DEFAULT_FRESHNESS_CONFIG,
        help="Freshness config path (default: configs/freshness.yaml)",
    )
    parser.add_argument(
        "--freshness-output",
        type=Path,
        default=DEFAULT_FRESHNESS_OUTPUT,
        help="Freshness artifact output (default: reports/_artifacts/monitors/freshness.json)",
    )
    parser.add_argument(
        "--proc-parquet",
        type=Path,
        default=DEFAULT_PROC_PARQUET,
        help="Per-tick parquet output (default: data/proc/polygon_index/snapshot_*.parquet)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Summary report output (default: reports/_artifacts/replay/polygon_index_replay.json)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override data root for snapshots/proc outputs (testing utility)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    alias_map = _resolved_aliases(DEFAULT_SYMBOLS)
    config = ReplayConfig(
        data_file=args.file,
        speed=max(float(args.speed), 0.0),
        start_time=_parse_clock(args.start) if args.start else None,
        end_time=_parse_clock(args.end) if args.end else None,
        freshness_config=args.freshness_config,
        freshness_output=args.freshness_output,
        proc_parquet=args.proc_parquet,
        alias_map=alias_map,
        channel_prefix=DEFAULT_CHANNEL,
        summary_path=args.summary,
        data_root=args.data_root,
    )
    _run_replay(config)


def _run_replay(config: ReplayConfig) -> None:
    if config.data_root is not None:
        _override_data_root(config.data_root)
    records = _load_records(config.data_file)
    filtered = _filter_window(records, config.start_time, config.end_time)
    if not filtered:
        raise SystemExit("No replay messages found in requested window")
    filtered.sort(key=lambda record: record.timestamp)

    summary = {
        "messages_processed": 0,
        "window_start": filtered[0].timestamp.astimezone(ET).isoformat(),
        "window_end": filtered[-1].timestamp.astimezone(ET).isoformat(),
        "speed": config.speed,
        "freshness_artifact": str(config.freshness_output),
        "proc_parquet": str(config.proc_parquet),
    }

    previous_ts: datetime | None = None
    for record in filtered:
        if previous_ts is not None and config.speed > 0:
            delta = max((record.timestamp - previous_ts).total_seconds(), 0.0)
            if delta > 0:
                time_module.sleep(delta / config.speed)
        _replay_step(
            payload=record.payload,
            timestamp=record.timestamp,
            config=config,
        )
        previous_ts = record.timestamp
        summary["messages_processed"] += 1

    artifact = freshness.load_artifact(config.freshness_output)
    stale_feeds = (artifact or {}).get("metrics", {}).get("stale_feeds") or []
    summary["stale_feeds"] = list(stale_feeds)
    summary["freshness_status"] = (artifact or {}).get("status", "UNKNOWN")
    summary_path = config.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _replay_step(*, payload: dict[str, object], timestamp: datetime, config: ReplayConfig) -> None:
    entries = _normalize_entries(payload)
    now_utc = timestamp.astimezone(UTC)
    _process_entries(
        entries=entries,
        alias_map=config.alias_map,
        channel_prefix=config.channel_prefix,
        now=now_utc,
        proc_parquet=config.proc_parquet,
        freshness_config=config.freshness_config,
        freshness_output=config.freshness_output,
    )


def _load_records(path: Path) -> list[ReplayRecord]:
    if not path.exists():
        raise SystemExit(f"Replay file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [_json_loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("records") or payload.get("messages") or []
    if not isinstance(payload, list):
        raise SystemExit("Replay payload must be a list of records")

    records: list[ReplayRecord] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        ts_raw = entry.get("ts") or entry.get("timestamp")
        msg_raw = entry.get("msg") or entry.get("payload")
        ts = _parse_timestamp(ts_raw)
        if ts is None:
            continue
        message = _ensure_message_dict(msg_raw)
        records.append(ReplayRecord(timestamp=ts, payload=message))
    return records


def _filter_window(
    records: Iterable[ReplayRecord],
    start_time: time | None,
    end_time: time | None,
) -> list[ReplayRecord]:
    if start_time is None and end_time is None:
        return list(records)
    results: list[ReplayRecord] = []
    for record in records:
        et_time = record.timestamp.astimezone(ET).time()
        if _in_window(et_time, start_time, end_time):
            results.append(record)
    return results


def _parse_clock(value: str | None) -> time:
    if value is None:
        raise ValueError("Clock value cannot be None")
    parts = value.strip().split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid clock value '{value}'")
    hour = int(parts[0]) % 24
    minute = int(parts[1]) % 60
    second = int(parts[2]) % 60 if len(parts) == 3 else 0
    return time(hour, minute, second)


def _parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = datetime.utcfromtimestamp(float(value))
                parsed = parsed.replace(tzinfo=UTC)
            except ValueError:
                return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _ensure_message_dict(raw: object) -> dict[str, object]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return _json_loads(raw)
    raise SystemExit("Replay message must be dict or JSON string")


def _json_loads(value: str) -> dict[str, object]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Invalid JSON in replay message: {exc}") from exc
    if isinstance(payload, dict):
        return payload
    return {"data": payload}


def _in_window(clock: time, start: time | None, end: time | None) -> bool:
    if start is None and end is None:
        return True
    if start is None:
        return clock <= end  # type: ignore[operator]
    if end is None:
        return clock >= start
    if start <= end:
        return start <= clock <= end
    return clock >= start or clock <= end


def _override_data_root(data_root: Path) -> None:
    data_root = data_root.resolve()
    raw_root = (data_root / "raw").resolve()
    proc_root = (data_root / "proc").resolve()
    bootstrap_root = (data_root / "bootstrap").resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    proc_root.mkdir(parents=True, exist_ok=True)
    bootstrap_root.mkdir(parents=True, exist_ok=True)

    from kalshi_alpha.datastore import paths as datastore_paths
    from kalshi_alpha.datastore import snapshots as datastore_snapshots

    datastore_paths.DATA_ROOT = data_root
    datastore_paths.RAW_ROOT = raw_root
    datastore_paths.PROC_ROOT = proc_root
    datastore_paths.BOOTSTRAP_ROOT = bootstrap_root

    datastore_snapshots.RAW_ROOT = raw_root

    from kalshi_alpha.drivers.polygon_index import client as polygon_client

    polygon_client.POLYGON_RAW_ROOT = (raw_root / "polygon").resolve()

    from kalshi_alpha.exec.collectors import polygon_ws

    polygon_ws.POLYGON_RAW_ROOT = (raw_root / "polygon").resolve()


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    main()
