"""Fetch Kalshi fee/rule URLs, compute checksums, and enforce manual ack."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import requests
import yaml

from kalshi_alpha.datastore import paths as datastore_paths
from kalshi_alpha.exec.monitors import fee_rules

PROJECT_ROOT = datastore_paths.PROJECT_ROOT
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "fee_rules_watch.yaml"
DEFAULT_STATE = PROJECT_ROOT / "data" / "proc" / "state" / "fee_rules_checksums.json"
DEFAULT_ARTIFACT = fee_rules.FEE_RULES_ARTIFACT


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Kalshi fee/rule URLs for changes.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config describing targets.")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE, help="State file storing acknowledged checksums.")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT, help="Monitor artifact output path.")
    parser.add_argument("--timeout", type=int, default=15, help="HTTP timeout in seconds per request.")
    parser.add_argument("--ack", action="store_true", help="Accept current content and update state (manual ack).")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = _load_config(args.config)
    current = _fetch_targets(config, timeout=args.timeout)
    state = _load_state(args.state)
    changed = _diff_state(state, current)
    now = datetime.now(tz=UTC)
    if args.ack:
        _write_state(args.state, current, now)
        changed = []
        status = "OK"
        message = "acknowledged"
    else:
        status = "OK" if not changed else "ALERT"
        message = None
    artifact_payload = {
        "generated_at": now.isoformat(),
        "status": status,
        "requires_ack": bool(changed),
        "message": message,
        "changed": changed,
        "targets": current,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact_payload, indent=2, sort_keys=True), encoding="utf-8")
    if changed and not args.ack:
        print("[fee_rules] change detected, run with --ack after review.")
        return 1
    if not changed and not args.ack and not state:
        print("[fee_rules] no baseline found; run with --ack to initialize.")
        return 1
    return 0


def _load_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"fee rules config missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("fee rules config requires a non-empty 'targets' list")
    normalized = []
    for entry in targets:
        if not isinstance(entry, dict):
            continue
        target_id = str(entry.get("id") or "").strip()
        url = str(entry.get("url") or "").strip()
        if not target_id or not url:
            continue
        normalized.append({"id": target_id, "url": url})
    if not normalized:
        raise ValueError("fee rules config produced no valid targets")
    return {"targets": normalized}


def _fetch_targets(config: dict[str, object], *, timeout: int) -> dict[str, dict[str, object]]:
    targets: dict[str, dict[str, object]] = {}
    for entry in config.get("targets", []):
        target_id = entry["id"]
        url = entry["url"]
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        checksum = hashlib.sha256(resp.content).hexdigest()
        targets[target_id] = {
            "url": url,
            "checksum": checksum,
            "content_length": len(resp.content),
            "fetched_at": datetime.now(tz=UTC).isoformat(),
        }
    return targets


def _load_state(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover
        return {}
    targets = payload.get("targets")
    if not isinstance(targets, dict):
        return {}
    return {str(key): value for key, value in targets.items() if isinstance(value, dict)}


def _write_state(path: Path, targets: dict[str, dict[str, object]], now: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": now.isoformat(),
        "targets": targets,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _diff_state(state: dict[str, dict[str, object]], current: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    changed: list[dict[str, object]] = []
    for key, info in current.items():
        previous = (state or {}).get(key, {})
        if not previous or previous.get("checksum") != info.get("checksum"):
            changed.append({
                "id": key,
                "previous_checksum": previous.get("checksum"),
                "current_checksum": info.get("checksum"),
            })
    return changed


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
