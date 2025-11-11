"""Helper utilities for fee/rule watcher artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from kalshi_alpha.datastore import paths as datastore_paths

FEE_RULES_ARTIFACT = datastore_paths.PROJECT_ROOT / "reports" / "_artifacts" / "monitors" / "fee_rules.json"


def load_status(path: Path | None = None) -> dict[str, object] | None:
    artifact_path = path or FEE_RULES_ARTIFACT
    if not artifact_path.exists():
        return None
    try:
        return json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover
        return None


def is_ready(payload: Mapping[str, object] | None = None) -> tuple[bool, str | None]:
    data = payload or load_status()
    if not data:
        return False, "fee_rules_artifact_missing"
    status = str(data.get("status", "NO_DATA")).upper()
    if status != "OK":
        message = str(data.get("message") or "fee_rules_pending_ack")
        return False, message
    if bool(data.get("requires_ack")):
        return False, "fee_rules_ack_required"
    return True, None


__all__ = ["FEE_RULES_ARTIFACT", "load_status", "is_ready"]
