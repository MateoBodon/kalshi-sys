"""Helpers for sigma drift monitor artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

SIGMA_DRIFT_ARTIFACT = Path("reports/_artifacts/monitors/sigma_drift.json")


def load_artifact(path: Path | None = None) -> dict[str, object] | None:
    artifact_path = path or SIGMA_DRIFT_ARTIFACT
    if not artifact_path.exists():
        return None
    try:
        return json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupted artifact
        return None


def shrink_for_series(series: str, *, artifact: Mapping[str, object] | None = None) -> float | None:
    payload = artifact or load_artifact()
    if not payload:
        return None
    series_payload = payload.get("series")
    if not isinstance(series_payload, Mapping):
        return None
    entry = series_payload.get(series.upper())
    if not isinstance(entry, Mapping):
        return None
    shrink = entry.get("shrink")
    try:
        value = float(shrink)
    except (TypeError, ValueError):
        return None
    if value <= 0.0:
        return None
    return min(value, 1.0)


__all__ = ["SIGMA_DRIFT_ARTIFACT", "load_artifact", "shrink_for_series"]
