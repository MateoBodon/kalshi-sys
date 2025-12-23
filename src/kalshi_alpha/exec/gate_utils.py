"""Shared helpers for quality gate configuration and artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from kalshi_alpha.core.gates import QualityGateResult


def resolve_quality_gate_config_path() -> Path:
    primary = Path("configs/quality_gates.yaml")
    if primary.exists():
        return primary
    fallback = Path("configs/quality_gates.example.yaml")
    if fallback.exists():
        return fallback
    return primary


def write_go_no_go(
    result: QualityGateResult,
    *,
    scope: str | None = None,
    scoped_blockers: list[str] | None = None,
    unscoped_blockers: list[str] | None = None,
    extra: dict[str, object] | None = None,
) -> Path:
    artifacts_dir = Path("reports/_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"go": bool(result.go), "reasons": list(result.reasons)}
    if scope:
        payload["scope"] = scope
        payload["scoped_blockers"] = list(scoped_blockers or result.reasons)
        payload["unscoped_blockers"] = list(unscoped_blockers or [])
    if extra:
        payload.update(extra)
    output_path = artifacts_dir / "go_no_go.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
