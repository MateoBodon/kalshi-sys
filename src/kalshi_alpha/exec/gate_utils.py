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


def write_go_no_go(result: QualityGateResult, *, scope: str | None = None) -> Path:
    artifacts_dir = Path("reports/_artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload = {"go": bool(result.go), "reasons": list(result.reasons)}
    if scope:
        payload["scope"] = str(scope)
    output_path = artifacts_dir / "go_no_go.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
