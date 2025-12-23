"""Telemetry run metadata writer for bounded TOB/quote-intent runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from kalshi_alpha.exec.collectors.tob_logger import (
    DEFAULT_INTENT_MAX_BYTES,
    DEFAULT_TOB_MAX_BYTES,
    DEFAULT_TOB_WINDOW_MAX_BYTES,
)


def write_telemetry_run_metadata(
    *,
    run_id: str,
    output_dir: Path,
    status: str,
    broker: str,
    telemetry_only: bool,
    series: Sequence[str] | None = None,
    window: Mapping[str, Any] | None = None,
    preflight: Mapping[str, Any] | None = None,
    tob_path: Path | None = None,
    quote_intents_path: Path | None = None,
    max_bytes_per_window: int = DEFAULT_TOB_WINDOW_MAX_BYTES,
    max_tob_bytes: int = DEFAULT_TOB_MAX_BYTES,
    max_intent_bytes: int = DEFAULT_INTENT_MAX_BYTES,
) -> Path:
    """Write a run metadata JSON describing the telemetry capture context."""

    if not run_id:
        raise ValueError("run_id is required for telemetry metadata")
    output_dir = Path(output_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    tob_path = tob_path or (output_dir / "tob" / f"{run_id}.jsonl.gz")
    quote_intents_path = quote_intents_path or (output_dir / "quote_intents" / f"{run_id}.jsonl.gz")
    payload = {
        "run_id": run_id,
        "status": status,
        "broker": broker,
        "telemetry_only": bool(telemetry_only),
        "series": list(series) if series else [],
        "window": dict(window) if window else {},
        "preflight": dict(preflight) if preflight else {},
        "bounds": {
            "max_bytes_per_window": int(max_bytes_per_window),
            "max_tob_bytes": int(max_tob_bytes),
            "max_quote_intent_bytes": int(max_intent_bytes),
        },
        "paths": {
            "tob": tob_path.as_posix(),
            "quote_intents": quote_intents_path.as_posix(),
        },
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    target = runs_dir / f"{run_id}.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


__all__ = ["write_telemetry_run_metadata"]
