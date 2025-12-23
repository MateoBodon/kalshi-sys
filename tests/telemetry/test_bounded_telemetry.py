from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

from kalshi_alpha.exec.telemetry.sink import REQUIRED_TELEMETRY_FIELDS, TelemetryJsonlSink


def test_bounded_telemetry_sink_writes_gz_and_bounds(tmp_path: Path) -> None:
    ts = datetime(2025, 11, 2, 12, 0, tzinfo=UTC).isoformat()
    payload = {
        "record_type": "tob_snapshot",
        "run_id": "RUN123",
        "window_id": "INXU:hourly-1200@2025-11-02T17:00:00+00:00",
        "ts": ts,
        "series": "INXU",
        "market_ticker": "INXU-24NOV02-5000",
        "bid_price": 0.41,
        "ask_price": 0.43,
    }
    line = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    max_bytes = len(line.encode("utf-8")) + 1

    sink = TelemetryJsonlSink(
        run_id="RUN123",
        stream="tob",
        base_dir=tmp_path,
        max_bytes_per_window=max_bytes + 1,
    )

    assert sink.emit(payload) is True
    assert sink.emit(payload) is False

    target = tmp_path / "tob" / "RUN123.jsonl.gz"
    assert target.exists()
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]

    assert len(lines) == 1
    for field in REQUIRED_TELEMETRY_FIELDS:
        assert field in lines[0]
    assert lines[0]["ts"] == ts
