from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kalshi_alpha.exec.telemetry.sink import EVENT_TYPES, TelemetrySink


def _clock_factory(times: list[datetime]) -> callable:
    counter = iter(times)

    def _next() -> datetime:
        try:
            return next(counter)
        except StopIteration:  # pragma: no cover - defensive
            return times[-1]

    return _next


def test_sink_writes_jsonl(tmp_path: Path) -> None:
    moment = datetime(2025, 11, 2, 12, tzinfo=UTC)
    sink = TelemetrySink(base_dir=tmp_path / "telemetry", clock=lambda: moment)

    sink.emit("sent", source="rest", data={"order_id": "O-1"})

    target = tmp_path / "telemetry" / "2025" / "11" / "02" / "exec.jsonl"
    assert target.exists()
    lines = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["event_type"] == "sent"
    assert lines[0]["data"]["order_id"] == "O-1"


def test_sink_rotates_daily(tmp_path: Path) -> None:
    moments = [
        datetime(2025, 11, 2, 23, tzinfo=UTC),
        datetime(2025, 11, 3, 0, tzinfo=UTC),
    ]
    sink = TelemetrySink(base_dir=tmp_path / "telemetry", clock=_clock_factory(moments))

    sink.emit("sent", source="rest", data={"idempotency_key": "abc"})
    sink.emit("ack", source="rest", data={"idempotency_key": "abc"})

    day1 = tmp_path / "telemetry" / "2025" / "11" / "02" / "exec.jsonl"
    day2 = tmp_path / "telemetry" / "2025" / "11" / "03" / "exec.jsonl"
    assert day1.exists()
    assert day2.exists()


def test_sink_rejects_unknown_event(tmp_path: Path) -> None:
    sink = TelemetrySink(base_dir=tmp_path / "telemetry", clock=lambda: datetime.now(tz=UTC))
    with pytest.raises(ValueError):
        sink.emit("unsupported", source="rest", data={})

    # sanity check that official events remain registered
    assert "sent" in EVENT_TYPES
