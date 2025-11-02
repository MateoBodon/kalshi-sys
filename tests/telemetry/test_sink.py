from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from kalshi_alpha.exec.telemetry import TelemetrySink, sanitize_book_snapshot
from kalshi_alpha.exec.telemetry.sink import EVENT_TYPES


def _clock_factory(times: list[datetime]) -> Callable[[], datetime]:
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

    sink.emit(
        "sent",
        source="rest",
        data={
            "order_id": "O-1",
            "signature": "SECRET123456",
            "snapshot": sanitize_book_snapshot(
                {"bids": [[0.42, 10, 3], [0.41, 8]], "asks": [[0.45, 6]]},
                depth=1,
            ),
        },
    )

    target = tmp_path / "telemetry" / "2025" / "11" / "02" / "exec.jsonl"
    assert target.exists()
    lines = [cast(dict[str, Any], json.loads(line)) for line in target.read_text(encoding="utf-8").splitlines()]
    first_data = cast(dict[str, Any], lines[0]["data"])
    assert lines[0]["event_type"] == "sent"
    assert first_data["order_id"] == "O-1"
    signature = cast(str, first_data["signature"])
    assert signature.endswith("3456")
    assert signature.startswith("***")
    snapshot = cast(dict[str, Any], first_data["snapshot"])
    assert snapshot["bids"] == [{"price": 0.42, "size": 10.0}]


def test_sink_rotates_daily(tmp_path: Path) -> None:
    moments = [
        datetime(2025, 11, 2, 23, tzinfo=UTC),
        datetime(2025, 11, 3, 0, tzinfo=UTC),
    ]
    sink = TelemetrySink(base_dir=tmp_path / "telemetry", clock=_clock_factory(moments))

    sink.emit("sent", source="rest", data={"idempotency_key": "abc123"})
    sink.emit("ack", source="rest", data={"idempotency_key": "abc123"})

    day1 = tmp_path / "telemetry" / "2025" / "11" / "02" / "exec.jsonl"
    day2 = tmp_path / "telemetry" / "2025" / "11" / "03" / "exec.jsonl"
    assert day1.exists()
    assert day2.exists()
    lines = [cast(dict[str, Any], json.loads(line)) for line in day1.read_text(encoding="utf-8").splitlines()]
    assert all(
        cast(dict[str, Any], entry["data"])["idempotency_key"].endswith("c123")
        for entry in lines
    )


def test_sink_rejects_unknown_event(tmp_path: Path) -> None:
    sink = TelemetrySink(base_dir=tmp_path / "telemetry", clock=lambda: datetime.now(tz=UTC))
    with pytest.raises(ValueError):
        sink.emit("unsupported", source="rest", data={})

    # sanity check that official events remain registered
    assert "sent" in EVENT_TYPES


def test_sanitize_book_snapshot_handles_sequences() -> None:
    snapshot = sanitize_book_snapshot(
        {
            "bids": [
                {"price": 0.52, "size": "11"},
                [0.51, 9],
                {"price": "0.50", "size": None},
            ],
            "asks": {"0.55": 7, "0.56": 10},
            "ts": "ignored",
        },
        depth=2,
    )
    assert snapshot is not None
    assert snapshot["bids"] == [{"price": 0.52, "size": 11.0}, {"price": 0.51, "size": 9.0}]
    assert snapshot["asks"] == [{"price": 0.55, "size": 7.0}, {"price": 0.56, "size": 10.0}]
    assert snapshot["ts"] == "ignored"
