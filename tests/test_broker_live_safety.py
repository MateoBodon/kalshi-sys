from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kalshi_alpha.brokers import create_broker
from kalshi_alpha.brokers.kalshi.base import BrokerOrder
from kalshi_alpha.brokers.kalshi.live import LiveBroker
from kalshi_alpha.exec.telemetry.sink import TelemetrySink


class _StubResponse:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code


class _RecordingHttpClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, object] | None = None,
        idempotency_key: str | None = None,
        **_: object,
    ) -> _StubResponse:
        self.requests.append(
            {
                "method": method,
                "path": path,
                "json": json_body,
                "idempotency_key": idempotency_key,
            }
        )
        return _StubResponse()



def _sample_order() -> BrokerOrder:
    return BrokerOrder(
        idempotency_key="demo-key-123",
        market_id="M-DEMO",
        strike=270.5,
        side="YES",
        price=0.42,
        contracts=3,
        probability=0.63,
        metadata={"order_id": "O-1"},
    )


def test_create_broker_requires_acknowledgement(tmp_path: Path) -> None:
    artifacts = tmp_path / "reports" / "_artifacts"
    audit_dir = tmp_path / "data" / "proc" / "audit"

    with pytest.raises(RuntimeError):
        create_broker(
            "live",
            artifacts_dir=artifacts,
            audit_dir=audit_dir,
            acknowledge_risks=False,
        )


def test_create_broker_live_with_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI", raising=False)

    artifacts = tmp_path / "reports" / "_artifacts"
    audit_dir = tmp_path / "data" / "proc" / "audit"
    http_client = _RecordingHttpClient()

    broker = create_broker(
        "live",
        artifacts_dir=artifacts,
        audit_dir=audit_dir,
        acknowledge_risks=True,
        live_kwargs={
            "http_client": http_client,
            "rate_limit_per_second": 10,
            "queue_capacity": 8,
            "max_retries": 1,
            "timeout": 0.1,
        },
    )
    assert isinstance(broker, LiveBroker)


def test_live_broker_does_not_log_secrets_and_writes_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI", raising=False)

    http_client = _RecordingHttpClient()
    telemetry_dir = tmp_path / "telemetry"
    telemetry = TelemetrySink(
        base_dir=telemetry_dir,
        clock=lambda: datetime(2025, 11, 2, 12, tzinfo=UTC),
    )
    broker = LiveBroker(
        artifacts_dir=tmp_path / "reports" / "_artifacts",
        audit_dir=tmp_path / "data" / "proc" / "audit",
        http_client=http_client,
        rate_limit_per_second=10,
        queue_capacity=8,
        max_retries=1,
        timeout=0.1,
        telemetry_sink=telemetry,
    )

    caplog.clear()
    order = _sample_order()
    broker.place([order])

    log_messages = " ".join(record.getMessage() for record in caplog.records)
    assert order.idempotency_key not in log_messages

    assert http_client.requests, "request should have been recorded"
    request_meta = http_client.requests[0]
    assert request_meta["idempotency_key"] == order.idempotency_key

    audit_files = sorted((tmp_path / "data" / "proc" / "audit").glob("live_orders_*.jsonl"))
    assert len(audit_files) == 1
    lines = [json.loads(line) for line in audit_files[0].read_text().splitlines()]
    assert lines[0]["action"] == "place_intent"
    assert lines[0]["idempotency_key"] == order.idempotency_key

    telemetry_files = sorted(telemetry_dir.rglob("exec.jsonl"))
    assert telemetry_files
    telemetry_lines = [json.loads(line) for line in telemetry_files[0].read_text().splitlines()]
    assert {entry["event_type"] for entry in telemetry_lines} == {"sent", "ack"}
    assert telemetry_lines[0]["data"]["idempotency_key"] == order.idempotency_key


def test_live_broker_cancel_serializes_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI", raising=False)

    http_client = _RecordingHttpClient()
    telemetry_dir = tmp_path / "telemetry"
    telemetry = TelemetrySink(
        base_dir=telemetry_dir,
        clock=lambda: datetime(2025, 11, 2, 13, tzinfo=UTC),
    )
    broker = LiveBroker(
        artifacts_dir=tmp_path / "reports" / "_artifacts",
        audit_dir=tmp_path / "data" / "proc" / "audit",
        http_client=http_client,
        rate_limit_per_second=10,
        queue_capacity=8,
        max_retries=1,
        timeout=0.1,
        telemetry_sink=telemetry,
    )

    broker.cancel(["ORD-1"])
    assert http_client.requests[0]["path"] == "/orders/ORD-1/cancel"
    audit_files = sorted((tmp_path / "data" / "proc" / "audit").glob("live_orders_*.jsonl"))
    assert audit_files, "cancel intent should have been recorded"
    lines = [json.loads(line) for line in audit_files[0].read_text().splitlines()]
    assert any(entry.get("action") == "cancel_intent" for entry in lines)
    telemetry_files = sorted(telemetry_dir.rglob("exec.jsonl"))
    assert telemetry_files
    telemetry_data = [json.loads(line) for line in telemetry_files[0].read_text().splitlines()]
    assert any(entry["event_type"] == "cancel" for entry in telemetry_data)
