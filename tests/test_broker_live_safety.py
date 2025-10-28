from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalshi_alpha.brokers import create_broker
from kalshi_alpha.brokers.kalshi.base import BrokerOrder
from kalshi_alpha.brokers.kalshi.live import LiveBroker


class _StubResponse:
    def __init__(self, status_code: int = 200, payload: dict[str, object] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {"ok": True}

    def json(self) -> dict[str, object]:
        return self._payload


class _StubSession:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.auth = None
        self.requests: list[dict[str, object]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> _StubResponse:
        self.requests.append(
            {
                "method": method,
                "url": url,
                "json": json,
                "headers": headers or {},
                "timeout": timeout,
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
    monkeypatch.setenv("KALSHI_API_KEY", "KEY123")
    monkeypatch.setenv("KALSHI_API_SECRET", "SECRET456")

    artifacts = tmp_path / "reports" / "_artifacts"
    audit_dir = tmp_path / "data" / "proc" / "audit"
    session = _StubSession()

    broker = create_broker(
        "live",
        artifacts_dir=artifacts,
        audit_dir=audit_dir,
        acknowledge_risks=True,
        live_kwargs={
            "session": session,
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
    monkeypatch.setenv("KALSHI_API_KEY", "SUPERSECRETKEY")
    monkeypatch.setenv("KALSHI_API_SECRET", "SUPERSECRETSECRET")

    session = _StubSession()
    broker = LiveBroker(
        artifacts_dir=tmp_path / "reports" / "_artifacts",
        audit_dir=tmp_path / "data" / "proc" / "audit",
        session=session,
        rate_limit_per_second=10,
        queue_capacity=8,
        max_retries=1,
        timeout=0.1,
    )

    caplog.clear()
    order = _sample_order()
    broker.place([order])

    log_messages = " ".join(record.getMessage() for record in caplog.records)
    assert "SUPERSECRETKEY" not in log_messages
    assert "SUPERSECRETSECRET" not in log_messages

    assert session.requests, "request should have been recorded"
    request_meta = session.requests[0]
    assert request_meta["headers"].get("Idempotency-Key") == order.idempotency_key

    audit_files = sorted((tmp_path / "data" / "proc" / "audit").glob("live_orders_*.jsonl"))
    assert len(audit_files) == 1
    lines = [json.loads(line) for line in audit_files[0].read_text().splitlines()]
    assert lines[0]["action"] == "place_intent"
    assert lines[0]["idempotency_key"] == order.idempotency_key


def test_live_broker_cancel_serializes_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("KALSHI_API_KEY", "KEY123")
    monkeypatch.setenv("KALSHI_API_SECRET", "SECRET456")

    session = _StubSession()
    broker = LiveBroker(
        artifacts_dir=tmp_path / "reports" / "_artifacts",
        audit_dir=tmp_path / "data" / "proc" / "audit",
        session=session,
        rate_limit_per_second=10,
        queue_capacity=8,
        max_retries=1,
        timeout=0.1,
    )

    broker.cancel(["ORD-1"])
    audit_files = sorted((tmp_path / "data" / "proc" / "audit").glob("live_orders_*.jsonl"))
    assert audit_files, "cancel intent should have been recorded"
    lines = [json.loads(line) for line in audit_files[0].read_text().splitlines()]
    assert any(entry.get("action") == "cancel_intent" for entry in lines)
