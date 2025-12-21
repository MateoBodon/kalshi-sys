from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from kalshi_alpha.brokers import create_broker
from kalshi_alpha.brokers.kalshi.base import BrokerOrder
from kalshi_alpha.brokers.kalshi.http_client import KalshiHttpClient
from kalshi_alpha.brokers.kalshi.live import LiveBroker
from kalshi_alpha.exec.pilot.config import PilotConfig
from kalshi_alpha.exec.telemetry import TelemetrySink


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


def _write_dummy_key(tmp_path: Path) -> Path:
    key_path = tmp_path / "dummy_kalshi_key.pem"
    key_path.write_text("dummy-key", encoding="utf-8")
    return key_path


def _set_live_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key_path = _write_dummy_key(tmp_path)
    monkeypatch.setenv("KALSHI_API_KEY_ID", "TESTKEY")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM_PATH", str(key_path))


def _sample_order() -> BrokerOrder:
    return BrokerOrder(
        idempotency_key="demo-key-123",
        market_id="M-DEMO",
        strike=270.5,
        side="YES",
        price=0.42,
        contracts=3,
        probability=0.63,
        metadata={"order_id": "O-1", "market_ticker": "TEST-TICKER", "liquidity": "maker"},
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
    _set_live_env(monkeypatch, tmp_path)

    artifacts = tmp_path / "reports" / "_artifacts"
    audit_dir = tmp_path / "data" / "proc" / "audit"
    client_stub = _RecordingHttpClient()
    http_client = cast(KalshiHttpClient, client_stub)

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
            "kill_switch_path": tmp_path / "kill_switch",
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
    _set_live_env(monkeypatch, tmp_path)

    client_stub = _RecordingHttpClient()
    http_client = cast(KalshiHttpClient, client_stub)
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
        acknowledge_risks=True,
        kill_switch_path=tmp_path / "kill_switch",
    )

    caplog.clear()
    order = _sample_order()
    broker.place([order])

    log_messages = " ".join(record.getMessage() for record in caplog.records)
    assert order.idempotency_key not in log_messages

    assert client_stub.requests, "request should have been recorded"
    request_meta = client_stub.requests[0]
    assert request_meta["idempotency_key"] == order.idempotency_key

    audit_files = sorted((tmp_path / "data" / "proc" / "audit").glob("live_orders_*.jsonl"))
    assert len(audit_files) == 1
    lines = [cast(dict[str, Any], json.loads(line)) for line in audit_files[0].read_text().splitlines()]
    assert lines[0]["action"] == "place_intent"
    assert lines[0]["idempotency_key"] == order.idempotency_key

    telemetry_files = sorted(telemetry_dir.rglob("exec.jsonl"))
    assert telemetry_files
    telemetry_lines = [cast(dict[str, Any], json.loads(line)) for line in telemetry_files[0].read_text().splitlines()]
    assert {entry["event_type"] for entry in telemetry_lines} == {"sent", "ack"}
    events = {entry["event_type"]: entry["data"] for entry in telemetry_lines}
    masked = events["sent"]["idempotency_key"]
    assert masked.endswith(order.idempotency_key[-4:])
    assert set(masked) >= {"*"}
    assert events["sent"]["side"] == order.side
    assert events["sent"]["size"] == order.contracts
    assert "latency_ms" not in events["sent"]
    ack_event = events["ack"]
    assert ack_event["status_code"] == 200
    assert ack_event["latency_ms"] >= 0.0
    assert ack_event["idempotency_key"].endswith(order.idempotency_key[-4:])


def test_live_broker_cancel_serializes_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI", raising=False)
    _set_live_env(monkeypatch, tmp_path)

    client_stub = _RecordingHttpClient()
    http_client = cast(KalshiHttpClient, client_stub)
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
        acknowledge_risks=True,
        kill_switch_path=tmp_path / "kill_switch",
    )

    broker.cancel(["ORD-1"])
    assert client_stub.requests[0]["path"] == "/portfolio/orders/ORD-1"
    audit_files = sorted((tmp_path / "data" / "proc" / "audit").glob("live_orders_*.jsonl"))
    assert audit_files, "cancel intent should have been recorded"
    lines = [cast(dict[str, Any], json.loads(line)) for line in audit_files[0].read_text().splitlines()]
    assert any(entry.get("action") == "cancel_intent" for entry in lines)
    telemetry_files = sorted(telemetry_dir.rglob("exec.jsonl"))
    assert telemetry_files
    telemetry_data = [cast(dict[str, Any], json.loads(line)) for line in telemetry_files[0].read_text().splitlines()]
    cancel_event = next(entry["data"] for entry in telemetry_data if entry["event_type"] == "cancel")
    assert cancel_event["order_id"] == "ORD-1"
    assert cancel_event["latency_ms"] >= 0.0


def test_live_broker_requires_acknowledgement_direct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI", raising=False)
    _set_live_env(monkeypatch, tmp_path)

    client_stub = _RecordingHttpClient()
    http_client = cast(KalshiHttpClient, client_stub)

    with pytest.raises(RuntimeError):
        LiveBroker(
            artifacts_dir=tmp_path / "reports" / "_artifacts",
            audit_dir=tmp_path / "data" / "proc" / "audit",
            http_client=http_client,
            rate_limit_per_second=10,
            queue_capacity=8,
            max_retries=1,
            timeout=0.1,
            kill_switch_path=tmp_path / "kill_switch",
        )


def test_pilot_boundary_rejects_crossing_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI", raising=False)
    _set_live_env(monkeypatch, tmp_path)

    client_stub = _RecordingHttpClient()
    http_client = cast(KalshiHttpClient, client_stub)
    pilot_config = PilotConfig(
        allowed_series=("INX",),
        max_contracts_per_order=1,
        max_unique_bins=2,
        require_live_broker=True,
        enforce_maker_only=True,
        require_acknowledgement=True,
        session_prefix="pilot",
    )
    broker = LiveBroker(
        artifacts_dir=tmp_path / "reports" / "_artifacts",
        audit_dir=tmp_path / "data" / "proc" / "audit",
        http_client=http_client,
        rate_limit_per_second=10,
        queue_capacity=8,
        max_retries=1,
        timeout=0.1,
        acknowledge_risks=True,
        pilot_mode=True,
        pilot_config=pilot_config,
        clock=lambda: datetime(2025, 11, 4, 20, 55, tzinfo=UTC),
        orders_state_path=tmp_path / "data" / "proc" / "state" / "orders.json",
        kill_switch_path=tmp_path / "kill_switch",
    )
    order = BrokerOrder(
        idempotency_key="crossing-1",
        market_id="M-INX-1",
        strike=5000.0,
        side="YES",
        price=0.45,
        contracts=1,
        probability=0.45,
        metadata={
            "market_ticker": "INX-TEST",
            "series": "INX",
            "liquidity": "maker",
            "book_snapshot_ts": "2025-11-04T20:55:00+00:00",
            "book_snapshot": {
                "bid": {"price": 0.44, "size": 5},
                "ask": {"price": 0.45, "size": 3},
            },
        },
    )

    with pytest.raises(RuntimeError) as excinfo:
        broker.place([order])
    assert "pilot_maker_only_crossing" in str(excinfo.value)
    assert not client_stub.requests


def test_pilot_boundary_rejects_stale_tob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI", raising=False)
    _set_live_env(monkeypatch, tmp_path)

    client_stub = _RecordingHttpClient()
    http_client = cast(KalshiHttpClient, client_stub)
    pilot_config = PilotConfig(
        allowed_series=("INX",),
        max_contracts_per_order=1,
        max_unique_bins=2,
        require_live_broker=True,
        enforce_maker_only=True,
        require_acknowledgement=True,
        session_prefix="pilot",
    )
    broker = LiveBroker(
        artifacts_dir=tmp_path / "reports" / "_artifacts",
        audit_dir=tmp_path / "data" / "proc" / "audit",
        http_client=http_client,
        rate_limit_per_second=10,
        queue_capacity=8,
        max_retries=1,
        timeout=0.1,
        acknowledge_risks=True,
        pilot_mode=True,
        pilot_config=pilot_config,
        clock=lambda: datetime(2025, 11, 4, 21, 0, tzinfo=UTC),
        orders_state_path=tmp_path / "data" / "proc" / "state" / "orders.json",
        kill_switch_path=tmp_path / "kill_switch",
        pilot_tob_staleness_seconds=1.0,
    )
    order = BrokerOrder(
        idempotency_key="stale-1",
        market_id="M-INX-2",
        strike=5000.0,
        side="YES",
        price=0.44,
        contracts=1,
        probability=0.45,
        metadata={
            "market_ticker": "INX-TEST",
            "series": "INX",
            "liquidity": "maker",
            "book_snapshot_ts": "2025-11-04T20:55:00+00:00",
            "book_snapshot": {
                "bid": {"price": 0.43, "size": 5},
                "ask": {"price": 0.46, "size": 3},
            },
        },
    )

    with pytest.raises(RuntimeError) as excinfo:
        broker.place([order])
    assert "pilot_maker_only_stale_tob" in str(excinfo.value)
    assert not client_stub.requests


def test_kill_switch_blocks_submit_and_cancel_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI", raising=False)
    _set_live_env(monkeypatch, tmp_path)

    client_stub = _RecordingHttpClient()
    http_client = cast(KalshiHttpClient, client_stub)
    kill_switch = tmp_path / "kill_switch"
    kill_switch.write_text("1", encoding="utf-8")

    broker = LiveBroker(
        artifacts_dir=tmp_path / "reports" / "_artifacts",
        audit_dir=tmp_path / "data" / "proc" / "audit",
        http_client=http_client,
        rate_limit_per_second=10,
        queue_capacity=8,
        max_retries=1,
        timeout=0.1,
        acknowledge_risks=True,
        kill_switch_path=kill_switch,
    )

    with pytest.raises(RuntimeError):
        broker.place([_sample_order()])

    broker.cancel(["ORD-1"])
    broker.replace([_sample_order()])
    assert broker.status()["queue_depth"] == 0
    assert not client_stub.requests


def test_kill_switch_blocks_queue_activity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI", raising=False)
    _set_live_env(monkeypatch, tmp_path)

    class _SpyQueue:
        def __init__(self) -> None:
            self.actions: list[str] = []

        def enqueue_cancel(self, order_id: str, cancel_fn):  # type: ignore[no-untyped-def]
            self.actions.append(f"cancel:{order_id}")
            cancel_fn(order_id)

        def enqueue_replace(self, **kwargs):  # type: ignore[no-untyped-def]
            self.actions.append("replace")

        def depth(self) -> int:
            return 0

    client_stub = _RecordingHttpClient()
    http_client = cast(KalshiHttpClient, client_stub)
    kill_switch = tmp_path / "kill_switch"
    kill_switch.write_text("1", encoding="utf-8")
    spy_queue = _SpyQueue()

    broker = LiveBroker(
        artifacts_dir=tmp_path / "reports" / "_artifacts",
        audit_dir=tmp_path / "data" / "proc" / "audit",
        http_client=http_client,
        rate_limit_per_second=10,
        queue_capacity=8,
        max_retries=1,
        timeout=0.1,
        acknowledge_risks=True,
        kill_switch_path=kill_switch,
        order_queue=spy_queue,
    )

    broker.cancel(["ORD-2"])
    broker.replace([_sample_order()])

    assert not spy_queue.actions
    assert not client_stub.requests
