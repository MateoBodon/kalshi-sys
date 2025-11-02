from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
import requests
import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from websockets.datastructures import Headers

from kalshi_alpha.brokers.kalshi.http_client import Clock, KalshiHttpClient
from kalshi_alpha.brokers.kalshi.ws_client import KalshiWebsocketClient
from kalshi_alpha.exec.telemetry import TelemetrySink


def _write_rsa_key(tmp_path: Path) -> Path:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path = tmp_path / "kalshi_test_key.pem"
    path.write_bytes(pem)
    return path


class _RealtimeClock(Clock):
    def now(self) -> datetime:
        return datetime.now(tz=UTC)


@pytest.mark.asyncio
async def test_websocket_client_connects_and_subscribes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_path = _write_rsa_key(tmp_path)
    private_key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    public_key = private_key.public_key()
    monkeypatch.setenv("KALSHI_API_KEY_ID", "ACCESS123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM_PATH", str(key_path))

    received_headers: Headers | None = None
    received_messages: list[str] = []

    async def handler(websocket: Any) -> None:
        nonlocal received_headers
        received_headers = websocket.request.headers
        try:
            message = await websocket.recv()
        except websockets.ConnectionClosedOK:  # pragma: no cover - defensive
            return
        received_messages.append(message)
        await websocket.send(
            json.dumps(
                {
                    "type": "order_update",
                    "status": "acknowledged",
                    "order_id": "O-1",
                    "idempotency_key": "idem-123",
                    "book_snapshot": {"best_bid": 0.45},
                    "ts_ms": 1700000000000,
                }
            )
        )
        await websocket.send(json.dumps({"type": "heartbeat", "ts": 1700000001000}))

    async with websockets.serve(handler, "localhost", 0) as server:
        sockets = list(server.sockets or [])
        port = sockets[0].getsockname()[1]
        ws_url = f"ws://localhost:{port}/trade-api/ws/v2"

        http_client = KalshiHttpClient(session=requests.Session(), clock=_RealtimeClock())
        telemetry = TelemetrySink(
            base_dir=tmp_path / "telemetry_ws",
            clock=lambda: datetime(2025, 11, 2, tzinfo=UTC),
        )
        ws_client = KalshiWebsocketClient(
            base_url=ws_url,
            http_client=http_client,
            max_retries=1,
            retry_backoff=0.05,
            telemetry_sink=telemetry,
        )

        await ws_client.subscribe({"type": "subscribe", "channels": ["markets"]})
        response = await ws_client.receive()
        assert response["status"] == "acknowledged"
        heartbeat = await ws_client.receive()
        assert heartbeat["type"] == "heartbeat"
        await ws_client.close()

    assert received_headers is not None
    assert received_headers.get("KALSHI-ACCESS-KEY") == "ACCESS123"
    timestamp_ms = received_headers.get("KALSHI-ACCESS-TIMESTAMP")
    assert timestamp_ms is not None
    signature_header = received_headers.get("KALSHI-ACCESS-SIGNATURE")
    assert signature_header is not None
    signature = base64.b64decode(signature_header)

    payload = f"{timestamp_ms}GET/trade-api/ws/v2".encode()
    rsa_public = cast(rsa.RSAPublicKey, public_key)
    rsa_public.verify(
        signature,
        payload,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )

    assert received_messages == [json.dumps({"type": "subscribe", "channels": ["markets"]})]

    telemetry_files = sorted((tmp_path / "telemetry_ws").rglob("exec.jsonl"))
    assert telemetry_files
    entries = [cast(dict[str, Any], json.loads(line)) for line in telemetry_files[0].read_text().splitlines()]
    event_map = {
        entry["event_type"]: cast(dict[str, Any], entry["data"])
        for entry in entries
    }
    ack_payload = event_map["ack"]
    assert ack_payload["order_id"] == "O-1"
    assert ack_payload["latency_ms"] >= 0.0
    assert ack_payload["book_snapshot"]["best_bid"] == 0.45
    heartbeat_payload = event_map["heartbeat"]
    assert heartbeat_payload["latency_ms"] >= 0.0
