from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, cast

import pytest
import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from websockets.datastructures import Headers

from kalshi_alpha.brokers.kalshi.http_client import KalshiHttpClient
from kalshi_alpha.core.ws import KalshiWebsocketClient


def _write_rsa_key(tmp_path: Path) -> Path:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path = tmp_path / "kalshi_test_ws.pem"
    path.write_bytes(pem)
    return path


@pytest.mark.asyncio
async def test_core_ws_client_signs_handshake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_path = _write_rsa_key(tmp_path)
    private_key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    public_key = private_key.public_key()
    monkeypatch.setenv("KALSHI_API_KEY_ID", "ACCESSID")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM_PATH", str(key_path))

    received_headers: Headers | None = None
    received_payloads: list[dict[str, Any]] = []

    async def handler(websocket: Any) -> None:
        nonlocal received_headers
        received_headers = websocket.request.headers
        try:
            message = await websocket.recv()
        except websockets.ConnectionClosedOK:  # pragma: no cover - defensive
            return
        received_payloads.append(json.loads(message))
        await websocket.send(json.dumps({"type": "ack", "ts_ms": 1700000000000}))

    async with websockets.serve(handler, "localhost", 0) as server:
        sockets = list(server.sockets or [])
        port = sockets[0].getsockname()[1]
        ws_url = f"ws://localhost:{port}/trade-api/ws/v2"

        http_client = KalshiHttpClient()
        client = KalshiWebsocketClient(base_url=ws_url, http_client=http_client)

        async with client.session() as websocket:
            await websocket.send(json.dumps({"type": "subscribe", "channels": ["heartbeat"]}))
            response = await websocket.recv()
            payload = json.loads(response)
            assert payload["type"] == "ack"

    assert received_headers is not None
    assert received_headers.get("KALSHI-ACCESS-KEY") == "ACCESSID"
    timestamp_ms = received_headers.get("KALSHI-ACCESS-TIMESTAMP")
    assert timestamp_ms is not None
    signature = received_headers.get("KALSHI-ACCESS-SIGNATURE")
    assert signature is not None
    decoded_signature = base64.b64decode(signature)

    payload = f"{timestamp_ms}GET/trade-api/ws/v2".encode()
    rsa_public = cast(rsa.RSAPublicKey, public_key)
    rsa_public.verify(
        decoded_signature,
        payload,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )

    assert received_payloads == [{"type": "subscribe", "channels": ["heartbeat"]}]
