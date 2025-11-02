from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import requests
import requests_mock
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kalshi_alpha.brokers.kalshi.base import BrokerOrder
from kalshi_alpha.brokers.kalshi.http_client import KalshiClockSkewError, KalshiHttpClient
from kalshi_alpha.brokers.kalshi.live import LiveBroker


class _FixedClock:
    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self) -> datetime:
        return self._moment


class _SkewedClock:
    def now(self) -> datetime:
        return datetime.now(tz=UTC) + timedelta(seconds=10)


def _write_rsa_key(tmp_path: Path) -> Path:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path = tmp_path / "kalshi_test_key.pem"
    path.write_bytes(pem)
    return path


def _load_private_key(path: Path) -> rsa.RSAPrivateKey:
    return serialization.load_pem_private_key(path.read_bytes(), password=None)


def _sample_order() -> BrokerOrder:
    return BrokerOrder(
        idempotency_key="idem-1234",
        market_id="M1",
        strike=270.5,
        side="YES",
        price=0.45,
        contracts=2,
        probability=0.6,
        metadata={"order_id": "ORD-1"},
    )


def test_request_signs_headers_without_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_path = _write_rsa_key(tmp_path)
    private_key = _load_private_key(key_path)
    monkeypatch.setenv("KALSHI_API_KEY_ID", "ACCESS123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM_PATH", str(key_path))

    with requests_mock.Mocker() as mocker:
        mocker.post(
            "https://api.elections.kalshi.com/trade-api/v2/orders",
            json={"order_id": "O-1"},
            status_code=201,
        )

        clock = _FixedClock(datetime.now(tz=UTC))
        client = KalshiHttpClient(session=requests.Session(), clock=clock)
        response = client.post(
            "/orders",
            json_body={"market_id": "M1"},
            idempotency_key="idem-1234",
        )

        assert response.status_code == 201
        assert len(mocker.request_history) == 1
        request = mocker.request_history[0]
        assert "Authorization" not in request.headers
        timestamp_ms = request.headers["KALSHI-ACCESS-TIMESTAMP"]
        assert len(timestamp_ms) == 13
        signature_b64 = request.headers["KALSHI-ACCESS-SIGNATURE"]
        payload = f"{timestamp_ms}POST/orders".encode()
        signature = base64.b64decode(signature_b64)
        private_key.public_key().verify(
            signature,
            payload,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )


def test_signature_excludes_query_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_path = _write_rsa_key(tmp_path)
    public_key = _load_private_key(key_path).public_key()
    monkeypatch.setenv("KALSHI_API_KEY_ID", "ACCESS123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM_PATH", str(key_path))

    with requests_mock.Mocker() as mocker:
        mocker.get(
            "https://api.elections.kalshi.com/trade-api/v2/markets",
            json={"markets": []},
        )

        client = KalshiHttpClient(
            session=requests.Session(),
            clock=_FixedClock(datetime.now(tz=UTC)),
        )
        response = client.get("/markets", params={"series": "CPI"})
        assert response.status_code == 200

        request = mocker.request_history[0]
        timestamp_ms = request.headers["KALSHI-ACCESS-TIMESTAMP"]
        signature = base64.b64decode(request.headers["KALSHI-ACCESS-SIGNATURE"])

        # Expected payload uses path only.
        valid_payload = f"{timestamp_ms}GET/markets".encode()
        public_key.verify(
            signature,
            valid_payload,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )

        # Incorrect payload including the query string must fail verification.
        invalid_payload = f"{timestamp_ms}GET/markets?series=CPI".encode()
        with pytest.raises(InvalidSignature):
            public_key.verify(
                signature,
                invalid_payload,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )


def test_signature_uses_millisecond_timestamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_path = _write_rsa_key(tmp_path)
    public_key = _load_private_key(key_path).public_key()
    monkeypatch.setenv("KALSHI_API_KEY_ID", "ACCESS123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM_PATH", str(key_path))

    with requests_mock.Mocker() as mocker:
        mocker.get(
            "https://api.elections.kalshi.com/trade-api/v2/portfolio/balance",
            json={"balance": 0},
        )

        client = KalshiHttpClient(
            session=requests.Session(),
            clock=_FixedClock(datetime.now(tz=UTC)),
        )
        response = client.get("/portfolio/balance")
        assert response.status_code == 200

        request = mocker.request_history[0]
        timestamp_ms_str = request.headers["KALSHI-ACCESS-TIMESTAMP"]
        assert len(timestamp_ms_str) == 13
        signature = base64.b64decode(request.headers["KALSHI-ACCESS-SIGNATURE"])

        # Correct millisecond payload verifies.
        valid_payload = f"{timestamp_ms_str}GET/portfolio/balance".encode()
        public_key.verify(
            signature,
            valid_payload,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )

        # Using seconds instead of milliseconds should fail verification.
        seconds_payload = f"{int(timestamp_ms_str) // 1000}GET/portfolio/balance".encode()
        with pytest.raises(InvalidSignature):
            public_key.verify(
                signature,
                seconds_payload,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )


def test_retry_on_retryable_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_path = _write_rsa_key(tmp_path)
    monkeypatch.setenv("KALSHI_API_KEY_ID", "ACCESS123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM_PATH", str(key_path))

    sleeps: list[float] = []

    def capture_sleep(value: float) -> None:
        sleeps.append(value)

    with requests_mock.Mocker() as mocker:
        mocker.post(
            "https://api.elections.kalshi.com/trade-api/v2/orders",
            [
                {"status_code": 500},
                {"json": {"order_id": "O-2"}, "status_code": 200},
            ],
        )
        client = KalshiHttpClient(
            session=requests.Session(),
            clock=_FixedClock(datetime.now(tz=UTC)),
            sleeper=capture_sleep,
            retry_backoff=0.1,
        )
        response = client.post("/orders", json_body={"market_id": "M2"})
        assert response.status_code == 200
        assert sleeps == [0.1]


def test_clock_skew_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_path = _write_rsa_key(tmp_path)
    monkeypatch.setenv("KALSHI_API_KEY_ID", "ACCESS123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM_PATH", str(key_path))

    client = KalshiHttpClient(session=requests.Session(), clock=_SkewedClock())

    with pytest.raises(KalshiClockSkewError):
        client.get("/markets")


def test_live_broker_order_lifecycle_with_mocked_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_path = _write_rsa_key(tmp_path)
    monkeypatch.setenv("KALSHI_API_KEY_ID", "ACCESS123")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM_PATH", str(key_path))
    monkeypatch.delenv("CI", raising=False)

    with requests_mock.Mocker() as mocker:
        mocker.post(
            "https://api.elections.kalshi.com/trade-api/v2/orders",
            [{"json": {"order_id": "ORD-1"}, "status_code": 201}],
        )
        mocker.post(
            "https://api.elections.kalshi.com/trade-api/v2/orders/ORD-1/cancel",
            [{"status_code": 200}],
        )

        session = requests.Session()
        http_client = KalshiHttpClient(
            session=session,
            clock=_FixedClock(datetime.now(tz=UTC)),
        )
        broker = LiveBroker(
            artifacts_dir=tmp_path / "reports" / "_artifacts",
            audit_dir=tmp_path / "data" / "proc" / "audit",
            http_client=http_client,
            rate_limit_per_second=100,
            queue_capacity=8,
            max_retries=2,
            timeout=1.0,
        )

        order = _sample_order()
        broker.place([order])
        broker.cancel(["ORD-1"])

        paths = [req.path for req in mocker.request_history]
        assert "/trade-api/v2/orders" in paths
        assert any(path.lower() == "/trade-api/v2/orders/ord-1/cancel" for path in paths)

        audit_files = sorted((tmp_path / "data" / "proc" / "audit").glob("live_orders_*.jsonl"))
        assert audit_files, "audit file should be written"
        entries = [line for line in audit_files[0].read_text().splitlines() if line]
        assert any('"action": "place_intent"' in line for line in entries)
        assert any('"action": "cancel_intent"' in line for line in entries)
