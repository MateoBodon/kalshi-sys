"""Shared Kalshi HTTP client implementing RSA-PSS auth, retries, and structured logging."""

from __future__ import annotations

import base64
import logging
import os
import time
from collections.abc import Callable, Mapping, MutableMapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from requests import Response
from requests.exceptions import RequestException

LOGGER = logging.getLogger(__name__)


_DEFAULT_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
_DEFAULT_TIMEOUT = 10.0
_DEFAULT_RETRIES = 3
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
_MAX_CLOCK_SKEW_SECONDS = 5


def _mask(value: str | None) -> str:
    if not value:
        return "***"
    return f"***{value[-4:]}"


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class Clock:
    """Small helper to allow deterministic clocks in tests."""

    def now(self) -> datetime:
        return datetime.now(tz=UTC)


class Sleep:
    """Wrapper so tests can bypass actual sleeping."""

    def __call__(self, seconds: float) -> None:  # pragma: no cover - simple delegation
        time.sleep(seconds)

class KalshiHttpError(RuntimeError):
    """Raised when the Kalshi API request exhausts retries."""


class KalshiClockSkewError(RuntimeError):
    """Raised when the local clock drifts too far from UTC."""


class KalshiHttpClient:
    """Resilient Kalshi HTTP client with header-based RSA-PSS signing."""

    def __init__(  # noqa: PLR0913 - configuration-heavy constructor
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        session: requests.Session | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_RETRIES,
        retry_backoff: float = 0.5,
        clock: Clock | None = None,
        sleeper: Callable[[float], None] | None = None,
        access_key_id: str | None = None,
        private_key_path: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._timeout = timeout
        self._max_retries = max(1, max_retries)
        self._retry_backoff = max(0.0, retry_backoff)
        self._clock = clock or Clock()
        self._sleep = sleeper or Sleep()

        parsed = urlparse(self._base_url)
        if not parsed.scheme or not parsed.netloc:
            raise RuntimeError(f"Invalid Kalshi base URL: {self._base_url}")
        self._base_path_prefix = parsed.path.rstrip("/")

        self._access_key_id = access_key_id or os.getenv("KALSHI_API_KEY_ID", "").strip()
        key_path = private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PEM_PATH", "").strip()
        if not self._access_key_id:
            raise RuntimeError("KALSHI_API_KEY_ID is required to authenticate with Kalshi.")
        if not key_path:
            raise RuntimeError(
                "KALSHI_PRIVATE_KEY_PEM_PATH is required to authenticate with Kalshi."
            )
        self._private_key = self._load_private_key(Path(key_path))

    # Public API ---------------------------------------------------------------------------------

    def get(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        return self.request("GET", path, params=params, headers=headers)

    def post(
        self,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> Response:
        return self.request(
            "POST",
            path,
            json_body=json_body,
            headers=headers,
            idempotency_key=idempotency_key,
        )

    def request(  # noqa: PLR0913 - mirrors `requests` signature
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        idempotency_key: str | None = None,
    ) -> Response:
        method_upper = method.upper()
        canonical_path = self._canonical_path(path)
        request_headers: MutableMapping[str, str] = {}
        if headers:
            request_headers.update(headers)
        if idempotency_key:
            request_headers.setdefault("Idempotency-Key", idempotency_key)

        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            signed_headers = self._auth_headers(method_upper, canonical_path)
            request_headers.update(signed_headers)
            request_headers["Content-Type"] = "application/json"

            url = f"{self._base_url}{canonical_path}"
            try:
                response = self._session.request(
                    method_upper,
                    url,
                    params=params,
                    json=json_body,
                    headers=request_headers,
                    timeout=self._timeout,
                )
            except RequestException as exc:
                last_error = exc
                self._log_failure(method_upper, canonical_path, attempt, exc=exc)
                if attempt >= self._max_retries:
                    break
                self._backoff(attempt)
                continue

            if 200 <= response.status_code < 300:
                self._log_success(
                    method_upper,
                    canonical_path,
                    response.status_code,
                    attempt,
                    idempotency_key,
                )
                return response

            if response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                self._log_failure(
                    method_upper,
                    canonical_path,
                    attempt,
                    status=response.status_code,
                )
                self._backoff(attempt)
                message = (
                    f"Kalshi API returned retryable status {response.status_code} "
                    f"for {canonical_path}"
                )
                last_error = KalshiHttpError(message)
                continue

            self._log_failure(
                method_upper,
                canonical_path,
                attempt,
                status=response.status_code,
            )
            message = self._build_error_message(response)
            raise KalshiHttpError(message)

        raise KalshiHttpError("Failed to execute Kalshi API request") from last_error

    # Internal helpers --------------------------------------------------------------------------

    def build_auth_headers(self, method: str, path: str, *, absolute: bool = False) -> dict[str, str]:
        method_upper = method.upper()
        canonical_path = self._canonical_path(path)
        return self._auth_headers(method_upper, canonical_path, absolute=absolute)

    def _sign(self, method: str, path: str) -> tuple[int, str]:
        now = self._clock.now()
        now_utc = _ensure_utc(now)
        current_ms = int(now_utc.timestamp() * 1000)
        utc_now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        if abs(current_ms - utc_now_ms) > _MAX_CLOCK_SKEW_SECONDS * 1000:
            raise KalshiClockSkewError(
                "Local clock skew exceeds 5 seconds. Sync system clock before trading."
            )
        payload = f"{current_ms}{method}{path}"
        signature_bytes = self._private_key.sign(
            payload.encode("utf-8"),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        signature_b64 = base64.b64encode(signature_bytes).decode("ascii")
        return current_ms, signature_b64

    def _auth_headers(
        self,
        method_upper: str,
        canonical_path: str,
        *,
        absolute: bool = False,
    ) -> dict[str, str]:
        signing_path = canonical_path
        if not absolute and self._base_path_prefix and not canonical_path.startswith(self._base_path_prefix):
            signing_path = f"{self._base_path_prefix}{canonical_path}"
        timestamp_ms, signature = self._sign(method_upper, signing_path)
        return {
            "KALSHI-ACCESS-KEY": self._access_key_id,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
            "KALSHI-ACCESS-SIGNATURE": signature,
        }

    @staticmethod
    def _load_private_key(path: Path) -> RSAPrivateKey:
        expanded = path.expanduser()
        if not expanded.exists():
            raise RuntimeError(f"Kalshi private key path does not exist: {expanded}")
        pem_data = expanded.read_bytes()
        try:
            private_key = load_pem_private_key(
                pem_data,
                password=None,
                backend=default_backend(),
            )
        except ValueError as exc:
            raise RuntimeError("Failed to parse Kalshi private key PEM.") from exc
        if not isinstance(private_key, RSAPrivateKey):
            raise RuntimeError("Kalshi private key must be RSA for RSA-PSS signing.")
        return private_key

    def _canonical_path(self, path: str) -> str:
        if not path:
            raise ValueError("Path must be provided when making Kalshi API requests.")
        if not path.startswith("/"):
            return f"/{path}"
        return path

    def _backoff(self, attempt: int) -> None:
        delay = self._retry_backoff * (2 ** (attempt - 1))
        self._sleep(delay)

    def _log_success(
        self,
        method: str,
        path: str,
        status: int,
        attempt: int,
        idempotency_key: str | None,
    ) -> None:
        LOGGER.info(
            "Kalshi request succeeded",
            extra={
                "kalshi": {
                    "method": method,
                    "path": path,
                    "status": status,
                    "attempt": attempt,
                    "idempotency_key_tail": _mask(idempotency_key),
                }
            },
        )

    def _log_failure(
        self,
        method: str,
        path: str,
        attempt: int,
        *,
        status: int | None = None,
        exc: Exception | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "method": method,
            "path": path,
            "attempt": attempt,
            "status": status,
        }
        if exc is not None:
            payload["error"] = str(exc)
        LOGGER.warning("Kalshi request attempt failed", extra={"kalshi": payload})

    def _build_error_message(self, response: Response) -> str:
        text = response.text[:256]
        request_meta = f"{response.request.method} {response.request.path_url}"
        return f"Kalshi API returned {response.status_code} for {request_meta}: {text}"
