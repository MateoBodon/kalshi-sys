from __future__ import annotations

import pytest

from kalshi_alpha.utils.secrets import ensure_safe_payload, redacted


def test_ensure_safe_payload_detects_secrets() -> None:
    with pytest.raises(ValueError):
        ensure_safe_payload({"message": "API_KEY=123"})
    with pytest.raises(ValueError):
        ensure_safe_payload({"message": "service_api_key=abc"})
    ensure_safe_payload({"message": "safe"})


def test_redacted_replaces_token() -> None:
    masked = redacted("service_api_key=supersecret")
    assert "[REDACTED]" in masked
    assert "supersecret" not in masked
    assert "api_key" not in masked.lower()
