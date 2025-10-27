from __future__ import annotations

import pytest

from kalshi_alpha.utils.secrets import ensure_safe_payload, redacted


def test_ensure_safe_payload_detects_secrets() -> None:
    with pytest.raises(ValueError):
        ensure_safe_payload({"message": "API_KEY=123"})
    ensure_safe_payload({"message": "safe"})


def test_redacted_replaces_token() -> None:
    assert "[REDACTED]" in redacted("API_KEY=123")
