"""Utilities for detecting and redacting sensitive strings."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

FORBIDDEN_TOKENS = [
    "API_KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
]

_TOKEN_PATTERN = re.compile(
    r"(?P<key>[A-Za-z0-9_\-]*(?:API_KEY|SECRET|TOKEN|PASSWORD))(?P<sep>\s*[=:]\s*)(?P<value>[^\s,;]+)",
    re.IGNORECASE,
)


def redacted(text: str) -> str:
    def _mask_match(match: re.Match[str]) -> str:
        key = match.group("key")
        sep = match.group("sep")
        safe_key = re.sub(r"API_KEY|SECRET|TOKEN|PASSWORD", "[REDACTED]", key, flags=re.IGNORECASE)
        return f"{safe_key}{sep}[REDACTED]"

    result = _TOKEN_PATTERN.sub(_mask_match, text)
    for token in FORBIDDEN_TOKENS:
        result = re.sub(token, "[REDACTED]", result, flags=re.IGNORECASE)
    return result


def ensure_safe_payload(payload: object) -> None:
    """Raise ValueError if payload contains forbidden tokens."""

    for token in FORBIDDEN_TOKENS:
        if _contains_token(payload, token):
            raise ValueError(f"Forbidden secret token detected: {token}")


def _contains_token(obj: object, token: str) -> bool:
    if isinstance(obj, str):
        lowered = obj.lower()
        if token.lower() in lowered:
            return True
        if _TOKEN_PATTERN.search(obj):
            return True
        return False
    if isinstance(obj, Mapping):
        return any(_contains_token(value, token) for value in obj.values())
    if isinstance(obj, Sequence) and not isinstance(obj, (bytes, bytearray)):
        return any(_contains_token(item, token) for item in obj)
    return False
