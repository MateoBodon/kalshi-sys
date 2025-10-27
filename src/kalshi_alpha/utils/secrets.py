"""Utilities for detecting and redacting sensitive strings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

FORBIDDEN_TOKENS = [
    "API_KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
]


def redacted(text: str) -> str:
    result = text
    for token in FORBIDDEN_TOKENS:
        if token in result:
            result = result.replace(token, "[REDACTED]")
    return result


def ensure_safe_payload(payload: object) -> None:
    """Raise ValueError if payload contains forbidden tokens."""

    for token in FORBIDDEN_TOKENS:
        if _contains_token(payload, token):
            raise ValueError(f"Forbidden secret token detected: {token}")


def _contains_token(obj: object, token: str) -> bool:
    if isinstance(obj, str):
        return token in obj
    if isinstance(obj, Mapping):
        return any(_contains_token(value, token) for value in obj.values())
    if isinstance(obj, Sequence) and not isinstance(obj, (bytes, bytearray)):
        return any(_contains_token(item, token) for item in obj)
    return False
