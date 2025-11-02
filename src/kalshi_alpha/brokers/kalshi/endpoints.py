"""Kalshi environment endpoint helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KalshiEndpoints:
    rest: str
    ws: str


_ENDPOINTS = {
    "prod": KalshiEndpoints(
        rest="https://api.elections.kalshi.com/trade-api/v2",
        ws="wss://api.elections.kalshi.com/trade-api/ws/v2",
    ),
    "demo": KalshiEndpoints(
        rest="https://demo-api.kalshi.co/trade-api/v2",
        ws="wss://demo-api.kalshi.co/trade-api/ws/v2",
    ),
}


def resolve(env: str | None) -> KalshiEndpoints:
    normalized = (env or "prod").strip().lower()
    if normalized not in _ENDPOINTS:
        available = ", ".join(sorted(_ENDPOINTS))
        raise ValueError(f"Unknown Kalshi environment '{env}'. Expected one of: {available}.")
    return _ENDPOINTS[normalized]
