from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kalshi_alpha.dev import sanity_check


class _StubResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200

    def json(self) -> dict[str, Any]:
        return self._payload


class _StubClient:
    def __init__(self, *, base_url: str) -> None:
        self.base_url = base_url
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def get(self, path: str, params: dict[str, Any] | None = None) -> _StubResponse:
        self.calls.append((path, params))
        if path == "/portfolio/balance":
            return _StubResponse({"balance": 100, "available": 80})
        if path == "/markets":
            return _StubResponse({"markets": [{"id": "M1"}]})
        raise AssertionError(f"Unexpected path {path}")


def test_run_live_smoke_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_load_env() -> None:
        return None

    def stub_client_factory(*, base_url: str) -> _StubClient:
        client = _StubClient(base_url=base_url)
        captured["client"] = client
        return client

    monkeypatch.setenv("KALSHI_ENV", "demo")
    monkeypatch.setenv("KALSHI_API_KEY_ID", "stub")
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PEM_PATH", str(tmp_path / "key.pem"))
    monkeypatch.setattr(sanity_check, "load_env", fake_load_env)
    monkeypatch.setattr(sanity_check, "KalshiHttpClient", stub_client_factory)

    assert sanity_check._run_live_smoke(env_override=None) == 0
    client = captured["client"]
    assert client.base_url == "https://demo-api.kalshi.co/trade-api/v2"
    assert client.calls == [
        ("/portfolio/balance", None),
        ("/markets", {"limit": 25}),
    ]


def test_run_live_smoke_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingClient:
        def __init__(self, **_: object) -> None:
            pass

        def get(self, *_: object, **__: object) -> _StubResponse:
            raise sanity_check.KalshiHttpError("boom")

    monkeypatch.setattr(sanity_check, "load_env", lambda: None)
    monkeypatch.setattr(sanity_check, "KalshiHttpClient", _FailingClient)

    assert sanity_check._run_live_smoke(env_override="prod") == 1
