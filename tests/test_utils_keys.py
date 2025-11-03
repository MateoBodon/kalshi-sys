from __future__ import annotations

from kalshi_alpha.utils import keys


def _reset_caches() -> None:
    keys.load_secret.cache_clear()
    keys.load_polygon_api_key.cache_clear()


def test_load_polygon_api_key_keychain_priority(monkeypatch):
    _reset_caches()
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setattr(keys, "_keychain_lookup", lambda label: "keychain-secret")
    assert keys.load_polygon_api_key() == "keychain-secret"


def test_load_polygon_api_key_env_fallback(monkeypatch):
    _reset_caches()
    monkeypatch.setattr(keys, "_keychain_lookup", lambda label: None)
    monkeypatch.setenv("POLYGON_API_KEY", "env-secret")
    assert keys.load_polygon_api_key() == "env-secret"
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    _reset_caches()
