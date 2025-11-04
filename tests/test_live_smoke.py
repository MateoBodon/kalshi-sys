from __future__ import annotations

from datetime import datetime

import pytest

from kalshi_alpha.exec import live_smoke


class StubOutstanding:
    def __init__(self, total: int) -> None:
        self._total = total

    def summary(self) -> dict[str, int]:
        return {"dry": self._total}

    def total(self) -> int:
        return self._total


class StubResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_live_smoke_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    target_label = "H1300"

    class StubClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        def get(self, path: str) -> StubResponse:
            if path == "/series":
                payload = {
                    "series": [
                        {"ticker": "INXU", "id": "INXU_SERIES"},
                        {"ticker": "NASDAQ100U", "id": "NASDAQ100U_SERIES"},
                    ]
                }
                return StubResponse(payload)
            if path == "/series/INXU_SERIES/events":
                payload = {"events": [{"ticker": f"KXINXU-25NOV03{target_label}"}]}
                return StubResponse(payload)
            if path == "/series/NASDAQ100U_SERIES/events":
                payload = {"events": [{"ticker": f"KXNASDAQ100U-25NOV03{target_label}"}]}
                return StubResponse(payload)
            raise AssertionError(f"unexpected path {path}")

    monkeypatch.setattr(live_smoke, "KalshiHttpClient", StubClient)
    monkeypatch.setattr(live_smoke, "_now_et", lambda: datetime(2025, 11, 3, 12, 55, tzinfo=live_smoke.ET))
    monkeypatch.setattr(
        live_smoke.OutstandingOrdersState,
        "load",
        classmethod(lambda cls: StubOutstanding(total=0)),
    )

    exit_code, payload = live_smoke.run_smoke(base_url="https://example.invalid")
    assert exit_code == 0
    reasons = payload["reasons"]
    assert not reasons
    assert all(entry["has_target"] for entry in payload["series"])


def test_live_smoke_missing_events(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        def get(self, path: str) -> StubResponse:
            if path == "/series":
                payload = {
                    "series": [
                        {"ticker": "INXU", "id": "INXU_SERIES"},
                        {"ticker": "NASDAQ100U", "id": "NASDAQ100U_SERIES"},
                    ]
                }
                return StubResponse(payload)
            # Return events that do not match the target label
            payload = {"events": [{"ticker": "KXINXU-25NOV03H1200"}]}
            return StubResponse(payload)

    monkeypatch.setattr(live_smoke, "KalshiHttpClient", StubClient)
    monkeypatch.setattr(live_smoke, "_now_et", lambda: datetime(2025, 11, 3, 12, 55, tzinfo=live_smoke.ET))
    monkeypatch.setattr(
        live_smoke.OutstandingOrdersState,
        "load",
        classmethod(lambda cls: StubOutstanding(total=2)),
    )

    exit_code, payload = live_smoke.run_smoke(base_url="https://example.invalid")
    assert exit_code == 1
    reasons = payload["reasons"]
    assert any("missing" in reason for reason in reasons)
    assert any("outstanding" in reason for reason in reasons)
