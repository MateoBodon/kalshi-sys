from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kalshi_alpha.dev import imbalance_snap


@pytest.mark.asyncio
async def _fake_stream(*args: Any, **kwargs: Any) -> dict[str, float]:  # pragma: no cover - replaced at runtime
    return {}


def test_imbalance_cli_invokes_stream(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    async def fake_stream(tickers, **kwargs):
        captured["tickers"] = tickers
        captured["kwargs"] = kwargs
        return {ticker: 0.42 for ticker in tickers}

    class FakeClient:
        def __init__(self, **_: Any) -> None:  # pragma: no cover - simple stub
            return

    monkeypatch.setattr(imbalance_snap, "KalshiWebsocketClient", FakeClient)
    monkeypatch.setattr(imbalance_snap.kalshi_ws, "stream_orderbook_imbalance", fake_stream)

    raw_root = tmp_path / "raw"
    proc_root = tmp_path / "proc"
    result = imbalance_snap.main(
        [
            "--tickers",
            "TNEY",
            "--duration-seconds",
            "0",
            "--raw-root",
            str(raw_root),
            "--proc-root",
            str(proc_root),
            "--quiet",
        ]
    )

    assert result == {"TNEY": 0.42}
    assert captured["tickers"] == ["TNEY"]
    assert raw_root.exists()
    assert proc_root.exists()
