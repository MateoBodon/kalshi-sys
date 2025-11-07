from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.exec.collectors import polygon_ws


def test_process_entries_writes_snapshots_and_freshness(monkeypatch, tmp_path: Path) -> None:
    captured = []

    def fake_write_snapshot(snapshot: object) -> None:
        captured.append(snapshot)

    freshness_calls: dict[str, object] = {}

    def fake_freshness_artifact(*, config_path: Path, output_path: Path, now: datetime) -> None:
        freshness_calls["config"] = config_path
        freshness_calls["output"] = output_path
        freshness_calls["now"] = now
        output_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(polygon_ws, "write_snapshot", fake_write_snapshot)
    monkeypatch.setattr(polygon_ws.freshness, "write_freshness_artifact", fake_freshness_artifact)

    proc_parquet = tmp_path / "snapshot.parquet"
    cfg_path = tmp_path / "freshness.yaml"
    output_path = tmp_path / "freshness.json"
    now = datetime.now(tz=UTC)

    payload = {
        "ev": "AM",
        "sym": "I:SPX",
        "c": 5025.0,
        "s": 1_700_000_000_000,
        "e": 1_700_000_060_000,
    }
    tracker = polygon_ws.CadenceTracker(log_threshold_seconds=60.0)
    polygon_ws._process_entries(  # type: ignore[attr-defined]
        entries=polygon_ws._normalize_entries(payload),
        alias_map={"I:SPX": ("INX", "INXU")},
        channel_prefix="AM",
        now=now,
        proc_parquet=proc_parquet,
        freshness_config=cfg_path,
        freshness_output=output_path,
        tracker=tracker,
    )

    assert len(captured) == 2
    assert proc_parquet.exists()
    frame = pl.read_parquet(proc_parquet)
    assert frame.height == 1
    assert freshness_calls.get("config") == cfg_path
    assert freshness_calls.get("output") == output_path
    assert output_path.exists()


def test_process_entries_updates_default_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[object] = []

    def fake_write_snapshot(snapshot: object) -> None:
        captured.append(snapshot)

    def fake_freshness_artifact(*, config_path: Path, output_path: Path, now: datetime) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(polygon_ws, "write_snapshot", fake_write_snapshot)
    monkeypatch.setattr(polygon_ws.freshness, "write_freshness_artifact", fake_freshness_artifact)
    monkeypatch.chdir(tmp_path)

    now = datetime.now(tz=UTC)
    proc_parquet = Path("data/proc/polygon_index/snapshot.parquet")
    freshness_output = Path("reports/_artifacts/monitors/freshness.json")

    payload = {
        "ev": "AM",
        "sym": "I:SPX",
        "c": 5000.0,
        "s": 1_700_000_123_000,
    }
    tracker = polygon_ws.CadenceTracker(log_threshold_seconds=60.0)
    polygon_ws._process_entries(  # type: ignore[attr-defined]
        entries=polygon_ws._normalize_entries(payload),
        alias_map={"I:SPX": ("INX",)},
        channel_prefix="AM",
        now=now,
        proc_parquet=proc_parquet,
        freshness_config=Path("configs/freshness.yaml"),
        freshness_output=freshness_output,
        tracker=tracker,
    )

    assert captured  # snapshot written
    assert proc_parquet.exists()
    assert freshness_output.exists()


def test_parse_args_uses_alias_overrides(monkeypatch) -> None:
    monkeypatch.setattr(polygon_ws, "load_polygon_api_key", lambda: "dummy-key")
    config = polygon_ws._parse_args(  # type: ignore[attr-defined]
        [
            "--symbols",
            "I:SPX,I:NDX",
            "--aliases",
            "I:SPX:INXU",
            "--aliases",
            "I:SPX:INX",
        ]
    )
    assert config.alias_map["I:SPX"] == ("INX", "INXU")
    assert config.ws_url.startswith("wss://")


class _StubWebsocket:
    def __init__(self, messages: list[str]) -> None:
        self._messages = iter(messages)

    async def recv(self) -> str:
        return next(self._messages)


@pytest.mark.asyncio
async def test_await_status_ignores_connected_and_returns_on_auth() -> None:
    ws = _StubWebsocket(
        [
            json.dumps([{"ev": "status", "status": "connected"}]),
            json.dumps([{"ev": "status", "status": "auth_success"}]),
        ]
    )
    buffer = await polygon_ws._await_status(ws, expected={"auth_success"})  # type: ignore[attr-defined]
    assert buffer == []


@pytest.mark.asyncio
async def test_await_status_raises_on_max_connections() -> None:
    ws = _StubWebsocket(
        [
            json.dumps(
                [
                    {
                        "ev": "status",
                        "status": "max_connections",
                        "message": "Maximum number of websocket connections exceeded.",
                    }
                ]
            )
        ]
    )
    with pytest.raises(polygon_ws.TooManyConnectionsError):  # type: ignore[attr-defined]
        await polygon_ws._await_status(ws, expected={"auth_success"})  # type: ignore[attr-defined]
