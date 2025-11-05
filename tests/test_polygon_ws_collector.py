from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

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

    monkeypatch.setattr(polygon_ws, "write_snapshot", fake_write_snapshot)
    monkeypatch.setattr(polygon_ws.freshness, "write_freshness_artifact", fake_freshness_artifact)

    proc_parquet = tmp_path / "snapshot.parquet"
    cfg_path = tmp_path / "freshness.yaml"
    output_path = tmp_path / "freshness.json"
    now = datetime.now(tz=UTC)

    entries = [
        {
            "ev": "A",
            "sym": "I:SPX",
            "c": 5025.0,
            "s": 1_700_000_000_000,
        }
    ]
    polygon_ws._process_entries(  # type: ignore[attr-defined]
        entries=entries,
        alias_map={"I:SPX": ("INX", "INXU")},
        channel_prefix="A",
        now=now,
        proc_parquet=proc_parquet,
        freshness_config=cfg_path,
        freshness_output=output_path,
    )

    assert len(captured) == 2
    assert proc_parquet.exists()
    frame = pl.read_parquet(proc_parquet)
    assert frame.height == 1
    assert freshness_calls.get("config") == cfg_path
    assert freshness_calls.get("output") == output_path


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
