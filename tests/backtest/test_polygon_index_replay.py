from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from kalshi_alpha.replay import polygon_index_replay

ET = ZoneInfo("America/New_York")


def _record(ts: datetime, symbol: str, price: float) -> dict[str, object]:
    millis = int(ts.timestamp() * 1000)
    return {
        "ts": ts.astimezone(UTC).isoformat(),
        "msg": {
            "ev": "A",
            "sym": symbol,
            "c": price,
            "s": millis,
        },
    }


def test_polygon_replay_generates_freshness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    start_et = datetime(2025, 11, 4, 11, 45, tzinfo=ET)
    records = [
        _record(start_et, "I:SPX", 5200.0),
        _record(start_et, "I:NDX", 18000.0),
        _record(start_et.replace(minute=50), "I:SPX", 5202.0),
        _record(start_et.replace(minute=50), "I:NDX", 18010.0),
    ]
    replay_file = tmp_path / "replay.json"
    replay_file.write_text(json.dumps(records, indent=2), encoding="utf-8")

    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    freshness_config = config_dir / "freshness.yaml"
    freshness_config.write_text(
        json.dumps(
            {
                "feeds": {
                    "polygon_index.websocket": {
                        "label": "Polygon replay",
                        "age_seconds": 120.0,
                        "required": True,
                        "namespace": "polygon_index",
                    }
                },
                "required_order": ["polygon_index.websocket"],
            }
        ),
        encoding="utf-8",
    )

    data_root = tmp_path / "data"
    freshness_output = tmp_path / "reports" / "_artifacts" / "monitors" / "freshness.json"
    proc_parquet = tmp_path / "reports" / "_artifacts" / "replay" / "proc.parquet"
    summary_path = tmp_path / "reports" / "_artifacts" / "replay" / "summary.json"

    # Eliminate real sleeping during tests
    monkeypatch.setattr(polygon_index_replay.time_module, "sleep", lambda *_args, **_kwargs: None)

    polygon_index_replay.main(
        [
            "--file",
            str(replay_file),
            "--speed",
            "10",
            "--start",
            "11:40",
            "--end",
            "12:05",
            "--freshness-config",
            str(freshness_config),
            "--freshness-output",
            str(freshness_output),
            "--proc-parquet",
            str(proc_parquet),
            "--summary",
            str(summary_path),
            "--data-root",
            str(data_root),
        ]
    )

    assert freshness_output.exists()
    payload = json.loads(freshness_output.read_text(encoding="utf-8"))
    assert "polygon_index.websocket" not in payload.get("metrics", {}).get("stale_feeds", ())

    assert proc_parquet.exists()
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["messages_processed"] == len(records)
