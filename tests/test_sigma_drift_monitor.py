from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from monitor import drift_sigma_tod


def _write_params(path: Path) -> None:
    payload = {
        "minutes_to_target": {
            "0": {"sigma": 0.0},
            "1": {"sigma": 1.0},
            "2": {"sigma": 1.2},
            "3": {"sigma": 1.1},
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_polygon(path: Path) -> None:
    timestamps = [
        datetime(2025, 11, 10, 14, 0, tzinfo=UTC),
        datetime(2025, 11, 10, 14, 1, tzinfo=UTC),
        datetime(2025, 11, 10, 14, 2, tzinfo=UTC),
        datetime(2025, 11, 10, 14, 3, tzinfo=UTC),
    ]
    closes = [5000.0, 5025.0, 4980.0, 5055.0]
    frame = pl.DataFrame({"timestamp": timestamps, "close": closes})
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def test_sigma_drift_generates_artifact(tmp_path: Path) -> None:
    sigma_root = tmp_path / "data" / "proc" / "calib" / "index"
    raw_root = tmp_path / "data" / "raw" / "polygon"
    _write_params(sigma_root / "spx" / "hourly" / "params.json")
    today = datetime.now(tz=UTC).date().isoformat()
    _write_polygon(raw_root / "I_SPX" / f"{today}.parquet")
    artifact = tmp_path / "sigma_drift.json"
    args = [
        "--lookback-days",
        "1",
        "--threshold",
        "0.1",
        "--series",
        "INXU",
        "--sigma-root",
        str(sigma_root),
        "--raw-root",
        str(raw_root),
        "--artifact",
        str(artifact),
    ]
    rc = drift_sigma_tod.main(args)
    assert artifact.exists()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    entry = payload["series"]["INXU"]
    assert entry["status"] in {"ALERT", "OK"}
    assert entry["forecast_sigma"] is not None
    assert entry["realized_sigma"] is not None
    if entry["status"] == "ALERT":
        assert entry["shrink"] < 1.0
        assert rc == 1
    else:
        assert entry["shrink"] == 1.0
