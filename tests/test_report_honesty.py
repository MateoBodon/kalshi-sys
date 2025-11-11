from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

import report.honesty as honesty


def _make_ledger(tmp_path: Path, series: str = "INXU") -> Path:
    now = datetime.now(tz=UTC)
    rows = []
    for idx in range(120):
        prob = 0.4 + 0.001 * idx
        pnl = 0.2 if idx % 2 == 0 else -0.1
        rows.append(
            {
                "series": series,
                "model_p": prob,
                "pnl_simulated": pnl,
                "side": "YES" if idx % 3 else "NO",
                "timestamp_et": now - timedelta(minutes=idx),
            }
        )
    frame = pl.DataFrame(rows)
    path = tmp_path / "ledger.parquet"
    frame.write_parquet(path)
    return path


def test_report_honesty_generates_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger_path = _make_ledger(tmp_path)
    artifact_root = tmp_path / "artifacts"
    clamp_path = artifact_root / "clamp.json"
    monkeypatch.setattr(honesty, "LEDGER_PATH", ledger_path)
    monkeypatch.setattr(honesty, "ARTIFACT_ROOT", artifact_root)
    monkeypatch.setattr(honesty, "CLAMP_ARTIFACT", clamp_path)

    honesty.main(["--window", "1", "--buckets", "5"])

    window_artifact = artifact_root / "honesty_window1.json"
    assert window_artifact.exists()
    payload = json.loads(window_artifact.read_text(encoding="utf-8"))
    assert payload["series"], "series metrics missing"
    inxu_metrics = payload["series"].get("INXU")
    assert inxu_metrics is not None
    assert inxu_metrics["clamp"] in {0.5, 0.75, 1.0}
    assert clamp_path.exists()
