from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
import json

import polars as pl

from kalshi_alpha.exec import scoreboard


def test_scoreboard_generates_markdown(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    ledger_dir = tmp_path / "data" / "proc"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=UTC)
    ledger = pl.DataFrame(
        {
            "series": ["CPI", "CLAIMS"],
            "ev_after_fees": [12.3, -5.4],
            "pnl_simulated": [11.0, -6.0],
            "expected_fills": [80, 40],
            "size": [100, 60],
            "fill_ratio": [0.8, 0.6],
            "timestamp_et": [now - timedelta(days=2), now - timedelta(days=6)],
        }
    )
    ledger.write_parquet(ledger_dir / "ledger_all.parquet")

    calib = pl.DataFrame(
        {
            "series": ["CPI", "CLAIMS"],
            "crps_advantage": [0.12, -0.03],
            "brier_advantage": [0.05, -0.01],
        }
    )
    calib.write_parquet(ledger_dir / "calibration_metrics.parquet")

    state_dir = ledger_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_dir.joinpath("fill_alpha.json").write_text(
        json.dumps(
            {
                "series": {
                    "CPI": {"alpha": 0.62, "ts": now.isoformat()},
                    "CLAIMS": {"alpha": 0.48, "ts": now.isoformat()},
                }
            }
        ),
        encoding="utf-8",
    )

    artifacts = tmp_path / "reports" / "_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    go_payload = {
        "go": False,
        "series": "CLAIMS",
        "timestamp": (now - timedelta(days=1)).isoformat(),
    }
    artifacts.joinpath("go_no_go_20250101.json").write_text(json.dumps(go_payload), encoding="utf-8")

    scoreboard.main([])

    report_7d = tmp_path / "reports" / "scoreboard_7d.md"
    report_30d = tmp_path / "reports" / "scoreboard_30d.md"
    assert report_7d.exists()
    assert report_30d.exists()

    contents = report_7d.read_text(encoding="utf-8")
    assert "## CPI" in contents
    assert "CRPS Advantage" in contents
    assert "Avg α" in contents
    assert "NO-GO Count" in contents
