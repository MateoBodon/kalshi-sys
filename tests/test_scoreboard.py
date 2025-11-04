from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from kalshi_alpha.exec import scoreboard


def _write_calibration(root: Path, slug: str, horizon: str, generated_at: datetime) -> None:
    path = root / slug / horizon / "params.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": generated_at.isoformat(),
        "minutes_to_target": {"0": {"sigma": 1.0, "drift": 0.0}},
        "residual_std": 0.5,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_freshness(path: Path, now: datetime) -> None:
    payload = {
        "name": "data_freshness",
        "status": "OK",
        "generated_at": now.isoformat(),
        "metrics": {
            "required_feeds_ok": True,
            "required_feeds": ["polygon.index_snapshot"],
            "stale_feeds": [],
            "feeds": [
                {
                    "id": "polygon.index_snapshot",
                    "label": "Polygon Snapshots",
                    "required": True,
                    "ok": True,
                    "age_minutes": 5.0,
                    "last_ts": now.isoformat(),
                    "reason": None,
                }
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_scoreboard_generates_markdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    ledger_dir = tmp_path / "data" / "proc"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=UTC)
    ledger = pl.DataFrame(
        {
            "series": ["INX", "INXU", "NASDAQ100", "NASDAQ100U"],
            "ev_after_fees": [15.2, 9.8, 11.4, 12.1],
            "pnl_simulated": [14.0, 8.5, 10.2, 11.0],
            "expected_fills": [320, 280, 340, 360],
            "size": [350, 300, 360, 380],
            "fill_ratio": [0.65, 0.58, 0.6, 0.63],
            "fill_ratio_observed": [0.6, 0.55, 0.58, 0.6],
            "alpha_target": [0.62, 0.57, 0.59, 0.61],
            "delta_p": [0.04, -0.02, 0.03, -0.01],
            "depth_fraction": [0.55, 0.5, 0.6, 0.52],
            "spread": [0.02, 0.025, 0.03, 0.028],
            "minutes_to_event": [30.0, 25.0, 40.0, 35.0],
            "t_fill_ms": [120.0, 140.0, 130.0, 125.0],
            "size_partial": [5, 6, 4, 5],
            "slippage_ticks": [0.5, 0.6, 0.4, 0.5],
            "ev_expected_bps": [120.0, 98.0, 110.0, 115.0],
            "ev_realized_bps": [130.0, 105.0, 118.0, 122.0],
            "fees_bps": [3.5, 3.2, 3.8, 3.6],
            "timestamp_et": [now - timedelta(days=2)] * 4,
        }
    )
    ledger.write_parquet(ledger_dir / "ledger_all.parquet")

    calib_root = tmp_path / "data" / "proc" / "calib" / "index"
    for slug, horizon in (("spx", "noon"), ("spx", "close"), ("ndx", "noon"), ("ndx", "close")):
        _write_calibration(calib_root, slug, horizon, now - timedelta(days=3))

    calib = pl.DataFrame(
        {
            "series": ["INX", "INXU", "NASDAQ100", "NASDAQ100U"],
            "crps_advantage": [0.12, 0.10, 0.08, 0.09],
            "brier_advantage": [0.05, 0.04, 0.03, 0.04],
        }
    )
    calib.write_parquet(ledger_dir / "calibration_metrics.parquet")

    state_dir = ledger_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_dir.joinpath("fill_alpha.json").write_text(
        json.dumps(
            {
                "series": {
                    "INX": {"alpha": 0.65, "ts": now.isoformat()},
                    "INXU": {"alpha": 0.58, "ts": now.isoformat()},
                    "NASDAQ100": {"alpha": 0.6, "ts": now.isoformat()},
                    "NASDAQ100U": {"alpha": 0.63, "ts": now.isoformat()},
                }
            }
        ),
        encoding="utf-8",
    )

    artifacts = tmp_path / "reports" / "_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    _write_freshness(artifacts / "monitors" / "freshness.json", now)
    go_payload = {
        "go": False,
        "series": "NASDAQ100",
        "timestamp": (now - timedelta(days=1)).isoformat(),
    }
    artifacts.joinpath("go_no_go_20250101.json").write_text(
        json.dumps(go_payload),
        encoding="utf-8",
    )

    scoreboard.main([])

    report_7d = tmp_path / "reports" / "scoreboard_7d.md"
    report_30d = tmp_path / "reports" / "scoreboard_30d.md"
    assert report_7d.exists()
    assert report_30d.exists()

    contents = report_7d.read_text(encoding="utf-8")
    assert "## INX" in contents
    assert "CRPS Advantage" in contents
    assert "Avg α" in contents
    assert "Avg Slippage (ticks)" in contents
    assert "Fill - α" in contents
    assert "Fill - model α" in contents
    assert "Slippage Δ (ticks)" in contents
    assert "NO-GO Count" in contents
    assert "Sample Size" in contents
    assert "Confidence" in contents

    pilot_report = tmp_path / "reports" / "pilot_readiness.md"
    assert pilot_report.exists()
    pilot_text = pilot_report.read_text(encoding="utf-8")
    assert pilot_text.startswith("# Pilot Readiness")
    assert "INXU —" in pilot_text
    assert "Reasons" in pilot_text
