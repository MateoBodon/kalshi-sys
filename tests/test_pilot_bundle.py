from __future__ import annotations

import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from kalshi_alpha.exec import pilot_bundle


def test_pilot_bundle_collects_artifacts(tmp_path: Path, monkeypatch) -> None:
    reports_dir = tmp_path / "reports"
    monitors_dir = reports_dir / "_artifacts" / "monitors"
    ladders_dir = reports_dir / "ladders" / "CPI"
    reports_dir.mkdir(parents=True)
    monitors_dir.mkdir(parents=True)
    ladders_dir.mkdir(parents=True)

    policy_payload = {
        "series": [
            {
                "series": "CPI",
                "recommendation": "GO",
                "fills": 10,
                "mean_delta_bps": 1.2,
                "t_stat": 2.1,
                "guardrail_breaches": 0,
                "drawdown_ok": True,
            }
        ],
        "overall": {"global_reasons": [], "ev_honesty_flags": {}},
        "drawdown": {"ok": True},
        "freshness": {
            "ledger_age_minutes": 10,
            "ledger_threshold_minutes": 120,
            "monitors_age_minutes": 5,
            "monitors_threshold_minutes": 30,
        },
        "monitors_summary": {
            "statuses": {"ev_seq_guard": "OK", "ws_disconnects": "OK", "auth_errors": "OK"}
        },
    }
    reports_dir.joinpath("pilot_ready.json").write_text(json.dumps(policy_payload), encoding="utf-8")
    reports_dir.joinpath("pilot_readiness.md").write_text("# Pilot Readiness", encoding="utf-8")
    reports_dir.joinpath("scoreboard_7d.md").write_text("# Scoreboard", encoding="utf-8")
    monitors_dir.joinpath("ev_gap.json").write_text("{}", encoding="utf-8")
    reports_dir.joinpath("_artifacts", "go_no_go.json").parent.mkdir(parents=True, exist_ok=True)
    reports_dir.joinpath("_artifacts", "go_no_go.json").write_text("{}", encoding="utf-8")
    reports_dir.joinpath("_artifacts", "pilot_session.json").write_text(
        json.dumps(
            {
                "session_id": "pilot-cpi-test",
                "started_at": "2025-11-02T12:00:00Z",
                "n_trades": 3,
                "mean_delta_bps_after_fees": 1.5,
                "cusum_status": "OK",
                "fill_realism_gap": 0.05,
                "alerts_summary": {"recent_alerts": []},
            }
        ),
        encoding="utf-8",
    )
    ladders_dir.joinpath("2025-11-02.md").write_text("# Ladder", encoding="utf-8")

    telemetry_dir = tmp_path / "data" / "raw" / "kalshi" / "2025" / "11" / "02"
    telemetry_dir.mkdir(parents=True)
    telemetry_path = telemetry_dir / "exec.jsonl"
    telemetry_path.write_text(
        json.dumps({"timestamp": datetime.now(tz=UTC).isoformat()}) + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "bundle.tar.gz"
    exit_code = pilot_bundle.main(
        [
            "--reports-dir",
            str(reports_dir),
            "--data-root",
            str(tmp_path / "data"),
            "--output",
            str(output_path),
            "--max-telemetry",
            "1",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()

    with tarfile.open(output_path, "r:gz") as archive:
        names = archive.getnames()
        assert "reports/pilot_ready.json" in names
        assert "reports/_artifacts/monitors/ev_gap.json" in names
        assert "reports/ladders/CPI/2025-11-02.md" in names
        assert "reports/_artifacts/pilot_session.json" in names
        assert "README_pilot.md" in names
        assert "telemetry/2025/11/02/exec.jsonl" in names
        manifest = archive.extractfile("manifest.json")
        assert manifest is not None
        payload = json.loads(manifest.read().decode("utf-8"))
        assert "reports/pilot_ready.json" in payload["files"]
        assert "reports/_artifacts/pilot_session.json" in payload["files"]
        assert "README_pilot.md" in payload["files"]
