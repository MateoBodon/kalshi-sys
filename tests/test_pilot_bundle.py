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

    reports_dir.joinpath("pilot_ready.json").write_text("{}", encoding="utf-8")
    reports_dir.joinpath("pilot_readiness.md").write_text("# Pilot Readiness", encoding="utf-8")
    reports_dir.joinpath("scoreboard_7d.md").write_text("# Scoreboard", encoding="utf-8")
    monitors_dir.joinpath("ev_gap.json").write_text("{}", encoding="utf-8")
    reports_dir.joinpath("_artifacts", "go_no_go.json").parent.mkdir(parents=True, exist_ok=True)
    reports_dir.joinpath("_artifacts", "go_no_go.json").write_text("{}", encoding="utf-8")
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
        assert "telemetry/2025/11/02/exec.jsonl" in names
        manifest = archive.extractfile("manifest.json")
        assert manifest is not None
        payload = json.loads(manifest.read().decode("utf-8"))
        assert "reports/pilot_ready.json" in payload["files"]
