from __future__ import annotations

from pathlib import Path

import scripts.pilot_teny as pilot


def test_build_steps_sequence() -> None:
    commands = pilot.build_steps("PY", ticker="TNEY-TEST")
    assert commands[0] == ["PY", "-m", "kalshi_alpha.datastore.ingest", "--all", "--online"]
    assert commands[1] == ["PY", "-m", "kalshi_alpha.drivers.macro_calendar.cli", "--days", "30"]
    assert commands[2][:5] == ["PY", "-m", "kalshi_alpha.dev.ws_smoke", "--tickers", "TNEY-TEST"]
    assert commands[-1][:3] == ["PY", "-m", "kalshi_alpha.exec.scoreboard"]


def test_run_pipeline_invokes_commands(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], Path]] = []

    def fake_run(cmd, check, cwd):  # type: ignore[no-untyped-def]
        calls.append((list(cmd), cwd))
        reports_dir = cwd / "reports"
        if cmd[2] == "kalshi_alpha.exec.pipelines.daily":
            report_dir = reports_dir / "TNEY"
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "2025-11-03.md").write_text("# report", encoding="utf-8")
            artifact_dir = reports_dir / "_artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "go_no_go.json").write_text("{}", encoding="utf-8")
        if cmd[2] == "kalshi_alpha.exec.scoreboard":
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / "scoreboard_7d.md").write_text("", encoding="utf-8")
            (reports_dir / "scoreboard_30d.md").write_text("", encoding="utf-8")
            (reports_dir / "pilot_readiness.md").write_text("", encoding="utf-8")

    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    monkeypatch.setattr(pilot, "resolve_tney_ticker", lambda now=None: "TNEY-TEST")
    monkeypatch.setattr(pilot.subprocess, "run", fake_run)
    latest_report = pilot.run_pipeline("PY", root=tmp_path)
    assert latest_report == tmp_path / "reports" / "TNEY" / "2025-11-03.md"
    assert len(calls) == 5
    assert calls[0][0][-1] == "--online"
