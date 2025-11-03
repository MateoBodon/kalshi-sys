from __future__ import annotations

from pathlib import Path

import scripts.pilot_teny as pilot


def test_build_steps_sequence() -> None:
    commands = pilot.build_steps("PY")
    assert commands[0] == ["PY", "-m", "kalshi_alpha.datastore.ingest", "--all", "--online"]
    assert commands[2][:5] == ["PY", "-m", "kalshi_alpha.dev.imbalance_snap", "--tickers", "TNEY"]
    assert commands[-1][-1] == "kalshi_alpha.exec.reports.ramp"


def test_run_pipeline_invokes_commands(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], Path]] = []

    def fake_run(cmd, check, cwd):  # type: ignore[no-untyped-def]
        calls.append((list(cmd), cwd))

    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    monkeypatch.setattr(pilot.subprocess, "run", fake_run)
    summary = pilot.run_pipeline("PY", root=tmp_path)
    assert summary == tmp_path / "reports/pilot_ready.json"
    assert len(calls) == 6
    assert calls[0][0][-1] == "--online"
