from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest

from kalshi_alpha.exec.pipelines import today


def _daily_stub_factory(
    tmp_path: Path,
    go_flag: bool,
    recorded: list[list[str]],
) -> Callable[[list[str]], None]:
    artifacts_dir = tmp_path / "reports" / "_artifacts"

    def _stub(argv: list[str]) -> None:
        recorded.append(argv)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        go_payload = {"go": go_flag, "reasons": []}
        (artifacts_dir / "go_no_go.json").write_text(json.dumps(go_payload, indent=2), encoding="utf-8")
        manifest_path = tmp_path / "data" / "raw" / "manifest.json"
        (artifacts_dir / "latest_manifest.txt").write_text(
            manifest_path.as_posix(),
            encoding="utf-8",
        )

    return _stub


def test_today_runner_passes_flags_and_prints_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorded: list[list[str]] = []
    stub = _daily_stub_factory(tmp_path, True, recorded)
    monkeypatch.setattr("kalshi_alpha.exec.pipelines.daily.main", stub)
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(today, "_plan_runs", lambda *_, **__: [today.ScheduledRun("pre_cpi", date.today())])

    today.main(
        [
            "--offline",
            "--report",
            "--paper-ledger",
            "--fill-alpha",
            "0.5",
            "--slippage-mode",
            "depth",
            "--impact-cap",
            "0.03",
            "--family",
            "macro",
        ]
    )

    out = capsys.readouterr().out
    assert "GO/NO-GO badge" in out
    assert "Archived manifest" in out
    assert recorded, "daily.main was not invoked"
    forwarded = recorded[0]
    assert "--fill-alpha" in forwarded and "0.5" in forwarded
    assert "--slippage-mode" in forwarded and "depth" in forwarded
    assert "--impact-cap" in forwarded and "0.03" in forwarded


def test_today_runner_exits_on_no_go(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorded: list[list[str]] = []
    stub = _daily_stub_factory(tmp_path, False, recorded)
    monkeypatch.setattr("kalshi_alpha.exec.pipelines.daily.main", stub)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(today, "_plan_runs", lambda *_, **__: [today.ScheduledRun("pre_cpi", date.today())])

    with pytest.raises(SystemExit) as excinfo:
        today.main(["--offline", "--report", "--paper-ledger", "--family", "macro"])

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "GO/NO-GO badge" in out
    assert recorded, "daily.main was not invoked"
