from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kalshi_alpha.exec.runners import micro_index


def test_micro_runner_invokes_scan_and_refits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_scan(args: list[str]) -> None:
        captured["scan_args"] = args

    monkeypatch.setattr(micro_index.scan_ladders, "main", fake_scan)

    def fake_tune(series: str, archives_dir: Path) -> float:
        captured["tune_series"] = series
        captured["tune_archives"] = archives_dir
        return 0.55

    monkeypatch.setattr(micro_index, "tune_alpha", fake_tune)

    def fake_fit(series: str, **kwargs) -> SimpleNamespace:
        captured["slippage_series"] = series
        return SimpleNamespace(impact_cap=0.01)

    monkeypatch.setattr(micro_index, "fit_slippage", fake_fit)

    scoreboard_calls: list[object] = []
    monkeypatch.setattr(micro_index.scoreboard, "main", lambda argv=None: scoreboard_calls.append(argv))

    fixtures_root = tmp_path / "fixtures"
    fixtures_root.mkdir(parents=True, exist_ok=True)
    args = [
        "--series",
        "INXU",
        "--offline",
        "--fixtures-root",
        str(fixtures_root),
        "--min-ev",
        "0.06",
        "--contracts",
        "1",
        "--regenerate-scoreboard",
        "--now",
        "2025-11-04T15:20:00+00:00",
    ]
    micro_index.main(args)

    scan_args = captured["scan_args"]
    assert scan_args[:2] == ["--series", "INXU"]
    assert "--contracts" in scan_args
    assert captured["tune_series"] == "INXU"
    assert "kalshi" in str(captured["tune_archives"]).lower()
    assert captured["slippage_series"] == "INXU"
    assert scoreboard_calls, "scoreboard should be regenerated"


def test_micro_runner_passes_quality_gate_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_scan(args: list[str]) -> None:
        captured["scan_args"] = args

    monkeypatch.setattr(micro_index.scan_ladders, "main", fake_scan)
    monkeypatch.setattr(micro_index, "tune_alpha", lambda *a, **k: None)
    monkeypatch.setattr(micro_index, "fit_slippage", lambda *a, **k: None)
    monkeypatch.setattr(micro_index, "_refit_execution_curves", lambda *a, **k: None)
    monkeypatch.setattr(micro_index.scoreboard, "main", lambda argv=None: None)

    fixtures_root = tmp_path / "fixtures"
    fixtures_root.mkdir(parents=True, exist_ok=True)
    quality_path = tmp_path / "configs" / "quality.yaml"
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text("{}", encoding="utf-8")

    args = [
        "--series",
        "NASDAQ100U",
        "--offline",
        "--fixtures-root",
        str(fixtures_root),
        "--quality-gates-config",
        str(quality_path),
    ]
    micro_index.main(args)

    scan_args = captured["scan_args"]
    assert "--quality-gates-config" in scan_args
    idx = scan_args.index("--quality-gates-config")
    assert scan_args[idx + 1] == str(quality_path)
