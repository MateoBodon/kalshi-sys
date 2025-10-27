from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kalshi_alpha.exec.pipelines import daily


def test_pipeline_offline_pre_cpi(
    tmp_path: Path,
    fixtures_root: Path,
    offline_fixtures_root: Path,
    isolated_data_roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, proc_root = isolated_data_roots
    project_root = Path(__file__).resolve().parents[1]
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    example_policy = project_root / "configs" / "pal_policy.example.yaml"
    configs_dir.joinpath("pal_policy.example.yaml").write_text(
        example_policy.read_text(encoding="utf-8"), encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(daily, "PROC_ROOT", proc_root)

    monkeypatch.setattr(
        daily,
        "evaluate_window",
        lambda mode, now: {"scan_allowed": True, "freeze_active": True},
    )

    args = [
        "--mode",
        "pre_cpi",
        "--offline",
        "--driver-fixtures",
        str(offline_fixtures_root),
        "--scanner-fixtures",
        str(fixtures_root),
        "--report",
        "--paper-ledger",
    ]

    daily.main(args)

    log_files = list((proc_root / "logs").rglob("*.json"))
    assert log_files, "Expected orchestration log"

    reports_dir = tmp_path / "reports"
    assert any(reports_dir.rglob("*.md")), "Expected markdown report"
