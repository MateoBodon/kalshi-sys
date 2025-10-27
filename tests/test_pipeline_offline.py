from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kalshi_alpha.exec.pipelines import daily
from kalshi_alpha.exec.pipelines.calendar import RunWindow


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

    configs_dir.joinpath("quality_gates.yaml").write_text(
        """
metrics:
  series:
    cpi:
      crps_advantage_min: -5000
    claims:
      crps_advantage_min: -5000
      brier_advantage_min: -5000
data_freshness:
  - name: cleveland
    namespace: cleveland_nowcast/monthly
    timestamp_field: as_of
    max_age_hours: 240
    require_et: false
  - name: treasury
    namespace: treasury_yields/daily
    timestamp_field: as_of
    max_age_hours: 240
    require_et: false
reconciliation:
  - name: t10_vs_dgs
    namespace: treasury_yields/daily
    par_maturity: "10 YR"
    dgs_maturity: "DGS10"
    tolerance_bps: 50
monitors:
  tz_not_et: 1
  non_monotone_ladders: 10
  negative_ev_after_fees: 5
""",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(daily, "PROC_ROOT", proc_root)

    def _always_open_window(*, mode: str, target_date: datetime.date, now: datetime, proc_root: Path):
        return RunWindow(
            mode=mode,
            freeze_start=now - timedelta(hours=2),
            scan_open=now - timedelta(hours=1),
            scan_close=now + timedelta(hours=1),
            reference=now,
            notes=[],
        )

    monkeypatch.setattr(daily, "resolve_run_window", _always_open_window)

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

    go_path = reports_dir / "_artifacts" / "go_no_go.json"
    assert go_path.exists(), "Expected go/no-go artifact"
    payload = json.loads(go_path.read_text(encoding="utf-8"))
    assert payload.get("go") is True
