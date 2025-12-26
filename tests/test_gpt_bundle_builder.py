from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from tools.gpt_bundle_builder import BundleBuildError, stage_bundle


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_run_log(root: Path, run_name: str, artifacts_md: str) -> None:
    run_dir = root / "docs" / "agent_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_text(run_dir / "RUN.md", "Run summary.\n")
    _write_text(run_dir / "NOTES.md", "Notes.\n")
    _write_text(run_dir / "COMMANDS.md", "Commands.\n")
    _write_text(run_dir / "TESTS.md", "Tests.\n")
    _write_text(run_dir / "RESULTS.md", "Results.\n")
    _write_text(run_dir / "FILES_TOUCHED.md", "Files.\n")
    _write_text(run_dir / "DIFF.patch", "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@\n+ok\n")
    _write_text(run_dir / "ARTIFACTS.md", artifacts_md)
    _write_text(run_dir / "prompt.md", "prompt text\n")
    meta = {"run_name": run_name, "ticket_id": "TICKET-108", "end_utc": "2025-12-26T00:00:00Z"}
    _write_text(run_dir / "META.json", json.dumps(meta))


def _write_required_files(root: Path) -> None:
    _write_text(root / "AGENTS.md", "agents\n")
    _write_text(root / "docs" / "PLAN_OF_RECORD.md", "plan\n")
    _write_text(root / "docs" / "DOCS_AND_LOGGING_SYSTEM.md", "docs\n")
    _write_text(root / "docs" / "CODEX_SPRINT_TICKETS.md", "tickets\n")
    _write_text(root / "docs" / "PROGRESS.md", "progress\n")
    _write_text(root / "project_state" / "CURRENT_RESULTS.md", "results\n")
    _write_text(root / "project_state" / "KNOWN_ISSUES.md", "issues\n")
    _write_text(root / "project_state" / "CONFIG_REFERENCE.md", "config\n")


def _write_zip_from_staging(zip_path: Path, staging_root: Path, workspace_root: Path) -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in staging_root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(workspace_root))


def test_stage_bundle_includes_fillcalib_and_readiness_artifacts(tmp_path: Path) -> None:
    run_name = "20251226T000000Z_TICKET-108_bundle_artifacts_fix"
    _write_required_files(tmp_path)

    _write_text(tmp_path / "data" / "proc" / "fillcalib" / "curves_SMOKE.json", "{\"ok\": true}\n")
    _write_text(tmp_path / "reports" / "fillcalib" / "SMOKE.md", "# Fillcalib\n")
    _write_text(tmp_path / "reports" / "pilot_ready.json", "{\"ok\": true}\n")
    _write_text(tmp_path / "reports" / "pilot_readiness.md", "# Pilot readiness\n")
    _write_text(tmp_path / "reports" / "calibration" / "SMOKE.md", "# Calibration\n")

    artifacts_md = "\n".join(
        [
            "# ARTIFACTS",
            "- data/proc/fillcalib/curves_SMOKE.json",
            "- reports/fillcalib/SMOKE.md",
            "- reports/pilot_ready.json",
            "- reports/pilot_readiness.md",
            "- reports/calibration/SMOKE.md",
            "",
        ]
    )
    _write_run_log(tmp_path, run_name, artifacts_md)

    staging_root = tmp_path / "docs" / "gpt_bundles" / run_name
    stage_bundle(tmp_path, run_name, staging_root)

    zip_path = tmp_path / "bundle.zip"
    _write_zip_from_staging(zip_path, staging_root, tmp_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert f"docs/gpt_bundles/{run_name}/data/proc/fillcalib/curves_SMOKE.json" in names
        assert f"docs/gpt_bundles/{run_name}/reports/fillcalib/SMOKE.md" in names
        assert f"docs/gpt_bundles/{run_name}/reports/pilot_ready.json" in names
        assert f"docs/gpt_bundles/{run_name}/reports/pilot_readiness.md" in names
        assert f"docs/gpt_bundles/{run_name}/reports/calibration/SMOKE.md" in names


def test_stage_bundle_fails_when_artifact_listed_missing(tmp_path: Path) -> None:
    run_name = "20251226T000001Z_TICKET-108_bundle_artifacts_fix"
    _write_required_files(tmp_path)

    _write_text(tmp_path / "reports" / "other" / "OMIT.md", "# Omit\n")
    artifacts_md = "\n".join(["# ARTIFACTS", "- reports/other/OMIT.md", ""])
    _write_run_log(tmp_path, run_name, artifacts_md)

    staging_root = tmp_path / "docs" / "gpt_bundles" / run_name
    with pytest.raises(BundleBuildError) as excinfo:
        stage_bundle(tmp_path, run_name, staging_root)
    assert "reports/other/OMIT.md" in str(excinfo.value)
