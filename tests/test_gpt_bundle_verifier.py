from __future__ import annotations

import json
import zipfile
from pathlib import Path

from tools.verify_gpt_bundle import verify_bundle


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def _base_files(run_name: str) -> tuple[dict[str, str], str, str]:
    root_prefix = f"docs/gpt_bundles/{run_name}"
    run_prefix = f"{root_prefix}/docs/agent_runs/{run_name}"
    diff_text = (
        "diff --git a/foo.txt b/foo.txt\n"
        "--- a/foo.txt\n"
        "+++ b/foo.txt\n"
        "@@\n"
        "+hello\n"
    )
    meta = {
        "run_name": run_name,
        "ticket_id": "TICKET-007_bundle_diff_hygiene",
        "end_utc": "2025-12-21T20:43:43Z",
    }
    files = {
        f"{root_prefix}/DIFF.patch": diff_text,
        f"{run_prefix}/README.md": "Goal: verify bundle hygiene.",
        f"{run_prefix}/RESULTS.md": "Bundle path: docs/gpt_bundles/sample.zip",
        f"{run_prefix}/META.json": json.dumps(meta),
        f"{run_prefix}/prompt.md": "prompt text",
        f"{run_prefix}/commands.log": "echo hi",
        f"{run_prefix}/artifacts.json": "[]",
        f"{run_prefix}/diff.patch": diff_text,
    }
    return files, root_prefix, run_prefix


def test_verify_bundle_happy_path(tmp_path: Path) -> None:
    run_name = "20251221_000000Z_TICKET-007_bundle_diff_hygiene"
    files, _, _ = _base_files(run_name)
    zip_path = tmp_path / "bundle.zip"
    _write_zip(zip_path, files)
    errors = verify_bundle(zip_path)
    assert errors == []


def test_verify_bundle_missing_required_files(tmp_path: Path) -> None:
    run_name = "20251221_000001Z_TICKET-007_bundle_diff_hygiene"
    files, _, run_prefix = _base_files(run_name)
    files.pop(f"{run_prefix}/RESULTS.md")
    zip_path = tmp_path / "bundle.zip"
    _write_zip(zip_path, files)
    errors = verify_bundle(zip_path)
    assert any("RESULTS.md" in error for error in errors)


def test_verify_bundle_rejects_placeholder_diff(tmp_path: Path) -> None:
    run_name = "20251221_000002Z_TICKET-007_bundle_diff_hygiene"
    files, _, run_prefix = _base_files(run_name)
    files[f"{run_prefix}/diff.patch"] = "...\n"
    zip_path = tmp_path / "bundle.zip"
    _write_zip(zip_path, files)
    errors = verify_bundle(zip_path)
    assert any("placeholder" in error or "missing patch hunks" in error for error in errors)


def _base_files_new(run_name: str, artifacts_content: str) -> tuple[dict[str, str], str, str]:
    root_prefix = f"docs/gpt_bundles/{run_name}"
    run_prefix = f"{root_prefix}/docs/agent_runs/{run_name}"
    diff_text = (
        "diff --git a/foo.txt b/foo.txt\n"
        "--- a/foo.txt\n"
        "+++ b/foo.txt\n"
        "@@\n"
        "+hello\n"
    )
    meta = {
        "run_name": run_name,
        "ticket": "TICKET-108",
        "end_utc": "2025-12-26T19:49:30Z",
    }
    files = {
        f"{root_prefix}/DIFF.patch": diff_text,
        f"{run_prefix}/RUN.md": "Goal: verify bundle completeness.",
        f"{run_prefix}/NOTES.md": "Notes.",
        f"{run_prefix}/COMMANDS.md": "Commands.",
        f"{run_prefix}/TESTS.md": "Tests.",
        f"{run_prefix}/RESULTS.md": "Results.",
        f"{run_prefix}/META.json": json.dumps(meta),
        f"{run_prefix}/ARTIFACTS.md": artifacts_content,
        f"{run_prefix}/FILES_TOUCHED.md": "Files.",
        f"{run_prefix}/DIFF.patch": diff_text,
        f"{run_prefix}/prompt.md": "prompt text",
    }
    return files, root_prefix, run_prefix


def test_verify_bundle_includes_fillcalib_and_readiness_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    run_name = "20251226T000000Z_TICKET-108_bundle_artifacts_fix"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/proc/fillcalib").mkdir(parents=True)
    (tmp_path / "reports/fillcalib").mkdir(parents=True)
    (tmp_path / "reports/calibration").mkdir(parents=True)
    (tmp_path / "data/proc/fillcalib/curves_SMOKE.json").write_text("{\"ok\":true}\n")
    (tmp_path / "reports/fillcalib/SMOKE.md").write_text("# Fillcalib\n")
    (tmp_path / "reports/pilot_ready.json").write_text("{\"ok\":true}\n")
    (tmp_path / "reports/pilot_readiness.md").write_text("# Pilot readiness\n")
    (tmp_path / "reports/calibration/SMOKE.md").write_text("# Calibration\n")

    artifacts_md = "\n".join(
        [
            "# Artifacts",
            "- data/proc/fillcalib/curves_SMOKE.json",
            "- reports/fillcalib/SMOKE.md",
            "- reports/pilot_ready.json",
            "- reports/pilot_readiness.md",
            "- reports/calibration/SMOKE.md",
            "",
        ]
    )
    files, root_prefix, _ = _base_files_new(run_name, artifacts_md)
    files[f"{root_prefix}/data/proc/fillcalib/curves_SMOKE.json"] = "{\"ok\":true}\n"
    files[f"{root_prefix}/reports/fillcalib/SMOKE.md"] = "# Fillcalib\n"
    files[f"{root_prefix}/reports/pilot_ready.json"] = "{\"ok\":true}\n"
    files[f"{root_prefix}/reports/pilot_readiness.md"] = "# Pilot readiness\n"
    files[f"{root_prefix}/reports/calibration/SMOKE.md"] = "# Calibration\n"

    zip_path = tmp_path / "bundle.zip"
    _write_zip(zip_path, files)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert f"{root_prefix}/data/proc/fillcalib/curves_SMOKE.json" in names
        assert f"{root_prefix}/reports/fillcalib/SMOKE.md" in names
        assert f"{root_prefix}/reports/pilot_ready.json" in names
        assert f"{root_prefix}/reports/pilot_readiness.md" in names
        assert f"{root_prefix}/reports/calibration/SMOKE.md" in names
    errors = verify_bundle(zip_path)
    assert errors == []


def test_verify_bundle_fails_when_artifact_listed_missing(tmp_path: Path, monkeypatch) -> None:
    run_name = "20251226T000001Z_TICKET-108_bundle_artifacts_fix"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports/calibration").mkdir(parents=True)
    (tmp_path / "reports/calibration/SMOKE.md").write_text("# Calibration\n")

    artifacts_md = "\n".join(
        [
            "# Artifacts",
            "- reports/calibration/SMOKE.md",
            "",
        ]
    )
    files, _, _ = _base_files_new(run_name, artifacts_md)
    zip_path = tmp_path / "bundle.zip"
    _write_zip(zip_path, files)
    errors = verify_bundle(zip_path)
    assert any("missing from bundle" in error for error in errors)
