#!/usr/bin/env python3
"""Stage GPT bundle contents with fail-closed artifact checks."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from tools.verify_gpt_bundle import _extract_artifact_paths


class BundleBuildError(RuntimeError):
    pass


def _copy_file(
    src: Path, dest: Path, staging_root: Path, staged_files: set[str]
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    staged_files.add(dest.relative_to(staging_root).as_posix())


def _copy_tree(
    src_dir: Path, dest_dir: Path, staging_root: Path, staged_files: set[str]
) -> None:
    if not src_dir.exists():
        return
    for path in src_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(src_dir)
            dest = dest_dir / rel
            _copy_file(path, dest, staging_root, staged_files)


def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise BundleBuildError(f"missing required {label}: {path}")


def _missing_artifacts_from_stage(
    artifacts_content: str, staged_files: set[str], workspace_root: Path
) -> list[str]:
    missing: list[str] = []
    for raw_path in _extract_artifact_paths(artifacts_content):
        rel_path = Path(raw_path)
        if rel_path.is_absolute():
            try:
                rel_path = rel_path.relative_to(workspace_root)
            except ValueError:
                continue
        if not rel_path.parts:
            continue
        if rel_path.parts[0] == "..":
            continue
        disk_path = workspace_root / rel_path
        if not disk_path.exists():
            continue
        rel_str = rel_path.as_posix()
        if disk_path.is_dir():
            prefix = rel_str.rstrip("/") + "/"
            present = any(
                path.startswith(prefix) or path == rel_str for path in staged_files
            )
        else:
            present = rel_str in staged_files
        if not present:
            missing.append(rel_str)
    return missing


def stage_bundle(workspace_root: Path, run_name: str, staging_root: Path) -> set[str]:
    staged_files: set[str] = set()

    docs_dir = workspace_root / "docs"
    project_state_dir = workspace_root / "project_state"
    run_log_dir = docs_dir / "agent_runs" / run_name
    systemd_unit = (
        workspace_root
        / "configs"
        / "systemd"
        / "kalshi-index-supervisor-paper.service"
    )
    cloudwatch_config = (
        workspace_root
        / "configs"
        / "cloudwatch"
        / "kalshi-supervisor-index.json"
    )
    aws_runbook = docs_dir / "runbooks" / "aws_supervisor_index.md"
    oncall_runbook = docs_dir / "runbooks" / "oncall_checks.md"

    _require_path(workspace_root / "AGENTS.md", "file")
    _require_path(docs_dir / "PLAN_OF_RECORD.md", "file")
    _require_path(docs_dir / "DOCS_AND_LOGGING_SYSTEM.md", "file")
    _require_path(docs_dir / "CODEX_SPRINT_TICKETS.md", "file")
    _require_path(docs_dir / "PROGRESS.md", "file")
    _require_path(project_state_dir / "CURRENT_RESULTS.md", "file")
    _require_path(project_state_dir / "KNOWN_ISSUES.md", "file")
    _require_path(project_state_dir / "CONFIG_REFERENCE.md", "file")
    _require_path(run_log_dir, "run log directory")
    _require_path(systemd_unit, "file")
    _require_path(cloudwatch_config, "file")
    _require_path(aws_runbook, "file")
    _require_path(oncall_runbook, "file")

    _copy_file(
        workspace_root / "AGENTS.md",
        staging_root / "AGENTS.md",
        staging_root,
        staged_files,
    )
    _copy_file(
        docs_dir / "PLAN_OF_RECORD.md",
        staging_root / "docs" / "PLAN_OF_RECORD.md",
        staging_root,
        staged_files,
    )
    _copy_file(
        docs_dir / "DOCS_AND_LOGGING_SYSTEM.md",
        staging_root / "docs" / "DOCS_AND_LOGGING_SYSTEM.md",
        staging_root,
        staged_files,
    )
    _copy_file(
        docs_dir / "CODEX_SPRINT_TICKETS.md",
        staging_root / "docs" / "CODEX_SPRINT_TICKETS.md",
        staging_root,
        staged_files,
    )
    _copy_file(
        docs_dir / "PROGRESS.md",
        staging_root / "docs" / "PROGRESS.md",
        staging_root,
        staged_files,
    )
    _copy_file(
        docs_dir / "PROGRESS.md",
        staging_root / "PROGRESS.md",
        staging_root,
        staged_files,
    )
    _copy_file(
        project_state_dir / "CURRENT_RESULTS.md",
        staging_root / "project_state" / "CURRENT_RESULTS.md",
        staging_root,
        staged_files,
    )
    _copy_file(
        project_state_dir / "KNOWN_ISSUES.md",
        staging_root / "project_state" / "KNOWN_ISSUES.md",
        staging_root,
        staged_files,
    )
    _copy_file(
        project_state_dir / "CONFIG_REFERENCE.md",
        staging_root / "project_state" / "CONFIG_REFERENCE.md",
        staging_root,
        staged_files,
    )
    _copy_file(
        systemd_unit,
        staging_root / "configs" / "systemd" / systemd_unit.name,
        staging_root,
        staged_files,
    )
    _copy_file(
        cloudwatch_config,
        staging_root / "configs" / "cloudwatch" / cloudwatch_config.name,
        staging_root,
        staged_files,
    )
    _copy_file(
        aws_runbook,
        staging_root / "docs" / "runbooks" / aws_runbook.name,
        staging_root,
        staged_files,
    )
    _copy_file(
        oncall_runbook,
        staging_root / "docs" / "runbooks" / oncall_runbook.name,
        staging_root,
        staged_files,
    )

    _copy_tree(
        run_log_dir,
        staging_root / "docs" / "agent_runs" / run_name,
        staging_root,
        staged_files,
    )

    telemetry_dir = workspace_root / "data" / "proc" / "telemetry"
    if telemetry_dir.exists():
        _copy_tree(
            telemetry_dir,
            staging_root / "data" / "proc" / "telemetry",
            staging_root,
            staged_files,
        )

    fillcalib_dir = workspace_root / "data" / "proc" / "fillcalib"
    if fillcalib_dir.exists():
        for path in fillcalib_dir.glob("*.json"):
            _copy_file(
                path,
                staging_root / "data" / "proc" / "fillcalib" / path.name,
                staging_root,
                staged_files,
            )

    ops_reports = workspace_root / "reports" / "ops"
    if ops_reports.exists():
        for path in ops_reports.glob("telemetry_volume_*.md"):
            _copy_file(
                path,
                staging_root / "reports" / "ops" / path.name,
                staging_root,
                staged_files,
            )
        for path in ops_reports.glob("aws_supervisor_dryrun_*.md"):
            _copy_file(
                path,
                staging_root / "reports" / "ops" / path.name,
                staging_root,
                staged_files,
            )
        for path in ops_reports.glob("supervisor_dryrun_*.md"):
            _copy_file(
                path,
                staging_root / "reports" / "ops" / path.name,
                staging_root,
                staged_files,
            )

    fillcalib_reports = workspace_root / "reports" / "fillcalib"
    if fillcalib_reports.exists():
        for path in fillcalib_reports.glob("*.md"):
            _copy_file(
                path,
                staging_root / "reports" / "fillcalib" / path.name,
                staging_root,
                staged_files,
            )

    pilot_ready = workspace_root / "reports" / "pilot_ready.json"
    if pilot_ready.exists():
        _copy_file(
            pilot_ready,
            staging_root / "reports" / "pilot_ready.json",
            staging_root,
            staged_files,
        )

    pilot_readiness = workspace_root / "reports" / "pilot_readiness.md"
    if pilot_readiness.exists():
        _copy_file(
            pilot_readiness,
            staging_root / "reports" / "pilot_readiness.md",
            staging_root,
            staged_files,
        )

    calibration_dir = workspace_root / "reports" / "calibration"
    if calibration_dir.exists():
        _copy_tree(
            calibration_dir,
            staging_root / "reports" / "calibration",
            staging_root,
            staged_files,
        )

    artifacts_path = run_log_dir / "ARTIFACTS.md"
    if artifacts_path.exists():
        content = artifacts_path.read_text(encoding="utf-8")
        missing = _missing_artifacts_from_stage(content, staged_files, workspace_root)
        if missing:
            missing_list = "\n".join(f"- {path}" for path in sorted(missing))
            raise BundleBuildError(
                "missing from bundle (listed in ARTIFACTS.md):\n" + missing_list
            )

    return staged_files


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage GPT bundle contents.")
    parser.add_argument("--run-name", required=True, help="Run name for bundle staging")
    parser.add_argument(
        "--staging",
        required=True,
        type=Path,
        help="Staging directory for bundle contents",
    )
    parser.add_argument(
        "--workspace",
        default=Path("."),
        type=Path,
        help="Workspace root (default: current directory)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        stage_bundle(args.workspace.resolve(), args.run_name, args.staging.resolve())
    except BundleBuildError as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
