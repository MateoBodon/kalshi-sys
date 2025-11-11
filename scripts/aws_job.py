#!/usr/bin/env python3
"""Rudimentary AWS job shim that can run locally or inside Docker."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

ARTIFACT_ROOT = Path("reports/_artifacts/aws_jobs")
DOCKERFILE = Path("docker/aws-jobs/Dockerfile")
IMAGE_NAME = "kalshi-aws-jobs:latest"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a job the same way our AWS batch wrapper does.")
    parser.add_argument("--job", required=True, help="Logical job name (e.g., calib_hourly, replay)")
    parser.add_argument("--command", required=True, help="Command to execute inside the job container/env")
    parser.add_argument("--artifact", action="append", default=[], help="Optional file/glob to copy into reports/_artifacts/aws_jobs")
    parser.add_argument("--use-docker", action="store_true", help="Build + run docker/aws-jobs/Dockerfile instead of executing locally")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    start = datetime.now(tz=UTC)
    exit_code = _run_command(args.command, use_docker=args.use_docker)
    duration = (datetime.now(tz=UTC) - start).total_seconds()
    metrics = {
        "job": args.job,
        "command": args.command,
        "use_docker": args.use_docker,
        "exit_code": exit_code,
        "duration_seconds": duration,
        "completed_at": datetime.now(tz=UTC).isoformat(),
    }
    metrics_path = ARTIFACT_ROOT / f"{args.job}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    if exit_code != 0:
        sys.exit(exit_code)
    for entry in args.artifact:
        _copy_artifact(entry, args.job)


def _run_command(command: str, *, use_docker: bool) -> int:
    if not use_docker:
        return subprocess.call(command, shell=True)
    if not DOCKERFILE.exists():
        raise SystemExit("docker/aws-jobs/Dockerfile missing; cannot build container")
    build = subprocess.call(["docker", "build", "-t", IMAGE_NAME, "-f", str(DOCKERFILE), "."])
    if build != 0:
        return build
    run_cmd = [
        "docker",
        "run",
        "--rm",
        IMAGE_NAME,
        "bash",
        "-lc",
        command,
    ]
    return subprocess.call(run_cmd)


def _copy_artifact(pattern: str, job: str) -> None:
    paths = list(Path().glob(pattern))
    if not paths:
        return
    for source in paths:
        if source.is_dir():
            archive = ARTIFACT_ROOT / f"{job}_{source.name}.tar.gz"
            shutil.make_archive(archive.with_suffix(""), "gztar", root_dir=source)
            continue
        dest = ARTIFACT_ROOT / f"{job}_{source.name}"
        shutil.copy2(source, dest)


if __name__ == "__main__":  # pragma: no cover
    main()
