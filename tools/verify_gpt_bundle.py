"""Verify GPT bundle completeness and diff hygiene."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

PLACEHOLDER_LINES = {"...", "content omitted", "(truncated)"}


class BundleVerificationError(Exception):
    pass


def _read_text(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8", errors="replace")


def _find_run_names(paths: list[str]) -> set[str]:
    run_names: set[str] = set()
    needle = "docs/agent_runs/"
    for path in paths:
        idx = path.find(needle)
        if idx == -1:
            continue
        tail = path[idx + len(needle) :]
        parts = tail.split("/", 1)
        if parts and parts[0]:
            run_names.add(parts[0])
    return run_names


def _placeholder_lines(content: str) -> list[str]:
    matches: list[str] = []
    for line in content.splitlines():
        if line[:1] in {"+", "-", " "}:
            # Ignore file content lines; placeholders there can be legitimate data.
            continue
        candidate = line.strip().lower()
        if candidate in PLACEHOLDER_LINES:
            matches.append(line)
    return matches


def _is_empty_or_pending(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    return lowered in {"pending", "tbd", "todo"}


def _has_patch_hunks(content: str) -> bool:
    return "diff --git " in content and ("+++ b/" in content or "+++ b\\" in content)


def verify_bundle(bundle_path: Path) -> list[str]:
    errors: list[str] = []
    if not bundle_path.exists():
        return [f"bundle not found: {bundle_path}"]
    if bundle_path.suffix.lower() != ".zip":
        return [f"bundle is not a .zip: {bundle_path}"]

    with zipfile.ZipFile(bundle_path) as zf:
        paths = [info.filename for info in zf.infolist()]
        if not paths:
            return ["bundle zip is empty"]

        run_names = _find_run_names(paths)
        if len(run_names) != 1:
            errors.append(f"expected 1 run log directory, found {sorted(run_names)}")
            return errors
        run_name = next(iter(run_names))

        run_log_prefix = None
        run_log_needle = f"docs/agent_runs/{run_name}/"
        for path in paths:
            if run_log_needle in path:
                run_log_prefix = path.split(run_log_needle)[0] + run_log_needle
                break
        if run_log_prefix is None:
            errors.append(f"run log path not found for {run_name}")
            return errors

        root_diff_candidates = [
            path
            for path in paths
            if path.endswith("/DIFF.patch") and "docs/agent_runs/" not in path
        ]
        if len(root_diff_candidates) != 1:
            errors.append(f"expected 1 root DIFF.patch, found {root_diff_candidates}")
            return errors
        root_diff_path = root_diff_candidates[0]

        root_diff_content = _read_text(zf, root_diff_path)
        if not root_diff_content.strip():
            errors.append("root DIFF.patch is empty")
        root_placeholder = _placeholder_lines(root_diff_content)
        if root_placeholder:
            errors.append("root DIFF.patch contains placeholder markers")

        required_files = [
            "README.md",
            "RESULTS.md",
            "META.json",
            "commands.log",
            "artifacts.json",
            "diff.patch",
        ]
        for filename in required_files:
            target = run_log_prefix + filename
            if target not in paths:
                errors.append(f"missing run log file: {filename}")

        prompt_candidates = [run_log_prefix + "prompt.md", run_log_prefix + "PROMPT.md"]
        if not any(candidate in paths for candidate in prompt_candidates):
            errors.append("missing run log file: prompt.md/PROMPT.md")

        if errors:
            return errors

        readme_content = _read_text(zf, run_log_prefix + "README.md")
        if _is_empty_or_pending(readme_content):
            errors.append("README.md is empty or placeholder")

        results_content = _read_text(zf, run_log_prefix + "RESULTS.md")
        if _is_empty_or_pending(results_content):
            errors.append("RESULTS.md is empty or placeholder")

        diff_content = _read_text(zf, run_log_prefix + "diff.patch")
        if not diff_content.strip():
            errors.append("run log diff.patch is empty")
        if not _has_patch_hunks(diff_content):
            errors.append("run log diff.patch missing patch hunks")
        diff_placeholder = _placeholder_lines(diff_content)
        if diff_placeholder:
            errors.append("run log diff.patch contains placeholder markers")

        meta_raw = _read_text(zf, run_log_prefix + "META.json")
        try:
            meta = json.loads(meta_raw)
        except json.JSONDecodeError as exc:
            errors.append(f"META.json invalid JSON: {exc}")
        else:
            end_utc = meta.get("end_utc")
            if not end_utc or not str(end_utc).strip():
                errors.append("META.json missing end_utc")

    return errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify GPT bundle completeness.")
    parser.add_argument("bundle", type=Path, help="Path to gpt-bundle zip file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    errors = verify_bundle(args.bundle)
    if errors:
        print("FAIL: GPT bundle verification failed.")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASS: GPT bundle verification OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
