#!/usr/bin/env python3
"""
gpt_bundle.py

Creates a zip bundle intended to be uploaded to GPT for review.

Outputs:
  artifacts/_local/gpt_bundles/gpt_bundle_<timestamp>[_<ticket>].zip

The bundle includes:
- artifacts/_local/repo_snapshot.md (auto-generated)
- git status, log, diffs
- ticket file (if present)
- selected small changed files (best-effort)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def run(cmd: list[str], cwd: Optional[Path] = None) -> Tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.STDOUT)
        return 0, out.decode("utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        return e.returncode, e.output.decode("utf-8", errors="replace")


def git_root(start: Path) -> Optional[Path]:
    code, out = run(["git", "-C", str(start), "rev-parse", "--show-toplevel"])
    if code != 0:
        return None
    return Path(out.strip())


def default_bundle_dir(repo: Path) -> Path:
    return repo / "artifacts" / "_local" / "gpt_bundles"


def default_bundle_path(repo: Path, ts: str, ticket: str) -> Path:
    suffix = f"_{ticket}" if ticket else ""
    return default_bundle_dir(repo) / f"gpt_bundle_{ts}{suffix}.zip"


def default_repo_snapshot_path(repo: Path) -> Path:
    return repo / "artifacts" / "_local" / "repo_snapshot.md"


def ensure_repo_snapshot(repo: Path) -> Optional[Path]:
    snap = default_repo_snapshot_path(repo)
    if snap.exists():
        return snap
    tool = repo / "tools" / "agentic" / "repo_snapshot.py"
    if tool.exists():
        snap.parent.mkdir(parents=True, exist_ok=True)
        code, _ = run([sys.executable, str(tool), "--out", str(snap)], cwd=repo)
        if code == 0 and snap.exists():
            return snap
    return None


def git_status_porcelain(repo: Path) -> str:
    code, out = run(["git", "-C", str(repo), "status", "--porcelain"])
    if code != 0:
        return ""
    return out


def is_dirty(status_porcelain: str) -> bool:
    return bool(status_porcelain.strip())


def ensure_allowed_output(repo: Path, out_zip: Path) -> Optional[str]:
    allowed_root = (repo / "artifacts" / "_local" / "gpt_bundles").resolve()
    candidate = out_zip.resolve()
    try:
        candidate.relative_to(allowed_root)
    except ValueError:
        return f"Output path must live under {allowed_root} (got {candidate})"
    return None


def stash_if_dirty(repo: Path, ticket: str, ts: str, allow_stash: bool) -> tuple[bool, bool, str, Optional[str], Optional[str]]:
    status_before = git_status_porcelain(repo)
    dirty = is_dirty(status_before)
    if not dirty or not allow_stash:
        return dirty, False, status_before, None, None

    label = f"temp: gpt_bundle {ticket or ts}"
    code, out = run(["git", "-C", str(repo), "stash", "push", "-u", "-m", label])
    if code != 0:
        return dirty, False, status_before, None, f"git stash push failed: {out.strip()}"

    code, out = run(["git", "-C", str(repo), "stash", "list", "-n", "1"])
    if code != 0 or not out.strip():
        return dirty, True, status_before, None, "Unable to locate stash reference after stash push."
    stash_ref = out.splitlines()[0].split(":", 1)[0].strip()

    status_after = git_status_porcelain(repo)
    if status_after.strip():
        return dirty, True, status_before, stash_ref, "Working tree still dirty after stashing."

    return dirty, True, status_before, stash_ref, None


def restore_stash(repo: Path, stash_ref: str, status_before: str) -> Optional[str]:
    code, out = run(["git", "-C", str(repo), "stash", "apply", "--index", stash_ref])
    if code != 0:
        return f"Failed to apply stash {stash_ref}. Resolve conflicts and re-apply if needed.\n{out.strip()}"

    status_after = git_status_porcelain(repo)
    if status_after != status_before:
        return (
            "Working tree did not match pre-bundle state after stash apply. "
            f"Stash preserved at {stash_ref}; resolve manually."
        )

    code, out = run(["git", "-C", str(repo), "stash", "drop", stash_ref])
    if code != 0:
        return f"Failed to drop stash {stash_ref}; drop it manually if safe.\n{out.strip()}"

    return None


def list_changed_files(repo: Path) -> list[str]:
    # Prefer git diff names for working tree
    _, out = run(["git", "-C", str(repo), "diff", "--name-only"])
    changed = [l.strip() for l in out.splitlines() if l.strip()]
    # Include staged
    _, out2 = run(["git", "-C", str(repo), "diff", "--cached", "--name-only"])
    for l in out2.splitlines():
        l = l.strip()
        if l and l not in changed:
            changed.append(l)
    return changed


def add_file_if_small(z: zipfile.ZipFile, repo: Path, rel_path: str, max_bytes: int = 120_000) -> None:
    p = repo / rel_path
    if not p.exists() or not p.is_file():
        return
    try:
        if p.stat().st_size > max_bytes:
            return
        z.write(p, arcname=str(p.relative_to(repo)))
    except Exception:
        return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true", help="Create zip bundle (default behavior).")
    ap.add_argument("--ticket", type=str, default=None, help="Ticket id to include (optional).")
    ap.add_argument("--out", type=str, default=None, help="Output zip path (optional).")
    ap.add_argument("--include-files", action="store_true", help="Include small changed files in addition to diffs.")
    ap.add_argument("--no-stash", action="store_true", help="Disable temporary stash even when the repo is dirty.")
    args = ap.parse_args()

    start = Path.cwd()
    repo = git_root(start) or start

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    ticket = (args.ticket or "").strip()
    out_zip = Path(args.out) if args.out else default_bundle_path(repo, ts, ticket)
    if not out_zip.is_absolute():
        out_zip = (repo / out_zip)
    err = ensure_allowed_output(repo, out_zip)
    if err:
        print(f"[error] {err}", file=sys.stderr)
        return 2
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    dirty, stash_used, status_before, stash_ref, stash_err = stash_if_dirty(
        repo, ticket, ts, allow_stash=not args.no_stash
    )
    print(f"[bundle] dirty: {'yes' if dirty else 'no'}")
    print(f"[bundle] stash: {'yes' if stash_used else 'no'}")
    if stash_err:
        print(f"[error] {stash_err}", file=sys.stderr)
        return 2

    restore_err = None
    try:
        # Ensure snapshot exists
        snap = ensure_repo_snapshot(repo)

        # Collect git info
        _, status = run(["git", "-C", str(repo), "status", "--porcelain=v1", "-b"])
        _, log = run(["git", "-C", str(repo), "log", "-n", "50", "--oneline", "--decorate"])
        _, diff = run(["git", "-C", str(repo), "diff"])
        _, diff_cached = run(["git", "-C", str(repo), "diff", "--cached"])
        _, diff_stat = run(["git", "-C", str(repo), "diff", "--stat"])
        changed = list_changed_files(repo)

        readme = f"""GPT Bundle

Generated: {ts}Z
Repo: {repo}
Ticket: {ticket or "(none)"}

Contents:
- artifacts/_local/repo_snapshot.md (if available)
- git_status.txt
- git_log.txt
- git_diff.patch (working tree)
- git_diff_cached.patch (staged)
- git_diff_stat.txt
- changed_files.txt
- ticket file (if present)
- small changed files (optional)
"""

        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("README.txt", readme)
            z.writestr("git_status.txt", status)
            z.writestr("git_log.txt", log)
            z.writestr("git_diff.patch", diff)
            z.writestr("git_diff_cached.patch", diff_cached)
            z.writestr("git_diff_stat.txt", diff_stat)
            z.writestr("changed_files.txt", "\n".join(changed) + ("\n" if changed else ""))

            if snap and snap.exists():
                z.write(snap, arcname=str(snap.relative_to(repo)))

            # Ticket file
            if ticket:
                tf = repo / "docs" / "tickets" / f"{ticket}.md"
                if tf.exists():
                    z.write(tf, arcname=str(tf.relative_to(repo)))

            if args.include_files:
                for rel in changed:
                    add_file_if_small(z, repo, rel)
    finally:
        if stash_used and stash_ref:
            restore_err = restore_stash(repo, stash_ref, status_before)

    if restore_err:
        print(f"[error] {restore_err}", file=sys.stderr)
        return 2

    print(f"[bundle] output: {out_zip}")
    print(str(out_zip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
