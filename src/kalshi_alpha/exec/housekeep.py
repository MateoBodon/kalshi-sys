"""Artifact retention housekeeping utility."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple

RETENTION_ROOTS = [
    Path("data/raw/kalshi"),
    Path("reports"),
    Path("reports/_artifacts"),
    Path("data/proc/logs"),
]


class Candidate(NamedTuple):
    path: Path
    mtime: datetime
    category: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune stale artifacts while keeping recent event windows.")
    parser.add_argument(
        "--keep-days",
        type=int,
        default=30,
        help="Retention window in days (default: 30).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cutoff = datetime.now(tz=UTC) - timedelta(days=max(args.keep_days, 0))
    total_deleted = 0
    for root in RETENTION_ROOTS:
        deleted = _prune_root(root, cutoff)
        total_deleted += deleted
    print(f"[housekeep] removed {total_deleted} artifacts older than {args.keep_days} days")


def _prune_root(root: Path, cutoff: datetime) -> int:
    if not root.exists():
        return 0
    candidates = list(_collect_candidates(root))
    preserved = _select_preserved(candidates)
    removed = 0
    for candidate in candidates:
        if candidate.path in preserved.values():
            continue
        if candidate.mtime >= cutoff:
            continue
        _delete_path(candidate.path)
        removed += 1
    return removed


def _collect_candidates(root: Path) -> Iterable[Candidate]:
    if not root.exists():
        return []
    entries: list[Path] = []
    if root.name == "reports":
        for entry in root.iterdir():
            if entry.is_dir() and entry.name.isupper():
                try:
                    for child in entry.iterdir():
                        entries.append(child)
                except FileNotFoundError:
                    continue
            else:
                entries.append(entry)
    else:
        try:
            entries = list(root.iterdir())
        except FileNotFoundError:
            entries = []
    candidates: list[Candidate] = []
    for entry in entries:
        try:
            stat = entry.stat()
        except FileNotFoundError:
            continue
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        candidates.append(Candidate(path=entry, mtime=mtime, category=_detect_category(entry)))
    return candidates


def _detect_category(path: Path) -> str | None:
    text = str(path).lower()
    if "teny" in text and "close" in text:
        return "teny_close"
    if "claims" in text and "freeze" in text:
        return "claims_freeze"
    if "claims" in text and ("print" in text or "release" in text or "pre" in text):
        return "claims_print"
    if "cpi" in text and "eve" in text:
        return "cpi_eve"
    if "cpi" in text and ("day" in text or "release" in text or "pre" in text):
        return "cpi_day"
    return None


def _select_preserved(candidates: Iterable[Candidate]) -> dict[str, Path]:
    preserved: dict[str, Candidate] = {}
    for candidate in candidates:
        if candidate.category is None:
            continue
        current = preserved.get(candidate.category)
        if current is None or candidate.mtime > current.mtime:
            preserved[candidate.category] = candidate
    return {category: cand.path for category, cand in preserved.items()}


def _delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
