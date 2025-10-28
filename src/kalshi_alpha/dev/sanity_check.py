"""Repository hygiene check for TODOs and unfinished code markers."""

from __future__ import annotations

import sys
from pathlib import Path

EXCLUDE_TOP_LEVEL = {"tests", "docs"}
FORBIDDEN_MARKERS = ("TODO", "NotImplementedError")
ENV_PRINT_TOKENS = ("KALSHI_", "API_KEY", "API_SECRET")


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    marker_violations: list[tuple[Path, list[str]]] = []
    print_violations: list[tuple[Path, str]] = []

    this_file = Path(__file__).resolve()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path == this_file:
            continue
        if _is_excluded(path, root):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:  # pragma: no cover - non-readable file
                continue
        text_lower = text.lower()
        hits = [marker for marker in FORBIDDEN_MARKERS if marker.lower() in text_lower]
        if hits:
            marker_violations.append((path.relative_to(root), hits))

        for line in text.splitlines():
            if "print(" not in line:
                continue
            if any(token in line for token in ENV_PRINT_TOKENS):
                print_violations.append((path.relative_to(root), line.strip()))

    had_error = False
    if marker_violations:
        had_error = True
        print("Sanity check failed; forbidden markers found outside tests/ and docs/:", file=sys.stderr)
        for rel_path, markers in marker_violations:
            joined = ", ".join(markers)
            print(f" - {rel_path} ({joined})", file=sys.stderr)

    if print_violations:
        had_error = True
        print("Sanity check failed; detected print statements referencing env var names:", file=sys.stderr)
        for rel_path, line in print_violations:
            print(f" - {rel_path}: {line}", file=sys.stderr)

    if had_error:
        return 1

    print("Sanity check passed: no forbidden markers or env var prints found.")
    return 0


def _is_excluded(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:  # pragma: no cover - should not happen
        return True
    parts = relative.parts
    if not parts:
        return False
    if parts[0] in EXCLUDE_TOP_LEVEL:
        return True
    if parts[0].startswith(".") or parts[0] == "__pycache__":
        return True
    if any(part.startswith(".") or part == "__pycache__" for part in parts[1:]):
        return True
    return False


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
