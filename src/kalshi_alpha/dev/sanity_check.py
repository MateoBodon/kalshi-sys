"""Repository hygiene check with optional live smoke test."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from kalshi_alpha.brokers.kalshi.endpoints import resolve
from kalshi_alpha.brokers.kalshi.http_client import KalshiHttpClient, KalshiHttpError
from kalshi_alpha.utils.env import load_env

EXCLUDE_TOP_LEVEL = {"tests", "docs"}
FORBIDDEN_MARKERS = ("TODO", "NotImplementedError")
ENV_PRINT_TOKENS = ("KALSHI_", "API_KEY", "API_SECRET")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[3]

    status = _run_repo_checks(root)
    if status != 0:
        return status

    if args.live_smoke:
        return _run_live_smoke(args.env)
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repository hygiene guardrails")
    parser.add_argument(
        "--live-smoke",
        action="store_true",
        help="Hit Kalshi REST endpoints (/portfolio/balance, /markets) without submitting orders.",
    )
    parser.add_argument(
        "--env",
        choices=["prod", "demo"],
        default=None,
        help="Kalshi environment for live smoke (defaults to KALSHI_ENV or prod).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _run_repo_checks(root: Path) -> int:  # noqa: PLR0912 - clarity over micro-branching
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
            except OSError:  # pragma: no cover - non-readable file
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
        print(
            "Sanity check failed; forbidden markers found outside tests/ and docs/:",
            file=sys.stderr,
        )
        for rel_path, markers in marker_violations:
            joined = ", ".join(markers)
            print(f" - {rel_path} ({joined})", file=sys.stderr)

    if print_violations:
        had_error = True
        print(
            "Sanity check failed; detected print statements referencing env var names:",
            file=sys.stderr,
        )
        for rel_path, line in print_violations:
            print(f" - {rel_path}: {line}", file=sys.stderr)

    if had_error:
        return 1

    print("Sanity check passed: no forbidden markers or env var prints found.")
    return 0


def _run_live_smoke(env_override: str | None) -> int:
    load_env()

    kalshi_env = (env_override or os.getenv("KALSHI_ENV") or "prod").strip().lower()
    try:
        endpoints = resolve(kalshi_env)
    except ValueError as exc:
        print(f"Live smoke failed: {exc}", file=sys.stderr)
        return 1

    client = KalshiHttpClient(base_url=endpoints.rest)

    try:
        balance_response = client.get("/portfolio/balance")
        markets_response = client.get("/markets", params={"limit": 25})
    except KalshiHttpError as exc:
        print(f"Live smoke failed: {exc}", file=sys.stderr)
        return 1

    try:
        balance_payload = balance_response.json()
    except ValueError:
        print("Live smoke failed: /portfolio/balance returned non-JSON payload", file=sys.stderr)
        return 1
    try:
        markets_payload = markets_response.json()
    except ValueError:
        print("Live smoke failed: /markets returned non-JSON payload", file=sys.stderr)
        return 1

    balance_summary = _safe_summary(balance_payload, keys=("balance", "available"))
    markets = markets_payload.get("markets")
    market_count = len(markets) if isinstance(markets, list) else 0

    print(f"Live smoke ({kalshi_env}) portfolio balance ok; snapshot keys={balance_summary}")
    print(f"Live smoke ({kalshi_env}) markets ok; count={market_count}")
    return 0


def _safe_summary(payload: object, *, keys: tuple[str, ...]) -> list[str]:
    if not isinstance(payload, dict):
        return []
    summary: list[str] = []
    for key in keys:
        if key in payload:
            summary.append(key)
    return summary


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
