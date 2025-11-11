"""Convert TOB snapshots into conservative fill probability curves."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Sequence

from kalshi_alpha.datastore.paths import PROC_ROOT

DEFAULT_OUTPUT = PROC_ROOT / "fill" / "index_fill_curve.json"
DEFAULT_TOB_DIR = Path("data/raw/kalshi/tob")
DEFAULT_CONTRACTS = 10.0
LATE_THRESHOLD_SECONDS = 120.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fill probability curves from TOB snapshots.")
    parser.add_argument(
        "--snapshots",
        nargs="+",
        default=None,
        help="Specific snapshot files to ingest (.jsonl). Defaults to all files under data/raw/kalshi/tob.",
    )
    parser.add_argument(
        "--tob-dir",
        type=Path,
        default=DEFAULT_TOB_DIR,
        help="Directory containing TOB jsonl files (default: data/raw/kalshi/tob).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON file for fill probabilities (default: data/proc/fill/index_fill_curve.json).",
    )
    parser.add_argument(
        "--contracts",
        type=float,
        default=DEFAULT_CONTRACTS,
        help="Reference contract size when computing depth ratios (default: %(default)s).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    files = _resolve_snapshot_files(args.snapshots, args.tob_dir)
    if not files:
        raise SystemExit("No TOB snapshot files found")
    entries = list(_iter_snapshots(files))
    if not entries:
        raise SystemExit("Snapshot files were empty")
    summary = _build_summary(entries, reference_contracts=max(args.contracts, 1.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[fill_model] wrote {args.output}")


def _resolve_snapshot_files(explicit: Sequence[str] | None, directory: Path) -> list[Path]:
    if explicit:
        return [Path(path) for path in explicit]
    if not directory.exists():
        return []
    return sorted(directory.glob("*.jsonl"))


def _iter_snapshots(files: Iterable[Path]) -> Iterable[dict[str, object]]:
    for path in files:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        except OSError:
            continue


def _build_summary(entries: Iterable[dict[str, object]], *, reference_contracts: float) -> dict[str, object]:
    buckets: dict[str, list[tuple[float, float | None]]] = defaultdict(list)
    for entry in entries:
        series = str(entry.get("series") or "").upper()
        if not series:
            continue
        bid_size = float(entry.get("best_bid_size") or 0.0)
        ask_size = float(entry.get("best_ask_size") or 0.0)
        avg_depth = max((bid_size + ask_size) / 2.0, 0.0)
        probability = max(0.1, min(1.0, avg_depth / reference_contracts))
        seconds_to_close = entry.get("seconds_to_close")
        try:
            seconds = float(seconds_to_close) if seconds_to_close is not None else None
        except (TypeError, ValueError):
            seconds = None
        buckets[series].append((probability, seconds))
    payload = {
        "version": 1,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "series": {},
    }
    for series, samples in buckets.items():
        if not samples:
            continue
        probs = [value for value, _ in samples]
        default_prob = sum(probs) / len(probs)
        late_samples = [value for value, seconds in samples if seconds is not None and seconds <= LATE_THRESHOLD_SECONDS]
        late_prob = sum(late_samples) / len(late_samples) if late_samples else None
        payload["series"][series] = {
            "default_probability": round(default_prob, 4),
            "late_probability": round(late_prob, 4) if late_prob is not None else None,
            "late_threshold_seconds": LATE_THRESHOLD_SECONDS,
            "sample_size": len(samples),
        }
    return payload


if __name__ == "__main__":  # pragma: no cover
    main()
