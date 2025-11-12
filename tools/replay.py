"""Re-run archived manifests to regenerate replay EV artifacts with parity filters."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import sys
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from kalshi_alpha.core.archive.replay import replay_manifest
from kalshi_alpha.utils.series import (
    INDEX_CANONICAL_SERIES,
    normalize_index_series,
    normalize_index_series_list,
)

DEFAULT_RAW_ROOT = Path("data/raw/kalshi")
DEFAULT_OUTPUT = Path("reports/_artifacts/replay_ev.parquet")
CENTS_PER_DOLLAR = 100.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay archived manifests for EV parity analysis.")
    parser.add_argument(
        "--date",
        default="yesterday",
        help="Trading day to backfill (YYYY-MM-DD | today | yesterday).",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=list(INDEX_CANONICAL_SERIES),
        help=f"Series tickers to process (default: {','.join(INDEX_CANONICAL_SERIES)}).",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Root directory containing archived Kalshi manifests (default: data/raw/kalshi).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write aggregated replay parquet (default: reports/_artifacts/replay_ev.parquet).",
    )
    parser.add_argument(
        "--epsilon-cents",
        type=float,
        default=5.0,
        help="Drop rows with per-contract ΔEV below this threshold (cents).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    target_date = _resolve_date(args.date)
    families = normalize_index_series_list(args.families)
    if not families:
        raise SystemExit("No valid series provided.")
    manifest_paths = _discover_manifests(args.raw_root, target_date, families)
    if not manifest_paths:
        raise SystemExit(f"No manifests found for {target_date.isoformat()} ({', '.join(families)}).")

    frames: list[pl.DataFrame] = []
    for manifest in manifest_paths:
        try:
            artifact_path = replay_manifest(manifest)
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"[replay] failed to replay {manifest}: {exc}", file=sys.stderr)
            continue
        output = Path(artifact_path)
        if not output.exists():
            continue
        frame = pl.read_parquet(output)
        if frame.is_empty():
            continue
        frames.append(
            frame.with_columns(
                pl.lit(str(manifest)).alias("manifest_path"),
            )
        )
    if not frames:
        raise SystemExit("Replay produced no rows; cannot write aggregate.")

    combined = pl.concat(frames, how="vertical_relaxed")
    combined = _normalize_series_column(combined)
    filtered = _apply_epsilon_filter(combined, args.epsilon_cents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_parquet(args.output)
    print(
        f"[replay] wrote {args.output} rows={filtered.height} "
        f"(manifests={len(manifest_paths)} threshold={args.epsilon_cents:.1f}¢)"
    )


def _resolve_date(value: str) -> date:
    candidate = (value or "").strip().lower()
    if not candidate or candidate == "today":
        return datetime.now().date()
    if candidate == "yesterday":
        return (datetime.now() - timedelta(days=1)).date()
    return date.fromisoformat(value)


def _discover_manifests(raw_root: Path, target_date: date, series: Sequence[str]) -> list[Path]:
    date_dir = raw_root / target_date.isoformat()
    if not date_dir.exists():
        return []
    manifests: list[Path] = []
    for timestamp_dir in sorted(filter(Path.is_dir, date_dir.iterdir())):
        for ticker in series:
            manifest = timestamp_dir / ticker / "manifest.json"
            if manifest.exists():
                manifests.append(manifest)
    return manifests


def _normalize_series_column(frame: pl.DataFrame) -> pl.DataFrame:
    if "series" not in frame.columns:
        return frame
    return frame.with_columns(
        pl.col("series")
        .cast(pl.Utf8)
        .map_elements(lambda value: normalize_index_series(value), return_dtype=pl.Utf8)
    )


def _apply_epsilon_filter(frame: pl.DataFrame, epsilon_cents: float) -> pl.DataFrame:
    if epsilon_cents is None or epsilon_cents <= 0:
        return frame
    per_contract_original = (
        pl.col("maker_ev_original") / pl.col("contracts").cast(pl.Float64).clip(lower_bound=1.0)
    )
    delta_cents = (
        (pl.col("maker_ev_per_contract_replay") - per_contract_original)
        .abs()
        .mul(CENTS_PER_DOLLAR)
        .alias("delta_cents")
    )
    enriched = frame.with_columns(delta_cents)
    return enriched.filter(pl.col("delta_cents") >= float(epsilon_cents))


if __name__ == "__main__":  # pragma: no cover
    main()
