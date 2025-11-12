#!/usr/bin/env python3
"""Fail fast if replay vs live EV parity drifts beyond the allowed epsilon."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

CENTS_PER_DOLLAR = 100.0
REPLAY_PATH = Path("reports/_artifacts/replay_ev.parquet")
DEFAULT_MONITOR_PATH = Path("reports/_artifacts/monitors/ev_gap.json")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ΔEV parity between replay and live proposals.")
    parser.add_argument(
        "--threshold-cents",
        type=float,
        default=15.0,
        help="Max allowed per-contract delta (cents).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--path", type=Path, default=REPLAY_PATH, help="Replay EV parquet path (default: reports/_artifacts/replay_ev.parquet).")
    parser.add_argument(
        "--window-column",
        default="window_type",
        help="Column name to group by when enforcing per-window parity (set to 'none' to disable).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_MONITOR_PATH,
        help="Path to write JSON summary for dashboards.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    threshold_cents = _resolve_threshold_cents(args)
    if not args.path.exists():
        raise SystemExit(f"Replay EV file missing: {args.path}")
    frame = pl.read_parquet(args.path)
    if frame.is_empty():
        print("[parity] replay file empty; treating as pass")
        return
    required = {"maker_ev_per_contract_replay", "maker_ev_original", "contracts"}
    if required.difference(frame.columns):
        missing = ", ".join(sorted(required.difference(frame.columns)))
        raise SystemExit(f"Replay file missing columns: {missing}")
    per_contract_original = (
        (pl.col("maker_ev_original") / pl.col("contracts").clip(lower_bound=1))
        .alias("maker_ev_per_contract_original")
    )
    delta_col = (
        (pl.col("maker_ev_per_contract_replay") - per_contract_original)
        .abs()
        .mul(CENTS_PER_DOLLAR)
        .alias("delta_cents")
    )
    enriched = frame.with_columns([per_contract_original, delta_col])
    window_map = _window_max(enriched, args.window_column)
    max_delta = max(window_map.values()) if window_map else 0.0
    for window, value in sorted(window_map.items()):
        print(
            f"[parity] window={window} max ΔEV per contract = {value:.2f}¢ "
            f"(threshold {threshold_cents:.2f}¢)"
        )
    if args.output_json:
        _write_summary(args.output_json, threshold_cents, window_map, max_delta)
    offenders = {window: value for window, value in window_map.items() if value > threshold_cents}
    if offenders:
        raise SystemExit(
            "per-window ΔEV parity exceeded threshold (cents): "
            + ", ".join(f"{window}={value:.2f}¢" for window, value in sorted(offenders.items()))
        )


def _window_max(frame: pl.DataFrame, column: str) -> dict[str, float]:
    if column and column.lower() != "none" and column in frame.columns:
        grouped = frame.group_by(column).agg(pl.max("delta_cents").alias("window_delta"))
        return {str(row[column]): float(row["window_delta"]) for row in grouped.to_dicts()}
    value = float(frame.select(pl.max("delta_cents")).item())
    return {"all": value}


def _write_summary(path: Path, threshold_cents: float, window_map: dict[str, float], max_delta: float) -> None:
    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "threshold_cents": threshold_cents,
        "threshold_usd": threshold_cents / CENTS_PER_DOLLAR,
        "max_delta_cents": max_delta,
        "max_delta_usd": max_delta / CENTS_PER_DOLLAR,
        "by_window": window_map,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_threshold_cents(args: argparse.Namespace) -> float:
    if args.threshold is not None:
        return float(args.threshold) * CENTS_PER_DOLLAR
    return float(args.threshold_cents)


if __name__ == "__main__":  # pragma: no cover
    main()
