#!/usr/bin/env python3
"""Fail fast if replay vs live EV parity drifts beyond the allowed epsilon."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

REPLAY_PATH = Path("reports/_artifacts/replay_ev.parquet")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ΔEV parity between replay and live proposals.")
    parser.add_argument("--threshold", type=float, default=0.15, help="Max allowed per-contract delta (USD).")
    parser.add_argument("--path", type=Path, default=REPLAY_PATH, help="Replay EV parquet path (default: reports/_artifacts/replay_ev.parquet).")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
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
    per_contract_original = (pl.col("maker_ev_original") / pl.col("contracts").clip(lower_bound=1)).alias("maker_ev_per_contract_original")
    delta_col = (
        (pl.col("maker_ev_per_contract_replay") - per_contract_original)
        .abs()
        .alias("delta")
    )
    enriched = frame.with_columns([per_contract_original, delta_col])
    max_delta = float(enriched.select(pl.max("delta")).item())
    print(f"[parity] max ΔEV per contract = {max_delta:.4f} (threshold {args.threshold:.4f})")
    if max_delta > args.threshold:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
