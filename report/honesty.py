"""Compute reliability, Brier, and ECE metrics for EV honesty clamps."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Sequence

import polars as pl

from kalshi_alpha.core.backtest import reliability_table

LEDGER_PATH = Path("data/proc/ledger_all.parquet")
ARTIFACT_ROOT = Path("reports/_artifacts/honesty")
CLAMP_ARTIFACT = ARTIFACT_ROOT / "honesty_clamp.json"
INDEX_SERIES = ("INXU", "NASDAQ100U", "INX", "NASDAQ100")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate honesty metrics and clamps.")
    parser.add_argument(
        "--window",
        type=int,
        action="append",
        default=[7, 30],
        help="Rolling window in days (default: 7 and 30).",
    )
    parser.add_argument(
        "--buckets",
        type=int,
        default=10,
        help="Number of reliability buckets (default: %(default)s).",
    )
    parser.add_argument(
        "--min-sample",
        type=int,
        default=50,
        help="Minimum trades per series to compute metrics (default: %(default)s).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ledger = _load_ledger()
    if ledger.is_empty():
        raise SystemExit("ledger_all.parquet missing or empty")
    windows = sorted({max(int(days), 1) for days in args.window})
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    clamp_payload: dict[str, object] | None = None
    for window in windows:
        summary = _window_summary(ledger, window, args.buckets, args.min_sample)
        if not summary["series"]:
            continue
        path = ARTIFACT_ROOT / f"honesty_window{window}.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[honesty] wrote {path}")
        if clamp_payload is None:
            clamp_payload = summary
    if clamp_payload:
        CLAMP_ARTIFACT.write_text(json.dumps(clamp_payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[honesty] wrote clamp artifact {CLAMP_ARTIFACT}")


def _load_ledger() -> pl.DataFrame:
    if not LEDGER_PATH.exists():
        return pl.DataFrame()
    frame = pl.read_parquet(LEDGER_PATH)
    if frame.is_empty():
        return frame
    if "timestamp_et" in frame.columns and frame["timestamp_et"].dtype == pl.Utf8:
        frame = frame.with_columns(
            pl.col("timestamp_et").str.strptime(pl.Datetime(time_zone="UTC"), strict=False)
        )
    return frame


def _window_summary(
    ledger: pl.DataFrame,
    window_days: int,
    buckets: int,
    min_sample: int,
) -> dict[str, object]:
    now = datetime.now(tz=UTC)
    cutoff = now - timedelta(days=window_days)
    if "timestamp_et" in ledger.columns:
        subset = ledger.filter(pl.col("timestamp_et") >= cutoff)
    else:
        subset = ledger
    payload = {
        "generated_at": now.isoformat(),
        "window_days": window_days,
        "series": {},
    }
    if subset.is_empty():
        return payload
    for series in INDEX_SERIES:
        series_subset = subset.filter(pl.col("series") == series)
        metrics = _series_metrics(series_subset, buckets, min_sample)
        if metrics is None:
            continue
        payload["series"][series] = metrics
    return payload


def _series_metrics(
    frame: pl.DataFrame,
    buckets: int,
    min_sample: int,
) -> dict[str, object] | None:
    if frame.is_empty():
        return None
    usable = frame.drop_nulls(["model_p", "pnl_simulated", "side"])
    if usable.is_empty():
        return None
    model_probs = usable["model_p"].cast(pl.Float64)
    probs = [float(max(0.0, min(1.0, value))) for value in model_probs.to_list()]
    outcomes = _infer_outcomes(usable)
    total = len(outcomes)
    if total < min_sample:
        return None
    brier = sum((p - o) ** 2 for p, o in zip(probs, outcomes, strict=True)) / total
    table = reliability_table(probs, outcomes, buckets=buckets)
    ece = _expected_calibration_error(table, total)
    clamp = _clamp_from_metrics(brier, ece, total)
    return {
        "sample_size": total,
        "brier": brier,
        "ece": ece,
        "clamp": clamp,
        "reliability": table,
    }


def _infer_outcomes(frame: pl.DataFrame) -> list[int]:
    results: list[int] = []
    for side, pnl in frame.select(["side", "pnl_simulated"]).iter_rows():
        pnl_value = float(pnl or 0.0)
        side_value = str(side or "YES").upper()
        if side_value == "YES":
            outcome = 1 if pnl_value >= 0 else 0
        else:
            outcome = 0 if pnl_value >= 0 else 1
        results.append(outcome)
    return results


def _expected_calibration_error(table: list[dict], total: int) -> float:
    if not table or total <= 0:
        return 0.0
    error = 0.0
    for bucket in table:
        count = int(bucket.get("count") or 0)
        if count <= 0:
            continue
        avg_probability = float(bucket.get("avg_probability") or 0.0)
        event_rate = float(bucket.get("event_rate") or 0.0)
        error += abs(event_rate - avg_probability) * (count / total)
    return error


def _clamp_from_metrics(brier: float, ece: float, sample_size: int) -> float:
    if sample_size < 100:
        return 1.0
    if ece <= 0.02 and brier <= 0.08:
        return 1.0
    if ece <= 0.05 and brier <= 0.12:
        return 0.75
    return 0.5


if __name__ == "__main__":  # pragma: no cover
    main()
