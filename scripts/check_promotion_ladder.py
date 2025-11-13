#!/usr/bin/env python3
"""Evaluate promotion readiness for the index size ladder."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

import polars as pl

from kalshi_alpha.config import size_ladder as size_ladder_config

ET = ZoneInfo("America/New_York")
DEFAULT_ARTIFACT_DIR = Path("reports/_artifacts")
DEFAULT_SLO_PATH = DEFAULT_ARTIFACT_DIR / "monitors" / "slo_selfcheck.json"
DEFAULT_SIZE_CONFIG = Path("configs/size_ladder.yaml")
PNL_GLOB = "pnl_window_*.parquet"
GREEN_DAYS_REQUIREMENTS = {"B": 7, "C": 14, "D": 30}
DELTA_EV_THRESHOLD = 3.0  # cents per lot
FILL_GAP_THRESHOLD = 10.0  # percentage points
HONESTY_MIN_RATIO = 1.0
LOSS_CAP_THRESHOLD = -50.0  # USD per day


@dataclass(slots=True)
class DayAggregate:
    day: date
    pnl: float
    ev: float
    delta_cents: list[float]
    fill_gaps: list[float]
    var_headroom_pct: list[float]


@dataclass(slots=True)
class EvaluationResult:
    current_stage: str
    recommended_stage: str
    stage_status: dict[str, bool]
    notes: list[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check promotion readiness for the size ladder.")
    parser.add_argument("--lookback", type=int, default=30, help="Days of history to include (default: 30)")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing pnl_window_*.parquet files",
    )
    parser.add_argument(
        "--slo",
        type=Path,
        default=DEFAULT_SLO_PATH,
        help="Path to slo_selfcheck.json",
    )
    parser.add_argument(
        "--size-config",
        type=Path,
        default=DEFAULT_SIZE_CONFIG,
        help="Size ladder configuration path",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=["INXU", "NASDAQ100U", "INX", "NASDAQ100"],
        help="Series to evaluate (default: all index families)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ladder = size_ladder_config.load_size_ladder(args.size_config)
    target_series = {s.upper() for s in args.series}
    pnl = _load_pnl_history(args.artifacts, target_series, lookback=max(args.lookback, 1))
    slo = _load_slo_metrics(args.slo)
    result = _evaluate(ladder, pnl, slo, target_series)
    _print_result(result)
    return 0


def _load_pnl_history(
    artifacts_dir: Path,
    target_series: set[str],
    *,
    lookback: int,
) -> list[DayAggregate]:
    today = datetime.now(tz=ET).date()
    cutoff = today - timedelta(days=lookback * 2)
    aggregates: dict[date, DayAggregate] = {}
    for path in sorted(artifacts_dir.glob(PNL_GLOB)):
        day = _extract_day(path)
        if day is None or day < cutoff:
            continue
        frame = pl.read_parquet(path)
        day_rows = frame.filter(pl.col("scope") == "day")
        for row in day_rows.iter_rows(named=True):
            series = str(row.get("series") or "").upper()
            if series not in target_series:
                continue
            aggregate = aggregates.get(day)
            if aggregate is None:
                aggregate = DayAggregate(day=day, pnl=0.0, ev=0.0, delta_cents=[], fill_gaps=[], var_headroom_pct=[])
                aggregates[day] = aggregate
            aggregate.pnl += float(row.get("pnl_realized") or 0.0)
            aggregate.ev += float(row.get("ev_after_fees") or 0.0)
            aggregate.delta_cents.append(float(row.get("delta_ev_cents_per_lot") or 0.0))
            aggregate.fill_gaps.append(float(row.get("fill_gap_pp") or 0.0))
            aggregate.var_headroom_pct.append(float(row.get("var_headroom_pct") or 100.0))
    return [aggregates[key] for key in sorted(aggregates)]


def _extract_day(path: Path) -> date | None:
    stem = path.stem
    if not stem.startswith("pnl_window_"):
        return None
    try:
        return date.fromisoformat(stem.replace("pnl_window_", ""))
    except ValueError:
        return None


def _load_slo_metrics(path: Path) -> dict[str, Mapping[str, object]]:
    if not path.exists():
        return {}
    metrics: dict[str, Mapping[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        series = str(payload.get("series") or "").upper()
        if series:
            metrics[series] = payload
    return metrics


def _evaluate(
    ladder: size_ladder_config.SizeLadderConfig,
    history: list[DayAggregate],
    slo_metrics: Mapping[str, Mapping[str, object]],
    target_series: set[str],
) -> EvaluationResult:
    stage_order = sorted(ladder.stages)
    stage_status: dict[str, bool] = {}
    recommended = ladder.current_stage
    notes: list[str] = []
    available_dates = [entry.day for entry in history]
    for stage in stage_order:
        if stage == "A":
            stage_status[stage] = True
            continue
        required = GREEN_DAYS_REQUIREMENTS.get(stage, 0)
        ok, detail = _meets_stage(stage, required, history, slo_metrics, target_series)
        stage_status[stage] = ok
        notes.append(f"{stage}: {detail}")
        if ok:
            recommended = stage
    return EvaluationResult(
        current_stage=ladder.current_stage,
        recommended_stage=recommended,
        stage_status=stage_status,
        notes=notes,
    )


def _meets_stage(
    stage: str,
    required_days: int,
    history: list[DayAggregate],
    slo_metrics: Mapping[str, Mapping[str, object]],
    target_series: set[str],
) -> tuple[bool, str]:
    if required_days <= 0:
        return True, "no lookback required"
    if len(history) < required_days:
        return False, f"need {required_days} green days, have {len(history)}"
    window = history[-required_days:]
    green = sum(1 for entry in window if entry.pnl >= 0.0)
    if green < required_days:
        return False, f"green days {green}/{required_days}"
    total_pnl = sum(entry.pnl for entry in window)
    total_ev = sum(entry.ev for entry in window)
    honesty = (total_pnl / total_ev) if total_ev > 0 else float("inf")
    if total_ev > 0 and honesty < HONESTY_MIN_RATIO:
        return False, f"honesty {honesty:.2f}< {HONESTY_MIN_RATIO:.0f}"
    max_delta = max(abs(value) for entry in window for value in entry.delta_cents or [0.0])
    if max_delta > DELTA_EV_THRESHOLD:
        return False, f"|ΔEV| {max_delta:.2f}c exceeds {DELTA_EV_THRESHOLD}c"
    max_fill = max(abs(value) for entry in window for value in entry.fill_gaps or [0.0])
    if max_fill > FILL_GAP_THRESHOLD:
        return False, f"|fill_gap| {max_fill:.1f}pp exceeds {FILL_GAP_THRESHOLD}pp"
    min_var = min((min(entry.var_headroom_pct or [100.0]) for entry in window), default=100.0)
    if min_var <= 0.0:
        return False, "VaR headroom exhausted"
    if any(entry.pnl <= LOSS_CAP_THRESHOLD for entry in window):
        return False, "hit PAL/stop"
    for series in target_series:
        reasons = slo_metrics.get(series, {}).get("no_go_reasons")
        if reasons:
            return False, f"NO-GO for {series}"
    return True, f"ready (honesty {honesty:.2f}, ΔEV {max_delta:.2f}c, fill_gap {max_fill:.1f}pp)"


def _print_result(result: EvaluationResult) -> None:
    print(f"Current stage: {result.current_stage}")
    print(f"Recommended stage: {result.recommended_stage}")
    print("")
    for stage, ok in sorted(result.stage_status.items()):
        status = "PASS" if ok else "HOLD"
        print(f"{stage}: {status}")
    if result.notes:
        print("")
        for note in result.notes:
            print(f"- {note}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
