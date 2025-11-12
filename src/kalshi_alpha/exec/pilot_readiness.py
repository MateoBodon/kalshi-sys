"""Compute pilot readiness for index ladders based on recent paper fills."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import polars.datatypes as pldt

from kalshi_alpha.exec.monitors.freshness import (
    FRESHNESS_ARTIFACT_PATH,
)
from kalshi_alpha.exec.monitors.freshness import (
    load_artifact as load_freshness_artifact,
)
from kalshi_alpha.exec.monitors.freshness import (
    summarize_artifact as summarize_freshness_artifact,
)

LEDGER_PATH = Path("data/proc/ledger_all.parquet")
REPORT_PATH = Path("reports/pilot_readiness.md")
INDEX_SERIES = ("INXU", "NASDAQ100U", "INX", "NASDAQ100")

WINDOW_DAYS_DEFAULT = 14
MIN_FILLS = 300.0
MIN_DELTA_BPS = 6.0
MIN_T_STAT = 2.0
MAX_ALPHA_GAP = 0.05
MAX_CALIBRATION_AGE_DAYS = 14.0

CALIBRATION_ROOT = Path("data/proc/calib/index")
CALIBRATION_TARGETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "INXU": ("spx", ("hourly", "noon")),
    "NASDAQ100U": ("ndx", ("hourly", "noon")),
    "INX": ("spx", ("close",)),
    "NASDAQ100": ("ndx", ("close",)),
}


@dataclass(slots=True)
class SeriesReadiness:
    series: str
    fills: float
    delta_bps: float
    t_stat: float
    alpha_gap: float
    sample_size: int
    reasons: tuple[str, ...]

    @property
    def go(self) -> bool:
        return not self.reasons


def _load_ledger(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Ledger parquet not found at {path}")
    frame = pl.read_parquet(path)
    if "timestamp_et" in frame.columns and frame["timestamp_et"].dtype == pl.Utf8:
        frame = frame.with_columns(
            pl.col("timestamp_et").str.strptime(
                pl.Datetime(time_zone="UTC"),
                format="%Y-%m-%dT%H:%M:%S%z",
                strict=False,
            )
        )
    return frame


def _series_fills(subset: pl.DataFrame) -> float:
    columns = set(subset.columns)
    if {"fill_ratio_observed", "size"}.issubset(columns):
        df = subset.select(
            (pl.col("fill_ratio_observed").fill_null(0.0) * pl.col("size").fill_null(0.0)).sum()
        )
        return float(df[0, 0]) if df.height else 0.0
    if "expected_fills" in columns:
        df = subset.select(pl.col("expected_fills").fill_null(0.0).sum())
        return float(df[0, 0]) if df.height else 0.0
    return 0.0


def _delta_stats(subset: pl.DataFrame) -> tuple[float, float]:
    columns = set(subset.columns)
    if {"ev_realized_bps", "ev_expected_bps"}.issubset(columns):
        delta_df = subset.select(
            (pl.col("ev_realized_bps") - pl.col("ev_expected_bps")).alias("delta")
        )
        delta_series = delta_df["delta"].fill_null(0.0)
        if delta_series.len():
            delta_mean = float(delta_series.mean())
            delta_std = float(delta_series.std(ddof=1)) if delta_series.len() > 1 else 0.0
            return delta_mean, delta_std
    return 0.0, 0.0


def _alpha_gap_mean(subset: pl.DataFrame) -> float:
    if "fill_ratio_observed" not in subset.columns:
        return 0.0
    observed = subset.get_column("fill_ratio_observed").fill_null(0.0)
    if not observed.len():
        return 0.0
    if "alpha_target" in subset.columns:
        target = subset.get_column("alpha_target").fill_null(0.0)
    elif "fill_ratio" in subset.columns:
        target = subset.get_column("fill_ratio").fill_null(0.0)
    else:
        return 0.0
    if not target.len():
        return 0.0
    diff = observed - target
    return float(diff.mean()) if diff.len() else 0.0


def _file_age_days(path: Path, now: datetime) -> float | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        generated_at = payload.get("generated_at")
        if isinstance(generated_at, str) and generated_at:
            generated_dt = datetime.fromisoformat(generated_at)
            if generated_dt.tzinfo is None:
                generated_dt = generated_dt.replace(tzinfo=UTC)
            age_seconds = (now.astimezone(UTC) - generated_dt.astimezone(UTC)).total_seconds()
            if age_seconds >= 0:
                return age_seconds / 86400.0
    except (json.JSONDecodeError, ValueError):
        pass
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    age_seconds = (now.astimezone(UTC) - mtime).total_seconds()
    return age_seconds / 86400.0 if age_seconds >= 0 else 0.0


def calibration_age_days(series: str, now: datetime) -> float | None:
    target = CALIBRATION_TARGETS.get(series.upper())
    if target is None:
        return None
    slug, horizons = target
    ages: list[float] = []
    for horizon in horizons:
        base_dir = CALIBRATION_ROOT / slug / horizon
        candidate_paths: list[Path] = []
        aggregated = base_dir / "params.json"
        candidate_paths.append(aggregated)
        if horizon == "hourly" and base_dir.exists():
            for child in base_dir.iterdir():
                if child.is_dir():
                    candidate_paths.append(child / "params.json")
        for path in candidate_paths:
            if not path.exists():
                continue
            age = _file_age_days(path, now)
            if age is not None:
                ages.append(age)
    if ages:
        return min(ages)
    return None


def freshness_status() -> tuple[bool, list[str]]:
    payload = load_freshness_artifact(FRESHNESS_ARTIFACT_PATH)
    summary = summarize_freshness_artifact(payload, artifact_path=FRESHNESS_ARTIFACT_PATH)
    status = summary.get("status", "MISSING")
    ok = bool(summary.get("required_feeds_ok", False)) and status != "MISSING"
    reasons: list[str] = []
    if status == "MISSING":
        reasons.append("data_freshness_missing")
    if not ok and status != "MISSING":
        reasons.append("data_freshness_alert")
    stale_feeds = summary.get("stale_feeds") or []
    for feed in stale_feeds:
        reasons.append(f"stale_feed:{feed}")
    return ok, reasons


def evaluate_readiness(
    frame: pl.DataFrame,
    *,
    now: datetime | None = None,
    window_days: int = WINDOW_DAYS_DEFAULT,
) -> list[SeriesReadiness]:
    now = now or datetime.now(tz=UTC)
    window_start = now - timedelta(days=max(window_days, 1))
    filtered = frame
    if "timestamp_et" in filtered.columns:
        ts_dtype = filtered.schema.get("timestamp_et")
        target_start = window_start
        if (
            isinstance(ts_dtype, pldt.Datetime)
            and ts_dtype.time_zone is None
            and window_start.tzinfo is not None
        ):
            target_start = window_start.replace(tzinfo=None)
        filtered = filtered.filter(pl.col("timestamp_et") >= target_start)

    results: list[SeriesReadiness] = []
    for series in INDEX_SERIES:
        subset = filtered.filter(pl.col("series") == series)
        if subset.is_empty():
            results.append(
                SeriesReadiness(
                    series=series,
                    fills=0.0,
                    delta_bps=0.0,
                    t_stat=0.0,
                    alpha_gap=0.0,
                    sample_size=0,
                    reasons=("insufficient_data",),
                )
            )
            continue

        sample_size = subset.height
        fills = _series_fills(subset)
        delta_mean, delta_std = _delta_stats(subset)
        if sample_size > 1 and delta_std > 0.0:
            t_stat = delta_mean / (delta_std / math.sqrt(sample_size))
        else:
            t_stat = 0.0
        alpha_gap = _alpha_gap_mean(subset)

        reasons: list[str] = []
        if fills < MIN_FILLS:
            reasons.append(f"fills {fills:.0f} < {int(MIN_FILLS)}")
        if delta_mean < MIN_DELTA_BPS:
            reasons.append(f"Δbps {delta_mean:.1f} < {MIN_DELTA_BPS}")
        if t_stat < MIN_T_STAT:
            reasons.append(f"t-stat {t_stat:.2f} < {MIN_T_STAT}")
        if abs(alpha_gap) > MAX_ALPHA_GAP:
            reasons.append(f"fill-α gap {alpha_gap:+.3f} exceeds {MAX_ALPHA_GAP}")
        calib_age = calibration_age_days(series, now)
        if calib_age is None:
            reasons.append("calibration_missing")
        elif calib_age > MAX_CALIBRATION_AGE_DAYS:
            reasons.append(f"calibration_age {calib_age:.1f}d > {MAX_CALIBRATION_AGE_DAYS:.0f}d")

        results.append(
            SeriesReadiness(
                series=series,
                fills=fills,
                delta_bps=delta_mean,
                t_stat=t_stat,
                alpha_gap=alpha_gap,
                sample_size=sample_size,
                reasons=tuple(reasons),
            )
        )
    return results


def render_markdown(
    results: list[SeriesReadiness],
    *,
    window_days: int = WINDOW_DAYS_DEFAULT,
    freshness_ok: bool = True,
    freshness_reasons: Iterable[str] | None = None,
) -> str:
    lines: list[str] = [f"# Pilot Readiness ({window_days}-day)", ""]
    go_count = sum(1 for entry in results if entry.go)
    lines.append(f"GO series: {go_count}/{len(results)}")
    lines.append("")
    lines.append(f"Data Freshness: {'OK' if freshness_ok else 'NO-GO'}")
    reasons_iter = list(freshness_reasons or [])
    if reasons_iter:
        lines.append("- Freshness Reasons:")
        for item in reasons_iter:
            lines.append(f"  - {item}")
    lines.append("")
    for entry in results:
        status = "GO" if entry.go else "NO-GO"
        lines.append(f"## {entry.series} — {status}")
        lines.append(f"- Fills: {entry.fills:.0f}")
        lines.append(f"- Δbps: {entry.delta_bps:.1f}")
        lines.append(f"- t-stat: {entry.t_stat:.2f}")
        lines.append(f"- Fill - α: {entry.alpha_gap:+.3f}")
        lines.append(f"- Sample Size: {entry.sample_size}")
        if entry.reasons:
            lines.append("- Reasons:")
            for reason in entry.reasons:
                lines.append(f"  - {reason}")
        else:
            lines.append("- Reasons: none")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def generate_report(
    *,
    ledger_path: Path = LEDGER_PATH,
    output_path: Path = REPORT_PATH,
    window_days: int = WINDOW_DAYS_DEFAULT,
    now: datetime | None = None,
) -> list[SeriesReadiness]:
    ledger = _load_ledger(ledger_path)
    results = evaluate_readiness(ledger, now=now, window_days=window_days)
    freshness_ok, freshness_reasons = freshness_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_markdown(
            results,
            window_days=window_days,
            freshness_ok=freshness_ok,
            freshness_reasons=freshness_reasons,
        ),
        encoding="utf-8",
    )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute pilot readiness for index series.")
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH, help="Ledger parquet path")
    parser.add_argument("--output", type=Path, default=REPORT_PATH, help="Markdown output path")
    parser.add_argument("--window", type=int, default=WINDOW_DAYS_DEFAULT, help="Lookback window in days")
    args = parser.parse_args(argv)

    results = generate_report(
        ledger_path=args.ledger,
        output_path=args.output,
        window_days=args.window,
    )
    freshness_ok, freshness_reasons = freshness_status()
    go_series = [entry.series for entry in results if entry.go]
    no_go_series = [entry.series for entry in results if not entry.go]
    print(f"[pilot_readiness] wrote {args.output}")
    if go_series:
        print(f"  GO: {', '.join(go_series)}")
    if no_go_series:
        print(f"  NO-GO: {', '.join(no_go_series)}")
    if not freshness_ok:
        suffix = ", ".join(freshness_reasons) if freshness_reasons else "unknown"
        print(f"  Freshness NO-GO: {suffix}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
