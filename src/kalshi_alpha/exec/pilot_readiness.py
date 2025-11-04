"""Compute pilot readiness for index ladders based on recent paper fills."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

LEDGER_PATH = Path("data/proc/ledger_all.parquet")
REPORT_PATH = Path("reports/pilot_readiness.md")
INDEX_SERIES = ("INXU", "NASDAQ100U", "INX", "NASDAQ100")

WINDOW_DAYS_DEFAULT = 14
MIN_FILLS = 300.0
MIN_DELTA_BPS = 6.0
MIN_T_STAT = 2.0
MAX_ALPHA_GAP = 0.05


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
        filtered = filtered.filter(pl.col("timestamp_et") >= window_start)

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


def render_markdown(results: list[SeriesReadiness], *, window_days: int = WINDOW_DAYS_DEFAULT) -> str:
    lines: list[str] = [f"# Pilot Readiness ({window_days}-day)", ""]
    go_count = sum(1 for entry in results if entry.go)
    lines.append(f"GO series: {go_count}/{len(results)}")
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(results, window_days=window_days), encoding="utf-8")
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
    go_series = [entry.series for entry in results if entry.go]
    no_go_series = [entry.series for entry in results if not entry.go]
    print(f"[pilot_readiness] wrote {args.output}")
    if go_series:
        print(f"  GO: {', '.join(go_series)}")
    if no_go_series:
        print(f"  NO-GO: {', '.join(no_go_series)}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
