"""Generate rolling performance scoreboards."""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

LEDGER_PATH = Path("data/proc/ledger_all.parquet")
CALIBRATION_PATH = Path("data/proc/calibration_metrics.parquet")
ALPHA_STATE_PATH = Path("data/proc/state/fill_alpha.json")
ARTIFACTS_DIR = Path("reports/_artifacts")
SCORECARD_DIR = ARTIFACTS_DIR / "scorecards"
LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 7-day and 30-day performance scoreboards."
    )
    parser.add_argument(
        "--window",
        type=int,
        action="append",
        default=[7, 30],
        help="Rolling window in days (default: 7 and 30). Can be supplied multiple times.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ledger = _load_ledger()
    calibrations = _load_calibrations()
    alpha_state = _load_alpha_state()
    windows = {int(max(days, 1)) for days in args.window}
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    pilot_window = 7
    pilot_summary: list[dict[str, object]] | None = None
    for window in sorted(windows):
        summary = _build_summary(ledger, calibrations, alpha_state, window)
        output = reports_dir / f"scoreboard_{window}d.md"
        _write_markdown(summary, window, output)
        print(f"[scoreboard] wrote {output}")
        if window == pilot_window:
            pilot_summary = summary

    if pilot_summary is None:
        pilot_summary = _build_summary(ledger, calibrations, alpha_state, pilot_window)
    pilot_report = _build_pilot_readiness(
        pilot_summary,
        ledger,
        alpha_state,
        pilot_window,
        replay_metrics=_load_replay_metrics(),
    )
    pilot_path = reports_dir / "pilot_readiness.md"
    _write_pilot_readiness(pilot_report, pilot_path)
    print(f"[scoreboard] wrote {pilot_path}")


def _load_ledger() -> pl.DataFrame:
    if not LEDGER_PATH.exists():
        return pl.DataFrame()
    frame = pl.read_parquet(LEDGER_PATH)
    if "timestamp_et" in frame.columns and frame["timestamp_et"].dtype == pl.Utf8:
        frame = frame.with_columns(pl.col("timestamp_et").str.strptime(pl.Datetime, strict=False))
    return frame


def _load_calibrations() -> pl.DataFrame:
    if not CALIBRATION_PATH.exists():
        return pl.DataFrame()
    frame = pl.read_parquet(CALIBRATION_PATH)
    normalized = frame.rename({col: col.lower() for col in frame.columns})
    expected_cols = {"series", "crps_advantage", "brier_advantage"}
    if not expected_cols.issubset(set(normalized.columns)):
        return pl.DataFrame()
    return normalized


def _load_alpha_state() -> dict[str, float]:
    if not ALPHA_STATE_PATH.exists():
        return {}
    try:
        payload = json.loads(ALPHA_STATE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    series_map = payload.get("series", {})
    result: dict[str, float] = {}
    for key, value in series_map.items():
        try:
            result[key.upper()] = float(value.get("alpha", value))
        except (TypeError, ValueError, AttributeError):
            continue
    return result


def _build_summary(
    ledger: pl.DataFrame,
    calibrations: pl.DataFrame,
    alpha_state: dict[str, float],
    window_days: int,
) -> list[dict[str, object]]:
    now = datetime.now(tz=UTC)
    window_start = now - timedelta(days=window_days)
    gate_metrics = _load_gate_metrics(window_days)
    if ledger.is_empty():
        return []
    filtered = ledger
    if "timestamp_et" in ledger.columns:
        filtered = ledger.filter(pl.col("timestamp_et") >= window_start)
    if filtered.is_empty():
        return []
    grouped = filtered.group_by("series").agg(
        pl.sum("ev_after_fees").alias("ev_after_fees"),
        pl.sum("pnl_simulated").alias("realized_pnl"),
        pl.sum("expected_fills").alias("expected_fills"),
        pl.sum("size").alias("requested_contracts"),
        pl.mean("fill_ratio").alias("avg_fill_ratio"),
        pl.len().alias("sample_size"),
        (pl.col("ev_realized_bps") - pl.col("ev_expected_bps"))
        .mean()
        .alias("ev_delta_mean"),
        (pl.col("ev_realized_bps") - pl.col("ev_expected_bps"))
        .std()
        .alias("ev_delta_std"),
        pl.mean("ev_expected_bps").alias("ev_expected_bps_mean"),
        pl.mean("ev_realized_bps").alias("ev_realized_bps_mean"),
    )
    records: list[dict[str, object]] = []
    calib_lookup = {}
    if not calibrations.is_empty():
        calib_lookup = {
            row["series"].upper(): row
            for row in calibrations.to_dicts()
        }
    for row in grouped.to_dicts():
        series = str(row["series"]).upper()
        requested = row.get("requested_contracts") or 0.0
        expected_fills = row.get("expected_fills") or 0.0
        fill_ratio = (expected_fills / requested) if requested else 0.0
        avg_alpha = alpha_state.get(series)
        gate_stats = gate_metrics.get(series, {"go": 0, "no_go": 0})
        sample_size = int(row.get("sample_size") or 0)
        ev_expected_mean = float(row.get("ev_expected_bps_mean") or 0.0)
        ev_realized_mean = float(row.get("ev_realized_bps_mean") or 0.0)
        delta_mean = float(row.get("ev_delta_mean") or 0.0)
        delta_std = float(row.get("ev_delta_std") or 0.0)
        t_stat = 0.0
        if sample_size > 1 and delta_std > 0:
            t_stat = delta_mean / (delta_std / math.sqrt(sample_size))
        badge = _confidence_badge(sample_size, t_stat)
        ev_plot_lines = _ev_plot_lines(ev_expected_mean, ev_realized_mean)
        metrics = {
            "series": series,
            "ev_after_fees": row.get("ev_after_fees", 0.0),
            "realized_pnl": row.get("realized_pnl", 0.0),
            "expected_fills": expected_fills,
            "requested_contracts": requested,
            "fill_ratio": fill_ratio,
            "avg_fill_ratio": row.get("avg_fill_ratio", 0.0),
            "avg_alpha": avg_alpha,
            "no_go_count": gate_stats.get("no_go", 0),
            "go_count": gate_stats.get("go", 0),
            "sample_size": sample_size,
            "ev_expected_bps_mean": ev_expected_mean,
            "ev_realized_bps_mean": ev_realized_mean,
            "t_stat": t_stat,
            "confidence_badge": badge,
            "ev_plot_lines": ev_plot_lines,
        }
        calib = calib_lookup.get(series)
        if calib:
            metrics["crps_advantage"] = calib.get("crps_advantage")
            metrics["brier_advantage"] = calib.get("brier_advantage")
        records.append(metrics)
    return sorted(records, key=lambda row: row["series"])


def _load_gate_metrics(window_days: int) -> dict[str, dict[str, int]]:
    metrics: dict[str, dict[str, int]] = defaultdict(lambda: {"go": 0, "no_go": 0})
    if not ARTIFACTS_DIR.exists():
        return metrics
    threshold = datetime.now(tz=UTC) - timedelta(days=window_days)
    for path in ARTIFACTS_DIR.glob("go_no_go*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        timestamp_text = data.get("timestamp")
        if timestamp_text:
            try:
                parsed = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
                mtime = parsed.astimezone(UTC)
            except ValueError:
                pass
        if mtime < threshold:
            continue
        series = str(data.get("series") or data.get("mode") or "GLOBAL").upper()
        if data.get("go", True):
            metrics[series]["go"] += 1
        else:
            metrics[series]["no_go"] += 1
    return metrics


def _confidence_badge(sample_size: int, t_stat: float) -> str:
    if sample_size >= 200 and t_stat >= 2.0:
        return "✓"
    if t_stat >= 1.0:
        return "△"
    return "✗"


def _ev_plot_lines(expected: float, realized: float) -> list[str]:
    scale = max(abs(expected), abs(realized), 1.0)

    def _bar(label: str, value: float) -> str:
        proportion = min(1.0, abs(value) / scale)
        length = max(1, int(round(proportion * 20)))
        bar = "█" * length
        sign = "+" if value >= 0 else "-"
        return f"{label:<9}: {sign}{bar:<20} {value:.1f} bps"

    return ["```", _bar("expected", expected), _bar("realized", realized), "```"]


def _write_markdown(summary: list[dict[str, object]], window_days: int, output: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Scoreboard ({window_days}-day)")
    lines.append("")
    if not summary:
        lines.append("_No ledger data available for this window._")
    for row in summary:
        lines.append(f"## {row['series']}")
        lines.append(f"- CRPS Advantage: {row.get('crps_advantage', 'n/a')}")
        lines.append(f"- Brier Advantage: {row.get('brier_advantage', 'n/a')}")
        lines.append(f"- EV (after fees): {row['ev_after_fees']:.2f}")
        lines.append(f"- Realized PnL: {row['realized_pnl']:.2f}")
        lines.append(
            "- Expected vs Requested Fills: "
            f"{row['expected_fills']:.1f} / {row['requested_contracts']:.1f}"
        )
        lines.append(f"- Avg Fill Ratio: {row['avg_fill_ratio']:.3f}")
        lines.append(f"- Sample Size: {row.get('sample_size', 0)} trades")
        lines.append(
            f"- Expected EV (bps): {row.get('ev_expected_bps_mean', 0.0):+.1f}"
        )
        lines.append(
            f"- Realized EV (bps): {row.get('ev_realized_bps_mean', 0.0):+.1f}"
        )
        t_stat = row.get("t_stat")
        badge = row.get("confidence_badge", "✗")
        if isinstance(t_stat, float):
            lines.append(f"- Confidence: {badge} (t={t_stat:.2f})")
        plot_lines = row.get("ev_plot_lines")
        if plot_lines:
            lines.extend(plot_lines)
        alpha = row.get("avg_alpha")
        alpha_text = f"{alpha:.3f}" if isinstance(alpha, (int, float)) else "n/a"
        lines.append(f"- Avg α: {alpha_text}")
        lines.append(f"- NO-GO Count: {row['no_go_count']}")
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def _load_replay_metrics() -> dict[str, dict[str, float]]:
    if not SCORECARD_DIR.exists():
        return {}
    metrics: dict[str, dict[str, float]] = {}
    for path in SCORECARD_DIR.glob("*_summary.parquet"):
        try:
            frame = pl.read_parquet(path)
        except Exception as exc:  # pragma: no cover - best-effort logging
            LOGGER.debug("Failed to load replay metrics from %s: %s", path, exc)
            continue
        if frame.is_empty():
            continue
        series = path.stem.replace("_summary", "").upper()
        metrics[series] = {
            "mean_abs_cdf_delta": float(frame["mean_abs_cdf_delta"].mean()),
            "max_abs_cdf_delta": float(frame["max_abs_cdf_delta"].max()),
        }
    return metrics


def _build_pilot_readiness(
    summary: list[dict[str, object]],
    ledger: pl.DataFrame,
    alpha_state: dict[str, float],
    window_days: int,
    *,
    replay_metrics: dict[str, dict[str, float]] | None = None,
) -> dict[str, object]:
    replay_metrics = replay_metrics or {}

    per_series: list[dict[str, object]] = []
    total_go = 0
    total_no_go = 0
    total_ev = 0.0
    total_expected = 0.0
    total_requested = 0.0
    alpha_weight_sum = 0.0
    alpha_weight_den = 0.0

    for row in summary:
        series = str(row.get("series", "")).upper()
        go = int(row.get("go_count", 0) or 0)
        no_go = int(row.get("no_go_count", 0) or 0)
        total = go + no_go
        go_rate = (go / total) if total else None
        ev = float(row.get("ev_after_fees", 0.0) or 0.0)
        expected = float(row.get("expected_fills", 0.0) or 0.0)
        requested = float(row.get("requested_contracts", 0.0) or 0.0)
        observed_fill = (expected / requested) if requested else row.get("fill_ratio") or 0.0
        alpha = row.get("avg_alpha")
        fill_gap = None
        if isinstance(alpha, (int, float)):
            fill_gap = observed_fill - float(alpha)
            if requested > 0:
                alpha_weight_sum += float(alpha) * requested
                alpha_weight_den += requested

        replay = replay_metrics.get(series, {})
        mean_delta = replay.get("mean_abs_cdf_delta")
        max_delta = replay.get("max_abs_cdf_delta")

        per_series.append(
            {
                "series": series,
                "go": go,
                "no_go": no_go,
                "go_rate": go_rate,
                "ev_after_fees": ev,
                "fill_ratio": observed_fill,
                "alpha": alpha,
                "fill_gap": fill_gap,
                "replay_mean": mean_delta,
                "replay_max": max_delta,
            }
        )

        total_go += go
        total_no_go += no_go
        total_ev += ev
        total_expected += expected
        total_requested += requested

    total_decisions = total_go + total_no_go
    overall_go_rate = (total_go / total_decisions) if total_decisions else None
    overall_fill = (total_expected / total_requested) if total_requested else None
    overall_alpha = (alpha_weight_sum / alpha_weight_den) if alpha_weight_den else None
    replay_means = [
        entry["replay_mean"]
        for entry in per_series
        if isinstance(entry.get("replay_mean"), (int, float))
    ]
    replay_maxes = [
        entry["replay_max"]
        for entry in per_series
        if isinstance(entry.get("replay_max"), (int, float))
    ]
    overall_replay_mean = sum(replay_means) / len(replay_means) if replay_means else None
    overall_replay_max = max(replay_maxes) if replay_maxes else None

    return {
        "overall": {
            "go_rate": overall_go_rate,
            "go": total_go,
            "no_go": total_no_go,
            "ev_after_fees": total_ev,
            "fill_ratio": overall_fill,
            "alpha": overall_alpha,
            "replay_mean": overall_replay_mean,
            "replay_max": overall_replay_max,
        },
        "series": sorted(per_series, key=lambda row: row["series"]),
    }


def _write_pilot_readiness(report: dict[str, object], output: Path) -> None:  # noqa: PLR0915
    lines: list[str] = ["# Pilot Readiness (last 7 days)", ""]
    overall = report.get("overall", {})
    series_rows: list[dict[str, object]] = report.get("series", [])  # type: ignore[assignment]

    if not series_rows:
        lines.append("_No execution data available in the last 7 days._")
    else:
        go_rate = overall.get("go_rate")
        go_total = overall.get("go", 0)
        no_go_total = overall.get("no_go", 0)
        decisions = go_total + no_go_total
        if isinstance(go_rate, (int, float)) and decisions:
            lines.append(f"- GO Rate: {go_rate:.1%} ({go_total}/{decisions})")
        lines.append(f"- EV after fees: {overall.get('ev_after_fees', 0.0):.2f} USD")
        fill_ratio = overall.get("fill_ratio")
        alpha = overall.get("alpha")
        if isinstance(fill_ratio, (int, float)) and isinstance(alpha, (int, float)):
            lines.append(f"- Fill realism: observed {fill_ratio:.3f} vs α {alpha:.3f}")
        elif isinstance(fill_ratio, (int, float)):
            lines.append(f"- Fill realism: observed {fill_ratio:.3f}")
        replay_mean = overall.get("replay_mean")
        replay_max = overall.get("replay_max")
        if isinstance(replay_mean, (int, float)) and isinstance(replay_max, (int, float)):
            lines.append(f"- Replay mean Δ: {replay_mean:.4f} (max {replay_max:.4f})")
        lines.append("")
        header = (
            "| Series | GO Rate | GO/Total | EV after fees | Fill ratio | α | "
            "Δfill | Replay mean Δ | Replay max Δ |"
        )
        lines.append(header)
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in series_rows:
            go = int(row.get("go", 0))
            no_go = int(row.get("no_go", 0))
            total = go + no_go
            go_rate_row = row.get("go_rate")
            go_rate_text = f"{go_rate_row:.1%}" if isinstance(go_rate_row, (int, float)) else "n/a"
            ev_text = f"{row.get('ev_after_fees', 0.0):.2f}"
            fill_text = f"{row.get('fill_ratio', 0.0):.3f}"
            alpha = row.get("alpha")
            alpha_text = f"{alpha:.3f}" if isinstance(alpha, (int, float)) else "n/a"
            gap = row.get("fill_gap")
            gap_text = f"{gap:+.3f}" if isinstance(gap, (int, float)) else "n/a"
            replay_mean = row.get("replay_mean")
            if isinstance(replay_mean, (int, float)):
                replay_mean_text = f"{replay_mean:.4f}"
            else:
                replay_mean_text = "n/a"
            replay_max = row.get("replay_max")
            if isinstance(replay_max, (int, float)):
                replay_max_text = f"{replay_max:.4f}"
            else:
                replay_max_text = "n/a"
            total_text = f"{go}/{total}" if total else "0/0"
            series_name = row.get("series", "-")
            row_line = (
                f"| {series_name} | {go_rate_text} | {total_text} | {ev_text} | "
                f"{fill_text} | {alpha_text} | {gap_text} | {replay_mean_text} | "
                f"{replay_max_text} |"
            )
            lines.append(row_line)

    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
