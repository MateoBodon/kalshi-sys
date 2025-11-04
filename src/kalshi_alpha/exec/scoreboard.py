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

from kalshi_alpha.core.execution import defaults as execution_defaults
from kalshi_alpha.core.execution.index_models import (
    AlphaCurve,
    SlippageCurve,
    load_alpha_curve,
    load_slippage_curve,
)
from kalshi_alpha.exec import pilot_readiness

LEDGER_PATH = Path("data/proc/ledger_all.parquet")
CALIBRATION_PATH = Path("data/proc/calibration_metrics.parquet")
ALPHA_STATE_PATH = Path("data/proc/state/fill_alpha.json")
ARTIFACTS_DIR = Path("reports/_artifacts")
LOGGER = logging.getLogger(__name__)
INDEX_SERIES_ORDER = ["INXU", "NASDAQ100U", "INX", "NASDAQ100"]
INDEX_SERIES = set(INDEX_SERIES_ORDER)


def _alpha_model_metrics(
    curve: AlphaCurve | None,
    subset: pl.DataFrame | None,
) -> tuple[float | None, float | None]:
    if curve is None or subset is None or subset.is_empty():
        return None, None
    required = {"depth_fraction", "delta_p", "minutes_to_event", "fill_ratio_observed"}
    if not required.issubset(set(subset.columns)):
        return None, None
    diffs: list[float] = []
    for depth, delta_p, minutes, observed in subset.select(
        ["depth_fraction", "delta_p", "minutes_to_event", "fill_ratio_observed"]
    ).iter_rows():
        pred = curve.predict(
            depth_fraction=float(depth or 0.0),
            delta_p=float(delta_p or 0.0),
            tau_minutes=float(minutes or 0.0),
        )
        obs = float(observed) if observed is not None else 0.0
        diffs.append(obs - pred)
    if not diffs:
        return None, None
    count = len(diffs)
    mean_diff = sum(diffs) / count
    abs_diff = sum(abs(value) for value in diffs) / count
    return mean_diff, abs_diff


def _slippage_model_metrics(
    curve: SlippageCurve | None,
    subset: pl.DataFrame | None,
) -> tuple[float | None, float | None]:
    if curve is None or subset is None or subset.is_empty():
        return None, None
    required = {"depth_fraction", "spread", "minutes_to_event", "slippage_ticks"}
    if not required.issubset(set(subset.columns)):
        return None, None
    diffs: list[float] = []
    for depth, spread, minutes, observed in subset.select(
        ["depth_fraction", "spread", "minutes_to_event", "slippage_ticks"]
    ).iter_rows():
        pred = curve.predict_ticks(
            depth_fraction=float(depth or 0.0),
            spread=float(spread or 0.0),
            tau_minutes=float(minutes or 0.0),
        )
        obs = abs(float(observed) if observed is not None else 0.0)
        diffs.append(obs - pred)
    if not diffs:
        return None, None
    count = len(diffs)
    mean_diff = sum(diffs) / count
    abs_diff = sum(abs(value) for value in diffs) / count
    return mean_diff, abs_diff


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
    for window in sorted(windows):
        summary = _build_summary(ledger, calibrations, alpha_state, window)
        output = reports_dir / f"scoreboard_{window}d.md"
        _write_markdown(summary, window, output)
        print(f"[scoreboard] wrote {output}")
    pilot_path = reports_dir / "pilot_readiness.md"
    pilot_readiness.generate_report(
        ledger_path=LEDGER_PATH,
        output_path=pilot_path,
        window_days=pilot_readiness.WINDOW_DAYS_DEFAULT,
    )
    print(f"[scoreboard] wrote {pilot_path}")


def _load_ledger() -> pl.DataFrame:
    if not LEDGER_PATH.exists():
        return pl.DataFrame()
    frame = pl.read_parquet(LEDGER_PATH)
    if "timestamp_et" in frame.columns and frame["timestamp_et"].dtype == pl.Utf8:
        frame = frame.with_columns(
            pl.col("timestamp_et").str.strptime(
                pl.Datetime(time_zone="UTC"),
                format="%Y-%m-%dT%H:%M:%S%z",
                strict=False,
            )
        )
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


def _build_summary(  # noqa: PLR0912, PLR0915
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
    if "slippage_ticks" not in filtered.columns:
        filtered = filtered.with_columns(pl.lit(0.0).alias("slippage_ticks"))
    if "fill_ratio_observed" not in filtered.columns:
        if "fill_ratio" in filtered.columns:
            filtered = filtered.with_columns(pl.col("fill_ratio").alias("fill_ratio_observed"))
        else:
            filtered = filtered.with_columns(pl.lit(0.0).alias("fill_ratio_observed"))
    if "timestamp_et" in filtered.columns:
        filtered = filtered.filter(pl.col("timestamp_et") >= window_start)
    if filtered.is_empty():
        return []
    frames_by_series = {
        series: filtered.filter(pl.col("series") == series)
        for series in INDEX_SERIES
    }
    alpha_curves = {series: load_alpha_curve(series) for series in INDEX_SERIES}
    slippage_curves = {series: load_slippage_curve(series) for series in INDEX_SERIES}
    grouped = filtered.group_by("series").agg(
        pl.sum("ev_after_fees").alias("ev_after_fees"),
        pl.sum("pnl_simulated").alias("realized_pnl"),
        pl.sum("expected_fills").alias("expected_fills"),
        pl.sum("size").alias("requested_contracts"),
        pl.mean("fill_ratio_observed").alias("avg_fill_ratio_observed"),
        pl.mean("fill_ratio").alias("avg_alpha_target"),
        pl.mean("slippage_ticks").alias("slippage_ticks_mean"),
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
        if series not in INDEX_SERIES:
            continue
        requested = row.get("requested_contracts") or 0.0
        expected_fills = row.get("expected_fills") or 0.0
        fill_ratio = float(row.get("avg_fill_ratio_observed") or 0.0)
        if fill_ratio == 0.0 and requested:
            fill_ratio = (expected_fills / requested)
        avg_alpha = row.get("avg_alpha_target")
        if avg_alpha is None:
            avg_alpha = alpha_state.get(series)
        if avg_alpha is None:
            avg_alpha = execution_defaults.default_alpha(series)
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
        slippage_ticks_mean = float(row.get("slippage_ticks_mean") or 0.0)
        fill_minus_alpha = None
        if isinstance(avg_alpha, (int, float)) and avg_alpha is not None:
            fill_minus_alpha = fill_ratio - float(avg_alpha)
        metrics = {
            "series": series,
            "ev_after_fees": row.get("ev_after_fees", 0.0),
            "realized_pnl": row.get("realized_pnl", 0.0),
            "expected_fills": expected_fills,
            "requested_contracts": requested,
            "fill_ratio": fill_ratio,
            "avg_fill_ratio": fill_ratio,
            "avg_alpha": avg_alpha,
            "fill_ratio_vs_alpha": fill_minus_alpha,
            "slippage_ticks_mean": slippage_ticks_mean,
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
        subset = frames_by_series.get(series)
        mean_fill_delta, abs_fill_delta = _alpha_model_metrics(alpha_curves.get(series), subset)
        if mean_fill_delta is not None:
            metrics["fill_minus_model"] = mean_fill_delta
        if abs_fill_delta is not None:
            metrics["fill_gap_model_abs"] = abs_fill_delta
        slip_delta_mean, slip_delta_abs = _slippage_model_metrics(slippage_curves.get(series), subset)
        if slip_delta_mean is not None:
            metrics["slippage_delta_mean"] = slip_delta_mean
        if slip_delta_abs is not None:
            metrics["slippage_delta_abs"] = slip_delta_abs
    return sorted(
        records,
        key=lambda row: (
            INDEX_SERIES_ORDER.index(row["series"])
            if row["series"] in INDEX_SERIES
            else row["series"]
        ),
    )


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
        if row.get("fill_ratio_vs_alpha") is not None:
            lines.append(f"- Fill - α: {row['fill_ratio_vs_alpha']:+.3f}")
        if row.get("fill_minus_model") is not None:
            lines.append(f"- Fill - model α: {row['fill_minus_model']:+.3f}")
        if row.get("fill_gap_model_abs") is not None:
            lines.append(f"- |Fill-model|: {row['fill_gap_model_abs']:.3f}")
        lines.append(f"- Sample Size: {row.get('sample_size', 0)} trades")
        lines.append(
            f"- Expected EV (bps): {row.get('ev_expected_bps_mean', 0.0):+.1f}"
        )
        lines.append(
            f"- Realized EV (bps): {row.get('ev_realized_bps_mean', 0.0):+.1f}"
        )
        if row.get("slippage_ticks_mean") is not None:
            lines.append(
                f"- Avg Slippage (ticks): {row.get('slippage_ticks_mean', 0.0):.3f}"
            )
        if row.get("slippage_delta_mean") is not None:
            lines.append(
                f"- Slippage Δ (ticks): {row['slippage_delta_mean']:+.3f}"
            )
        if row.get("slippage_delta_abs") is not None:
            lines.append(
                f"- |Slippage Δ|: {row['slippage_delta_abs']:.3f}"
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



if __name__ == "__main__":
    main()
