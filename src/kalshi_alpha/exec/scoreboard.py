"""Generate rolling performance scoreboards."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

LEDGER_PATH = Path("data/proc/ledger_all.parquet")
CALIBRATION_PATH = Path("data/proc/calibration_metrics.parquet")
ALPHA_STATE_PATH = Path("data/proc/state/fill_alpha.json")
ARTIFACTS_DIR = Path("reports/_artifacts")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 7-day and 30-day performance scoreboards.")
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
    no_go_counts = _load_no_go_counts(window_days)
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
        metrics = {
            "series": series,
            "ev_after_fees": row.get("ev_after_fees", 0.0),
            "realized_pnl": row.get("realized_pnl", 0.0),
            "expected_fills": expected_fills,
            "requested_contracts": requested,
            "fill_ratio": fill_ratio,
            "avg_fill_ratio": row.get("avg_fill_ratio", 0.0),
            "avg_alpha": avg_alpha,
            "no_go_count": no_go_counts.get(series, 0),
        }
        calib = calib_lookup.get(series)
        if calib:
            metrics["crps_advantage"] = calib.get("crps_advantage")
            metrics["brier_advantage"] = calib.get("brier_advantage")
        records.append(metrics)
    return sorted(records, key=lambda row: row["series"])


def _load_no_go_counts(window_days: int) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    if not ARTIFACTS_DIR.exists():
        return counts
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
                mtime = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00")).astimezone(UTC)
            except ValueError:
                pass
        if mtime < threshold:
            continue
        if data.get("go", True):
            continue
        series = data.get("series") or data.get("mode") or "GLOBAL"
        counts[str(series).upper()] += 1
    return counts


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
        alpha = row.get("avg_alpha")
        alpha_text = f"{alpha:.3f}" if isinstance(alpha, (int, float)) else "n/a"
        lines.append(f"- Avg α: {alpha_text}")
        lines.append(f"- NO-GO Count: {row['no_go_count']}")
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
