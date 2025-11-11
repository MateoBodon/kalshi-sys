"""Sigma time-of-day drift monitor that emits shrink factors per series."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Sequence

import polars as pl

from kalshi_alpha.exec.monitors import sigma_drift

SERIES_CONFIG = {
    "INXU": {"slug": "spx", "horizon": "hourly", "symbol": "I_SPX"},
    "NASDAQ100U": {"slug": "ndx", "horizon": "hourly", "symbol": "I_NDX"},
    "INX": {"slug": "spx", "horizon": "close", "symbol": "I_SPX"},
    "NASDAQ100": {"slug": "ndx", "horizon": "close", "symbol": "I_NDX"},
}
DEFAULT_LOOKBACK = 7
DEFAULT_THRESHOLD = 0.25


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute realized vs forecast sigma drift shrink factors.")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK, help="History window in trading days.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Relative drift (fraction) before shrink kicks in (default: 0.25 -> 25%%).",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=list(SERIES_CONFIG),
        help="Series tickers to evaluate (default: INX/NDX hourly + close).",
    )
    parser.add_argument(
        "--sigma-root",
        type=Path,
        default=Path("data/proc/calib/index"),
        help="Calibration root containing <slug>/<horizon>/params.json files.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/polygon"),
        help="Polygon parquet root (expects symbol/YYYY-MM-DD.parquet).",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=sigma_drift.SIGMA_DRIFT_ARTIFACT,
        help="Output artifact path.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    now = datetime.now(tz=UTC)
    lookback_days = max(int(args.lookback_days), 1)
    threshold = max(float(args.threshold), 0.0)
    results: dict[str, dict[str, object]] = {}
    overall_status = "NO_DATA"
    for series in args.series:
        series_upper = series.upper()
        cfg = SERIES_CONFIG.get(series_upper)
        if not cfg:
            print(f"[sigma_drift] skipping unknown series {series}")
            continue
        forecast = _load_forecast(args.sigma_root, cfg)
        realized, samples = _realized_sigma(cfg["symbol"], args.raw_root, lookback_days)
        entry: dict[str, object] = {
            "forecast_sigma": forecast,
            "realized_sigma": realized,
            "samples": samples,
            "status": "NO_DATA",
            "shrink": 1.0,
        }
        if forecast is None or realized is None or forecast <= 0.0:
            results[series_upper] = entry
            continue
        ratio = realized / forecast
        entry["ratio"] = ratio
        entry["drift"] = ratio - 1.0
        if ratio - 1.0 >= threshold:
            shrink = max(min(forecast / realized, 1.0), 0.25)
            entry["shrink"] = shrink
            entry["status"] = "ALERT"
            overall_status = "ALERT"
        else:
            entry["shrink"] = 1.0
            entry["status"] = "OK"
            if overall_status != "ALERT":
                overall_status = "OK"
        results[series_upper] = entry
    payload = {
        "generated_at": now.isoformat(),
        "lookback_days": lookback_days,
        "threshold": threshold,
        "status": overall_status,
        "series": results,
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[sigma_drift] wrote {args.artifact}")
    return 1 if overall_status == "ALERT" else 0


def _load_forecast(root: Path, cfg: dict[str, str]) -> float | None:
    params_path = root / cfg["slug"] / cfg["horizon"] / "params.json"
    if not params_path.exists():
        return None
    try:
        payload = json.loads(params_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    minutes = payload.get("minutes_to_target")
    if not isinstance(minutes, dict):
        return None
    sigma_values = []
    for entry in minutes.values():
        if not isinstance(entry, dict):
            continue
        try:
            sigma_values.append(float(entry.get("sigma", 0.0)))
        except (TypeError, ValueError):
            continue
    if not sigma_values:
        return None
    return sum(sigma_values) / len(sigma_values)


def _realized_sigma(symbol: str, root: Path, lookback_days: int) -> tuple[float | None, int]:
    days = _recent_days(lookback_days)
    diffs: list[float] = []
    for day in days:
        path = root / symbol / f"{day.isoformat()}.parquet"
        if not path.exists():
            continue
        try:
            frame = pl.read_parquet(path)
        except Exception:  # noqa: BLE001 - tolerate corrupt files
            continue
        if frame.is_empty() or "close" not in frame.columns:
            continue
        ordered = frame.sort("timestamp") if "timestamp" in frame.columns else frame
        closes = ordered["close"].cast(pl.Float64)
        diff = closes.diff().drop_nulls()
        if diff.is_empty():
            continue
        diffs.extend(float(val) for val in diff.to_list())
    if len(diffs) < 2:
        return None, len(diffs)
    mean = sum(diffs) / len(diffs)
    variance = sum((value - mean) ** 2 for value in diffs) / (len(diffs) - 1)
    return math.sqrt(max(variance, 0.0)), len(diffs)


def _recent_days(window: int) -> list[date]:
    today = datetime.now(tz=UTC).date()
    return [today - timedelta(days=offset) for offset in range(window)]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
