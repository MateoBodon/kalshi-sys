"""Derive σ_tod + drift curves for intraday ("U") index ladders."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from kalshi_alpha.datastore.paths import CALIB_REPORTS_ROOT, INDEX_PMF_ROOT
from kalshi_alpha.drivers.polygon_index.client import MinuteBar, PolygonIndicesClient
from kalshi_alpha.drivers.polygon_index.symbols import resolve_series
from jobs._index_calibration import (
    ET,
    build_sigma_curve,
    extend_calibration_window,
    latest_observation_date,
    minutes_curves,
    symbol_slug,
)

DEFAULT_SERIES = ("INXU", "NASDAQ100U")
DEFAULT_TARGET_HOURS = (10, 11, 12, 13, 14, 15)
RESIDUAL_WINDOW_MINUTES = 5
DEFAULT_DAYS = 35
OUTPUT_VERSION = 1
MIN_STD_DEFAULT = 0.65


@dataclass(frozen=True)
class CalibrationWindow:
    requested_start: date
    requested_end: date
    start: date
    end: date


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate hourly σ_tod for index ladders.")
    parser.add_argument("--series", nargs="+", default=list(DEFAULT_SERIES), help="Kalshi series symbols (default: INXU NASDAQ100U).")
    parser.add_argument("--target-hours", nargs="+", type=int, default=list(DEFAULT_TARGET_HOURS), metavar="HOUR", help="One or more ET hours to calibrate (default: 10 11 12 13 14 15).")
    parser.add_argument("--start", type=_parse_date, help="Optional start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=_parse_date, help="Optional end date (YYYY-MM-DD).")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Trailing trading days when start/end omitted (default: %(default)s).")
    parser.add_argument("--months", type=int, help="Approximate trailing months (overrides --days; assumes 21 trading days/month).")
    parser.add_argument("--output", type=Path, default=INDEX_PMF_ROOT, help="Output root for PMF parameter JSON (default: data/proc/calib/index_pmf).")
    parser.add_argument("--reports-dir", type=Path, default=CALIB_REPORTS_ROOT, help="Report/plot output root (default: reports/calib).")
    parser.add_argument("--min-std", type=float, default=MIN_STD_DEFAULT, help="Minimum σ safeguard when generating PMFs (default: %(default)s).")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation (useful for CI).")
    parser.add_argument("--skip-snapshots", action="store_true", help="Skip writing raw polygon snapshots to data/raw.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    window = _resolve_window(args.start, args.end, args.days, args.months)
    target_hours = sorted({int(hour) % 24 for hour in (args.target_hours or DEFAULT_TARGET_HOURS)})
    client = PolygonIndicesClient()
    symbols, alias_map = _resolve_inputs(args.series)
    start_ts, end_ts = _time_bounds(window)
    records_written = 0

    for symbol in symbols:
        bars = client.fetch_minute_bars(symbol, start_ts, end_ts)
        if not bars:
            continue
        if not args.skip_snapshots:
            _write_snapshots(symbol, bars)
        slug = symbol_slug(symbol)
        for hour in target_hours:
            frame, records = build_sigma_curve({symbol: bars}, target_time=time(hour, 0), residual_window=RESIDUAL_WINDOW_MINUTES, return_records=True)
            if frame.is_empty():
                continue
            sigma_curve, drift_curve, residual_std = minutes_curves(frame, symbol)
            if not sigma_curve:
                continue
            payload = _build_payload(
                series_aliases=alias_map.get(symbol, (symbol,)),
                symbol=symbol,
                slug=slug,
                target_hour=hour,
                sigma_curve=sigma_curve,
                drift_curve=drift_curve,
                residual_std=residual_std,
                min_std=args.min_std,
                window=window,
                records=records,
            )
            for alias, alias_payload in payload.items():
                target_dir = args.output / alias.upper() / "hourly"
                target_dir.mkdir(parents=True, exist_ok=True)
                file_path = target_dir / f"{alias_payload['target_label']}.json"
                _write_payload(alias_payload, file_path)
                if not args.skip_plots:
                    _write_plot(alias_payload, args.reports_dir / alias.upper() / "hourly")
                records_written += 1

    if records_written == 0:
        raise SystemExit("No hourly calibration payloads generated")
    print(f"[calib_hourly] wrote {records_written} payloads to {args.output}")


def _resolve_inputs(series_list: Sequence[str]) -> tuple[list[str], dict[str, tuple[str, ...]]]:
    symbols: dict[str, set[str]] = {}
    for series in series_list or ():
        try:
            meta = resolve_series(series)
        except KeyError:
            continue
        symbols.setdefault(meta.polygon_ticker.upper(), set()).add(meta.kalshi_series.upper())
    ordered_symbols = sorted(symbols)
    alias_map = {symbol: tuple(sorted(values)) for symbol, values in symbols.items()}
    return ordered_symbols, alias_map


def _build_payload(
    *,
    series_aliases: Sequence[str],
    symbol: str,
    slug: str,
    target_hour: int,
    sigma_curve: Mapping[int, float],
    drift_curve: Mapping[int, float],
    residual_std: float,
    min_std: float,
    window: CalibrationWindow,
    records: pl.DataFrame,
) -> dict[str, dict[str, object]]:
    now = datetime.now(tz=UTC)
    minutes_payload = {str(minute): {"sigma": value, "drift": drift_curve.get(minute, 0.0)} for minute, value in sigma_curve.items()}
    observation_date = latest_observation_date(records, symbol)
    checksum_source = {
        "version": OUTPUT_VERSION,
        "symbol": symbol,
        "target_type": "hourly",
        "target_label": f"{target_hour:02d}00",
        "sigma_curve": sigma_curve,
        "drift_curve": drift_curve,
        "residual_std": residual_std,
        "min_std": min_std,
    }
    checksum = hashlib.sha256(json.dumps(checksum_source, sort_keys=True).encode("utf-8")).hexdigest()
    metadata = {
        "window": {
            "requested_start": window.requested_start.isoformat(),
            "requested_end": window.requested_end.isoformat(),
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
        },
        "slug": slug,
        "records": int(records.height),
        "latest_observation": observation_date.isoformat() if observation_date else None,
    }
    payloads: dict[str, dict[str, object]] = {}
    for alias in series_aliases or (symbol,):
        alias_payload = {
            "version": OUTPUT_VERSION,
            "series": alias.upper(),
            "symbol": symbol,
            "target_type": "hourly",
            "target_label": f"{target_hour:02d}00",
            "generated_at": now.isoformat(),
            "sigma_curve": {str(k): v for k, v in sigma_curve.items()},
            "drift_curve": {str(k): drift_curve.get(k, 0.0) for k in sigma_curve},
            "minutes_to_target": minutes_payload,
            "residual_std": float(residual_std),
            "min_std": float(min_std),
            "metadata": metadata,
            "checksum": {
                "algorithm": "sha256",
                "value": checksum,
            },
        }
        payloads[alias.upper()] = alias_payload
    return payloads


def _write_payload(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[calib_hourly] wrote {path}")


def _write_plot(payload: dict[str, object], reports_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - matplotlib optional
        print(f"[calib_hourly] skipping plot (matplotlib unavailable: {exc})")
        return
    curves = payload.get("sigma_curve") or {}
    if not isinstance(curves, dict):  # pragma: no cover - defensive
        return
    minutes = sorted(int(key) for key in curves.keys())
    if not minutes:
        return
    sigma = [float(curves[str(minute)]) for minute in minutes]
    reports_dir.mkdir(parents=True, exist_ok=True)
    plot_path = reports_dir / f"{payload['target_label']}.png"
    plt.figure(figsize=(7, 3))
    plt.plot(minutes, sigma, label="sigma_tod")
    plt.title(f"{payload['series']} {payload['target_label']} σ_tod")
    plt.xlabel("Minutes to target")
    plt.ylabel("σ (points)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"[calib_hourly] wrote plot {plot_path}")


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _resolve_window(start: date | None, end: date | None, days: int, months: int | None) -> CalibrationWindow:
    today = datetime.now(tz=ET).date()
    requested_end = end or today
    if start:
        requested_start = start
    else:
        trailing_days = max(days, 1)
        if months is not None and months > 0:
            trailing_days = max(trailing_days, months * 21)
        requested_start = requested_end - timedelta(days=trailing_days)
    if requested_start > requested_end:
        raise ValueError("start date must be on or before end date")
    adjusted_start, adjusted_end = extend_calibration_window(requested_start, requested_end)
    return CalibrationWindow(
        requested_start=requested_start,
        requested_end=requested_end,
        start=adjusted_start,
        end=adjusted_end,
    )


def _time_bounds(window: CalibrationWindow) -> tuple[datetime, datetime]:
    start_dt = datetime.combine(window.start, time(0, 0), tzinfo=ET)
    end_dt = datetime.combine(window.end + timedelta(days=1), time(0, 0), tzinfo=ET)
    return start_dt.astimezone(UTC), end_dt.astimezone(UTC)


def _write_snapshots(symbol: str, bars: Sequence[MinuteBar]) -> None:
    from kalshi_alpha.drivers.polygon_index.snapshots import write_minute_bars

    try:
        write_minute_bars(symbol, list(bars))
    except Exception as exc:  # pragma: no cover - snapshots best effort
        print(f"[calib_hourly] failed to write snapshots for {symbol}: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
