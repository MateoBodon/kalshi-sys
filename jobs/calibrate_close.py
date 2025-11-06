"""Build close calibration curves for index ladders using Polygon minute bars."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import polars as pl

from kalshi_alpha.drivers.polygon_index.client import MinuteBar, PolygonIndicesClient
from kalshi_alpha.drivers.polygon_index.snapshots import write_minute_bars
from kalshi_alpha.drivers.polygon_index.symbols import resolve_series
from kalshi_alpha.strategies.index.close_range import CLOSE_CALIBRATION_PATH

from ._index_calibration import (
    ET,
    build_sigma_curve,
    compute_pit_bias,
    event_tail_multiplier,
    extend_calibration_window,
    late_day_lambda,
    latest_observation_date,
    minutes_curves,
    symbol_slug,
)

TARGET_TIME = time(16, 0)
RESIDUAL_WINDOW_MINUTES = 15
EVENT_WINDOW_MINUTES = 60
EVENT_KAPPA_CLAMP = (1.0, 1.75)
EVENT_TAGS = ("CPI", "FOMC")
LATE_DAY_WINDOW_MINUTES = 10
DEFAULT_DAYS = 55
HORIZON = "close"


@dataclass(frozen=True)
class CalibrationWindow:
    requested_start: date
    requested_end: date
    start: date
    end: date


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate close index distributions from Polygon minute bars.")
    parser.add_argument("--start", type=_parse_date, help="Optional start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=_parse_date, help="Optional end date (YYYY-MM-DD).")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Trailing trading-day window when --start/--end not supplied (default: %(default)s).",
    )
    parser.add_argument(
        "--months",
        type=int,
        help="Approximate calibration window in calendar months (overrides --days, assumes 21 trading days/month).",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=["INX", "NASDAQ100"],
        help="Kalshi series symbols to calibrate (resolved via drivers.polygon_index.symbols).",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Optional explicit Polygon tickers (e.g. I:SPX I:NDX).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CLOSE_CALIBRATION_PATH,
        help="Output directory root for calibration parameters (default: data/proc/calib/index).",
    )
    parser.add_argument(
        "--skip-snapshots",
        action="store_true",
        help="Skip writing raw datastore snapshots under data/raw/.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    window = _resolve_window(args.start, args.end, args.days, args.months)
    tickers, series_aliases = _resolve_inputs(args.series, args.symbols)

    client = PolygonIndicesClient()
    start_ts, end_ts = _time_bounds(window)
    bars_by_symbol: dict[str, list[MinuteBar]] = defaultdict(list)
    for ticker in tickers:
        bars = client.fetch_minute_bars(ticker, start_ts, end_ts)
        if not bars:
            continue
        if not args.skip_snapshots:
            write_minute_bars(ticker, bars)
        bars_by_symbol[ticker].extend(bars)

    if not bars_by_symbol:
        raise RuntimeError("No minute bars fetched for calibration")

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    payloads_written = 0

    for symbol, bars in bars_by_symbol.items():
        slug = symbol_slug(symbol)
        frame, records = build_sigma_curve(
            {symbol: bars},
            target_time=TARGET_TIME,
            residual_window=RESIDUAL_WINDOW_MINUTES,
            return_records=True,
        )
        if frame.is_empty():
            continue
        extras = _derive_extras(records, frame, symbol)
        payload = _build_close_payload(
            symbol=symbol,
            slug=slug,
            frame=frame,
            records=records,
            window=window,
            series_aliases=series_aliases,
            extras=extras,
        )
        _write_payload(payload, output_root / slug / HORIZON)
        payloads_written += 1

    if payloads_written == 0:
        raise RuntimeError("No calibration payloads generated for the requested inputs")


def _resolve_inputs(
    series_list: Sequence[str],
    symbol_list: Sequence[str] | None,
) -> tuple[list[str], dict[str, tuple[str, ...]]]:
    tickers: set[str] = set()
    aliases: dict[str, set[str]] = {}

    for series in series_list or ():
        try:
            meta = resolve_series(series)
        except KeyError:
            continue
        ticker = meta.polygon_ticker.upper()
        tickers.add(ticker)
        aliases.setdefault(ticker, set()).add(meta.kalshi_series.upper())

    for symbol in symbol_list or ():
        normalized = symbol.strip().upper()
        if not normalized:
            continue
        tickers.add(normalized)
        aliases.setdefault(normalized, aliases.get(normalized, set()))

    sorted_tickers = sorted(tickers)
    alias_map = {
        ticker: tuple(sorted(values)) if values else tuple()
        for ticker, values in aliases.items()
    }
    return sorted_tickers, alias_map


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _resolve_window(
    start: date | None,
    end: date | None,
    days: int,
    months: int | None,
) -> CalibrationWindow:
    today = datetime.now(tz=ET).date()
    requested_end = end or today
    if start is not None:
        requested_start = start
    else:
        trailing_days = max(days, 1)
        if months is not None and months > 0:
            trailing_days = max(trailing_days, months * 21)
        requested_start = requested_end - timedelta(days=trailing_days)
    if requested_start > requested_end:
        raise ValueError("start date must be on or before end date")
    adjusted_start, adjusted_end = extend_calibration_window(requested_start, requested_end, tags=EVENT_TAGS)
    return CalibrationWindow(
        requested_start=requested_start,
        requested_end=requested_end,
        start=adjusted_start,
        end=adjusted_end,
    )


def _time_bounds(window: CalibrationWindow) -> tuple[datetime, datetime]:
    start_dt = datetime.combine(window.start, time(8, 0), tzinfo=ET).astimezone(UTC)
    end_dt = datetime.combine(window.end + timedelta(days=1), time(6, 0), tzinfo=ET).astimezone(UTC)
    return start_dt, end_dt


def _derive_extras(records: pl.DataFrame, frame: pl.DataFrame, symbol: str) -> dict[str, object]:
    if records.is_empty():
        return {}
    kappa = event_tail_multiplier(
        records,
        symbol,
        window=EVENT_WINDOW_MINUTES,
        tags=EVENT_TAGS,
        clamp=EVENT_KAPPA_CLAMP,
    )
    late_day = late_day_lambda(
        records,
        frame,
        symbol,
        window=LATE_DAY_WINDOW_MINUTES,
        tags=EVENT_TAGS,
    )
    extras: dict[str, object] = {
        "event_tail": {
            "tags": [tag.upper() for tag in EVENT_TAGS],
            "kappa": round(float(kappa), 4),
        },
        "kappa_event": round(float(kappa), 4),
    }
    if late_day is not None:
        extras["late_day_variance"] = {
            "minutes_threshold": int(late_day["minutes_threshold"]),
            "lambda": round(float(late_day["lambda"]), 6),
        }
        extras["lambda_close"] = round(float(late_day["lambda"]), 6)
    return extras


def _build_close_payload(  # noqa: PLR0913
    *,
    symbol: str,
    slug: str,
    frame: pl.DataFrame,
    records: pl.DataFrame,
    window: CalibrationWindow,
    series_aliases: Mapping[str, Sequence[str]],
    extras: Mapping[str, object] | None = None,
) -> dict[str, object]:
    sigma_curve, drift_curve, residual_std = minutes_curves(frame, symbol)
    if not sigma_curve:
        raise ValueError(f"No sigma curve computed for {symbol}")

    minutes_map = {
        str(minute): {
            "sigma": float(sigma_curve.get(minute, 0.0)),
            "drift": float(drift_curve.get(minute, 0.0)),
        }
        for minute in sorted(sigma_curve)
    }
    sigma_now = float(sigma_curve.get(0, 0.0))
    if sigma_now <= 0.0:
        positive_sigmas = [value for value in sigma_curve.values() if value > 0.0]
        if positive_sigmas:
            sigma_now = float(sorted(positive_sigmas)[0])
    sigma_now = max(sigma_now, 0.0)

    m_tod = {
        key: (value / sigma_now if sigma_now > 0.0 else 1.0)
        for key, value in ((str(minute), float(sigma_curve.get(minute, 0.0))) for minute in sorted(sigma_curve))
    }
    micro_drift = {
        str(minute): float(drift_curve.get(minute, 0.0))
        for minute in sorted(sigma_curve)
    }

    subset_records = records.filter(pl.col("symbol") == symbol)
    pit_bias = compute_pit_bias(
        subset_records,
        symbol,
        sigma_curve=sigma_curve,
        drift_curve=drift_curve,
    )
    latest_date = latest_observation_date(records, symbol)
    now_utc = datetime.now(tz=UTC)
    now_et_date = now_utc.astimezone(ET).date()
    age_days = None
    if latest_date is not None:
        age_days = max((now_et_date - latest_date).days, 0)

    payload: dict[str, object] = {
        "symbol": symbol,
        "slug": slug,
        "series": list(series_aliases.get(symbol.upper(), ())),
        "horizon": HORIZON,
        "target": {"hour_et": TARGET_TIME.strftime("%H:%M"), "label": "close"},
        "generated_at": now_utc.isoformat(),
        "source_window": {
            "requested_start": window.requested_start.isoformat(),
            "requested_end": window.requested_end.isoformat(),
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
        },
        "age": {
            "days": age_days,
            "last_observation": latest_date.isoformat() if latest_date else None,
        },
        "sample_count": int(subset_records.height),
        "minutes_to_target": minutes_map,
        "sigma_now": float(sigma_now),
        "m_tod": {key: float(value) for key, value in m_tod.items()},
        "micro_drift": micro_drift,
        "residual_std": float(residual_std),
        "pit_bias": None if pit_bias is None else float(pit_bias),
    }

    if extras:
        for key, value in extras.items():
            payload[key] = value
        if "lambda_close" not in payload:
            payload["lambda_close"] = 0.0
    else:
        payload["lambda_close"] = 0.0

    return payload


def _write_payload(payload: Mapping[str, object], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    checksum = hashlib.sha256(canonical_bytes).hexdigest()
    payload_with_checksum = dict(payload)
    payload_with_checksum["checksum"] = {
        "algorithm": "sha256",
        "value": checksum,
    }
    path = directory / "params.json"
    path.write_text(json.dumps(payload_with_checksum, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
