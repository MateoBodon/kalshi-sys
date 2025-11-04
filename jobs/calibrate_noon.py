"""Build noon calibration curves for index ladders using Polygon minute bars."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from kalshi_alpha.drivers.polygon_index.client import MinuteBar, PolygonIndicesClient
from kalshi_alpha.drivers.polygon_index.snapshots import write_minute_bars
from kalshi_alpha.strategies.index.noon_above_below import NOON_CALIBRATION_PATH

from ._index_calibration import build_sigma_curve
from .calibrate_hourly import (  # type: ignore[attr-defined]
    RESIDUAL_WINDOW_MINUTES,
    TARGET_TIME,
    _derive_event_extras,
    _resolve_tickers,
    _resolve_window,
    _time_bounds,
    parse_args,
    _write_params,
)

HORIZON = "noon"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start_date, end_date = _resolve_window(args.start, args.end, args.days)
    tickers = _resolve_tickers(args.series)

    client = PolygonIndicesClient()
    bars_by_symbol: dict[str, list[MinuteBar]] = defaultdict(list)
    for ticker in tickers:
        start_ts, end_ts = _time_bounds(start_date, end_date)
        bars = client.fetch_minute_bars(ticker, start_ts, end_ts)
        if not args.skip_snapshots:
            write_minute_bars(ticker, bars)
        bars_by_symbol[ticker].extend(bars)

    if not bars_by_symbol:
        raise RuntimeError("No minute bars fetched for calibration")

    frame, records = build_sigma_curve(
        bars_by_symbol,
        target_time=TARGET_TIME,
        residual_window=RESIDUAL_WINDOW_MINUTES,
        return_records=True,
    )
    extras = _derive_event_extras(records)
    output_root = args.output or NOON_CALIBRATION_PATH
    _write_params(frame, output_root, horizon=HORIZON, extras=extras)


__all__ = ["main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
