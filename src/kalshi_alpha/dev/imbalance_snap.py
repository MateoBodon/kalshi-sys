"""Capture Kalshi orderbook imbalance metrics for the TENY close window."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from kalshi_alpha.core import kalshi_ws
from kalshi_alpha.core.ws import DEFAULT_WS_URL, KalshiWebsocketClient

ET_ZONE = ZoneInfo("America/New_York")


def _parse_time(value: str) -> time:
    try:
        hour, minute = (int(part) for part in value.split(":", 1))
    except ValueError as exc:  # pragma: no cover - argparse validation
        raise argparse.ArgumentTypeError(f"Invalid time '{value}'; expected HH:MM") from exc
    return time(hour, minute)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - argparse validation
        raise argparse.ArgumentTypeError(f"Invalid date '{value}'; expected YYYY-MM-DD") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture Kalshi orderbook imbalance snapshots.")
    parser.add_argument("--tickers", required=True, help="Comma-separated list of ladder tickers (e.g. TNEY).")
    parser.add_argument("--duration-seconds", type=float, help="Override the capture duration in seconds.")
    parser.add_argument("--window-start", default="15:00", help="ET window start time (HH:MM).")
    parser.add_argument("--window-end", default="15:25", help="ET window end time (HH:MM).")
    parser.add_argument("--as-of-date", type=_parse_date, help="Override ET trading date (YYYY-MM-DD).")
    parser.add_argument("--depth", type=int, default=3, help="Orderbook depth used when computing imbalance.")
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=30,
        help="Rolling window length (seconds) for imbalance averaging.",
    )
    parser.add_argument("--raw-root", type=Path, help="Override raw snapshot output directory.")
    parser.add_argument("--proc-root", type=Path, help="Override processed imbalance output directory.")
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL, help="Kalshi websocket endpoint.")
    parser.add_argument("--ping-interval", type=float, default=30.0, help="Websocket ping interval (seconds).")
    parser.add_argument("--quiet", action="store_true", help="Suppress status logging.")
    return parser


def _resolve_window(args: argparse.Namespace, now_utc: datetime) -> tuple[datetime, datetime]:
    now_et = now_utc.astimezone(ET_ZONE)
    trading_date = args.as_of_date or now_et.date()
    start_clock = _parse_time(args.window_start)
    end_clock = _parse_time(args.window_end)
    start_et = datetime.combine(trading_date, start_clock, tzinfo=ET_ZONE)
    end_et = datetime.combine(trading_date, end_clock, tzinfo=ET_ZONE)
    if end_et <= start_et:
        end_et += timedelta(days=1)
    return start_et.astimezone(UTC), end_et.astimezone(UTC)


def _apply_root_overrides(args: argparse.Namespace) -> None:
    if args.raw_root:
        root = Path(args.raw_root)
        root.mkdir(parents=True, exist_ok=True)
        kalshi_ws.RAW_ORDERBOOK_ROOT = root
    if args.proc_root:
        root = Path(args.proc_root)
        root.mkdir(parents=True, exist_ok=True)
        kalshi_ws.PROC_IMBALANCE_ROOT = root


async def _capture(args: argparse.Namespace) -> dict[str, float]:
    tickers = [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
    if not tickers:
        raise ValueError("At least one ticker must be provided.")

    _apply_root_overrides(args)

    def clock() -> datetime:
        return datetime.now(tz=UTC)
    run_seconds: float
    start_delay = 0.0

    if args.duration_seconds is not None:
        run_seconds = max(0.0, float(args.duration_seconds))
    else:
        start_utc, end_utc = _resolve_window(args, clock())
        now_utc = clock()
        start_delay = max(0.0, (start_utc - now_utc).total_seconds())
        run_seconds = max(0.0, (end_utc - start_utc).total_seconds())

    if start_delay > 0:
        if not args.quiet:
            print(f"Sleeping {start_delay:.0f}s until window start...")
        await asyncio.sleep(start_delay)

    client = KalshiWebsocketClient(base_url=args.ws_url, ping_interval=float(args.ping_interval))
    results = await kalshi_ws.stream_orderbook_imbalance(
        tickers,
        depth=args.depth,
        window_seconds=args.window_seconds,
        client=client,
        run_seconds=run_seconds,
        now_fn=clock,
    )

    if not args.quiet:
        if results:
            for ticker, value in results.items():
                print(f"{ticker}: imbalance={value:.3f}")
        else:
            print("No imbalance updates captured.")
    return results


def main(argv: Sequence[str] | None = None) -> dict[str, float]:
    args = _build_parser().parse_args(argv)
    return asyncio.run(_capture(args))


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
