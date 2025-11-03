"""Developer CLI to smoke-test Kalshi websocket imbalance streaming."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from kalshi_alpha.core import kalshi_ws

DEFAULT_RUN_SECONDS = 300
DEFAULT_DEPTH = 3
DEFAULT_WINDOW_SECONDS = 30


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream Kalshi orderbook imbalance metrics.")
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="One or more Kalshi ladder tickers (e.g. TNEY-20251103).",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=DEFAULT_RUN_SECONDS,
        help="Total duration to stream before exiting (seconds).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=DEFAULT_DEPTH,
        help="Orderbook depth used when computing imbalance.",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=DEFAULT_WINDOW_SECONDS,
        help="Rolling average window length for imbalance (seconds).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress status output.")
    return parser


def _normalize_tickers(raw: Sequence[str]) -> list[str]:
    tickers: list[str] = []
    for item in raw:
        stripped = item.strip().upper()
        if stripped:
            tickers.append(stripped)
    return tickers


def _latest_snapshot_path(ticker: str) -> Path | None:
    directory = kalshi_ws.RAW_ORDERBOOK_ROOT / ticker.upper()
    if not directory.exists():
        return None
    latest: Path | None = None
    latest_mtime = float("-inf")
    for candidate in directory.glob("*.jsonl"):
        try:
            mtime = candidate.stat().st_mtime
        except OSError:  # pragma: no cover - filesystem race
            continue
        if mtime >= latest_mtime:
            latest = candidate
            latest_mtime = mtime
    return latest


async def _execute(args: argparse.Namespace) -> dict[str, float]:
    tickers = _normalize_tickers(args.tickers)
    if not tickers:
        raise SystemExit("At least one ticker must be provided.")
    results = await kalshi_ws.stream_orderbook_imbalance(
        tickers,
        depth=max(1, int(args.depth)),
        window_seconds=max(1, int(args.window_seconds)),
        run_seconds=max(0.0, float(args.run_seconds)),
    )
    if args.quiet:
        return results

    if results:
        print("Recorded imbalance metrics:")
        for ticker in sorted(tickers):
            metric_path = kalshi_ws.PROC_IMBALANCE_ROOT / f"{ticker}.json"
            snapshot_path = _latest_snapshot_path(ticker)
            if snapshot_path is not None:
                print(f"  {ticker}: metric={metric_path} snapshot={snapshot_path}")
            else:
                print(f"  {ticker}: metric={metric_path} snapshot=<missing>")
    else:
        print("No imbalance metrics recorded.")
    return results


def main(argv: Sequence[str] | None = None) -> dict[str, float]:
    args = _build_parser().parse_args(argv)
    return asyncio.run(_execute(args))


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
