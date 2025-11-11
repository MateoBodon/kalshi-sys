"""Capture Kalshi top-of-book snapshots for fill modeling."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from kalshi_alpha.core.kalshi_api import KalshiPublicClient, Market
from kalshi_alpha.datastore.paths import RAW_ROOT

DEFAULT_OUTPUT_DIR = RAW_ROOT / "kalshi" / "tob"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record Kalshi top-of-book snapshots.")
    parser.add_argument("--series", nargs="+", default=["INXU", "NASDAQ100U", "INX", "NASDAQ100"], help="Kalshi series tickers to capture.")
    parser.add_argument("--fixtures-root", type=Path, default=Path("tests/data_fixtures"), help="Offline fixtures root for --offline mode (default: tests/data_fixtures).")
    parser.add_argument("--offline", action="store_true", help="Use offline fixtures instead of live API.")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between successive captures (default: %(default)s).")
    parser.add_argument("--iterations", type=int, default=1, help="Number of capture iterations (default: %(default)s).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for JSONL snapshot files (default: data/raw/kalshi/tob).")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    client = _build_client(args.fixtures_root, args.offline)
    markets = _resolve_markets(client, args.series)
    if not markets:
        raise SystemExit("No markets resolved for requested series")
    args.output.mkdir(parents=True, exist_ok=True)
    snapshots: list[dict[str, object]] = []
    for iteration in range(max(args.iterations, 1)):
        now = datetime.now(tz=UTC)
        for market in markets:
            try:
                orderbook = client.get_orderbook(market.id)
            except Exception as exc:  # pragma: no cover - defensive network handling
                print(f"[tob] failed to fetch orderbook {market.id}: {exc}")
                continue
            snapshot = _snapshot_from_orderbook(market, orderbook, now)
            snapshots.append(snapshot)
        if iteration + 1 < args.iterations:
            time.sleep(max(args.interval, 0.5))
    if not snapshots:
        raise SystemExit("No TOB snapshots captured")
    output_path = args.output / f"tob_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in snapshots:
            handle.write(json.dumps(entry, sort_keys=True))
            handle.write("\n")
    print(f"[tob] wrote {len(snapshots)} snapshots to {output_path}")


def _build_client(fixtures_root: Path, offline: bool) -> KalshiPublicClient:
    fixtures = fixtures_root / "kalshi"
    return KalshiPublicClient(offline_dir=fixtures, use_offline=offline)


def _resolve_markets(client: KalshiPublicClient, series_list: Sequence[str]) -> list[Market]:
    resolved: list[Market] = []
    available_series = {entry.ticker.upper(): entry for entry in client.get_series()}
    for series in series_list:
        metadata = available_series.get(series.upper())
        if metadata is None:
            continue
        events = client.get_events(metadata.id)
        for event in events:
            markets = client.get_markets(event.id)
            resolved.extend(markets)
    return resolved


def _snapshot_from_orderbook(market: Market, orderbook, captured_at: datetime) -> dict[str, object]:
    best_bid = _best_entry(orderbook.bids)
    best_ask = _best_entry(orderbook.asks, side="ask")
    close_time = market.close_time
    seconds_to_close = None
    if isinstance(close_time, datetime):
        seconds_to_close = max((close_time - captured_at).total_seconds(), 0.0)
    ticker_root = (market.ticker or "").split("-", 1)[0]
    if ticker_root.startswith("KX"):
        ticker_root = ticker_root[2:]
    return {
        "captured_at": captured_at.isoformat(),
        "series": ticker_root,
        "market_id": market.id,
        "market_ticker": market.ticker,
        "best_bid_price": best_bid.get("price"),
        "best_bid_size": best_bid.get("size"),
        "best_ask_price": best_ask.get("price"),
        "best_ask_size": best_ask.get("size"),
        "seconds_to_close": seconds_to_close,
    }


def _best_entry(entries: Sequence[dict[str, object]], side: str | None = None) -> dict[str, float]:
    if not entries:
        return {"price": None, "size": 0.0}
    if side == "ask":
        best = min(entries, key=lambda item: float(item.get("price", 0.0)))
    else:
        best = max(entries, key=lambda item: float(item.get("price", 0.0)))
    price = float(best.get("price", 0.0))
    size = float(best.get("size", 0.0))
    return {"price": price, "size": size}


if __name__ == "__main__":  # pragma: no cover
    main()
