"""Download Polygon index history into the raw datastore."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from kalshi_alpha.drivers.polygon_index.client import PolygonIndicesClient


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Polygon minute history for I:SPX and I:NDX.")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--symbols", nargs="+", default=["I:SPX", "I:NDX"], help="Polygon tickers to download")
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/polygon"), help="Output root directory")
    return parser.parse_args(list(argv) if argv is not None else None)


def _parse_date(label: str) -> datetime:
    try:
        return datetime.fromisoformat(label).replace(tzinfo=UTC)
    except ValueError as exc:
        raise SystemExit(f"Invalid date '{label}', expected YYYY-MM-DD") from exc


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    start_date = _parse_date(args.start)
    end_date = _parse_date(args.end)
    if end_date <= start_date:
        raise SystemExit("--end must be after --start")
    client = PolygonIndicesClient()
    for symbol in args.symbols:
        print(f"[ingest] downloading {symbol} from {start_date.date()} to {end_date.date()}")
        paths = client.download_minute_history(symbol, start_date, end_date, output_root=args.output_root)
        if paths:
            target_dir = args.output_root / symbol.replace(":", "_").upper()
            print(f"[ingest] wrote {len(paths)} parquet files under {target_dir}")
        else:
            print(f"[ingest] no bars returned for {symbol}")


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    main()
