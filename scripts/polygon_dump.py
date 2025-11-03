#!/usr/bin/env python
"""Dump Polygon index aggregates to Parquet with metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from kalshi_alpha.drivers.polygon_index.client import PolygonIndicesClient

ET_ZONE = ZoneInfo("America/New_York")


def _parse_et_timestamp(label: str) -> datetime:
    text = label.strip().replace(" ", "T")
    try:
        return datetime.fromisoformat(text).replace(tzinfo=ET_ZONE)
    except ValueError as exc:  # pragma: no cover - argument parsing
        raise argparse.ArgumentTypeError(f"Invalid timestamp: {label}") from exc


def dump_polygon_data(  # noqa: PLR0913
    *,
    symbol: str,
    start_et: datetime,
    end_et: datetime,
    output: Path,
    adjusted: bool,
    timespan: str,
) -> Path:
    client = PolygonIndicesClient()
    start_utc = start_et.astimezone(UTC)
    end_utc = end_et.astimezone(UTC)
    if timespan == "second":
        bars = client.fetch_second_bars(symbol, start_utc, end_utc, adjusted=adjusted)
    else:
        bars = client.fetch_minute_bars(symbol, start_utc, end_utc, adjusted=adjusted)
    if not bars:
        raise RuntimeError("No aggregate data returned")
    frame = pl.DataFrame(
        {
            "timestamp": [entry.timestamp for entry in bars],
            "open": [entry.open for entry in bars],
            "high": [entry.high for entry in bars],
            "low": [entry.low for entry in bars],
            "close": [entry.close for entry in bars],
            "volume": [entry.volume for entry in bars],
            "vwap": [entry.vwap for entry in bars],
            "trades": [entry.trades for entry in bars],
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {
        "symbol": symbol,
        "start_et": start_et.isoformat(),
        "end_et": end_et.isoformat(),
        "rows": frame.height,
        "adjusted": adjusted,
        "timespan": timespan,
        "checksum": checksum,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump Polygon index aggregates to Parquet")
    parser.add_argument("symbol", help="Polygon ticker symbol (e.g., I:SPX)")
    parser.add_argument("start", type=_parse_et_timestamp, help="Start timestamp (ET)")
    parser.add_argument("end", type=_parse_et_timestamp, help="End timestamp (ET)")
    parser.add_argument("output", type=Path, help="Output Parquet path")
    parser.add_argument(
        "--timespan",
        choices=["minute", "second"],
        default="minute",
        help="Aggregate granularity (default: minute)",
    )
    parser.add_argument(
        "--unadjusted",
        action="store_true",
        help="Disable Polygon adjusted bars (default: adjusted)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    start_et: datetime = args.start
    end_et: datetime = args.end
    if end_et <= start_et:
        parser.error("end timestamp must be after start timestamp")
    dump_polygon_data(
        symbol=args.symbol,
        start_et=start_et,
        end_et=end_et,
        output=args.output,
        adjusted=not args.unadjusted,
        timespan=args.timespan,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
