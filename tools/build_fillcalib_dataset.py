"""Build fill calibration datasets + conservative maker fill curves from telemetry."""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

try:  # optional dependency for parquet output
    import polars as pl
except Exception:  # pragma: no cover - optional
    pl = None

SERIES_CHOICES = ("INXU", "NASDAQ100U")
DEFAULT_HORIZON_SECONDS = 30
DEFAULT_MIN_SAMPLES = 200
DEFAULT_SCALER = 0.25
DEFAULT_MAX_FILL = 0.25

DISTANCE_BINS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
TIME_BINS_MINUTES = (0, 5, 15, 30, 60, 120, 240)


@dataclass(frozen=True)
class TobSnapshot:
    ts: datetime
    best_bid: float | None
    best_ask: float | None


@dataclass(frozen=True)
class QuoteIntent:
    ts: datetime
    series: str
    window_id: str | None
    window_ts_utc: datetime | None
    market_ticker: str
    side: str
    price: float
    size: int
    tob_ts: datetime | None


@dataclass(frozen=True)
class BucketKey:
    series: str
    window_id: str
    side: str
    distance_bin: str
    time_bin: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build proxy fill curves from TOB + quote intent telemetry.")
    parser.add_argument("--series", required=True, choices=SERIES_CHOICES, help="Series to ingest (INXU/NASDAQ100U).")
    parser.add_argument("--from", dest="date_from", required=True, help="Start date (YYYY-MM-DD), inclusive.")
    parser.add_argument("--to", dest="date_to", required=True, help="End date (YYYY-MM-DD), inclusive.")
    parser.add_argument(
        "--telemetry-root",
        type=Path,
        default=Path("data/proc/telemetry"),
        help="Telemetry root directory (default: data/proc/telemetry).",
    )
    parser.add_argument(
        "--output-curves",
        type=Path,
        default=None,
        help="Optional override path for curves JSON output.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional override path for Markdown report output.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=None,
        help="Optional override path for dataset parquet output.",
    )
    parser.add_argument(
        "--horizon-seconds",
        type=int,
        default=DEFAULT_HORIZON_SECONDS,
        help="Lookahead horizon for TOB crossing proxy (seconds).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="Minimum samples per bucket before allowing non-zero p_fill.",
    )
    parser.add_argument(
        "--scaler",
        type=float,
        default=DEFAULT_SCALER,
        help="Conservative scaler applied to proxy fill rates.",
    )
    parser.add_argument(
        "--max-fill",
        type=float,
        default=DEFAULT_MAX_FILL,
        help="Upper cap on conservative p_fill.",
    )
    parser.add_argument(
        "--write-parquet",
        action="store_true",
        help="Write dataset parquet output (skipped if polars unavailable).",
    )
    parser.add_argument(
        "--max-parquet-rows",
        type=int,
        default=50000,
        help="Only emit parquet if dataset rows <= this threshold.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def build_fillcalib_dataset(
    *,
    series: str,
    date_from: date,
    date_to: date,
    telemetry_root: Path,
    horizon_seconds: int = DEFAULT_HORIZON_SECONDS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    scaler: float = DEFAULT_SCALER,
    max_fill: float = DEFAULT_MAX_FILL,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    series = series.strip().upper()
    if series not in SERIES_CHOICES:
        raise ValueError(f"Unsupported series: {series}")
    if date_to < date_from:
        raise ValueError("--to must be on/after --from")

    tob_snapshots, tob_count = _load_tob_snapshots(
        telemetry_root=telemetry_root,
        series=series,
        date_from=date_from,
        date_to=date_to,
    )
    quote_intents = _load_quote_intents(
        telemetry_root=telemetry_root,
        series=series,
        date_from=date_from,
        date_to=date_to,
    )
    if not quote_intents:
        raise ValueError("No quote intents found for requested range")

    tob_index = _build_tob_index(tob_snapshots)
    rows: list[dict[str, object]] = []
    buckets: dict[BucketKey, dict[str, object]] = {}

    horizon_seconds = max(int(horizon_seconds), 1)
    min_samples = max(int(min_samples), 1)
    scaler = max(0.0, float(scaler))
    max_fill = max(0.0, float(max_fill))

    for intent in quote_intents:
        if intent.market_ticker not in tob_index:
            continue
        times, snapshots = tob_index[intent.market_ticker]
        proxy_fill = _proxy_fill(intent, times, snapshots, horizon_seconds)
        distance = _quote_distance_to_touch(intent, times, snapshots)
        distance_bin = _bin_distance(distance)
        time_to_expiry_seconds, time_bin = _time_to_expiry(intent)
        window_id = intent.window_id or "unknown"
        side = intent.side.upper()
        row = {
            "series": intent.series,
            "window_id": window_id,
            "market_ticker": intent.market_ticker,
            "side": side,
            "quote_price": intent.price,
            "quote_size": intent.size,
            "quote_ts": intent.ts.isoformat(),
            "tob_ts": intent.tob_ts.isoformat() if intent.tob_ts is not None else None,
            "quote_distance_to_touch": distance,
            "quote_distance_to_touch_bin": distance_bin,
            "time_to_expiry_seconds": time_to_expiry_seconds,
            "time_to_expiry_bin": time_bin,
            "proxy_fill": proxy_fill,
        }
        rows.append(row)

        key = BucketKey(series=series, window_id=window_id, side=side, distance_bin=distance_bin, time_bin=time_bin)
        bucket = buckets.setdefault(
            key,
            {
                "series": series,
                "window_id": window_id,
                "side": side,
                "quote_distance_to_touch_bin": distance_bin,
                "time_to_expiry_bin": time_bin,
                "n": 0,
                "proxy_fill_count": 0,
            },
        )
        bucket["n"] = int(bucket["n"]) + 1
        bucket["proxy_fill_count"] = int(bucket["proxy_fill_count"]) + int(proxy_fill)

    bucket_list: list[dict[str, object]] = []
    total_samples = 0
    total_proxy_fills = 0
    weighted_p_fill = 0.0
    for bucket in buckets.values():
        n = int(bucket["n"])
        proxy_fills = int(bucket["proxy_fill_count"])
        total_samples += n
        total_proxy_fills += proxy_fills
        proxy_rate = (proxy_fills / n) if n > 0 else 0.0
        if n < min_samples:
            p_fill = 0.0
            reason = "insufficient_samples"
        else:
            p_fill = min(max(proxy_rate * scaler, 0.0), max_fill)
            reason = "scaled" if p_fill > 0.0 else "zero"
        bucket.update(
            {
                "proxy_fill_rate": round(proxy_rate, 6),
                "p_fill": round(p_fill, 6),
                "min_samples": min_samples,
                "scaler": round(scaler, 6),
                "cap": round(max_fill, 6),
                "reason": reason,
            }
        )
        bucket_list.append(bucket)
        weighted_p_fill += p_fill * n

    default_probability = (weighted_p_fill / total_samples) if total_samples > 0 else 0.0
    proxy_fill_rate = (total_proxy_fills / total_samples) if total_samples > 0 else 0.0

    payload = {
        "version": 1,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "asof_date": date_to.isoformat(),
        "series": {
            series: {
                "data_window": {
                    "from": date_from.isoformat(),
                    "to": date_to.isoformat(),
                },
                "params": {
                    "horizon_seconds": horizon_seconds,
                    "min_samples": min_samples,
                    "conservative_scaler": round(scaler, 6),
                    "max_fill_probability": round(max_fill, 6),
                    "distance_bins": list(DISTANCE_BINS),
                    "time_to_expiry_bins_minutes": list(TIME_BINS_MINUTES),
                    "proxy_method": "tob_crossing",
                },
                "summary": {
                    "quote_intents": len(rows),
                    "tob_snapshots": tob_count,
                    "buckets": len(bucket_list),
                    "proxy_fill_rate": round(proxy_fill_rate, 6),
                },
                "default_probability": round(default_probability, 6),
                "buckets": sorted(
                    bucket_list,
                    key=lambda entry: (
                        str(entry.get("window_id")),
                        str(entry.get("side")),
                        str(entry.get("quote_distance_to_touch_bin")),
                        str(entry.get("time_to_expiry_bin")),
                    ),
                ),
            }
        },
        "notes": {
            "proxy_disclaimer": (
                "Proxy fill indicator derived from TOB crossings (not actual fills). "
                "No queue position or cancel/replace latency modeling."
            )
        },
    }
    return payload, rows


def write_outputs(
    *,
    payload: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    series: str,
    date_to: date,
    output_curves: Path | None,
    output_report: Path | None,
    output_parquet: Path | None,
    write_parquet: bool,
    max_parquet_rows: int,
) -> dict[str, Path | None]:
    series = series.strip().upper()
    asof = date_to.isoformat()
    default_curves = Path("data/proc/fillcalib") / f"curves_{asof}.json"
    default_report = Path("reports/fillcalib") / f"{asof}.md"
    default_parquet = Path("data/proc/fillcalib") / f"dataset_{asof}.parquet"

    curves_path = output_curves or default_curves
    report_path = output_report or default_report
    parquet_path = output_parquet or default_parquet

    curves_path.parent.mkdir(parents=True, exist_ok=True)
    curves_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(payload, rows, series=series), encoding="utf-8")

    parquet_written: Path | None = None
    if write_parquet and rows and len(rows) <= max_parquet_rows and pl is not None:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pl.DataFrame(list(rows))
        frame.write_parquet(parquet_path)
        parquet_written = parquet_path

    return {
        "curves": curves_path,
        "report": report_path,
        "parquet": parquet_written,
    }


def _render_report(payload: Mapping[str, object], rows: Sequence[Mapping[str, object]], *, series: str) -> str:
    series_block = payload.get("series", {}).get(series, {}) if isinstance(payload.get("series"), Mapping) else {}
    params = series_block.get("params", {}) if isinstance(series_block, Mapping) else {}
    summary = series_block.get("summary", {}) if isinstance(series_block, Mapping) else {}
    data_window = series_block.get("data_window", {}) if isinstance(series_block, Mapping) else {}
    buckets = series_block.get("buckets", []) if isinstance(series_block, Mapping) else []
    lines = [
        f"# Fill Calibration Report — {series}",
        "",
        "## Summary",
        f"- Asof date: {payload.get('asof_date')}",
        f"- Data window: {data_window.get('from')} → {data_window.get('to')}",
        f"- Quote intents: {summary.get('quote_intents')}",
        f"- TOB snapshots: {summary.get('tob_snapshots')}",
        f"- Buckets: {summary.get('buckets')}",
        f"- Proxy fill rate: {summary.get('proxy_fill_rate')}",
        f"- Horizon seconds: {params.get('horizon_seconds')}",
        f"- Min samples: {params.get('min_samples')}",
        f"- Conservative scaler: {params.get('conservative_scaler')}",
        f"- Max fill prob: {params.get('max_fill_probability')}",
        "",
        "## Proxy disclaimer",
        "- Proxy fill indicator is derived from TOB crossings only (upper bound).",
        "- No queue position, cancel/replace latency, or partial fill realism is modeled.",
        "",
        "## Bucket summary (top 50 by samples)",
        "| window_id | side | distance_bin | time_bin | n | proxy_fill_rate | p_fill | reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    bucket_rows = sorted(
        (bucket for bucket in buckets if isinstance(bucket, Mapping)),
        key=lambda entry: int(entry.get("n", 0)),
        reverse=True,
    )
    for bucket in bucket_rows[:50]:
        lines.append(
            "| {window_id} | {side} | {distance_bin} | {time_bin} | {n} | {proxy} | {p_fill} | {reason} |".format(
                window_id=bucket.get("window_id"),
                side=bucket.get("side"),
                distance_bin=bucket.get("quote_distance_to_touch_bin"),
                time_bin=bucket.get("time_to_expiry_bin"),
                n=bucket.get("n"),
                proxy=bucket.get("proxy_fill_rate"),
                p_fill=bucket.get("p_fill"),
                reason=bucket.get("reason"),
            )
        )
    if len(bucket_rows) > 50:
        lines.append("")
        lines.append(f"- Truncated {len(bucket_rows) - 50} buckets; see curves JSON for full list.")

    if not rows:
        lines.append("")
        lines.append("- No dataset rows produced.")

    return "\n".join(lines)


def _load_tob_snapshots(
    *,
    telemetry_root: Path,
    series: str,
    date_from: date,
    date_to: date,
) -> tuple[dict[str, list[TobSnapshot]], int]:
    tob_dir = telemetry_root / "tob"
    snapshots: dict[str, list[TobSnapshot]] = {}
    count = 0
    if not tob_dir.exists():
        return snapshots, count
    for path in _iter_paths(tob_dir):
        for record in _iter_json_lines(path):
            if not _record_type_ok(record, "tob_snapshot"):
                continue
            if str(record.get("series") or "").upper() != series:
                continue
            ts = _parse_ts(record.get("ts_utc") or record.get("ts"))
            if ts is None or not _within_range(ts, date_from, date_to):
                continue
            ticker = str(record.get("market_ticker") or "").strip()
            if not ticker:
                continue
            bid = _safe_float(record.get("best_bid_price") or record.get("bid_price"))
            ask = _safe_float(record.get("best_ask_price") or record.get("ask_price"))
            snapshots.setdefault(ticker, []).append(TobSnapshot(ts=ts, best_bid=bid, best_ask=ask))
            count += 1
    return snapshots, count


def _load_quote_intents(
    *,
    telemetry_root: Path,
    series: str,
    date_from: date,
    date_to: date,
) -> list[QuoteIntent]:
    intents: list[QuoteIntent] = []
    intent_dir = telemetry_root / "quote_intents"
    if not intent_dir.exists():
        return intents
    for path in _iter_paths(intent_dir):
        for record in _iter_json_lines(path):
            if not _record_type_ok(record, "quote_intent"):
                continue
            if str(record.get("series") or "").upper() != series:
                continue
            ts = _parse_ts(record.get("ts_utc") or record.get("ts"))
            if ts is None or not _within_range(ts, date_from, date_to):
                continue
            ticker = str(record.get("market_ticker") or "").strip()
            if not ticker:
                continue
            side = str(record.get("quote_side") or "").upper().strip() or "UNKNOWN"
            price = _safe_float(record.get("quote_price"))
            if price is None:
                continue
            size = int(record.get("quote_size") or 0)
            window_id = str(record.get("window_id") or "").strip() or None
            window_ts_utc = _parse_ts(record.get("window_ts_utc"))
            tob_ts = _parse_ts(record.get("tob_ts"))
            intents.append(
                QuoteIntent(
                    ts=ts,
                    series=series,
                    window_id=window_id,
                    window_ts_utc=window_ts_utc,
                    market_ticker=ticker,
                    side=side,
                    price=float(price),
                    size=size,
                    tob_ts=tob_ts,
                )
            )
    return intents


def _build_tob_index(tob_snapshots: Mapping[str, Sequence[TobSnapshot]]) -> dict[str, tuple[list[datetime], list[TobSnapshot]]]:
    index: dict[str, tuple[list[datetime], list[TobSnapshot]]] = {}
    for ticker, snapshots in tob_snapshots.items():
        ordered = sorted(snapshots, key=lambda snap: snap.ts)
        times = [snap.ts for snap in ordered]
        index[ticker] = (times, ordered)
    return index


def _proxy_fill(
    intent: QuoteIntent,
    times: Sequence[datetime],
    snapshots: Sequence[TobSnapshot],
    horizon_seconds: int,
) -> int:
    if not times:
        return 0
    start = intent.ts
    end = start + timedelta(seconds=horizon_seconds)
    idx = bisect.bisect_left(times, start)
    side = intent.side.upper()
    for snap in snapshots[idx:]:
        if snap.ts > end:
            break
        if side == "YES":
            if snap.best_ask is not None and snap.best_ask <= intent.price:
                return 1
        elif side == "NO":
            if snap.best_bid is not None and snap.best_bid >= intent.price:
                return 1
    return 0


def _quote_distance_to_touch(
    intent: QuoteIntent,
    times: Sequence[datetime],
    snapshots: Sequence[TobSnapshot],
) -> float | None:
    if not times:
        return None
    target = intent.tob_ts or intent.ts
    idx = bisect.bisect_right(times, target) - 1
    if idx < 0:
        return None
    snap = snapshots[idx]
    side = intent.side.upper()
    if side == "YES":
        if snap.best_bid is None:
            return None
        return max(0.0, float(snap.best_bid) - intent.price)
    if side == "NO":
        if snap.best_ask is None:
            return None
        return max(0.0, intent.price - float(snap.best_ask))
    return None


def _time_to_expiry(intent: QuoteIntent) -> tuple[float | None, str]:
    if intent.window_ts_utc is None:
        return None, "unknown"
    seconds = max((intent.window_ts_utc - intent.ts).total_seconds(), 0.0)
    minutes = seconds / 60.0
    return seconds, _bin_time(minutes)


def _bin_distance(distance: float | None) -> str:
    if distance is None:
        return "unknown"
    value = max(0.0, float(distance))
    for low, high in zip(DISTANCE_BINS, DISTANCE_BINS[1:]):
        if value < high:
            return f"{low:.2f}-{high:.2f}"
    return f">={DISTANCE_BINS[-1]:.2f}"


def _bin_time(minutes: float | None) -> str:
    if minutes is None:
        return "unknown"
    value = max(0.0, float(minutes))
    for low, high in zip(TIME_BINS_MINUTES, TIME_BINS_MINUTES[1:]):
        if value < high:
            return f"{int(low)}-{int(high)}m"
    return f">={int(TIME_BINS_MINUTES[-1])}m"


def _iter_paths(root: Path) -> Iterable[Path]:
    return sorted([path for path in root.glob("*.jsonl*") if path.is_file()])


def _iter_json_lines(path: Path) -> Iterator[dict[str, object]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record
    except OSError:
        return


def _record_type_ok(record: Mapping[str, object], expected: str) -> bool:
    record_type = record.get("record_type")
    if record_type is None:
        return True
    return str(record_type) == expected


def _parse_ts(value: object) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, (int, float)):
        try:
            ts = datetime.fromtimestamp(float(value), tz=UTC)
        except (OSError, ValueError):
            return None
    elif isinstance(value, str):
        try:
            ts = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _within_range(ts: datetime, date_from: date, date_to: date) -> bool:
    ts_date = ts.date()
    return date_from <= ts_date <= date_to


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    date_from = date.fromisoformat(str(args.date_from))
    date_to = date.fromisoformat(str(args.date_to))
    payload, rows = build_fillcalib_dataset(
        series=args.series,
        date_from=date_from,
        date_to=date_to,
        telemetry_root=args.telemetry_root,
        horizon_seconds=args.horizon_seconds,
        min_samples=args.min_samples,
        scaler=args.scaler,
        max_fill=args.max_fill,
    )
    outputs = write_outputs(
        payload=payload,
        rows=rows,
        series=args.series,
        date_to=date_to,
        output_curves=args.output_curves,
        output_report=args.output_report,
        output_parquet=args.output_parquet,
        write_parquet=args.write_parquet,
        max_parquet_rows=args.max_parquet_rows,
    )
    print(f"[fillcalib] wrote curves: {outputs['curves']}")
    print(f"[fillcalib] wrote report: {outputs['report']}")
    if outputs["parquet"]:
        print(f"[fillcalib] wrote parquet: {outputs['parquet']}")


if __name__ == "__main__":  # pragma: no cover
    main()
