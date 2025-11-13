#!/usr/bin/env python3
"""Proof-of-fill and PnL accounting helper for Kalshi index ladders."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

try:  # pragma: no cover - handled via tests that import polars
    import polars as pl
except ModuleNotFoundError as exc:  # pragma: no cover - surfaced early for operators
    raise SystemExit("scripts/proof_of_fill.py requires the 'polars' package") from exc

from kalshi_alpha.brokers.kalshi import endpoints as kalshi_endpoints
from kalshi_alpha.brokers.kalshi.http_client import KalshiHttpClient
from kalshi_alpha.risk import var_index
from kalshi_alpha.sched import windows as sched_windows

ET = ZoneInfo("America/New_York")
DEFAULT_LEDGER_PATH = Path("data/proc/ledger_all.parquet")
DEFAULT_OUTPUT_DIR = Path("reports/_artifacts")
DEFAULT_REPORTS_ROOT = Path("reports")
ORDERS_PAGE_LIMIT = 500
WINDOW_SCOPE = "window"
DAY_SCOPE = "day"


@dataclass(slots=True)
class OrderFill:
    order_id: str | None
    market_id: str | None
    side: str
    price: float
    contracts: int
    filled_contracts: int
    avg_fill_price: float | None
    status: str
    created_at: datetime
    updated_at: datetime | None
    idempotency_key: str | None = None
    strike: float | None = None

    def fill_ratio(self) -> float:
        if self.contracts <= 0:
            return 0.0
        return max(0.0, min(1.0, self.filled_contracts / self.contracts))


@dataclass(slots=True)
class LedgerEntry:
    series: str
    market_ticker: str
    price: float
    side: str
    contracts: int
    ev_after_fees: float
    pnl_simulated: float
    fill_ratio_model: float
    fill_ratio_observed: float
    timestamp: datetime
    manifest_path: Path | None
    ledger_row: Mapping[str, object]
    market_id: str | None = None
    trading_day: date | None = None
    window_label: str | None = None

    def max_loss(self) -> float:
        price = max(0.0, min(1.0, float(self.price)))
        size = max(0, int(self.contracts))
        if size <= 0:
            return 0.0
        if self.side == "YES":
            return size * price
        if self.side == "NO":
            return size * (1.0 - price)
        return 0.0


@dataclass(slots=True)
class WindowSummary:
    day: date
    series: str
    label: str
    ev_after_fees: float = 0.0
    pnl_realized: float = 0.0
    contracts: int = 0
    fill_gap_pp: float = 0.0
    samples: int = 0
    var_exposure: float = 0.0

    def add_entry(self, entry: LedgerEntry) -> None:
        self.ev_after_fees += float(entry.ev_after_fees)
        self.pnl_realized += float(entry.pnl_simulated)
        self.contracts += int(entry.contracts)
        gap = float(entry.fill_ratio_observed) - float(entry.fill_ratio_model)
        self.fill_gap_pp += gap * 100.0
        self.samples += 1
        self.var_exposure += entry.max_loss()

    def delta_ev_cents_per_lot(self) -> float:
        if self.contracts <= 0:
            return 0.0
        difference = self.pnl_realized - self.ev_after_fees
        return (difference / self.contracts) * 100.0

    def fill_gap_avg(self) -> float:
        if self.samples <= 0:
            return 0.0
        return self.fill_gap_pp / self.samples


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proof-of-fill and PnL summary")
    parser.add_argument("--start", type=str, default="yesterday", help="Start date (YYYY-MM-DD|today|yesterday)")
    parser.add_argument("--end", type=str, default=None, help="End date inclusive (defaults to start date)")
    parser.add_argument("--series", nargs="+", help="Optional subset of series tickers")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH, help="Ledger parquet path")
    parser.add_argument(
        "--orders-json",
        type=Path,
        action="append",
        help="Optional local orders JSON file(s) for offline analysis",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=DEFAULT_REPORTS_ROOT,
        help="Root used to resolve proposal manifests",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Artifact directory for parquet exports",
    )
    parser.add_argument(
        "--kalshi-env",
        type=str,
        default=None,
        help="Optional Kalshi environment (prod/demo) for Get Orders",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Max API pages to query when pulling live orders (default: 10)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dates = _date_range(args.start, args.end)
    ledger_frame = _load_ledger(args.ledger)
    if ledger_frame.is_empty():
        raise SystemExit(f"ledger is empty: {args.ledger}")
    series_filter = {s.upper() for s in args.series} if args.series else None
    orders = _load_orders(
        dates,
        json_paths=args.orders_json,
        kalshi_env=args.kalshi_env,
        max_pages=max(args.max_pages, 1),
    )
    entries = _extract_entries(ledger_frame, dates, series_filter, reports_root=args.reports_root)
    if not entries:
        raise SystemExit("no ledger entries found for requested dates")
    summaries = _summaries(entries)
    _write_daily_parquet(summaries, output_dir=args.output_dir)
    _print_window_table(summaries)
    _print_order_table(orders, entries)
    return 0


def _date_range(start_value: str, end_value: str | None) -> list[date]:
    start = _resolve_date(start_value)
    end = _resolve_date(end_value) if end_value else start
    if end < start:
        raise ValueError("end date precedes start date")
    days: list[date] = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _resolve_date(value: str | None) -> date:
    if not value or value.lower() == "today":
        return datetime.now(tz=ET).date()
    if value.lower() == "yesterday":
        return (datetime.now(tz=ET) - timedelta(days=1)).date()
    return date.fromisoformat(value)


def _load_ledger(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pl.read_parquet(path)
    if "timestamp_et" in frame.columns and frame["timestamp_et"].dtype == pl.Utf8:
        frame = frame.with_columns(
            pl.col("timestamp_et").str.strptime(
                pl.Datetime(time_zone="UTC"),
                format="%Y-%m-%dT%H:%M:%S%z",
                strict=False,
            )
        )
    return frame


def _subset_frame(frame: pl.DataFrame, days: Sequence[date]) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    start_dt = datetime.combine(min(days), time.min, tzinfo=ET).astimezone(UTC)
    end_dt = datetime.combine(max(days) + timedelta(days=1), time.min, tzinfo=ET).astimezone(UTC)
    filtered = frame.filter(
        (pl.col("timestamp_et") >= start_dt) & (pl.col("timestamp_et") < end_dt)
    )
    return filtered


def _extract_entries(
    frame: pl.DataFrame,
    days: Sequence[date],
    series_filter: set[str] | None,
    *,
    reports_root: Path,
) -> list[LedgerEntry]:
    subset = _subset_frame(frame, days)
    if subset.is_empty():
        return []
    manifest_cache: dict[str, dict[str, str]] = {}
    repo_root = Path(__file__).resolve().parents[1]
    entries: list[LedgerEntry] = []
    for row in subset.iter_rows(named=True):
        series = str(row.get("series") or "").upper()
        if series_filter and series not in series_filter:
            continue
        timestamp_raw = row.get("timestamp_et")
        timestamp = _ensure_timestamp(timestamp_raw)
        manifest_path = _manifest_path(row.get("manifest_path"))
        market_ticker = str(row.get("market") or row.get("event") or "")
        entry = LedgerEntry(
            series=series,
            market_ticker=market_ticker,
            price=float(row.get("price") or 0.0),
            side=str(row.get("side") or "YES").upper(),
            contracts=int(row.get("size") or 0),
            ev_after_fees=float(row.get("ev_after_fees") or 0.0),
            pnl_simulated=float(row.get("pnl_simulated") or 0.0),
            fill_ratio_model=float(row.get("fill_ratio") or 0.0),
            fill_ratio_observed=float(row.get("fill_ratio_observed") or 0.0),
            timestamp=timestamp,
            manifest_path=manifest_path,
            ledger_row=row,
        )
        entry.trading_day = timestamp.astimezone(ET).date()
        entry.window_label = _window_label(entry.series, timestamp)
        if manifest_path:
            proposals = manifest_cache.get(str(manifest_path))
            if proposals is None:
                proposals = _load_proposal_map(manifest_path, reports_root=reports_root, repo_root=repo_root)
                manifest_cache[str(manifest_path)] = proposals
            entry.market_id = proposals.get(entry.market_ticker)
        entries.append(entry)
    return entries


def _manifest_path(value: object) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.exists():
        return path
    return None


def _load_proposal_map(manifest_path: Path, *, reports_root: Path, repo_root: Path) -> dict[str, str]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    proposals_path_raw = payload.get("proposals_path")
    if not isinstance(proposals_path_raw, str):
        return {}
    proposals_path = Path(proposals_path_raw)
    if not proposals_path.is_absolute():
        candidates = [repo_root / proposals_path, manifest_path.parent / proposals_path, reports_root / proposals_path]
        for candidate in candidates:
            if candidate.exists():
                proposals_path = candidate
                break
    if not proposals_path.exists():
        return {}
    try:
        proposals_payload = json.loads(proposals_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    proposals = proposals_payload.get("proposals") if isinstance(proposals_payload, Mapping) else None
    if not isinstance(proposals, list):
        return {}
    mapping: dict[str, str] = {}
    for proposal in proposals:
        if not isinstance(proposal, Mapping):
            continue
        ticker = proposal.get("market_ticker") or proposal.get("market")
        market_id = proposal.get("market_id")
        if not ticker or not market_id:
            continue
        mapping[str(ticker)] = str(market_id)
    return mapping


def _window_label(series: str, moment: datetime) -> str:
    reference = moment.astimezone(ET)
    windows = sched_windows.windows_for_day(reference.date())
    for window in windows:
        if series.upper() not in window.series:
            continue
        if window.start_et <= reference <= window.target_et:
            return window.label
    return f"adhoc-{series.upper()}"


def _ensure_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return datetime.now(tz=UTC)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    raise TypeError("timestamp_et column must contain datetime-compatible values")


def _summaries(entries: Sequence[LedgerEntry]) -> dict[tuple[date, str, str], WindowSummary]:
    summaries: dict[tuple[date, str, str], WindowSummary] = {}
    for entry in entries:
        if entry.trading_day is None:
            continue
        label = entry.window_label or f"adhoc-{entry.series}"
        key = (entry.trading_day, label, entry.series)
        summary = summaries.get(key)
        if summary is None:
            summary = WindowSummary(day=entry.trading_day, series=entry.series, label=label)
            summaries[key] = summary
        summary.add_entry(entry)
    return summaries


def _write_daily_parquet(summaries: Mapping[tuple[date, str, str], WindowSummary], *, output_dir: Path) -> None:
    if not summaries:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: defaultdict[date, list[WindowSummary]] = defaultdict(list)
    for summary in summaries.values():
        grouped[summary.day].append(summary)
    limits = var_index.load_family_limits()
    for day, rows in grouped.items():
        records: list[dict[str, object]] = []
        family_records: defaultdict[str, float] = defaultdict(float)
        for summary in rows:
            family = var_index.SERIES_FAMILY.get(summary.series.upper(), summary.series.upper())
            limit = float(limits.get(family, 0.0) or 0.0)
            headroom_pct = 100.0
            if limit > 0.0:
                headroom_pct = max(0.0, (limit - summary.var_exposure) / limit * 100.0)
            records.append(
                {
                    "scope": WINDOW_SCOPE,
                    "window_date": day.isoformat(),
                    "window_label": summary.label,
                    "series": summary.series,
                    "ev_after_fees": summary.ev_after_fees,
                    "pnl_realized": summary.pnl_realized,
                    "delta_ev_cents_per_lot": summary.delta_ev_cents_per_lot(),
                    "fill_gap_pp": summary.fill_gap_avg(),
                    "contracts": summary.contracts,
                    "var_exposure": summary.var_exposure,
                    "var_headroom_pct": headroom_pct,
                }
            )
            family_records[summary.series.upper()] += summary.var_exposure

        for series, exposure in family_records.items():
            family = var_index.SERIES_FAMILY.get(series, series)
            limit = float(limits.get(family, 0.0) or 0.0)
            headroom_pct = 100.0
            if limit > 0.0:
                headroom_pct = max(0.0, (limit - exposure) / limit * 100.0)
            day_rows = [row for row in rows if row.series.upper() == series]
            ev_total = sum(row.ev_after_fees for row in day_rows)
            pnl_total = sum(row.pnl_realized for row in day_rows)
            contracts = sum(row.contracts for row in day_rows)
            fill_gap_pp = 0.0
            total_samples = sum(row.samples for row in day_rows)
            if total_samples > 0:
                fill_gap_pp = sum(row.fill_gap_pp for row in day_rows) / total_samples
            records.append(
                {
                    "scope": DAY_SCOPE,
                    "window_date": day.isoformat(),
                    "window_label": f"daily-{series}",
                    "series": series,
                    "ev_after_fees": ev_total,
                    "pnl_realized": pnl_total,
                    "delta_ev_cents_per_lot": (0.0 if contracts == 0 else (pnl_total - ev_total) / contracts * 100.0),
                    "fill_gap_pp": fill_gap_pp,
                    "contracts": contracts,
                    "var_exposure": exposure,
                    "var_headroom_pct": headroom_pct,
                }
            )

        output_path = output_dir / f"pnl_window_{day.isoformat()}.parquet"
        pl.DataFrame(records).write_parquet(output_path)
        print(f"[proof-of-fill] wrote {output_path}")


def _print_window_table(summaries: Mapping[tuple[date, str, str], WindowSummary]) -> None:
    if not summaries:
        return
    headers = (
        "Date",
        "Window",
        "Series",
        "EV($)",
        "PnL($)",
        "ΔEV(c/L)",
        "Fill Gap(pp)",
        "VaR(USD)",
    )
    rows = [headers]
    for key in sorted(summaries):
        summary = summaries[key]
        rows.append(
            (
                summary.day.isoformat(),
                summary.label,
                summary.series,
                f"{summary.ev_after_fees:+.2f}",
                f"{summary.pnl_realized:+.2f}",
                f"{summary.delta_ev_cents_per_lot():+.2f}",
                f"{summary.fill_gap_avg():+.2f}",
                f"{summary.var_exposure:.2f}",
            )
        )
    _print_table(rows, title="Window Summaries")


def _print_order_table(orders: Sequence[OrderFill], entries: Sequence[LedgerEntry]) -> None:
    if not orders:
        print("[proof-of-fill] no orders retrieved for requested window")
        return
    ledger_index: defaultdict[tuple[str, str], list[LedgerEntry]] = defaultdict(list)
    for entry in entries:
        if entry.market_id:
            ledger_index[(entry.market_id, entry.side)].append(entry)
    headers = (
        "Created",
        "ID",
        "Market",
        "Series",
        "Side",
        "Contracts",
        "Filled",
        "Status",
        "Fill%",
        "EV($)",
        "PnL($)",
    )
    rows = [headers]
    for order in sorted(orders, key=lambda o: o.created_at):
        series = _series_from_market_id(order.market_id)
        ledger_entry = None
        if order.market_id:
            bucket = ledger_index.get((order.market_id, order.side.upper()))
            if bucket:
                ledger_entry = bucket.pop(0)
        rows.append(
            (
                order.created_at.astimezone(ET).strftime("%Y-%m-%d %H:%M"),
                order.order_id or (order.idempotency_key or "?"),
                order.market_id or "?",
                series or "?",
                order.side,
                f"{order.contracts}",
                f"{order.filled_contracts}",
                order.status.upper(),
                f"{order.fill_ratio()*100:.1f}",
                f"{(ledger_entry.ev_after_fees if ledger_entry else 0.0):+.2f}",
                f"{(ledger_entry.pnl_simulated if ledger_entry else 0.0):+.2f}",
            )
        )
    _print_table(rows, title="Orders & Fills")


def _print_table(rows: Sequence[Sequence[str]], *, title: str) -> None:
    widths = [max(len(str(row[idx])) for row in rows) for idx in range(len(rows[0]))]
    print(f"\n== {title} ==")
    for idx, row in enumerate(rows):
        formatted = " | ".join(str(value).ljust(widths[col]) for col, value in enumerate(row))
        print(formatted)
        if idx == 0:
            print("-+-".join("-" * width for width in widths))


def _series_from_market_id(market_id: str | None) -> str | None:
    if not market_id:
        return None
    parts = market_id.split("_")
    if len(parts) >= 2:
        return parts[1]
    return None


def _load_orders(
    days: Sequence[date],
    *,
    json_paths: Sequence[Path] | None,
    kalshi_env: str | None,
    max_pages: int,
) -> list[OrderFill]:
    if json_paths:
        return _orders_from_files(json_paths)
    return _orders_from_kalshi(days, env=kalshi_env, max_pages=max_pages)


def _orders_from_files(paths: Sequence[Path]) -> list[OrderFill]:
    orders: list[OrderFill] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            candidates = payload.get("orders")
        else:
            candidates = payload
        if not isinstance(candidates, list):
            continue
        for entry in candidates:
            record = _order_from_payload(entry)
            if record:
                orders.append(record)
    return orders


def _orders_from_kalshi(days: Sequence[date], *, env: str | None, max_pages: int) -> list[OrderFill]:
    client = _build_kalshi_client(env=env)
    if client is None:
        print("[proof-of-fill] skipping Get Orders (kalshi credentials missing)")
        return []
    start = min(days)
    end = max(days) + timedelta(days=1)
    start_ts = int(datetime.combine(start, time.min, tzinfo=ET).astimezone(UTC).timestamp())
    end_ts = int(datetime.combine(end, time.min, tzinfo=ET).astimezone(UTC).timestamp())
    orders: list[OrderFill] = []
    cursor: str | None = None
    pages = 0
    while pages < max_pages:
        params = {"min_ts": start_ts, "max_ts": end_ts, "limit": ORDERS_PAGE_LIMIT}
        if cursor:
            params["cursor"] = cursor
        response = client.get("/portfolio/orders", params=params)
        payload = response.json()
        candidates = payload.get("orders") if isinstance(payload, Mapping) else None
        if not isinstance(candidates, list):
            break
        for entry in candidates:
            record = _order_from_payload(entry)
            if record:
                orders.append(record)
        cursor = payload.get("next_cursor") if isinstance(payload, Mapping) else None
        pages += 1
        if not cursor:
            break
    return orders


def _build_kalshi_client(env: str | None) -> KalshiHttpClient | None:
    try:
        base_url = None
        if env:
            endpoints = kalshi_endpoints.resolve(env)
            base_url = endpoints.rest
        return KalshiHttpClient(base_url=base_url) if base_url else KalshiHttpClient()
    except Exception as exc:  # pragma: no cover - network dependency
        print(f"[proof-of-fill] Kalshi client unavailable: {exc}")
        return None


def _order_from_payload(payload: object) -> OrderFill | None:
    if not isinstance(payload, Mapping):
        return None
    order_id = payload.get("order_id") or payload.get("id")
    market_id = payload.get("market_id") or payload.get("market")
    side = str(payload.get("side") or payload.get("direction") or "YES").upper()
    price = _safe_float(payload.get("price") or payload.get("yes_price") or 0.0)
    contracts = int(payload.get("contracts") or payload.get("quantity") or payload.get("size") or 0)
    filled_contracts = int(payload.get("filled_contracts") or payload.get("filled") or 0)
    avg_fill_price = payload.get("avg_fill_price") or payload.get("average_price")
    avg_fill_price = _safe_float(avg_fill_price) if avg_fill_price is not None else None
    status = str(payload.get("status") or "submitted")
    created_at = _ensure_utc(_parse_ts(payload.get("created_time") or payload.get("created_at")))
    updated_at_raw = payload.get("updated_time") or payload.get("updated_at")
    updated_at = _ensure_utc(_parse_ts(updated_at_raw)) if updated_at_raw else None
    idempotency = payload.get("idempotency_key")
    strike = payload.get("strike") or payload.get("strike_price")
    strike_value = _safe_float(strike) if strike is not None else None
    return OrderFill(
        order_id=str(order_id) if order_id else None,
        market_id=str(market_id) if market_id else None,
        side=side,
        price=price,
        contracts=contracts,
        filled_contracts=max(filled_contracts, 0),
        avg_fill_price=avg_fill_price,
        status=status,
        created_at=created_at,
        updated_at=updated_at,
        idempotency_key=str(idempotency) if idempotency else None,
        strike=strike_value,
    )


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_ts(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return datetime.strptime(value, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return datetime.now(tz=UTC)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed
    return datetime.now(tz=UTC)


def _ensure_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
