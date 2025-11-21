"""Scoreboard for index paper (dry) ledger entries."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence
from zoneinfo import ZoneInfo

from kalshi_alpha.datastore.paths import REPORTS_ROOT
from kalshi_alpha.exec.index_paper_ledger import (
    DEFAULT_LEDGER_PATH,
    INDEX_SERIES,
    LEDGER_ENV_KEY,
)

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class LedgerEntry:
    timestamp_et: datetime
    trade_date: date
    series: str
    window: str | None
    ev_after_fees_cents: float


def _resolve_ledger_path(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg)
    env_override = os.getenv(LEDGER_ENV_KEY)
    if env_override:
        return Path(env_override)
    return DEFAULT_LEDGER_PATH


def _default_date_range(days: int = 7) -> tuple[date, date]:
    end_date = datetime.now(tz=ET).date()
    start_date = end_date - timedelta(days=max(days, 1) - 1)
    return start_date, end_date


def _parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        stamp = value if value.tzinfo else value.replace(tzinfo=UTC)
        return stamp.astimezone(ET)
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ET)
    return parsed.astimezone(ET)


def _parse_record(payload: Mapping[str, object]) -> LedgerEntry | None:
    timestamp = _parse_timestamp(payload.get("timestamp_et"))
    if timestamp is None:
        return None
    series_raw = str(payload.get("series") or "").strip().upper()
    if series_raw not in INDEX_SERIES:
        return None
    ev_value = payload.get("ev_after_fees_cents")
    try:
        ev_cents = float(ev_value)
    except (TypeError, ValueError):
        return None
    window_raw = payload.get("window")
    window_label = str(window_raw) if window_raw else None
    return LedgerEntry(
        timestamp_et=timestamp,
        trade_date=timestamp.date(),
        series=series_raw,
        window=window_label,
        ev_after_fees_cents=ev_cents,
    )


def load_entries(ledger_path: Path, *, start_date: date, end_date: date) -> list[LedgerEntry]:
    if not ledger_path.exists():
        return []
    entries: list[LedgerEntry] = []
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:  # pragma: no cover - tolerate malformed lines
                continue
            if not isinstance(payload, Mapping):
                continue
            parsed = _parse_record(payload)
            if parsed is None:
                continue
            if start_date <= parsed.trade_date <= end_date:
                entries.append(parsed)
    return entries


def aggregate(entries: Sequence[LedgerEntry]) -> list[dict[str, object]]:
    buckets: dict[tuple[date, str, str | None], dict[str, float | int | date | str | None]] = {}
    for entry in entries:
        key = (entry.trade_date, entry.series, entry.window)
        bucket = buckets.setdefault(
            key,
            {"date": entry.trade_date, "series": entry.series, "window": entry.window, "trades": 0, "ev_sum": 0.0},
        )
        bucket["trades"] = int(bucket["trades"]) + 1
        bucket["ev_sum"] = float(bucket["ev_sum"]) + float(entry.ev_after_fees_cents)
    summaries: list[dict[str, object]] = []
    for (_date, _series, _window), values in buckets.items():
        trades = int(values["trades"])
        ev_sum = float(values["ev_sum"])
        avg_ev = ev_sum / trades if trades else 0.0
        summaries.append(
            {
                "date": values["date"],
                "series": values["series"],
                "window": values["window"],
                "trades": trades,
                "ev_sum_cents": ev_sum,
                "avg_ev_cents": avg_ev,
            }
        )
    summaries.sort(key=lambda row: (row["date"], row["series"], row["window"] or ""))
    return summaries


def render_markdown(
    summaries: Sequence[Mapping[str, object]],
    *,
    start_date: date,
    end_date: date,
) -> str:
    lines: list[str] = []
    lines.append(f"# Index Paper Scoreboard ({start_date.isoformat()} → {end_date.isoformat()})")
    total_trades = sum(int(row.get("trades", 0)) for row in summaries)
    total_ev = sum(float(row.get("ev_sum_cents", 0.0)) for row in summaries)
    lines.append("")
    lines.append(f"- Trades: {total_trades}")
    lines.append(f"- EV sum (cents): {total_ev:.2f}")
    lines.append("")
    if not summaries:
        lines.append("_No paper trades recorded for this range._")
        return "\n".join(lines) + "\n"
    lines.append("| Date | Series | Window | Trades | EV sum (cents) | Avg EV (cents) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in summaries:
        window = row.get("window") or "unspecified"
        lines.append(
            "| {date} | {series} | {window} | {trades} | {ev_sum:.2f} | {avg_ev:.2f} |".format(
                date=row["date"],
                series=row["series"],
                window=window,
                trades=int(row.get("trades", 0)),
                ev_sum=float(row.get("ev_sum_cents", 0.0)),
                avg_ev=float(row.get("avg_ev_cents", 0.0)),
            )
        )
    return "\n".join(lines) + "\n"


def _build_output_path(output_arg: str | None, end_date: date) -> Path:
    if output_arg:
        return Path(output_arg)
    target_dir = REPORTS_ROOT / "index_paper"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{end_date.strftime('%Y%m%d')}_scoreboard.md"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate index paper ledger scoreboard.")
    parser.add_argument("--start-date", help="Inclusive start date YYYY-MM-DD (default: last 7 days).")
    parser.add_argument("--end-date", help="Inclusive end date YYYY-MM-DD (default: today ET).")
    parser.add_argument("--days", type=int, default=7, help="Fallback lookback window when no dates are provided.")
    parser.add_argument("--ledger", help="Override path to index_paper.jsonl.")
    parser.add_argument("--output", help="Output markdown path (default: reports/index_paper/<end>_scoreboard.md).")
    args = parser.parse_args(argv)

    if args.start_date or args.end_date:
        try:
            start_date = date.fromisoformat(args.start_date) if args.start_date else None
            end_date = date.fromisoformat(args.end_date) if args.end_date else None
        except ValueError as exc:  # pragma: no cover - defensive
            raise SystemExit(f"Invalid date format: {exc}") from exc
        if start_date is None or end_date is None:
            today = datetime.now(tz=ET).date()
            if start_date is None:
                start_date = today - timedelta(days=max(args.days, 1) - 1)
            if end_date is None:
                end_date = today
    else:
        start_date, end_date = _default_date_range(args.days)

    ledger_path = _resolve_ledger_path(args.ledger)
    entries = load_entries(ledger_path, start_date=start_date, end_date=end_date)
    summaries = aggregate(entries)
    markdown = render_markdown(summaries, start_date=start_date, end_date=end_date)
    output_path = _build_output_path(args.output, end_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote index paper scoreboard to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["LedgerEntry", "aggregate", "load_entries", "main", "render_markdown"]
