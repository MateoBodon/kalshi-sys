"""Generate daily execution digest markdown + plot with optional S3 push."""

from __future__ import annotations

import argparse
import base64
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import polars as pl

from kalshi_alpha.exec import slo
from kalshi_alpha.exec.monitors.summary import (
    MONITOR_ARTIFACTS_DIR,
    summarize_monitor_artifacts,
)

LEDGER_PATH = Path("data/proc/ledger_all.parquet")
REPORTS_DIR = Path("reports")
DIGEST_DIR = REPORTS_DIR / "digests"
PNG_PLACEHOLDER = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAAAASUVORK5CYII="
)
ET = ZoneInfo("America/New_York")


@dataclass(slots=True)
class DigestArtifacts:
    markdown_path: Path
    plot_path: Path | None
    s3_objects: list[str]


@dataclass(slots=True)
class SeriesDigest:
    series: str
    trades: int
    ev_usd: float
    pnl_usd: float
    ev_delta_bps: float | None
    fill_ratio: float | None
    fill_minus_alpha: float | None
    slippage_ticks: float | None
    var_latest: float | None
    monitors: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "series": self.series,
            "trades": self.trades,
            "ev_usd": self.ev_usd,
            "pnl_usd": self.pnl_usd,
            "ev_delta_bps": self.ev_delta_bps,
            "fill_ratio": self.fill_ratio,
            "fill_minus_alpha": self.fill_minus_alpha,
            "slippage_ticks": self.slippage_ticks,
            "var_latest": self.var_latest,
            "monitors": list(self.monitors),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily digest markdown + PNG plot.")
    parser.add_argument("--date", type=str, default="yesterday", help="Target date (YYYY-MM-DD|today|yesterday).")
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH, help="Ledger parquet path.")
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR, help="Reports directory root.")
    parser.add_argument("--output", type=Path, default=DIGEST_DIR, help="Digest output directory.")
    parser.add_argument(
        "--s3",
        type=str,
        default=None,
        help="Optional s3://bucket/prefix URI for uploads (requires boto3).",
    )
    parser.add_argument("--no-write", action="store_true", help="Skip writing files; print markdown to stdout.")
    parser.add_argument("--lookback", type=int, default=7, help="Lookback days for SLO metrics (default: 7).")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    target_date = _resolve_date(args.date)
    ledger = _load_ledger(args.ledger)
    day_frame = _subset_day(ledger, target_date)
    if day_frame.is_empty():
        raise SystemExit(f"no ledger rows for {target_date} (ET)")
    series_rows = _series_summaries(day_frame, args.reports, target_date)
    slo_metrics = slo.collect_metrics(
        [row.as_dict() for row in series_rows],
        reports_root=args.reports,
        raw_root=Path("data/raw"),
        lookback_days=max(int(args.lookback), 1),
        now=datetime.combine(target_date, time(23, 59, tzinfo=ET)).astimezone(UTC),
    )
    digest_md = _render_markdown(target_date, series_rows, slo_metrics)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"digest_{target_date.isoformat()}"
    markdown_path = output_dir / f"{filename}.md"
    plot_path = output_dir / f"{filename}.png"
    s3_objects: list[str] = []
    if args.no_write:
        print(digest_md)
        plot_path = None
    else:
        markdown_path.write_text(digest_md, encoding="utf-8")
        _render_plot(series_rows, plot_path)
        print(f"[digest] wrote {markdown_path}")
        print(f"[digest] wrote {plot_path}")
    if args.s3 and not args.no_write:
        uploads = _upload_to_s3(args.s3, {markdown_path, plot_path})
        s3_objects.extend(uploads)
    _summarize_monitors(target_date)
    artifacts = DigestArtifacts(markdown_path=markdown_path, plot_path=plot_path, s3_objects=s3_objects)
    if s3_objects:
        print("[digest] uploaded:")
        for obj in s3_objects:
            print(f"  - {obj}")
    return 0 if artifacts.markdown_path.exists() or args.no_write else 1


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
        frame = frame.with_columns(pl.col("timestamp_et").str.strptime(pl.Datetime(time_zone="UTC"), strict=False))
    return frame


def _subset_day(frame: pl.DataFrame, target_date: date) -> pl.DataFrame:
    if "timestamp_et" not in frame.columns:
        raise ValueError("ledger missing timestamp_et column")
    start = datetime.combine(target_date, time(0, 0, tzinfo=ET)).astimezone(UTC)
    end = start + timedelta(days=1)
    return frame.filter((pl.col("timestamp_et") >= start) & (pl.col("timestamp_et") < end))


def _series_summaries(frame: pl.DataFrame, reports_dir: Path, target_date: date) -> list[SeriesDigest]:
    if frame.is_empty():
        return []
    aggregated = frame.group_by("series").agg(
        pl.len().alias("trades"),
        pl.sum("ev_after_fees").alias("ev_usd"),
        pl.sum("pnl_simulated").alias("pnl_usd"),
        (pl.col("ev_realized_bps") - pl.col("ev_expected_bps")).mean().alias("ev_delta_bps"),
        pl.mean("fill_ratio_observed").alias("fill_ratio"),
        (pl.col("fill_ratio_observed") - pl.col("alpha_target")).mean().alias("fill_minus_alpha"),
        pl.mean("slippage_ticks").alias("slippage_ticks"),
    )
    summaries: list[SeriesDigest] = []
    for row in aggregated.iter_rows(named=True):
        series = str(row["series"]).upper()
        report_path = _series_report_path(reports_dir, series, target_date)
        var_value = _extract_numeric(report_path, "Portfolio VaR") if report_path else None
        monitors: list[str] = []
        if report_path and report_path.exists():
            monitors = _extract_monitor_lines(report_path)
        summaries.append(
            SeriesDigest(
                series=series,
                trades=int(row.get("trades") or 0),
                ev_usd=float(row.get("ev_usd") or 0.0),
                pnl_usd=float(row.get("pnl_usd") or 0.0),
                ev_delta_bps=_safe_float(row.get("ev_delta_bps")),
                fill_ratio=_safe_float(row.get("fill_ratio")),
                fill_minus_alpha=_safe_float(row.get("fill_minus_alpha")),
                slippage_ticks=_safe_float(row.get("slippage_ticks")),
                var_latest=var_value,
                monitors=monitors,
            )
        )
    summaries.sort(key=lambda entry: entry.series)
    return summaries


def _series_report_path(reports_dir: Path, series: str, target_date: date) -> Path | None:
    candidate = reports_dir / series.upper() / f"{target_date.isoformat()}.md"
    if candidate.exists():
        return candidate
    indexed = reports_dir / "index_ladders" / series.upper() / f"{target_date.isoformat()}.md"
    if indexed.exists():
        return indexed
    return None


def _extract_numeric(path: Path | None, label: str) -> float | None:
    if path is None or not path.exists():
        return None
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError:
        return None
    needle = f"{label}:"
    for line in contents.splitlines():
        if needle not in line:
            continue
        tokens = line.split(needle, 1)[1].strip().split()
        if not tokens:
            continue
        try:
            value = float(tokens[0])
            return value
        except ValueError:
            continue
    return None


def _extract_monitor_lines(path: Path) -> list[str]:
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError:
        return []
    lines: list[str] = []
    capture = False
    for raw in contents.splitlines():
        if raw.startswith("## Monitors"):
            capture = True
            continue
        if capture:
            if raw.startswith("## "):
                break
            if raw.strip():
                lines.append(raw.strip())
    return lines


def _render_markdown(
    target_date: date,
    series_rows: Sequence[SeriesDigest],
    slo_metrics: Mapping[str, slo.SLOSeriesMetrics],
) -> str:
    total_trades = sum(row.trades for row in series_rows)
    total_ev = sum(row.ev_usd for row in series_rows)
    total_pnl = sum(row.pnl_usd for row in series_rows)
    lines: list[str] = []
    lines.append(f"# Daily Digest — {target_date.isoformat()} (ET)")
    lines.append("")
    lines.append("**Overview**")
    lines.append(f"- Trades: {total_trades}")
    lines.append(f"- EV (after fees): {total_ev:.2f} USD")
    lines.append(f"- Realized PnL: {total_pnl:.2f} USD")
    lines.append("")
    header = "| Series | Trades | EV (USD) | PnL (USD) | ΔEV (bps) | Fill-α (pp) | VaR (USD) |"
    lines.append(header)
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in series_rows:
        delta = f"{row.ev_delta_bps:+.1f}" if row.ev_delta_bps is not None else "n/a"
        fill_gap = f"{(row.fill_minus_alpha or 0)*100:+.2f}" if row.fill_minus_alpha is not None else "n/a"
        var_text = f"{row.var_latest:.2f}" if row.var_latest is not None else "n/a"
        lines.append(
            f"| {row.series} | {row.trades} | {row.ev_usd:.2f} | {row.pnl_usd:.2f} | {delta} | {fill_gap} | {var_text} |"
        )
    lines.append("")
    lines.append("## SLO Snapshots")
    for row in series_rows:
        entry = slo_metrics.get(row.series)
        if not entry:
            continue
        pieces = []
        if entry.freshness_p95_ms is not None:
            pieces.append(f"freshness p95={entry.freshness_p95_ms:.0f} ms")
        if entry.time_at_risk_p95_s is not None:
            pieces.append(f"time-at-risk p95={entry.time_at_risk_p95_s:.1f} s")
        if entry.var_headroom_usd is not None:
            pieces.append(f"VaR headroom={entry.var_headroom_usd:.1f} USD")
        if entry.ev_gap_bps is not None:
            pieces.append(f"ΔEV={entry.ev_gap_bps:+.1f} bps")
        if not pieces:
            pieces.append("no telemetry on file")
        lines.append(f"- **{row.series}**: {', '.join(pieces)}")
    lines.append("")
    lines.append("## Monitor Status")
    monitor_summary = summarize_monitor_artifacts(
        MONITOR_ARTIFACTS_DIR,
        now=datetime.now(tz=UTC),
        window=timedelta(minutes=30),
    )
    if not monitor_summary.statuses:
        lines.append("- No monitor artifacts found.")
    else:
        for name, status in sorted(monitor_summary.statuses.items()):
            lines.append(f"- {name}: {status}")
    lines.append("")
    lines.append("## Notes")
    for row in series_rows:
        if not row.monitors:
            continue
        lines.append(f"- {row.series} monitors:")
        for entry in row.monitors:
            lines.append(f"  - {entry}")
    if not any(row.monitors for row in series_rows):
        lines.append("- No additional monitor notes.")
    lines.append("")
    lines.append("Generated via `python -m report.digest`. Attach the PNG plot alongside this markdown when archiving.")
    return "\n".join(lines)


def _render_plot(series_rows: Sequence[SeriesDigest], output: Path) -> None:
    if not series_rows:
        output.write_bytes(PNG_PLACEHOLDER)
        return
    series_labels = [row.series for row in series_rows]
    ev_values = [row.ev_usd for row in series_rows]
    pnl_values = [row.pnl_usd for row in series_rows]
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        output.write_bytes(PNG_PLACEHOLDER)
        return
    fig, ax = plt.subplots(figsize=(max(4, len(series_rows)), 3))
    index = range(len(series_rows))
    ax.bar(index, ev_values, label="EV", color="#1f77b4", alpha=0.7)
    ax.bar(index, pnl_values, label="PnL", color="#ff7f0e", alpha=0.7)
    ax.set_xticks(list(index))
    ax.set_xticklabels(series_labels, rotation=30)
    ax.set_ylabel("USD")
    ax.set_title("Daily EV vs Realized")
    ax.legend()
    ax.axhline(0, color="#333", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _upload_to_s3(uri: str, paths: Iterable[Path | None]) -> list[str]:
    bucket, prefix = _parse_s3_uri(uri)
    try:
        import boto3
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("boto3 required for S3 uploads") from exc
    client = boto3.client("s3")
    uploaded: list[str] = []
    for path in paths:
        if path is None:
            continue
        key = f"{prefix.rstrip('/')}/{path.name}" if prefix else path.name
        client.upload_file(str(path), bucket, key)
        uploaded.append(f"s3://{bucket}/{key}")
    return uploaded


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError("expected s3://bucket/prefix URI")
    remainder = uri[5:]
    if "/" not in remainder:
        return remainder, ""
    bucket, prefix = remainder.split("/", 1)
    return bucket, prefix


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
        if math.isnan(numeric):
            return None
        return numeric
    except (TypeError, ValueError):
        return None


def _summarize_monitors(target_date: date) -> None:
    summary_path = REPORTS_DIR / "_artifacts" / "digests" / f"digest_{target_date.isoformat()}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=UTC)
    summary = summarize_monitor_artifacts(
        MONITOR_ARTIFACTS_DIR,
        now=now,
        window=timedelta(minutes=30),
    )
    payload = {
        "generated_at": now.isoformat(),
        "target_date": target_date.isoformat(),
        "file_count": summary.file_count,
        "max_age_minutes": summary.max_age_minutes,
        "statuses": summary.statuses,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
