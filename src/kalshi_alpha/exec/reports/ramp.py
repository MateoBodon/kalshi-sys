"""Compute pilot ramp readiness reports."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from kalshi_alpha.core.risk import drawdown

LEDGER_PATH = Path("data/proc/ledger_all.parquet")
GO_NO_GO_DIR = Path("reports/_artifacts")
JSON_OUTPUT = Path("reports/pilot_ready.json")
MARKDOWN_OUTPUT = Path("reports/pilot_readiness.md")


@dataclass(slots=True)
class RampPolicyConfig:
    lookback_days: int = 14
    min_fills: int = 300
    min_delta_bps: float = 6.0
    min_t_stat: float = 2.0
    go_multiplier: float = 1.5
    base_multiplier: float = 1.0
    daily_loss_cap: float = 2000.0
    weekly_loss_cap: float = 6000.0


def compute_ramp_policy(
    *,
    ledger_path: Path = LEDGER_PATH,
    artifacts_dir: Path = GO_NO_GO_DIR,
    drawdown_state_dir: Path | None = None,
    config: RampPolicyConfig | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    cfg = config or RampPolicyConfig()
    moment = _ensure_utc(now or datetime.now(tz=UTC))

    ledger = _load_ledger(ledger_path)
    guardrails = _load_guardrail_events(artifacts_dir, since=moment - timedelta(days=cfg.lookback_days))
    drawdown_status = drawdown.check_limits(
        cfg.daily_loss_cap,
        cfg.weekly_loss_cap,
        now=moment,
        state_dir=drawdown_state_dir,
    )

    series_stats = _aggregate_series(ledger, cfg, moment)
    results: list[dict[str, Any]] = []
    go_count = 0
    for stats in series_stats:
        breaches = guardrails.get(stats["series"], 0)
        reasons: list[str] = []
        if stats["fills"] < cfg.min_fills:
            reasons.append(f"fills<{cfg.min_fills}")
        if stats["mean_delta_bps"] < cfg.min_delta_bps:
            reasons.append(f"Δbps<{cfg.min_delta_bps}")
        if stats["t_stat"] < cfg.min_t_stat:
            reasons.append(f"t<{cfg.min_t_stat}")
        if breaches > 0:
            reasons.append(f"guardrail_breaches={breaches}")
        if not drawdown_status.ok:
            reasons.append("drawdown")

        go = not reasons
        multiplier = cfg.go_multiplier if go else cfg.base_multiplier
        if go:
            go_count += 1
        record = {
            **stats,
            "guardrail_breaches": breaches,
            "drawdown_ok": drawdown_status.ok,
            "recommendation": "GO" if go else "NO_GO",
            "size_multiplier": multiplier,
            "reasons": reasons,
        }
        results.append(record)

    policy = {
        "generated_at": moment.isoformat(),
        "criteria": {
            "min_fills": cfg.min_fills,
            "min_delta_bps": cfg.min_delta_bps,
            "min_t_stat": cfg.min_t_stat,
            "go_multiplier": cfg.go_multiplier,
            "base_multiplier": cfg.base_multiplier,
            "lookback_days": cfg.lookback_days,
            "daily_loss_cap": cfg.daily_loss_cap,
            "weekly_loss_cap": cfg.weekly_loss_cap,
        },
        "drawdown": {
            "ok": drawdown_status.ok,
            "metrics": drawdown_status.metrics,
            "reasons": drawdown_status.reasons,
        },
        "series": results,
        "overall": {
            "go": go_count,
            "no_go": len(results) - go_count,
        },
    }
    return policy


def write_ramp_outputs(
    policy: dict[str, Any],
    *,
    json_path: Path = JSON_OUTPUT,
    markdown_path: Path = MARKDOWN_OUTPUT,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Pilot Ramp Readiness", ""]
    generated = policy.get("generated_at")
    if generated:
        lines.append(f"_Generated {generated}_")
        lines.append("")

    criteria = policy.get("criteria", {})
    lines.append("**Criteria**")
    lines.append(
        "- Min fills: {min_fills}\n"
        "- Min Δbps: {min_delta_bps}\n"
        "- Min t-stat: {min_t_stat}\n"
        "- Drawdown caps (daily/weekly): {daily_loss_cap}/{weekly_loss_cap}".format(
            min_fills=criteria.get("min_fills"),
            min_delta_bps=criteria.get("min_delta_bps"),
            min_t_stat=criteria.get("min_t_stat"),
            daily_loss_cap=criteria.get("daily_loss_cap"),
            weekly_loss_cap=criteria.get("weekly_loss_cap"),
        )
    )
    lines.append("")

    lines.append(
        "| Series | Fills | Δbps | t-stat | Guardrail breaches | Drawdown | Recommendation | Multiplier |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for entry in policy.get("series", []):
        row_line = (
            "| {series} | {fills} | {delta:.2f} | {t_stat:.2f} | {breaches} | {drawdown} | "
            "{rec} | {multiplier:.2f} |"
        ).format(
            series=entry.get("series"),
            fills=int(entry.get("fills", 0)),
            delta=float(entry.get("mean_delta_bps", 0.0)),
            t_stat=float(entry.get("t_stat", 0.0)),
            breaches=entry.get("guardrail_breaches", 0),
            drawdown="OK" if entry.get("drawdown_ok") else "NO",
            rec=entry.get("recommendation"),
            multiplier=float(entry.get("size_multiplier", 1.0)),
        )
        lines.append(row_line)

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate pilot ramp readiness outputs.")
    parser.add_argument("--ledger-path", type=Path, default=LEDGER_PATH)
    parser.add_argument("--artifacts-dir", type=Path, default=GO_NO_GO_DIR)
    parser.add_argument("--drawdown-state-dir", type=Path, default=None)
    parser.add_argument("--json-path", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--markdown-path", type=Path, default=MARKDOWN_OUTPUT)
    parser.add_argument("--lookback-days", type=int, default=RampPolicyConfig().lookback_days)
    parser.add_argument("--min-fills", type=int, default=RampPolicyConfig().min_fills)
    parser.add_argument("--min-delta-bps", type=float, default=RampPolicyConfig().min_delta_bps)
    parser.add_argument("--min-t-stat", type=float, default=RampPolicyConfig().min_t_stat)
    parser.add_argument("--go-multiplier", type=float, default=RampPolicyConfig().go_multiplier)
    parser.add_argument("--base-multiplier", type=float, default=RampPolicyConfig().base_multiplier)
    parser.add_argument("--daily-loss-cap", type=float, default=RampPolicyConfig().daily_loss_cap)
    parser.add_argument("--weekly-loss-cap", type=float, default=RampPolicyConfig().weekly_loss_cap)
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = RampPolicyConfig(
        lookback_days=args.lookback_days,
        min_fills=args.min_fills,
        min_delta_bps=args.min_delta_bps,
        min_t_stat=args.min_t_stat,
        go_multiplier=args.go_multiplier,
        base_multiplier=args.base_multiplier,
        daily_loss_cap=args.daily_loss_cap,
        weekly_loss_cap=args.weekly_loss_cap,
    )
    policy = compute_ramp_policy(
        ledger_path=args.ledger_path,
        artifacts_dir=args.artifacts_dir,
        drawdown_state_dir=args.drawdown_state_dir,
        config=config,
    )
    write_ramp_outputs(policy, json_path=args.json_path, markdown_path=args.markdown_path)
    print(json.dumps(policy["overall"], indent=2, sort_keys=True))
    return 0


def _aggregate_series(ledger: pl.DataFrame, cfg: RampPolicyConfig, moment: datetime) -> list[dict[str, Any]]:
    if ledger.is_empty():
        return []
    window_start = moment - timedelta(days=cfg.lookback_days)
    filtered = ledger
    if "timestamp_et" in ledger.columns:
        filtered = ledger.filter(pl.col("timestamp_et") >= window_start)
    if filtered.is_empty():
        return []

    delta = (pl.col("ev_realized_bps") - pl.col("ev_expected_bps")).alias("delta_bps")
    grouped = (
        filtered.with_columns(delta)
        .group_by(pl.col("series").str.to_uppercase())
        .agg(
            pl.sum("expected_fills").alias("fills"),
            pl.len().alias("trades"),
            pl.mean("delta_bps").alias("mean_delta_bps"),
            pl.std("delta_bps").alias("delta_std"),
        )
        .to_dicts()
    )

    stats: list[dict[str, Any]] = []
    for row in grouped:
        fills = float(row.get("fills") or 0.0)
        trades = int(row.get("trades") or 0)
        mean_delta = float(row.get("mean_delta_bps") or 0.0)
        std_delta = float(row.get("delta_std") or 0.0)
        t_stat = 0.0
        if std_delta > 0.0 and trades > 1:
            t_stat = mean_delta / (std_delta / math.sqrt(trades))
        stats.append(
            {
                "series": str(row["series"]),
                "fills": int(round(fills)),
                "trades": trades,
                "mean_delta_bps": mean_delta,
                "t_stat": t_stat,
            }
        )
    return sorted(stats, key=lambda item: item["series"])


def _load_ledger(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    frame = pl.read_parquet(path)
    if "timestamp_et" in frame.columns and frame["timestamp_et"].dtype == pl.Utf8:
        frame = frame.with_columns(pl.col("timestamp_et").str.strptime(pl.Datetime, strict=False))
    return frame


def _load_guardrail_events(artifacts_dir: Path, *, since: datetime) -> dict[str, int]:
    counters: dict[str, int] = {}
    if not artifacts_dir.exists():
        return counters
    for path in artifacts_dir.glob("go_no_go*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        series = str(payload.get("series", "")).upper()
        ts_text = payload.get("timestamp")
        ts = _parse_timestamp(ts_text)
        if ts is None or ts < since:
            continue
        go = bool(payload.get("go", True))
        if not go and series:
            counters[series] = counters.get(series, 0) + 1
    return counters


def _parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return _ensure_utc(parsed)
    return None


def _ensure_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
