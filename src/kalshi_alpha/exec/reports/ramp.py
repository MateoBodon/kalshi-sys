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
from kalshi_alpha.exec.monitors.sequential import (
    DEFAULT_SEQ_DRIFT,
    DEFAULT_SEQ_MIN_SAMPLE,
    DEFAULT_SEQ_THRESHOLD,
    SequentialGuardParams,
    evaluate_sequential_guard,
)
from kalshi_alpha.exec.monitors.summary import (
    DEFAULT_MONITOR_MAX_AGE_MINUTES,
    DEFAULT_PANIC_ALERT_THRESHOLD,
    DEFAULT_PANIC_ALERT_WINDOW_MINUTES,
    MONITOR_ARTIFACTS_DIR,
    summarize_monitor_artifacts,
)
from kalshi_alpha.exec.policy.freeze import FreezeEvaluation, evaluate_freeze_for_series

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
    ledger_max_age_minutes: int = 120
    monitor_max_age_minutes: int = DEFAULT_MONITOR_MAX_AGE_MINUTES
    panic_alert_threshold: int = DEFAULT_PANIC_ALERT_THRESHOLD
    panic_alert_window_minutes: int = DEFAULT_PANIC_ALERT_WINDOW_MINUTES
    seq_guard_threshold: float = DEFAULT_SEQ_THRESHOLD
    seq_guard_drift: float = DEFAULT_SEQ_DRIFT
    seq_guard_min_sample: int = DEFAULT_SEQ_MIN_SAMPLE


def compute_ramp_policy(
    *,
    ledger_path: Path = LEDGER_PATH,
    artifacts_dir: Path = GO_NO_GO_DIR,
    monitor_artifacts_dir: Path = MONITOR_ARTIFACTS_DIR,
    drawdown_state_dir: Path | None = None,
    config: RampPolicyConfig | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    cfg = config or RampPolicyConfig()
    moment = _ensure_utc(now or datetime.now(tz=UTC))

    ledger = _load_ledger(ledger_path)
    panic_window = max(cfg.panic_alert_window_minutes, 0)
    monitor_summary = summarize_monitor_artifacts(
        monitor_artifacts_dir,
        now=moment,
        window=timedelta(minutes=panic_window),
    )
    seq_params = SequentialGuardParams(
        threshold=cfg.seq_guard_threshold,
        drift=cfg.seq_guard_drift,
        min_sample=cfg.seq_guard_min_sample,
    )
    seq_result = evaluate_sequential_guard(
        ledger,
        params=seq_params,
        window_start=moment - timedelta(days=cfg.lookback_days),
    )
    seq_alerts_by_series: dict[str, list[dict[str, Any]]] = {}
    for trigger in seq_result.triggers:
        series_name = str(trigger.get("series", "")).upper()
        if not series_name:
            continue
        seq_alerts_by_series.setdefault(series_name, []).append(trigger)
    ledger_age_minutes = _file_age_minutes(ledger_path, moment)
    guardrails = _load_guardrail_events(artifacts_dir, since=moment - timedelta(days=cfg.lookback_days))
    drawdown_status = drawdown.check_limits(
        cfg.daily_loss_cap,
        cfg.weekly_loss_cap,
        now=moment,
        state_dir=drawdown_state_dir,
    )

    freshness_summary: dict[str, Any] = {
        "ledger_path": ledger_path.as_posix(),
        "ledger_minutes": _format_minutes(ledger_age_minutes),
        "ledger_threshold_minutes": cfg.ledger_max_age_minutes,
        "monitors_dir": monitor_artifacts_dir.as_posix(),
        "monitors_minutes": _format_minutes(monitor_summary.max_age_minutes),
        "monitors_threshold_minutes": cfg.monitor_max_age_minutes,
    }
    if monitor_summary.latest_generated_at is not None:
        freshness_summary["monitors_generated_at"] = monitor_summary.latest_generated_at.isoformat()

    global_reasons: list[str] = []
    if ledger_age_minutes is None:
        global_reasons.append("ledger_missing")
    elif ledger_age_minutes > cfg.ledger_max_age_minutes:
        global_reasons.append("ledger_stale")

    if monitor_summary.file_count == 0:
        global_reasons.append("monitors_missing")
    else:
        monitors_age = monitor_summary.max_age_minutes
        if monitors_age is not None and monitors_age > cfg.monitor_max_age_minutes:
            global_reasons.append("monitors_stale")

    if monitor_summary.statuses.get("kill_switch") == "ALERT":
        global_reasons.append("kill_switch_engaged")

    panic_backoff = False
    if cfg.panic_alert_threshold > 0:
        panic_backoff = len(monitor_summary.alerts_recent) >= cfg.panic_alert_threshold
        if panic_backoff:
            global_reasons.append("panic_backoff")

    global_reasons = _dedupe_reasons(global_reasons)

    monitors_summary: dict[str, Any] = {
        "statuses": monitor_summary.statuses,
        "max_age_minutes": _format_minutes(monitor_summary.max_age_minutes),
        "latest_generated_at": monitor_summary.latest_generated_at.isoformat()
        if monitor_summary.latest_generated_at
        else None,
        "file_count": monitor_summary.file_count,
        "alerts_recent": sorted(monitor_summary.alerts_recent),
        "panic_backoff": panic_backoff,
        "panic_threshold": cfg.panic_alert_threshold,
        "panic_window_minutes": cfg.panic_alert_window_minutes,
    }
    if monitor_summary.metrics:
        monitors_summary["metrics"] = monitor_summary.metrics
        kill_metrics = monitor_summary.metrics.get("kill_switch")
        if isinstance(kill_metrics, dict):
            path_value = kill_metrics.get("path")
            if isinstance(path_value, str):
                monitors_summary.setdefault("kill_switch_path", path_value)
    monitors_summary["sequential_triggers"] = seq_result.triggers
    monitors_summary["sequential_max_stat"] = seq_result.max_stat
    monitors_summary["sequential_params"] = {
        "threshold": seq_params.threshold,
        "drift": seq_params.drift,
        "min_sample": seq_params.min_sample,
    }
    monitors_summary["sequential_series_stats"] = seq_result.series_stats

    series_stats = _aggregate_series(ledger, cfg, moment)
    freeze_evaluations: dict[str, FreezeEvaluation] = {}
    freeze_violations: set[str] = set()
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

        seq_stats = seq_result.series_stats.get(stats["series"])
        record_seq_stats: dict[str, Any] | None = None
        if seq_stats:
            record_seq_stats = dict(seq_stats)
            if seq_stats.get("insufficient"):
                record_seq_stats.setdefault("note", "insufficient_samples")
        seq_alerts = seq_alerts_by_series.get(stats["series"], [])
        if seq_alerts:
            reasons.append("sequential_alert")

        freeze_eval = freeze_evaluations.get(stats["series"])
        if freeze_eval is None:
            freeze_eval = evaluate_freeze_for_series(stats["series"], now=moment)
            freeze_evaluations[stats["series"]] = freeze_eval
        if freeze_eval.freeze_active:
            reasons.append("freeze_window")
            freeze_violations.add(stats["series"])

        combined_reasons = _dedupe_reasons([*reasons, *global_reasons])
        go = not combined_reasons
        multiplier = cfg.go_multiplier if go else cfg.base_multiplier
        if go:
            go_count += 1
        record = {
            **stats,
            "guardrail_breaches": breaches,
            "drawdown_ok": drawdown_status.ok,
            "recommendation": "GO" if go else "NO_GO",
            "size_multiplier": multiplier,
            "reasons": combined_reasons,
        }
        if record_seq_stats is not None:
            record["sequential_stats"] = record_seq_stats
        if seq_alerts:
            record["sequential_alerts"] = seq_alerts
        record["freeze"] = freeze_eval.to_dict()
        results.append(record)

    monitors_summary["freeze_status"] = {
        series: evaluation.to_dict() for series, evaluation in freeze_evaluations.items()
    }
    monitors_summary["freeze_violation_series"] = sorted(freeze_violations)

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
            "ledger_max_age_minutes": cfg.ledger_max_age_minutes,
            "monitor_max_age_minutes": cfg.monitor_max_age_minutes,
            "panic_alert_threshold": cfg.panic_alert_threshold,
            "panic_alert_window_minutes": cfg.panic_alert_window_minutes,
            "seq_guard_threshold": cfg.seq_guard_threshold,
            "seq_guard_drift": cfg.seq_guard_drift,
            "seq_guard_min_sample": cfg.seq_guard_min_sample,
        },
        "drawdown": {
            "ok": drawdown_status.ok,
            "metrics": drawdown_status.metrics,
            "reasons": drawdown_status.reasons,
        },
        "freshness": freshness_summary,
        "monitors_summary": monitors_summary,
        "series": results,
        "overall": {
            "go": go_count,
            "no_go": len(results) - go_count,
            "global_reasons": global_reasons,
            "panic_backoff": panic_backoff,
            "sequential_alert_series": sorted(seq_alerts_by_series.keys()),
            "freeze_violation_series": sorted(freeze_violations),
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

    overall = policy.get("overall", {})
    global_reasons = [str(reason) for reason in overall.get("global_reasons", [])]
    if global_reasons:
        lines.append("**Global NO-GO reasons:** " + ", ".join(global_reasons))
        lines.append("")

    freshness = policy.get("freshness", {})
    if freshness:
        lines.append("**Freshness**")
        ledger_age = freshness.get("ledger_minutes")
        monitors_age = freshness.get("monitors_minutes")
        ledger_limit = freshness.get("ledger_threshold_minutes")
        monitors_limit = freshness.get("monitors_threshold_minutes")
        ledger_age_str = _format_number_for_markdown(ledger_age)
        ledger_limit_str = _format_number_for_markdown(ledger_limit)
        monitors_age_str = _format_number_for_markdown(monitors_age)
        monitors_limit_str = _format_number_for_markdown(monitors_limit)
        lines.append(f"- Ledger age: {ledger_age_str} min (limit {ledger_limit_str})")
        lines.append(f"- Monitors age: {monitors_age_str} min (limit {monitors_limit_str})")
        lines.append("")

    sequential_triggers = (
        policy.get("monitors_summary", {}).get("sequential_triggers") or []
    )
    if sequential_triggers:
        lines.append("**Sequential Guard Alerts**")
        for trigger in sequential_triggers:
            series = trigger.get("series", "?")
            direction = trigger.get("direction", "?")
            stat = trigger.get("stat", 0.0)
            threshold = trigger.get("threshold", 0.0)
            ts = trigger.get("timestamp") or "n/a"
            lines.append(
                f"- {series}: {direction} stat {stat:.2f} (threshold {threshold:.2f}) at {ts}"
            )
        lines.append("")

    freeze_status = policy.get("monitors_summary", {}).get("freeze_status", {})
    if freeze_status:
        lines.append("**Freeze Windows**")
        for series, status in freeze_status.items():
            freeze_active = bool(status.get("freeze_active"))
            state_label = "ACTIVE" if freeze_active else "clear"
            freeze_start = status.get("freeze_start", {}) or {}
            freeze_et = freeze_start.get("et") or "n/a"
            lines.append(f"- {series}: {state_label} (freeze starts {freeze_et})")
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
    parser.add_argument("--monitor-artifacts-dir", type=Path, default=MONITOR_ARTIFACTS_DIR)
    parser.add_argument(
        "--max-ledger-age-minutes",
        type=int,
        default=RampPolicyConfig().ledger_max_age_minutes,
    )
    parser.add_argument(
        "--max-monitor-age-minutes",
        type=int,
        default=RampPolicyConfig().monitor_max_age_minutes,
    )
    parser.add_argument(
        "--panic-alert-threshold",
        type=int,
        default=RampPolicyConfig().panic_alert_threshold,
    )
    parser.add_argument(
        "--panic-alert-window-minutes",
        type=int,
        default=RampPolicyConfig().panic_alert_window_minutes,
    )
    parser.add_argument(
        "--seq-guard-threshold",
        type=float,
        default=RampPolicyConfig().seq_guard_threshold,
    )
    parser.add_argument(
        "--seq-guard-drift",
        type=float,
        default=RampPolicyConfig().seq_guard_drift,
    )
    parser.add_argument(
        "--seq-guard-min-sample",
        type=int,
        default=RampPolicyConfig().seq_guard_min_sample,
    )
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
        ledger_max_age_minutes=args.max_ledger_age_minutes,
        monitor_max_age_minutes=args.max_monitor_age_minutes,
        panic_alert_threshold=args.panic_alert_threshold,
        panic_alert_window_minutes=args.panic_alert_window_minutes,
        seq_guard_threshold=args.seq_guard_threshold,
        seq_guard_drift=args.seq_guard_drift,
        seq_guard_min_sample=args.seq_guard_min_sample,
    )
    policy = compute_ramp_policy(
        ledger_path=args.ledger_path,
        artifacts_dir=args.artifacts_dir,
        monitor_artifacts_dir=args.monitor_artifacts_dir,
        drawdown_state_dir=args.drawdown_state_dir,
        config=config,
    )
    write_ramp_outputs(policy, json_path=args.json_path, markdown_path=args.markdown_path)
    print(json.dumps(policy["overall"], indent=2, sort_keys=True))
    return 0


def _format_minutes(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return round(value, 2)


def _format_number_for_markdown(value: float | int | str | None) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "n/a"
    if abs(numeric - round(numeric)) < 0.1:
        return f"{numeric:.0f}"
    return f"{numeric:.1f}"


def _file_age_minutes(path: Path, moment: datetime) -> float | None:
    if not path.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return None
    if mtime > moment:
        return 0.0
    delta = moment - mtime
    return delta.total_seconds() / 60.0


def _dedupe_reasons(reasons: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for reason in reasons:
        if reason not in seen:
            ordered.append(reason)
            seen.add(reason)
    return ordered


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
