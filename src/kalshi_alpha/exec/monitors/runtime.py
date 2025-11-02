"""Compute runtime execution monitors and persist artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import polars as pl

from kalshi_alpha.core.risk import drawdown
from kalshi_alpha.exec.heartbeat import kill_switch_engaged, resolve_kill_switch_path
from kalshi_alpha.exec.monitors.sequential import (
    DEFAULT_SEQ_DRIFT,
    DEFAULT_SEQ_MIN_SAMPLE,
    DEFAULT_SEQ_THRESHOLD,
    SequentialGuardParams,
    SequentialGuardResult,
    evaluate_sequential_guard,
)
from kalshi_alpha.exec.monitors.summary import MONITOR_ARTIFACTS_DIR
from kalshi_alpha.exec.policy.freeze import FreezeEvaluation, evaluate_freeze_for_series

TELEMETRY_ROOT = Path("data/raw/kalshi")
LEDGER_PATH = Path("data/proc/ledger_all.parquet")
ALPHA_STATE_PATH = Path("data/proc/state/fill_alpha.json")


MonitorStatus = Literal["OK", "ALERT", "NO_DATA"]

DEFAULT_FREEZE_SERIES = ["CPI", "CLAIMS", "TENY"]


@dataclass(slots=True)
class MonitorResult:
    """Container for an individual monitor evaluation."""

    name: str
    status: MonitorStatus
    metrics: dict[str, Any]
    message: str | None = None


@dataclass(slots=True)
class RuntimeMonitorConfig:
    """Configuration knobs for runtime monitor evaluation."""

    telemetry_lookback_hours: int = 6
    ledger_lookback_days: int = 3
    ev_gap_alert_bps: float = -5.0
    ev_gap_alert_tstat: float = -2.0
    ev_gap_min_sample: int = 5
    fill_gap_alert_pp: float = -10.0
    fill_min_contracts: int = 100
    daily_loss_cap: float = 2000.0
    weekly_loss_cap: float = 6000.0
    ws_disconnect_rate_threshold: float = 1.0
    auth_error_streak_threshold: int = 3
    kill_switch_path: Path | None = None
    seq_cusum_threshold: float = DEFAULT_SEQ_THRESHOLD
    seq_cusum_drift: float = DEFAULT_SEQ_DRIFT
    seq_min_sample: int = DEFAULT_SEQ_MIN_SAMPLE


def compute_runtime_monitors(
    *,
    config: RuntimeMonitorConfig | None = None,
    telemetry_root: Path = TELEMETRY_ROOT,
    ledger_path: Path = LEDGER_PATH,
    alpha_state_path: Path = ALPHA_STATE_PATH,
    drawdown_state_dir: Path | None = None,
    now: datetime | None = None,
) -> list[MonitorResult]:
    """Evaluate runtime monitors and return structured results."""

    cfg = config or RuntimeMonitorConfig()
    moment = _ensure_utc(now or datetime.now(tz=UTC))

    telemetry_events = _load_telemetry_events(
        telemetry_root,
        since=moment - timedelta(hours=cfg.telemetry_lookback_hours),
    )
    ledger_frame = _load_ledger(ledger_path)
    alpha_state = _load_alpha_state(alpha_state_path)

    results: list[MonitorResult] = []
    results.append(_monitor_ev_gap(ledger_frame, cfg, moment))
    results.append(_monitor_fill_vs_alpha(ledger_frame, alpha_state, cfg, moment))
    seq_params = SequentialGuardParams(
        threshold=cfg.seq_cusum_threshold,
        drift=cfg.seq_cusum_drift,
        min_sample=cfg.seq_min_sample,
    )
    seq_result = evaluate_sequential_guard(
        ledger_frame,
        params=seq_params,
        window_start=moment - timedelta(days=cfg.ledger_lookback_days),
    )
    results.append(_monitor_ev_sequential(seq_result, seq_params))
    freeze_series = set(DEFAULT_FREEZE_SERIES)
    if "series" in ledger_frame.columns:
        derived = (
            ledger_frame.select(pl.col("series").str.to_uppercase().alias("series"))
            .unique()
            .to_series()
            .to_list()
        )
        freeze_series.update(derived)
    freeze_evaluations = [
        evaluate_freeze_for_series(series, now=moment)
        for series in sorted(freeze_series)
    ]
    results.append(_monitor_freeze_windows(freeze_evaluations))
    results.append(
        _monitor_drawdown(drawdown_state_dir, cfg, moment)
    )
    results.append(_monitor_ws_disconnects(telemetry_events, cfg, moment))
    results.append(_monitor_auth_errors(telemetry_events, cfg))
    results.append(_monitor_kill_switch(cfg.kill_switch_path))

    return results


def write_monitor_artifacts(
    results: Iterable[MonitorResult],
    *,
    artifacts_dir: Path = MONITOR_ARTIFACTS_DIR,
    generated_at: datetime | None = None,
) -> list[Path]:
    """Persist monitor results to JSON artifacts."""

    timestamp = _ensure_utc(generated_at or datetime.now(tz=UTC)).isoformat()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for result in results:
        payload = {
            "name": result.name,
            "status": result.status,
            "metrics": result.metrics,
            "message": result.message,
            "generated_at": timestamp,
        }
        path = artifacts_dir / f"{result.name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        written.append(path)
    return written


def build_report_summary(results: Iterable[MonitorResult]) -> list[str]:
    """Build human-readable summary lines for REPORT.md."""

    lines: list[str] = []
    for result in results:
        icon = "⚠️" if result.status == "ALERT" else "✅" if result.status == "OK" else "⬜"
        headline = f"{icon} {result.name}: {result.status}"
        if result.message:
            lines.append(f"- {headline} — {result.message}")
        else:
            lines.append(f"- {headline}")
    return lines


# --- Monitor implementations ------------------------------------------------


def _monitor_ev_gap(
    ledger: pl.DataFrame,
    cfg: RuntimeMonitorConfig,
    moment: datetime,
) -> MonitorResult:
    if ledger.is_empty():
        return MonitorResult("ev_gap", "NO_DATA", {"reason": "ledger_empty"})

    window_start = moment - timedelta(days=cfg.ledger_lookback_days)
    filtered = ledger
    if "timestamp_et" in ledger.columns:
        filtered = ledger.filter(pl.col("timestamp_et") >= window_start)
    if filtered.is_empty() or filtered.height < cfg.ev_gap_min_sample:
        return MonitorResult(
            "ev_gap",
            "NO_DATA",
            {"reason": "insufficient_samples", "count": int(filtered.height)},
        )

    delta = filtered["ev_realized_bps"] - filtered["ev_expected_bps"]
    mean_raw = delta.mean()
    mean_delta = float(mean_raw) if mean_raw is not None else 0.0
    std_raw = delta.std()
    std_delta = float(std_raw) if std_raw is not None else 0.0
    t_stat = 0.0
    if std_delta > 0 and filtered.height > 1:
        t_stat = mean_delta / (std_delta / math.sqrt(filtered.height))

    status: MonitorStatus = "OK"
    message: str | None = None
    if mean_delta <= cfg.ev_gap_alert_bps or t_stat <= cfg.ev_gap_alert_tstat:
        status = "ALERT"
        message = (
            f"Δbps {mean_delta:.2f} (t={t_stat:.2f}) below guardrails"
        )

    metrics = {
        "mean_delta_bps": mean_delta,
        "t_stat": t_stat,
        "sample_size": int(filtered.height),
    }
    return MonitorResult("ev_gap", status, metrics, message)


def _monitor_fill_vs_alpha(
    ledger: pl.DataFrame,
    alpha_state: dict[str, float],
    cfg: RuntimeMonitorConfig,
    moment: datetime,
) -> MonitorResult:
    if ledger.is_empty() or not alpha_state:
        return MonitorResult("fill_vs_alpha", "NO_DATA", {"reason": "missing_data"})

    window_start = moment - timedelta(days=cfg.ledger_lookback_days)
    filtered = ledger
    if "timestamp_et" in ledger.columns:
        filtered = ledger.filter(pl.col("timestamp_et") >= window_start)
    if filtered.is_empty():
        return MonitorResult("fill_vs_alpha", "NO_DATA", {"reason": "no_recent_trades"})

    grouped = (
        filtered.group_by(pl.col("series").str.to_uppercase())
        .agg(
            pl.sum("expected_fills").alias("fills"),
            pl.sum("size").alias("requested"),
        )
        .to_dicts()
    )

    worst_gap = 0.0
    worst_series = None
    series_metrics: list[dict[str, Any]] = []
    for row in grouped:
        series = str(row["series"])
        requested = float(row.get("requested") or 0.0)
        fills = float(row.get("fills") or 0.0)
        if requested < cfg.fill_min_contracts:
            continue
        observed = fills / requested if requested else 0.0
        alpha = alpha_state.get(series)
        if alpha is None:
            continue
        gap_pp = (observed - alpha) * 100.0
        series_metrics.append(
            {
                "series": series,
                "observed": observed,
                "alpha": alpha,
                "gap_pp": gap_pp,
            }
        )
        if gap_pp < worst_gap:
            worst_gap = gap_pp
            worst_series = series

    if not series_metrics:
        return MonitorResult("fill_vs_alpha", "NO_DATA", {"reason": "no_series_above_min"})

    status: MonitorStatus = "OK"
    message: str | None = None
    if worst_gap <= cfg.fill_gap_alert_pp:
        status = "ALERT"
        message = f"{worst_series} fill gap {worst_gap:.1f}pp below α"

    metrics = {
        "series": series_metrics,
        "worst_gap_pp": worst_gap,
        "worst_series": worst_series,
    }
    return MonitorResult("fill_vs_alpha", status, metrics, message)


def _monitor_ev_sequential(
    result: SequentialGuardResult,
    params: SequentialGuardParams,
) -> MonitorResult:
    if not result.series_stats:
        return MonitorResult(
            "ev_seq_guard",
            "NO_DATA",
            {"reason": "ledger_empty"},
        )

    sufficient_data = any(
        stats.get("sample_size", 0) >= params.min_sample
        and not stats.get("insufficient", False)
        for stats in result.series_stats.values()
    )

    if not sufficient_data:
        return MonitorResult(
            "ev_seq_guard",
            "NO_DATA",
            {
                "reason": "insufficient_samples",
                "min_sample": params.min_sample,
                "series_stats": result.series_stats,
            },
        )

    status: MonitorStatus = "ALERT" if result.alert else "OK"
    message = None
    if result.alert:
        impacted = sorted({trigger["series"] for trigger in result.triggers})
        message = "Sequential EV drift in " + ", ".join(impacted)

    metrics = {
        "threshold": params.threshold,
        "drift": params.drift,
        "min_sample": params.min_sample,
        "max_stat": result.max_stat,
        "triggers": result.triggers,
        "series_stats": result.series_stats,
    }
    return MonitorResult("ev_seq_guard", status, metrics, message)


def _monitor_freeze_windows(
    evaluations: list[FreezeEvaluation],
) -> MonitorResult:
    if not evaluations:
        return MonitorResult(
            "freeze_window",
            "NO_DATA",
            {"reason": "no_series"},
        )

    metrics = {
        "evaluations": [evaluation.to_dict() for evaluation in evaluations],
    }
    active_series = [evaluation.series for evaluation in evaluations if evaluation.freeze_active]
    if not active_series:
        return MonitorResult("freeze_window", "OK", metrics)
    message = "Freeze active for " + ", ".join(sorted(active_series))
    return MonitorResult("freeze_window", "ALERT", metrics, message)


def _monitor_drawdown(
    drawdown_state_dir: Path | None,
    cfg: RuntimeMonitorConfig,
    moment: datetime,
) -> MonitorResult:
    status = drawdown.check_limits(
        cfg.daily_loss_cap,
        cfg.weekly_loss_cap,
        now=moment,
        state_dir=drawdown_state_dir,
    )
    monitor_status: MonitorStatus = "OK" if status.ok else "ALERT"
    message = None
    if not status.ok:
        message = ", ".join(status.reasons)
    metrics = status.metrics | {
        "daily_cap": cfg.daily_loss_cap,
        "weekly_cap": cfg.weekly_loss_cap,
    }
    return MonitorResult("drawdown", monitor_status, metrics, message)


def _monitor_ws_disconnects(
    events: list[dict[str, Any]],
    cfg: RuntimeMonitorConfig,
    moment: datetime,
) -> MonitorResult:
    relevant = {
        "ws_disconnect",
        "ws_heartbeat_timeout",
    }
    disconnects = [event for event in events if event["event_type"] in relevant]
    hours = max(cfg.telemetry_lookback_hours, 1)
    rate = len(disconnects) / hours
    status: MonitorStatus = "OK"
    message = None
    if rate > cfg.ws_disconnect_rate_threshold:
        status = "ALERT"
        message = f"{len(disconnects)} disconnects in last {hours}h"
    metrics = {
        "count": len(disconnects),
        "rate_per_hour": rate,
        "lookback_hours": cfg.telemetry_lookback_hours,
    }
    return MonitorResult("ws_disconnect_rate", status, metrics, message)


def _monitor_auth_errors(
    events: list[dict[str, Any]],
    cfg: RuntimeMonitorConfig,
) -> MonitorResult:
    max_streak = 0
    current_streak = 0
    last_error_ts: datetime | None = None
    for event in events:
        if event["event_type"] != "reject":
            current_streak = 0
            continue
        payload = event.get("data") or {}
        if not isinstance(payload, dict):
            current_streak = 0
            continue
        texts = [str(payload.get("error", "")), str(payload.get("error_cause", ""))]
        normalized = " ".join(texts).lower()
        if any(token in normalized for token in ("401", "unauthorized", "auth", "signature")):
            current_streak += 1
            last_error_ts = event["timestamp"]
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    status: MonitorStatus = "OK"
    message = None
    if max_streak >= cfg.auth_error_streak_threshold:
        status = "ALERT"
        message = f"auth rejects streak {max_streak} (threshold {cfg.auth_error_streak_threshold})"

    metrics = {
        "max_streak": max_streak,
        "current_streak": current_streak,
        "last_error_ts": last_error_ts.isoformat() if last_error_ts else None,
    }
    return MonitorResult("auth_error_streak", status, metrics, message)


def _monitor_kill_switch(path: Path | None) -> MonitorResult:
    resolved = resolve_kill_switch_path(path)
    engaged = kill_switch_engaged(resolved)
    status: MonitorStatus = "ALERT" if engaged else "OK"
    message = "Kill switch engaged" if engaged else None
    metrics = {
        "path": resolved.as_posix(),
        "engaged": engaged,
    }
    return MonitorResult("kill_switch", status, metrics, message)


# --- data loading helpers ---------------------------------------------------


def _load_telemetry_events(base_dir: Path, *, since: datetime) -> list[dict[str, Any]]:
    if not base_dir.exists():
        return []
    events: list[dict[str, Any]] = []
    for path in base_dir.rglob("exec.jsonl"):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            timestamp = _parse_timestamp(record.get("timestamp"))
            if timestamp is None or timestamp < since:
                continue
            events.append(
                {
                    "timestamp": timestamp,
                    "event_type": str(record.get("event_type") or ""),
                    "data": record.get("data"),
                }
            )
    events.sort(key=lambda item: item["timestamp"])
    return events


def _load_ledger(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    frame = pl.read_parquet(path)
    if "timestamp_et" in frame.columns and frame["timestamp_et"].dtype == pl.Utf8:
        frame = frame.with_columns(pl.col("timestamp_et").str.strptime(pl.Datetime, strict=False))
    return frame


def _load_alpha_state(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    series_map = payload.get("series", {})
    result: dict[str, float] = {}
    for key, value in series_map.items():
        alpha = value
        if isinstance(value, dict):
            alpha = value.get("alpha")
        try:
            result[str(key).upper()] = float(alpha)
        except (TypeError, ValueError):
            continue
    return result


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
