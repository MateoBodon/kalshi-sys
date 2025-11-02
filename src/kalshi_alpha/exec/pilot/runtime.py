"""Runtime helpers for configuring pilot sessions."""

from __future__ import annotations

import argparse
import json
import math
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kalshi_alpha.exec.ledger import PaperLedger
from kalshi_alpha.exec.monitors.summary import MonitorArtifactsSummary

from .config import PilotConfig, load_pilot_config

TokenFactory = Callable[[int], str]


@dataclass(frozen=True)
class PilotSession:
    """Captures identifying metadata for a pilot run."""

    session_id: str
    series: str
    started_at: datetime
    config: PilotConfig

    def metadata(self) -> dict[str, object]:
        started_at = self.started_at.isoformat()
        return {
            "session_id": self.session_id,
            "series": self.series,
            "family": self.series,
            "started_at": started_at,
            "max_contracts_per_order": self.config.max_contracts_per_order,
            "max_unique_bins": self.config.max_unique_bins,
            "maker_only": self.config.enforce_maker_only,
            "require_live_broker": self.config.require_live_broker,
        }


def _generate_session_id(*, prefix: str, series: str, now: datetime, token_factory: TokenFactory | None) -> str:
    token_fn = token_factory or secrets.token_hex
    return f"{prefix}-{series.lower()}-{now.strftime('%Y%m%dT%H%M%SZ')}-{token_fn(3)}"


def apply_pilot_mode(
    args: argparse.Namespace,
    *,
    now: datetime | None = None,
    token_factory: TokenFactory | None = None,
) -> PilotSession | None:
    """Mutate CLI args in-place to enforce pilot constraints."""

    if not getattr(args, "pilot", False):
        return None

    config_path = getattr(args, "pilot_config", None)
    config = load_pilot_config(config_path)

    series = getattr(args, "series", None)
    if not series:
        raise ValueError("Pilot mode requires --series")
    normalized_series = series.strip().upper()
    if config.allowed_series and normalized_series not in config.allowed_series:
        allowed = ", ".join(config.allowed_series)
        raise ValueError(f"Series {normalized_series} not permitted for pilot mode (allowed: {allowed})")

    broker = getattr(args, "broker", "dry").strip().lower()
    paper_ledger = bool(getattr(args, "paper_ledger", False))
    if config.require_live_broker and broker != "live" and not paper_ledger:
        raise ValueError("Pilot mode requires --broker live")
    needs_ack = config.require_acknowledgement and not (paper_ledger and broker != "live")
    if needs_ack and not getattr(args, "i_understand_the_risks", False):
        raise ValueError("Pilot mode requires --i-understand-the-risks acknowledgement")
    if getattr(args, "offline", False):
        raise ValueError("Pilot mode cannot run with --offline fixtures")

    args.online = True

    if config.enforce_maker_only:
        args.maker_only = True

    current_contracts = getattr(args, "contracts", config.max_contracts_per_order)
    sanitized_contracts = max(1, min(int(current_contracts), config.max_contracts_per_order))
    args.contracts = sanitized_contracts

    sizing_mode = getattr(args, "sizing", "fixed")
    if sizing_mode != "fixed":
        args.sizing = "fixed"

    max_legs = getattr(args, "max_legs", None)
    if max_legs is not None:
        args.max_legs = min(int(max_legs), config.max_unique_bins)

    for cap_attr, limit in (("daily_loss_cap", config.max_daily_loss), ("weekly_loss_cap", config.max_weekly_loss)):
        current_limit = getattr(args, cap_attr, None)
        if limit is None:
            continue
        if current_limit is None:
            setattr(args, cap_attr, limit)
        else:
            setattr(args, cap_attr, min(float(current_limit), limit))

    session_started = now or datetime.now(tz=UTC)
    session_id = _generate_session_id(
        prefix=config.session_prefix,
        series=normalized_series,
        now=session_started,
        token_factory=token_factory,
    )
    session = PilotSession(
        session_id=session_id,
        series=normalized_series,
        started_at=session_started,
        config=config,
    )
    args.pilot_session_id = session.session_id
    args.pilot_session_started_at = session.started_at.isoformat()
    args.pilot_max_unique_bins = config.max_unique_bins
    args.pilot_max_contracts = config.max_contracts_per_order
    return session


def _delta_stats(records: Sequence[object]) -> tuple[float | None, float | None, int]:
    deltas: list[float] = []
    for record in records:
        realized = getattr(record, "ev_realized_bps", None)
        expected = getattr(record, "ev_expected_bps", None)
        if realized is None or expected is None:
            continue
        try:
            realized_val = float(realized)
            expected_val = float(expected)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if math.isnan(realized_val) or math.isnan(expected_val):  # pragma: no cover - defensive
            continue
        deltas.append(realized_val - expected_val)
    if not deltas:
        return None, None, 0
    mean_delta = sum(deltas) / len(deltas)
    if len(deltas) < 2:
        return mean_delta, None, len(deltas)
    variance = sum((value - mean_delta) ** 2 for value in deltas) / (len(deltas) - 1)
    std_dev = math.sqrt(max(variance, 0.0))
    if std_dev == 0.0:
        return mean_delta, 0.0, len(deltas)
    t_stat = mean_delta / (std_dev / math.sqrt(len(deltas)))
    return mean_delta, t_stat, len(deltas)


def _resolve_fill_gap(
    monitors: dict[str, object] | None,
    summary: MonitorArtifactsSummary | None,
) -> float | None:
    if summary is not None:
        fill_metrics = summary.metrics.get("fill_vs_alpha") if summary.metrics else None
        if isinstance(fill_metrics, dict):
            gap = fill_metrics.get("worst_gap_pp")
            if isinstance(gap, (int, float)):
                return float(gap)
    if monitors:
        gap = monitors.get("fill_realism_median")
        if isinstance(gap, (int, float)):
            return float(gap)
    return None


def build_pilot_session_payload(  # noqa: PLR0913
    *,
    session: PilotSession,
    ledger: PaperLedger | None,
    monitors: dict[str, object] | None,
    monitor_summary: MonitorArtifactsSummary | None,
    broker_status: dict[str, object] | None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    timestamp = generated_at or datetime.now(tz=UTC)
    records = ledger.records if isinstance(ledger, PaperLedger) else getattr(ledger, "records", []) or []
    mean_delta, t_stat, sample_size = _delta_stats(records)
    trades = len(records)
    orders_recorded = None
    broker_mode = None
    if broker_status:
        orders_value = broker_status.get("orders_recorded")
        if isinstance(orders_value, (int, float)):
            orders_recorded = int(orders_value)
        broker_mode = broker_status.get("mode")
    if trades == 0 and orders_recorded is not None:
        trades = orders_recorded

    fill_gap = _resolve_fill_gap(monitors or {}, monitor_summary)
    cusum_status = None
    if monitor_summary is not None:
        cusum_status = monitor_summary.statuses.get("ev_seq_guard")

    alerts_summary: dict[str, Any] = {}
    if monitor_summary is not None:
        alerts_summary = {
            "recent_alerts": sorted(monitor_summary.alerts_recent),
            "statuses": dict(monitor_summary.statuses),
            "max_age_minutes": monitor_summary.max_age_minutes,
        }
        if monitor_summary.metrics:
            alerts_summary["metrics"] = monitor_summary.metrics

    payload: dict[str, Any] = {
        **session.metadata(),
        "family": session.series,
        "generated_at": timestamp.isoformat(),
        "n_trades": trades,
        "sample_size": sample_size,
        "mean_delta_bps_after_fees": mean_delta,
        "t_stat": t_stat,
        "cusum_state": cusum_status,
        "fill_realism_gap": fill_gap,
        "alerts_summary": alerts_summary,
    }
    if cusum_status is not None:
        payload.setdefault("cusum_status", cusum_status)
    if monitors:
        ev_table = monitors.get("ev_honesty_table")
        if isinstance(ev_table, list):
            payload["ev_honesty_table"] = ev_table
        threshold_value = monitors.get("ev_honesty_threshold")
        if isinstance(threshold_value, (int, float)):
            payload["ev_honesty_threshold"] = float(threshold_value)
        max_delta_value = monitors.get("ev_honesty_max_delta")
        if isinstance(max_delta_value, (int, float)):
            payload["ev_honesty_max_delta"] = float(max_delta_value)
        if "ev_honesty_no_go" in monitors:
            payload["ev_honesty_no_go"] = bool(monitors.get("ev_honesty_no_go"))
    if orders_recorded is not None:
        payload["orders_recorded"] = orders_recorded
    if broker_mode is not None:
        payload["broker_mode"] = broker_mode
    return payload


def write_pilot_session_artifact(  # noqa: PLR0913
    *,
    session: PilotSession,
    ledger: PaperLedger | None,
    monitors: dict[str, object] | None,
    monitor_summary: MonitorArtifactsSummary | None,
    broker_status: dict[str, object] | None,
    artifacts_dir: Path | None = None,
    generated_at: datetime | None = None,
) -> Path:
    output_dir = Path(artifacts_dir) if artifacts_dir is not None else Path("reports/_artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_pilot_session_payload(
        session=session,
        ledger=ledger,
        monitors=monitors,
        monitor_summary=monitor_summary,
        broker_status=broker_status,
        generated_at=generated_at,
    )
    path = output_dir / "pilot_session.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
