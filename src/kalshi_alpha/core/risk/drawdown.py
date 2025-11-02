"""Persistent drawdown guard for paper PnL caps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from kalshi_alpha.datastore.paths import PROC_ROOT

ET = ZoneInfo("America/New_York")


@dataclass
class DrawdownStatus:
    ok: bool
    reasons: list[str]
    metrics: dict[str, float]


def record_pnl(
    pnl: float,
    *,
    timestamp: datetime | None = None,
    state_dir: Path | None = None,
) -> None:
    """Persist paper ledger PnL into daily/weekly aggregates."""

    ts = timestamp.astimezone(UTC) if timestamp else datetime.now(tz=UTC)
    et_date = ts.astimezone(ET).date()
    week_info = ts.astimezone(ET).isocalendar()
    week_key = f"{week_info.year}-W{week_info.week:02d}"
    state = _load_state(state_dir)

    daily_totals = state.setdefault("daily", {})
    daily_key = et_date.isoformat()
    daily_totals[daily_key] = float(daily_totals.get(daily_key, 0.0) + pnl)

    weekly_totals = state.setdefault("weekly", {})
    weekly_totals[week_key] = float(weekly_totals.get(week_key, 0.0) + pnl)

    _save_state(state, state_dir)


def check_limits(
    daily_cap: float | None,
    weekly_cap: float | None,
    *,
    now: datetime | None = None,
    state_dir: Path | None = None,
) -> DrawdownStatus:
    """Evaluate whether drawdown caps are respected."""

    ts = now.astimezone(UTC) if now else datetime.now(tz=UTC)
    state = _load_state(state_dir)
    reasons: list[str] = []
    metrics: dict[str, float] = {}

    et_date = ts.astimezone(ET).date()
    daily_totals = state.get("daily", {})
    daily_pnl = float(daily_totals.get(et_date.isoformat(), 0.0))
    metrics["daily_pnl"] = daily_pnl
    if daily_cap is not None and daily_cap > 0:
        if daily_pnl <= -float(daily_cap):
            reasons.append(f"drawdown.daily:{daily_pnl:.2f}<=-{float(daily_cap):.2f}")

    week_info = ts.astimezone(ET).isocalendar()
    week_key = f"{week_info.year}-W{week_info.week:02d}"
    weekly_totals = state.get("weekly", {})
    weekly_pnl = float(weekly_totals.get(week_key, 0.0))
    metrics["weekly_pnl"] = weekly_pnl
    if weekly_cap is not None and weekly_cap > 0:
        if weekly_pnl <= -float(weekly_cap):
            reasons.append(f"drawdown.weekly:{weekly_pnl:.2f}<=-{float(weekly_cap):.2f}")

    return DrawdownStatus(ok=not reasons, reasons=reasons, metrics=metrics)


# --- internal helpers ------------------------------------------------------


def _state_directory(override: Path | None = None) -> Path:
    base = override if override is not None else PROC_ROOT
    return Path(base) / "state"


def _state_path(override: Path | None = None) -> Path:
    directory = _state_directory(override)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "drawdown.json"


def _load_state(override: Path | None = None) -> dict[str, Any]:
    path = _state_path(override)
    if not path.exists():
        return {"daily": {}, "weekly": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"daily": {}, "weekly": {}}


def _save_state(state: dict[str, Any], override: Path | None = None) -> None:
    path = _state_path(override)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
