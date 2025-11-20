"""Shared helpers for gating index runners to ET trading windows."""

from __future__ import annotations

from datetime import UTC, datetime

from kalshi_alpha.sched import TradingWindow, current_window, next_window_for_series


def parse_now_override(now_text: str | None) -> datetime:
    """Return a timezone-aware reference timestamp."""

    if not now_text:
        return datetime.now(tz=UTC)
    parsed = datetime.fromisoformat(now_text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def guard_series_window(
    series: str,
    *,
    now: datetime | None = None,
    quiet: bool = False,
) -> tuple[bool, TradingWindow | None, TradingWindow | None]:
    """Check whether *series* is inside an active trading window.

    Returns a tuple ``(allowed, active_window, next_window)``. ``allowed`` is True
    when the current time is inside the configured window for *series*.
    """

    reference = now or datetime.now(tz=UTC)
    active = current_window(series, reference)
    if active is not None:
        return True, active, None
    upcoming = next_window_for_series(series, reference)
    if not quiet:
        if upcoming is None:
            print(f"[window] No configured window found for {series}")
        else:
            print(
                f"[window] {series.upper()} closed; next window {upcoming.label} "
                f"starts at {upcoming.start_et.isoformat()} (target {upcoming.target_et.isoformat()})"
            )
    return False, None, upcoming


__all__ = ["guard_series_window", "parse_now_override"]
