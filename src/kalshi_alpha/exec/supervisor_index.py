"""Supervisor for index ladder windows with preflight and WS freshness gating."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Sequence
from zoneinfo import ZoneInfo

from kalshi_alpha.data import WSFreshnessSentry
from kalshi_alpha.drivers.polygon_index_ws import (
    PolygonIndexWSConfig,
    close_shared_connection,
    polygon_index_ws,
)
from kalshi_alpha.exec.preflight_index import PreflightResult, run_preflight
from kalshi_alpha.exec.runners import micro_index
from kalshi_alpha.sched import TradingWindow, next_windows, windows_for_day

ET = ZoneInfo("America/New_York")

DEFAULT_WS_SOFT_MS = 1500.0
DEFAULT_WS_STRICT_MS = 800.0
DEFAULT_SLEEP_SECONDS = 20.0


@dataclass(slots=True)
class SupervisorIndexConfig:
    loop: bool = False
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS
    broker: str = "dry"
    offline: bool = False
    listen_ws: bool = True
    ws_soft_ms: float = DEFAULT_WS_SOFT_MS
    ws_strict_ms: float = DEFAULT_WS_STRICT_MS
    kill_switch_file: Path | None = None
    now: datetime | None = None
    quiet: bool = False
    preflight_retry_interval: float = 60.0  # seconds

    def normalized_broker(self) -> str:
        return (self.broker or "dry").strip().lower()


class WSListener:
    """Lightweight index websocket listener feeding a freshness sentry."""

    def __init__(
        self,
        *,
        soft_ms: float,
        strict_ms: float,
        enabled: bool = True,
        ws_config: PolygonIndexWSConfig | None = None,
    ) -> None:
        self.enabled = enabled
        self._sentry = WSFreshnessSentry(soft_threshold_ms=soft_ms, strict_threshold_ms=strict_ms)
        self._task: asyncio.Task[None] | None = None
        self._config = ws_config or PolygonIndexWSConfig()

    async def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._consume(), name="polygon-index-ws")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        await close_shared_connection()

    async def _consume(self) -> None:
        backoff = 1.0
        while True:
            try:
                async with polygon_index_ws(self._config) as stream:
                    async for message in stream:
                        _ = message  # content unused; freshness comes from timing
                        self._sentry.record_timestamp(datetime.now(tz=UTC))
                        backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive network loop
                await close_shared_connection()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def freshness(self, *, strict: bool, now: datetime | None = None) -> tuple[bool, float | None]:
        if not self.enabled:
            return True, None
        age_ms = self._sentry.age_ms(now)
        if age_ms is None:
            return False, None
        ok = self._sentry.is_fresh(strict=strict, now=now)
        return ok, age_ms


def _log(message: str, *, quiet: bool = False) -> None:
    if quiet:
        return
    stamp = datetime.now(tz=UTC).isoformat(timespec="seconds")
    print(f"[supervisor_index] {stamp} {message}", flush=True)


def _pick_window(now_et: datetime) -> TradingWindow | None:
    """Return the active window or the next upcoming window."""

    active = None
    for window in windows_for_day(now_et.date()):
        if window.contains(now_et):
            active = window
            break
    if active:
        return active
    upcoming = next_windows(now_et, limit=1)
    return upcoming[0] if upcoming else None


def _series_to_run(window: TradingWindow) -> tuple[str, ...]:
    return tuple(series.upper() for series in window.series)


def _default_runner(
    series: str,
    window: TradingWindow,
    config: SupervisorIndexConfig,
    now_et: datetime,
) -> None:
    args: list[str] = [
        "--series",
        series,
        "--broker",
        config.normalized_broker(),
        "--quiet",
        "--now",
        now_et.isoformat(),
    ]
    if config.offline:
        args.append("--offline")
    else:
        args.append("--online")
    if config.kill_switch_file:
        args.extend(["--kill-switch-file", str(config.kill_switch_file)])
    micro_index.main(args)


async def _run_window(
    window: TradingWindow,
    *,
    now_et: datetime,
    config: SupervisorIndexConfig,
    preflight_fn: Callable[[datetime], PreflightResult],
    ws_listener: WSListener,
    runner: Callable[[str, TradingWindow, SupervisorIndexConfig, datetime], None],
) -> tuple[bool, bool]:
    """Return (ran, terminal). terminal=True marks window complete."""

    if now_et < window.start_et:
        return False, False
    if now_et > window.freeze_et:
        _log(f"skip {window.label}: past cancel buffer ({window.freeze_et.isoformat()})", quiet=config.quiet)
        return False, True

    preflight = preflight_fn(now_et)
    if not preflight.go:
        _log(f"NO-GO {window.label}: {', '.join(preflight.reasons)}", quiet=config.quiet)
        if _is_transient_preflight(preflight.reasons) and config.preflight_retry_interval > 0:
            return False, False
        return False, True

    strict = now_et >= window.freshness_strict_et
    ws_ok, age_ms = ws_listener.freshness(strict=strict, now=now_et.astimezone(UTC))
    if not ws_ok:
        age_desc = "unknown" if age_ms is None else f"{age_ms:.0f}ms"
        _log(f"skip {window.label}: polygon WS stale (age={age_desc})", quiet=config.quiet)
        return False, False
    if age_ms is not None:
        _log(
            f"fresh WS ok age={age_ms:.0f}ms strict={strict} window={window.label}",
            quiet=config.quiet,
        )

    tasks = [
        asyncio.to_thread(runner, series, window, config, now_et) for series in _series_to_run(window)
    ]
    if tasks:
        _log(f"running window {window.label} for series {','.join(_series_to_run(window))}", quiet=config.quiet)
        await asyncio.gather(*tasks)
    return True, True


async def _run_once(
    config: SupervisorIndexConfig,
    *,
    preflight_fn: Callable[[datetime], PreflightResult],
    ws_factory: Callable[[], WSListener],
    runner: Callable[[str, TradingWindow, SupervisorIndexConfig, datetime], None],
) -> None:
    now_reference = config.now or datetime.now(tz=UTC)
    now_et = now_reference.astimezone(ET)
    window = _pick_window(now_et)
    if window is None:
        _log("no upcoming index window found", quiet=config.quiet)
        return

    ws_listener = ws_factory()
    await ws_listener.start()
    try:
        await _run_window(
            window,
            now_et=now_et,
            config=config,
            preflight_fn=preflight_fn,
            ws_listener=ws_listener,
            runner=runner,
        )
    finally:
        await ws_listener.stop()


async def _run_loop(
    config: SupervisorIndexConfig,
    *,
    preflight_fn: Callable[[datetime], PreflightResult],
    ws_factory: Callable[[], WSListener],
    runner: Callable[[str, TradingWindow, SupervisorIndexConfig, datetime], None],
) -> None:
    ws_listener = ws_factory()
    await ws_listener.start()
    try:
        current_day = None
        completed: set[tuple[str, datetime.date]] = set()
        while True:
            now_reference = datetime.now(tz=UTC)
            now_et = now_reference.astimezone(ET)
            if current_day != now_et.date():
                current_day = now_et.date()
                completed.clear()

            windows_today = windows_for_day(now_et.date())
            if not windows_today:
                if not config.loop:
                    break
                await asyncio.sleep(max(config.sleep_seconds, 1.0))
                continue
            for window in windows_today:
                key = (window.label, window.target_et.date())
                if key in completed:
                    continue
                ran, terminal = await _run_window(
                    window,
                    now_et=now_et,
                    config=config,
                    preflight_fn=preflight_fn,
                    ws_listener=ws_listener,
                    runner=runner,
                )
                if terminal or now_et > window.target_et or ran:
                    completed.add(key)

            if not config.loop:
                break
            last_target = windows_today[-1].target_et
            if len(completed) >= len(windows_today) and now_et > last_target:
                break
            await asyncio.sleep(max(config.sleep_seconds, 1.0))
    finally:
        await ws_listener.stop()


def _parse_now(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _is_transient_preflight(reasons: Sequence[str]) -> bool:
    transient_tags = {"polygon_unreachable"}
    return all(reason in transient_tags for reason in reasons)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Index ladder supervisor (paper, maker-only).")
    parser.add_argument("--loop", action="store_true", help="Run continuously across windows until close.")
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS, help="Loop sleep interval.")
    parser.add_argument("--broker", default="dry", choices=["dry", "live"], help="Execution broker (default dry).")
    parser.add_argument("--offline", action="store_true", help="Use offline fixtures (skips WS gating).")
    parser.add_argument("--no-ws-listen", action="store_true", help="Disable local Polygon WS listener.")
    parser.add_argument("--ws-soft-ms", type=float, default=DEFAULT_WS_SOFT_MS, help="WS freshness soft threshold.")
    parser.add_argument("--ws-strict-ms", type=float, default=DEFAULT_WS_STRICT_MS, help="WS freshness strict threshold.")
    parser.add_argument("--kill-switch-file", type=Path, help="Override kill switch sentinel path.")
    parser.add_argument("--now", help="Override current time (ISO-8601, default: now).")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging.")
    parser.add_argument(
        "--preflight-retry-interval",
        type=float,
        default=60.0,
        help="Seconds before retrying a transient preflight failure inside the same window (0 to disable).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    config = SupervisorIndexConfig(
        loop=bool(args.loop),
        sleep_seconds=max(1.0, float(args.sleep_seconds)),
        broker=args.broker,
        offline=bool(args.offline),
        listen_ws=not bool(args.no_ws_listen) and not bool(args.offline),
        ws_soft_ms=max(1.0, float(args.ws_soft_ms)),
        ws_strict_ms=max(1.0, float(args.ws_strict_ms)),
        kill_switch_file=args.kill_switch_file,
        now=_parse_now(args.now),
        quiet=bool(args.quiet),
        preflight_retry_interval=max(0.0, float(args.preflight_retry_interval)),
    )

    ws_factory = lambda: WSListener(
        soft_ms=config.ws_soft_ms,
        strict_ms=config.ws_strict_ms,
        enabled=config.listen_ws,
    )

    runner = _default_runner
    preflight_fn = lambda moment: run_preflight(
        moment,
        kill_switch_file=config.kill_switch_file,
    )

    try:
        if config.loop:
            asyncio.run(
                _run_loop(
                    config,
                    preflight_fn=preflight_fn,
                    ws_factory=ws_factory,
                    runner=runner,
                )
            )
        else:
            asyncio.run(
                _run_once(
                    config,
                    preflight_fn=preflight_fn,
                    ws_factory=ws_factory,
                    runner=runner,
                )
            )
    except KeyboardInterrupt:  # pragma: no cover - operator convenience
        _log("shutdown requested (Ctrl-C)", quiet=config.quiet)


__all__ = ["SupervisorIndexConfig", "main"]
