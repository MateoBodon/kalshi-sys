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
from kalshi_alpha.exec.collectors.tob_logger import (
    DEFAULT_TOB_DEPTH,
    DEFAULT_TOB_DIR,
    DEFAULT_TOB_MAX_BYTES,
)
from kalshi_alpha.exec.telemetry.run_metadata import write_telemetry_run_metadata
from kalshi_alpha.exec.preflight_index import (
    PreflightResult,
    format_preflight_summary,
    run_preflight,
    write_go_no_go_artifact,
)
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
    record_tob: bool = False
    tob_run_id: str | None = None
    tob_output_dir: Path | None = None
    tob_depth: int = DEFAULT_TOB_DEPTH
    tob_max_bytes: int = DEFAULT_TOB_MAX_BYTES
    telemetry_only: bool = False
    max_runtime_seconds: float | None = None
    skip_preflight: bool = False
    series_filter: tuple[str, ...] | None = None

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


def _emit_preflight_summary(result: PreflightResult, *, config: SupervisorIndexConfig) -> None:
    write_go_no_go_artifact(result, source="supervisor_index")
    summary = format_preflight_summary(
        result,
        label="SUPERVISOR preflight",
        series=config.series_filter,
        broker=config.normalized_broker(),
    )
    print(summary, flush=True)


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


def _series_to_run(window: TradingWindow, *, series_filter: Sequence[str] | None = None) -> tuple[str, ...]:
    candidates = [series.upper() for series in window.series]
    if series_filter:
        allowed = {series.upper() for series in series_filter}
        candidates = [series for series in candidates if series in allowed]
    return tuple(candidates)


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
        "--quality-gates-scope",
        "index",
    ]
    if config.offline:
        args.append("--offline")
    else:
        args.append("--online")
    if config.kill_switch_file:
        args.extend(["--kill-switch-file", str(config.kill_switch_file)])
    if config.record_tob:
        args.append("--record-tob")
        if config.tob_run_id:
            args.extend(["--tob-run-id", str(config.tob_run_id)])
        if config.tob_output_dir:
            args.extend(["--tob-output-dir", str(config.tob_output_dir)])
        args.extend(["--tob-depth", str(int(config.tob_depth))])
        args.extend(["--tob-max-bytes", str(int(config.tob_max_bytes))])
    if config.telemetry_only:
        args.append("--telemetry-only")
    micro_index.main(args)


async def _run_window(
    window: TradingWindow,
    *,
    now_et: datetime,
    config: SupervisorIndexConfig,
    preflight_fn: Callable[[datetime], PreflightResult],
    ws_listener: WSListener,
    runner: Callable[[str, TradingWindow, SupervisorIndexConfig, datetime], None],
    preflight_observer: Callable[[PreflightResult], None] | None = None,
    preflight_override: PreflightResult | None = None,
) -> tuple[bool, bool]:
    """Return (ran, terminal). terminal=True marks window complete."""

    if now_et < window.start_et:
        return False, False
    if now_et > window.freeze_et:
        _log(f"skip {window.label}: past cancel buffer ({window.freeze_et.isoformat()})", quiet=config.quiet)
        return False, True

    if preflight_override is not None:
        preflight = preflight_override
    else:
        if config.skip_preflight:
            preflight = PreflightResult(go=True, reasons=[], details={"skipped": True})
        else:
            preflight = preflight_fn(now_et)
        if preflight_observer is not None:
            preflight_observer(preflight)
    series_to_run = _series_to_run(window, series_filter=config.series_filter)
    if config.record_tob and config.tob_run_id:
        window_payload = {
            "label": window.label,
            "target_type": window.target_type,
            "target_et": window.target_et.isoformat(),
            "start_et": window.start_et.isoformat(),
            "freeze_et": window.freeze_et.isoformat(),
        }
        preflight_payload = {"go": bool(preflight.go), "reasons": list(preflight.reasons)}
        if config.telemetry_only and not preflight.go:
            preflight_payload["override"] = "telemetry_only"
        status = "GO" if preflight.go else "NO-GO"
        write_telemetry_run_metadata(
            run_id=config.tob_run_id,
            output_dir=config.tob_output_dir or DEFAULT_TOB_DIR,
            status=status,
            broker=config.normalized_broker(),
            telemetry_only=bool(config.telemetry_only),
            series=series_to_run,
            window=window_payload,
            preflight=preflight_payload,
        )
    if not preflight.go:
        _log(f"NO-GO {window.label}: {', '.join(preflight.reasons)}", quiet=config.quiet)
        if config.telemetry_only:
            _log(
                f"telemetry-only override enabled for {window.label} (dry-run, NO-GO)",
                quiet=config.quiet,
            )
        else:
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
        asyncio.to_thread(runner, series, window, config, now_et) for series in series_to_run
    ]
    if tasks:
        _log(f"running window {window.label} for series {','.join(series_to_run)}", quiet=config.quiet)
        await asyncio.gather(*tasks)
    return True, True


async def _run_once(
    config: SupervisorIndexConfig,
    *,
    preflight_fn: Callable[[datetime], PreflightResult],
    ws_factory: Callable[[], WSListener],
    runner: Callable[[str, TradingWindow, SupervisorIndexConfig, datetime], None],
    preflight_observer: Callable[[PreflightResult], None] | None = None,
) -> None:
    now_reference = config.now or datetime.now(tz=UTC)
    now_et = now_reference.astimezone(ET)
    preflight_override: PreflightResult | None = None
    if preflight_observer is not None:
        if config.skip_preflight:
            preflight_override = PreflightResult(go=True, reasons=[], details={"skipped": True})
        else:
            preflight_override = preflight_fn(now_et)
        preflight_observer(preflight_override)
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
            preflight_observer=None if preflight_override else preflight_observer,
            preflight_override=preflight_override,
        )
    finally:
        await ws_listener.stop()


async def _run_loop(
    config: SupervisorIndexConfig,
    *,
    preflight_fn: Callable[[datetime], PreflightResult],
    ws_factory: Callable[[], WSListener],
    runner: Callable[[str, TradingWindow, SupervisorIndexConfig, datetime], None],
    preflight_observer: Callable[[PreflightResult], None] | None = None,
) -> None:
    ws_listener = ws_factory()
    await ws_listener.start()
    try:
        start_time = datetime.now(tz=UTC)
        current_day = None
        completed: set[tuple[str, datetime.date]] = set()
        while True:
            if config.max_runtime_seconds and config.max_runtime_seconds > 0:
                elapsed = (datetime.now(tz=UTC) - start_time).total_seconds()
                if elapsed >= config.max_runtime_seconds:
                    _log(
                        f"max runtime reached ({config.max_runtime_seconds:.0f}s); stopping loop",
                        quiet=config.quiet,
                    )
                    break
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
                    preflight_observer=preflight_observer,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index ladder supervisor (paper, maker-only).")
    parser.add_argument("--loop", action="store_true", help="Run continuously across windows until close.")
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS, help="Loop sleep interval.")
    parser.add_argument(
        "--series",
        nargs="+",
        help="Restrict supervisor to specific series (INX/INXU/NASDAQ100/NASDAQ100U).",
    )
    parser.add_argument("--broker", default="dry", choices=["dry", "live"], help="Execution broker (default dry).")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --broker dry (safe default).")
    parser.add_argument("--offline", action="store_true", help="Use offline fixtures (skips WS gating).")
    parser.add_argument("--no-ws-listen", action="store_true", help="Disable local Polygon WS listener.")
    parser.add_argument("--ws-soft-ms", type=float, default=DEFAULT_WS_SOFT_MS, help="WS freshness soft threshold.")
    parser.add_argument("--ws-strict-ms", type=float, default=DEFAULT_WS_STRICT_MS, help="WS freshness strict threshold.")
    parser.add_argument("--kill-switch-file", type=Path, help="Override kill switch sentinel path.")
    parser.add_argument("--now", help="Override current time (ISO-8601, default: now).")
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout logging.")
    parser.add_argument("--record-tob", action="store_true", help="Record TOB snapshots for scans.")
    parser.add_argument(
        "--telemetry-only",
        action="store_true",
        help="Allow dry-run telemetry capture even if preflight is NO-GO (requires --dry-run and --record-tob).",
    )
    parser.add_argument("--tob-run-id", help="Run identifier for TOB snapshots (default: autogenerated).")
    parser.add_argument(
        "--tob-output-dir",
        type=Path,
        default=DEFAULT_TOB_DIR,
        help="Directory for TOB + quote intent telemetry (default: data/proc/telemetry).",
    )
    parser.add_argument(
        "--tob-depth",
        type=int,
        default=DEFAULT_TOB_DEPTH,
        help="Depth per side to record (max 5, default: %(default)s).",
    )
    parser.add_argument(
        "--tob-max-bytes",
        type=int,
        default=DEFAULT_TOB_MAX_BYTES,
        help="Approx max bytes per snapshot record (default: %(default)s).",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=0.0,
        help="Maximum runtime for loop mode (0 disables).",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip preflight checks (offline/dry-run only; default: off).",
    )
    parser.add_argument(
        "--preflight-retry-interval",
        type=float,
        default=60.0,
        help="Seconds before retrying a transient preflight failure inside the same window (0 to disable).",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    return parser.parse_args(list(argv) if argv is not None else None)


def _build_config(args: argparse.Namespace) -> SupervisorIndexConfig:
    broker = args.broker
    if args.dry_run:
        if args.broker != "dry":
            raise ValueError("--dry-run cannot be combined with --broker live.")
        broker = "dry"
    series_filter = tuple(str(series).upper() for series in args.series) if args.series else None
    config = SupervisorIndexConfig(
        loop=bool(args.loop),
        sleep_seconds=max(1.0, float(args.sleep_seconds)),
        broker=broker,
        offline=bool(args.offline),
        listen_ws=not bool(args.no_ws_listen) and not bool(args.offline),
        ws_soft_ms=max(1.0, float(args.ws_soft_ms)),
        ws_strict_ms=max(1.0, float(args.ws_strict_ms)),
        kill_switch_file=args.kill_switch_file,
        now=_parse_now(args.now),
        quiet=bool(args.quiet),
        preflight_retry_interval=max(0.0, float(args.preflight_retry_interval)),
        record_tob=bool(args.record_tob),
        tob_run_id=str(args.tob_run_id) if args.tob_run_id else None,
        tob_output_dir=Path(args.tob_output_dir) if args.tob_output_dir else None,
        tob_depth=max(1, int(args.tob_depth)),
        tob_max_bytes=max(256, int(args.tob_max_bytes)),
        telemetry_only=bool(args.telemetry_only),
        max_runtime_seconds=float(args.max_runtime_seconds) if args.max_runtime_seconds else None,
        skip_preflight=bool(args.skip_preflight),
        series_filter=series_filter,
    )
    if config.record_tob and not config.tob_run_id:
        config.tob_run_id = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%SZ")
    if config.skip_preflight and not config.offline:
        raise ValueError("--skip-preflight is only allowed with --offline")
    if config.telemetry_only:
        if not args.dry_run:
            raise ValueError("--telemetry-only requires --dry-run.")
        if config.normalized_broker() != "dry":
            raise ValueError("--telemetry-only requires --broker dry.")
        if not config.record_tob:
            raise ValueError("--telemetry-only requires --record-tob.")
    return config


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _build_config(args)

    ws_factory = lambda: WSListener(
        soft_ms=config.ws_soft_ms,
        strict_ms=config.ws_strict_ms,
        enabled=config.listen_ws,
    )

    runner = _default_runner
    preflight_fn = lambda moment: run_preflight(
        moment,
        kill_switch_file=config.kill_switch_file,
        require_kalshi=not config.offline,
        require_polygon=not config.offline,
        series=config.series_filter,
    )
    preflight_observer = lambda result: _emit_preflight_summary(result, config=config)

    try:
        if config.loop:
            asyncio.run(
                _run_loop(
                    config,
                    preflight_fn=preflight_fn,
                    ws_factory=ws_factory,
                    runner=runner,
                    preflight_observer=preflight_observer,
                )
            )
        else:
            asyncio.run(
                _run_once(
                    config,
                    preflight_fn=preflight_fn,
                    ws_factory=ws_factory,
                    runner=runner,
                    preflight_observer=preflight_observer,
                )
            )
    except KeyboardInterrupt:  # pragma: no cover - operator convenience
        _log("shutdown requested (Ctrl-C)", quiet=config.quiet)


__all__ = ["SupervisorIndexConfig", "main"]


if __name__ == "__main__":  # pragma: no cover
    main()
