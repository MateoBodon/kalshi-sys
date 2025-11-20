"""24/7 supervisor daemon that orchestrates live index ladder scans."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from kalshi_alpha.brokers import create_broker
from kalshi_alpha.data import WSFreshnessSentry
from kalshi_alpha.drivers.polygon_index.client import INDICES_WS_URL, PolygonIndicesClient
from kalshi_alpha.exec.heartbeat import resolve_kill_switch_path, write_heartbeat
from kalshi_alpha.exec.runners import scan_ladders
from kalshi_alpha.exec.scanners import scan_index_close
from kalshi_alpha.utils.env import load_env

ET = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
HOURLY_TRIGGER_MINUTE = 55
DAILY_CLOSE_TRIGGER = time(15, 50)
DEFAULT_WS_SYMBOLS: tuple[str, ...] = ("I:SPX", "I:NDX")
DEFAULT_HOURLY_SERIES: tuple[str, ...] = ("INXU", "NASDAQ100U")
DEFAULT_CLOSE_SERIES: tuple[str, ...] = ("INX", "NASDAQ100")


def _default_kill_switch_path() -> Path:
    return resolve_kill_switch_path()


@dataclass(frozen=True)
class SupervisorConfig:
    broker_mode: str = "dry"
    acknowledge_risks: bool = False
    sniper: bool = False
    sniper_threshold: float = 0.05
    poll_seconds: float = 15.0
    offline: bool = False
    ws_latency_kill_ms: float = 500.0
    ws_symbols: tuple[str, ...] = DEFAULT_WS_SYMBOLS
    hourly_series: tuple[str, ...] = DEFAULT_HOURLY_SERIES
    close_series: tuple[str, ...] = DEFAULT_CLOSE_SERIES
    kill_switch_path: Path = field(default_factory=_default_kill_switch_path)
    ws_url: str = INDICES_WS_URL

    def normalized_broker(self) -> str:
        return (self.broker_mode or "dry").strip().lower()


class Supervisor:
    def __init__(self, config: SupervisorConfig) -> None:
        self.config = config
        self._ws_sentry = WSFreshnessSentry(strict_threshold_ms=config.ws_latency_kill_ms)
        self._ws_task: asyncio.Task[None] | None = None
        self._scan_lock = asyncio.Lock()
        self._last_hourly_trigger: tuple[datetime.date, int] | None = None
        self._last_close_trigger: datetime.date | None = None
        self._kill_switch_triggered = False
        self._last_ws_latency_ms: float | None = None
        self._last_ws_tick: datetime | None = None

    async def run(self) -> None:
        load_env()
        while True:
            now_utc = datetime.now(tz=UTC)
            now_et = now_utc.astimezone(ET)
            await self._ensure_ws_task()
            engaged = self._kill_switch_engaged()
            ws_age_ms = self._ws_sentry.age_ms()
            if ws_age_ms is not None and ws_age_ms > self.config.ws_latency_kill_ms:
                self._engage_kill_switch("polygon_ws_stale")
            if self._market_open(now_et) and not engaged:
                await self._ensure_broker()
                await self._maybe_trigger_hourly(now_et)
                await self._maybe_trigger_close(now_et)
            await self._write_heartbeat(now_utc, now_et, ws_age_ms)
            await asyncio.sleep(max(1.0, self.config.poll_seconds))

    async def _ensure_ws_task(self) -> None:
        if self._ws_task is None or self._ws_task.done():
            self._ws_task = asyncio.create_task(self._stream_polygon_ws(), name="polygon-ws")

    async def _stream_polygon_ws(self) -> None:
        client = PolygonIndicesClient(ws_url=self.config.ws_url)
        symbols = list(self.config.ws_symbols)
        backoff = 1.0
        while True:
            try:
                async for message in client.stream_aggregates(
                    symbols,
                    timespan="second",
                    reconnect_attempts=None,
                ):
                    self._handle_ws_message(message)
                    backoff = 1.0
            except Exception as exc:  # pragma: no cover - resilience
                self._log(f"[ws] stream error: {exc}; backoff {backoff:.1f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def _handle_ws_message(self, message: object) -> None:
        if not isinstance(message, dict):
            return
        ts_raw = message.get("s") or message.get("e") or message.get("t")
        tick_ts = None
        if isinstance(ts_raw, (int, float)):
            try:
                tick_ts = datetime.fromtimestamp(float(ts_raw) / 1000.0, tz=UTC)
            except (OSError, ValueError):
                tick_ts = None
        if tick_ts is None:
            tick_ts = datetime.now(tz=UTC)
        latency_ms = self._ws_sentry.record_timestamp(tick_ts)
        self._last_ws_latency_ms = latency_ms
        self._last_ws_tick = tick_ts
        if latency_ms > self.config.ws_latency_kill_ms:
            self._engage_kill_switch("polygon_ws_latency_exceeded")

    async def _ensure_broker(self) -> None:
        normalized = self.config.normalized_broker()
        try:
            await asyncio.to_thread(
                create_broker,
                normalized,
                artifacts_dir=Path("reports/_artifacts"),
                audit_dir=Path("data/proc/audit"),
                acknowledge_risks=self.config.acknowledge_risks,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self._log(f"[broker] failed to initialize broker ({normalized}): {exc}")

    async def _maybe_trigger_hourly(self, now_et: datetime) -> None:
        if now_et.minute < HOURLY_TRIGGER_MINUTE:
            return
        key = (now_et.date(), now_et.hour)
        if self._last_hourly_trigger == key:
            return
        await self._run_scan_ladders(label="hourly")
        self._last_hourly_trigger = key

    async def _maybe_trigger_close(self, now_et: datetime) -> None:
        if now_et.time() < DAILY_CLOSE_TRIGGER:
            return
        if self._last_close_trigger == now_et.date():
            return
        await self._run_scan_close()
        self._last_close_trigger = now_et.date()

    async def _run_scan_ladders(self, *, label: str) -> None:
        async with self._scan_lock:
            if self._kill_switch_engaged():
                self._log("[scan] kill switch engaged; skipping scan_ladders")
                return
            series_list = self.config.hourly_series or ("INXU",)
            for series in series_list:
                args: list[str] = ["--series", series, "--online", "--broker", self.config.normalized_broker()]
                args.append("--clear-dry-orders-start")
                args.append("--quiet")
                if self.config.sniper:
                    args.append("--sniper")
                    args.extend(["--sniper-threshold", f"{self.config.sniper_threshold:.4f}"])
                if self.config.normalized_broker() == "live" and self.config.acknowledge_risks:
                    args.append("--i-understand-the-risks")
                if self.config.offline:
                    args.append("--offline")
                    if "--online" in args:
                        args.remove("--online")
                args.extend(["--kill-switch-file", self.config.kill_switch_path.as_posix()])
                self._log(f"[scan] running scan_ladders ({label}) series={series} args={args}")
                await asyncio.to_thread(scan_ladders.main, args)

    async def _run_scan_close(self) -> None:
        async with self._scan_lock:
            if self._kill_switch_engaged():
                self._log("[scan] kill switch engaged; skipping scan_index_close")
                return
            series_list = self.config.close_series or ("INX", "NASDAQ100")
            args: list[str] = ["--series", *series_list, "--quiet"]
            if self.config.offline:
                args.append("--offline")
            self._log(f"[scan] running scan_index_close args={args}")
            await asyncio.to_thread(scan_index_close.main, args)

    def _market_open(self, now_et: datetime) -> bool:
        if now_et.weekday() >= 5:
            return False
        current = now_et.time()
        return MARKET_OPEN <= current < MARKET_CLOSE

    def _kill_switch_engaged(self) -> bool:
        return self.config.kill_switch_path.exists() or self._kill_switch_triggered

    def _engage_kill_switch(self, reason: str) -> None:
        if self._kill_switch_triggered:
            return
        self._kill_switch_triggered = True
        path = self.config.kill_switch_path
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).isoformat()
        path.write_text(f"{timestamp} {reason}\n", encoding="utf-8")
        self._log(f"[kill-switch] engaged ({reason}) at {path.as_posix()}")

    async def _write_heartbeat(self, now_utc: datetime, now_et: datetime, ws_age_ms: float | None) -> None:
        monitors = {
            "ws_latency_ms": self._last_ws_latency_ms,
            "ws_last_tick": self._last_ws_tick.isoformat() if self._last_ws_tick else None,
            "ws_age_ms": ws_age_ms,
            "sniper_enabled": self.config.sniper,
            "sniper_threshold": self.config.sniper_threshold if self.config.sniper else None,
            "kill_switch_path": self.config.kill_switch_path.as_posix(),
            "kill_switch_engaged": self._kill_switch_engaged(),
        }
        write_heartbeat(
            mode="supervisor",
            monitors=monitors,
            extra={"broker": self.config.broker_mode},
            now=now_utc,
        )
        self._log(
            "[hb] {et} ws_latency_ms={lat:.1f} kill={kill}".format(
                et=now_et.isoformat(timespec="seconds"),
                lat=self._last_ws_latency_ms or -1.0,
                kill=self._kill_switch_engaged(),
            )
        )

    def _log(self, message: str) -> None:
        timestamp = datetime.now(tz=UTC).isoformat(timespec="seconds")
        print(f"[supervisor] {timestamp} {message}", flush=True)


def _parse_series(value: str | None, *, default: Iterable[str]) -> tuple[str, ...]:
    if not value:
        return tuple(default)
    parts = [part.strip().upper() for part in value.split(",") if part.strip()]
    return tuple(parts) or tuple(default)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Kalshi Alpha supervisor daemon.")
    parser.add_argument("--broker", default="dry", choices=["dry", "live"], help="Broker mode (default: dry).")
    parser.add_argument("--ack-risks", "--i-understand-the-risks", dest="ack_risks", action="store_true")
    parser.add_argument("--sniper", action="store_true", help="Enable taker sniper mode for scan_ladders.")
    parser.add_argument(
        "--sniper-threshold",
        type=float,
        default=0.05,
        help="Absolute probability gap required to take liquidity (default: 0.05).",
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0, help="Supervisor loop sleep (sec).")
    parser.add_argument("--offline", action="store_true", help="Use offline fixtures instead of live data.")
    parser.add_argument(
        "--ws-latency-kill-ms",
        type=float,
        default=500.0,
        help="Kill-switch latency threshold for Polygon websocket (ms).",
    )
    parser.add_argument(
        "--hourly-series",
        help="Comma-separated hourly series tickers (default: INXU,NASDAQ100U).",
    )
    parser.add_argument(
        "--close-series",
        help="Comma-separated close series tickers (default: INX,NASDAQ100).",
    )
    parser.add_argument(
        "--kill-switch-path",
        type=Path,
        help="Override kill switch sentinel path (default: data/proc/state/kill_switch).",
    )
    parser.add_argument("--ws-url", default=INDICES_WS_URL, help="Polygon indices websocket URL.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = SupervisorConfig(
        broker_mode=args.broker,
        acknowledge_risks=bool(args.ack_risks),
        sniper=bool(args.sniper),
        sniper_threshold=float(args.sniper_threshold),
        poll_seconds=max(1.0, float(args.poll_seconds)),
        offline=bool(args.offline),
        ws_latency_kill_ms=max(1.0, float(args.ws_latency_kill_ms)),
        hourly_series=_parse_series(args.hourly_series, default=DEFAULT_HOURLY_SERIES),
        close_series=_parse_series(args.close_series, default=DEFAULT_CLOSE_SERIES),
        kill_switch_path=resolve_kill_switch_path(args.kill_switch_path),
        ws_url=args.ws_url,
    )

    supervisor = Supervisor(config)
    try:
        asyncio.run(supervisor.run())
    except KeyboardInterrupt:  # pragma: no cover - interactive convenience
        supervisor._log("shutdown requested (Ctrl-C)")


if __name__ == "__main__":  # pragma: no cover
    main()
