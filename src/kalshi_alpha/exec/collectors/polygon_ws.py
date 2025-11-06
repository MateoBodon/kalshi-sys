"""Massive (Polygon) websocket collector for index ladders."""

from __future__ import annotations

import argparse
import asyncio
import json
import ssl
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

import certifi
import polars as pl
import websockets
from websockets.asyncio.client import ClientConnection

from kalshi_alpha.drivers.polygon_index.client import IndexSnapshot
from kalshi_alpha.drivers.polygon_index.snapshots import write_snapshot
from kalshi_alpha.exec.monitors import freshness
from kalshi_alpha.utils.keys import load_polygon_api_key

DEFAULT_WS_URL = "wss://socket.massive.com/indices"
DEFAULT_SYMBOLS = ("I:SPX", "I:NDX")
DEFAULT_CHANNEL = "AM"
CHANNEL_EVENT_ALIASES: dict[str, tuple[str, ...]] = {
    "AM": ("AM", "XA", "A"),
    "AS": ("AS", "XS"),
}
DEFAULT_FRESHNESS_CONFIG = Path("/tmp/index_freshness.yaml")
DEFAULT_FRESHNESS_OUTPUT = Path("reports/_artifacts/monitors/freshness.json")
DEFAULT_PROC_PARQUET = Path("data/proc/polygon_index/snapshot_2025-11-04.parquet")

AliasMap = dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class CollectorConfig:
    ws_url: str
    api_key: str
    channel_prefix: str
    alias_map: AliasMap
    freshness_config: Path
    freshness_output: Path
    proc_parquet: Path
    sleep_seconds: float
    max_runtime: float | None


def _resolved_aliases(raw: Sequence[str]) -> AliasMap:
    defaults = {
        "I:SPX": ("INX", "INXU"),
        "I:NDX": ("NASDAQ100", "NASDAQ100U"),
    }
    result: AliasMap = {}
    for entry in raw:
        symbol = entry.strip()
        if not symbol:
            continue
        result[symbol] = defaults.get(symbol, (symbol.replace("I:", "INX"),))
    return result or defaults


def _parse_args(argv: Sequence[str] | None = None) -> CollectorConfig:
    parser = argparse.ArgumentParser(description="Stream Massive index websocket into readiness artifacts.")
    parser.add_argument(
        "--ws-url",
        default=DEFAULT_WS_URL,
        help="Websocket endpoint (default: wss://socket.massive.com/indices)",
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated Polygon index symbols (default: I:SPX,I:NDX)",
    )
    parser.add_argument(
        "--aliases",
        action="append",
        help="Optional overrides in the form SYMBOL:SERIES. Repeat for multiple aliases.",
    )
    parser.add_argument(
        "--channel-prefix",
        default=DEFAULT_CHANNEL,
        help="Channel prefix for Massive aggregate updates (default: AM)",
    )
    parser.add_argument(
        "--freshness-config",
        type=Path,
        default=DEFAULT_FRESHNESS_CONFIG,
        help="YAML config to reuse when writing freshness artifacts.",
    )
    parser.add_argument(
        "--freshness-output",
        type=Path,
        default=DEFAULT_FRESHNESS_OUTPUT,
        help="Target freshness artifact path.",
    )
    parser.add_argument(
        "--proc-parquet",
        type=Path,
        default=DEFAULT_PROC_PARQUET,
        help="Parquet heartbeat updated per tick for quality gates.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.25,
        help="Delay between retries on failure (exponential backoff base).",
    )
    parser.add_argument(
        "--max-runtime",
        type=float,
        help="Optional maximum runtime in seconds (test harness).",
    )
    parser.add_argument(
        "--api-key",
        help="Override Polygon API key (default: load from Keychain/env).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    alias_map = _resolved_aliases(symbols)
    if args.aliases:
        for entry in args.aliases:
            if ":" not in entry:
                continue
            polygon, series = entry.split(":", 1)
            polygon = polygon.strip()
            series = series.strip()
            if not polygon or not series:
                continue
            existing = list(alias_map.get(polygon, ()))
            if series not in existing:
                existing.append(series)
            alias_map[polygon] = tuple(existing)
    if not alias_map:
        raise SystemExit("No valid Polygon symbols supplied to websocket collector")

    api_key = (args.api_key or load_polygon_api_key() or "").strip()
    if not api_key:
        raise SystemExit("Polygon API key missing (set POLYGON_API_KEY or Keychain kalshi-sys:POLYGON_API_KEY)")

    return CollectorConfig(
        ws_url=args.ws_url,
        api_key=api_key,
        channel_prefix=(args.channel_prefix.strip() or DEFAULT_CHANNEL).upper(),
        alias_map=alias_map,
        freshness_config=args.freshness_config,
        freshness_output=args.freshness_output,
        proc_parquet=args.proc_parquet,
        sleep_seconds=max(0.1, float(args.sleep_seconds)),
        max_runtime=float(args.max_runtime) if args.max_runtime is not None else None,
    )


def _normalize_entries(payload: object) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        yield payload
    elif isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict):
                yield entry


def _process_entries(
    *,
    entries: Iterable[dict[str, Any]],
    alias_map: AliasMap,
    channel_prefix: str,
    now: datetime,
    proc_parquet: Path,
    freshness_config: Path,
    freshness_output: Path,
) -> None:
    latest_ts: datetime | None = None
    normalized_prefix = channel_prefix.upper()
    accepted_events = set(CHANNEL_EVENT_ALIASES.get(normalized_prefix, (normalized_prefix,)))
    for entry in entries:
        event = str(entry.get("ev") or "").upper()
        if event not in accepted_events:
            continue
        symbol = str(entry.get("sym") or "").strip()
        aliases = alias_map.get(symbol)
        if not aliases:
            continue
        close = entry.get("c")
        try:
            price = float(close)
        except (TypeError, ValueError):
            continue
        source_ts = entry.get("s")
        if not isinstance(source_ts, (int, float)):
            source_ts = entry.get("e")
        if isinstance(source_ts, (int, float)):
            try:
                tick_ts = datetime.fromtimestamp(float(source_ts) / 1000.0, tz=UTC)
            except (ValueError, OSError):
                tick_ts = now
        else:
            tick_ts = now
        for alias in aliases:
            write_snapshot(
                IndexSnapshot(
                    ticker=alias,
                    last_price=price,
                    change=None,
                    change_percent=None,
                    previous_close=None,
                    timestamp=tick_ts,
                )
            )
        latest_ts = tick_ts if latest_ts is None else max(latest_ts, tick_ts)

    if latest_ts is None:
        return
    frame = pl.DataFrame(
        {
            "snapshot_ts": [latest_ts],
            "generated_at": [now],
            "source": ["massive_ws"],
        }
    )
    proc_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(proc_parquet)
    freshness.write_freshness_artifact(
        config_path=freshness_config,
        output_path=freshness_output,
        now=now,
    )


class TooManyConnectionsError(RuntimeError):
    """Raised when Massive websocket rejects the connection due to max_connections."""


def _iter_status(payload: object) -> Iterable[dict[str, Any]]:
    yield from (
        entry
        for entry in _normalize_entries(payload)
        if isinstance(entry, dict) and entry.get("ev") == "status"
    )


async def _await_status(
    websocket: ClientConnection,
    *,
    expected: set[str],
) -> list[object]:
    """Drain websocket until one of the expected status codes (case-insensitive) arrives.

    Returns any non-status payloads consumed during the wait so callers can process them.
    """
    buffer: list[object] = []
    while True:
        raw = await websocket.recv()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        statuses = list(_iter_status(payload))
        handled_expected = False
        for status_entry in statuses:
            status = str(status_entry.get("status") or "").lower()
            message = str(status_entry.get("message") or "").strip()
            if status in expected:
                handled_expected = True
            elif status == "connected":
                continue  # initial handshake banner
            elif status == "max_connections":
                raise TooManyConnectionsError(
                    "Massive websocket rejected connection: max_connections limit reached"
                )
            elif status in {"auth_failed", "authentication_failed", "unauthorized"}:
                raise RuntimeError(f"Massive websocket authentication failed: {message or status}")
            elif status in {"error", "subscription_error"}:
                raise RuntimeError(f"Massive websocket error: {message or status}")
        if handled_expected:
            return buffer
        if not statuses:
            buffer.append(payload)


async def _connect_once(
    config: CollectorConfig,
    *,
    ssl_context: ssl.SSLContext,
    connection_factory: Callable[..., Awaitable[ClientConnection]] | None = None,
) -> None:
    factory = connection_factory or websockets.connect
    channel_prefix = config.channel_prefix
    subscribe_payload = ",".join(f"{channel_prefix}.{symbol}" for symbol in config.alias_map)
    async with factory(
        config.ws_url,
        ping_interval=60.0,
        ping_timeout=60.0,
        ssl=ssl_context,
    ) as websocket:
        await websocket.send(json.dumps({"action": "auth", "params": config.api_key}))
        prebuffer = await _await_status(
            websocket,
            expected={"auth_success", "authenticated"},
        )
        await websocket.send(json.dumps({"action": "subscribe", "params": subscribe_payload}))
        prebuffer.extend(
            await _await_status(
                websocket,
                expected={"success", "subscribed", "subscribed to"},
            )
        )

        start = datetime.now(tz=UTC).timestamp()
        for payload in prebuffer:
            now = datetime.now(tz=UTC)
            _process_entries(
                entries=_normalize_entries(payload),
                alias_map=config.alias_map,
                channel_prefix=channel_prefix,
                now=now,
                proc_parquet=config.proc_parquet,
                freshness_config=config.freshness_config,
                freshness_output=config.freshness_output,
            )
        async for raw in websocket:
            now = datetime.now(tz=UTC)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            for status_entry in _iter_status(payload):
                status = str(status_entry.get("status") or "").lower()
                message = str(status_entry.get("message") or "").strip()
                if status == "max_connections":
                    raise TooManyConnectionsError(
                        "Massive websocket rejected connection: max_connections limit reached"
                    )
                if status in {"auth_failed", "authentication_failed", "unauthorized"}:
                    raise RuntimeError(f"Massive websocket authentication failed: {message or status}")
                if status in {"error", "subscription_error"}:
                    raise RuntimeError(f"Massive websocket error: {message or status}")
            _process_entries(
                entries=_normalize_entries(payload),
                alias_map=config.alias_map,
                channel_prefix=channel_prefix,
                now=now,
                proc_parquet=config.proc_parquet,
                freshness_config=config.freshness_config,
                freshness_output=config.freshness_output,
            )
            if config.max_runtime is not None and (now.timestamp() - start) >= config.max_runtime:
                break


async def _run_forever(config: CollectorConfig) -> None:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    backoff = config.sleep_seconds
    while True:
        try:
            await _connect_once(config, ssl_context=ssl_context)
        except asyncio.CancelledError:
            raise
        except TooManyConnectionsError as exc:
            print(f"[polygon-ws] error: {exc!s}; stopping collector", flush=True)
            break
        except Exception as exc:  # pragma: no cover - network failures
            print(f"[polygon-ws] warning: {exc!s}; reconnecting in {backoff:.1f}s", flush=True)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)
            continue
        backoff = config.sleep_seconds
        if config.max_runtime is not None:
            break


def main(argv: Sequence[str] | None = None) -> None:
    config = _parse_args(argv)
    try:
        asyncio.run(_run_forever(config))
    except KeyboardInterrupt:  # pragma: no cover - interactive run
        return


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    main()
