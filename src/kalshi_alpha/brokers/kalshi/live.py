"""Live Kalshi broker adapter with rate limiting, backoff, and auditing."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from requests import Response

from kalshi_alpha.brokers.kalshi.base import Broker, BrokerOrder, ensure_directory
from kalshi_alpha.brokers.kalshi.http_client import (
    KalshiClockSkewError,
    KalshiHttpClient,
    KalshiHttpError,
)
from kalshi_alpha.core.execution.order_queue import OrderQueue
from kalshi_alpha.core.kalshi_api import KalshiPublicClient, Orderbook
from kalshi_alpha.exec.heartbeat import kill_switch_engaged, resolve_kill_switch_path
from kalshi_alpha.exec.index_paper_ledger import INDEX_SERIES
from kalshi_alpha.exec.pilot.config import PilotConfig, load_pilot_config
from kalshi_alpha.exec.state.orders import OutstandingOrdersState
from kalshi_alpha.exec.telemetry import TelemetrySink, sanitize_book_snapshot
from kalshi_alpha.sched import current_window as scheduler_current_window
from kalshi_alpha.utils.env import load_env

LOGGER = logging.getLogger(__name__)
OrderbookFetcher = Callable[[str], Orderbook | None]
_BID_KEYS = ("bid", "best_bid", "best_bid_price")
_ASK_KEYS = ("ask", "best_ask", "best_ask_price")


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, str):
            return float(value)
    except (TypeError, ValueError):
        return None
    return None


def _price_from_entry(entry: object) -> float | None:
    if isinstance(entry, Mapping):
        return _safe_float(entry.get("price"))
    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, bytearray)) and entry:
        return _safe_float(entry[0])
    return _safe_float(entry)


def _best_price(levels: object, *, side: str) -> float | None:
    if not levels:
        return None
    best: float | None = None
    entries = levels if isinstance(levels, Sequence) else [levels]
    for entry in entries:
        price = _price_from_entry(entry)
        if price is None:
            continue
        if best is None:
            best = price
        elif side == "bid":
            best = max(best, price)
        else:
            best = min(best, price)
    return best


def _extract_best_prices(snapshot: object) -> tuple[float | None, float | None]:
    if snapshot is None:
        return None, None
    if isinstance(snapshot, Orderbook):
        return _best_price(snapshot.bids, side="bid"), _best_price(snapshot.asks, side="ask")
    if isinstance(snapshot, Mapping):
        bid = None
        ask = None
        for key in _BID_KEYS:
            if key in snapshot:
                bid = _price_from_entry(snapshot.get(key))
                if bid is not None:
                    break
        for key in _ASK_KEYS:
            if key in snapshot:
                ask = _price_from_entry(snapshot.get(key))
                if ask is not None:
                    break
        if bid is None and "bid" in snapshot:
            bid = _price_from_entry(snapshot.get("bid"))
        if ask is None and "ask" in snapshot:
            ask = _price_from_entry(snapshot.get("ask"))
        if bid is not None or ask is not None:
            return bid, ask
        bids = snapshot.get("bids")
        asks = snapshot.get("asks")
        return _best_price(bids, side="bid"), _best_price(asks, side="ask")
    return None, None


def _validate_live_environment() -> None:
    access_key = os.getenv("KALSHI_API_KEY_ID", "").strip()
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PEM_PATH", "").strip()
    if not access_key:
        raise RuntimeError("KALSHI_API_KEY_ID is required to authenticate live trading.")
    if not key_path:
        raise RuntimeError("KALSHI_PRIVATE_KEY_PEM_PATH is required to authenticate live trading.")
    resolved = Path(key_path).expanduser()
    if not resolved.exists():
        raise RuntimeError(f"Kalshi private key path does not exist: {resolved}")


class _RateLimiter:
    """Token bucket rate limiter used to throttle Trading API calls."""

    def __init__(self, max_calls: int, per_seconds: float) -> None:
        self._max_calls = max(1, max_calls)
        self._per_seconds = max(per_seconds, 0.1)
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until a slot becomes available."""
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self._per_seconds:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return
                earliest = self._timestamps[0]
                sleep_for = max(0.0, self._per_seconds - (now - earliest))
            if sleep_for <= 0:
                sleep_for = self._per_seconds / self._max_calls
            time.sleep(sleep_for)


class LiveBroker(Broker):
    """Networked broker adapter for the Kalshi trading API."""

    mode = "live"

    def __init__(  # noqa: PLR0913 - broker wiring requires multiple knobs
        self,
        *,
        artifacts_dir: Path,
        audit_dir: Path,
        session: requests.Session | None = None,
        base_url: str = "https://api.elections.kalshi.com/trade-api/v2",
        rate_limit_per_second: int = 5,
        queue_capacity: int = 64,
        max_retries: int = 3,
        timeout: float = 10.0,
        retry_backoff: float = 0.5,
        http_client: KalshiHttpClient | None = None,
        order_queue: OrderQueue | None = None,
        telemetry_sink: TelemetrySink | None = None,
        acknowledge_risks: bool = False,
        pilot_mode: bool = False,
        pilot_config: PilotConfig | None = None,
        kill_switch_path: Path | str | None = None,
        orderbook_client: KalshiPublicClient | None = None,
        orderbook_fetcher: OrderbookFetcher | None = None,
        clock: Callable[[], datetime] | None = None,
        orders_state_path: Path | None = None,
    ) -> None:
        if not acknowledge_risks:
            raise RuntimeError(
                "Live broker requires explicit acknowledgement via --i-understand-the-risks."
            )
        if os.environ.get("CI"):
            raise RuntimeError("Live broker is disabled while running under CI.")

        load_env()
        _validate_live_environment()
        self._artifacts_dir = ensure_directory(artifacts_dir)
        self._audit_dir = ensure_directory(audit_dir)
        self._http = http_client or KalshiHttpClient(
            base_url=base_url,
            session=session,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        self._rate_limiter = _RateLimiter(rate_limit_per_second, 1.0)
        self._seen_idempotency: set[str] = set()
        self._lock = threading.Lock()
        self._order_queue = order_queue or OrderQueue(
            capacity=queue_capacity,
            max_retries=max_retries,
            audit_callback=self._queue_audit,
        )
        self._telemetry = telemetry_sink
        self._pilot_mode = bool(pilot_mode or pilot_config is not None)
        if self._pilot_mode:
            self._pilot_config = pilot_config or load_pilot_config(None)
        else:
            self._pilot_config = None
        self._kill_switch_path = resolve_kill_switch_path(kill_switch_path)
        self._orderbook_client = orderbook_client
        self._orderbook_fetcher = orderbook_fetcher
        self._clock = clock or (lambda: datetime.now(tz=UTC))
        self._orders_state_path = orders_state_path

        LOGGER.info("Live broker initialized; submissions remain feature-gated.")

    # Broker interface ----------------------------------------------------------------------------

    def place(self, orders: Sequence[BrokerOrder]) -> None:
        accepted: list[BrokerOrder] = []
        for order in orders:
            with self._lock:
                if order.idempotency_key in self._seen_idempotency:
                    LOGGER.debug(
                        "Skipping duplicate order with idempotency %s",
                        order.idempotency_key,
                    )
                    continue
                self._seen_idempotency.add(order.idempotency_key)
            accepted.append(order)
        if not accepted:
            return
        if self._kill_switch_blocked("submit"):
            raise RuntimeError(
                f"Kill switch engaged at {self._kill_switch_path.as_posix()}; refusing submit"
            )
        self._enforce_pilot_boundary(accepted)

        for order in accepted:
            payload = self._order_payload(order)
            event_payload = self._telemetry_payload(order, payload)
            self._emit_telemetry("sent", event_payload)
            start_ns = time.perf_counter_ns()
            try:
                response = self._request(
                    "POST",
                    "/portfolio/orders",
                    json_body=payload,
                    idempotency_key=order.idempotency_key,
                )
            except Exception as exc:  # pragma: no cover - verified via tests
                error_payload = dict(event_payload)
                error_payload["error"] = str(exc)
                if exc.__cause__ is not None:
                    error_payload["error_cause"] = str(exc.__cause__)
                error_payload["latency_ms"] = self._elapsed_ms(start_ns)
                self._emit_telemetry("reject", error_payload)
                raise
            ack_payload = dict(event_payload)
            ack_payload["status_code"] = response.status_code
            ack_payload["latency_ms"] = self._elapsed_ms(start_ns)
            self._emit_telemetry("ack", ack_payload)
            self._write_audit("place_intent", order)

    def cancel(self, order_ids: Sequence[str]) -> None:
        if self._kill_switch_blocked("cancel"):
            return
        for order_id in order_ids:
            self._order_queue.enqueue_cancel(order_id, self._submit_cancel)

    def replace(self, orders: Sequence[BrokerOrder]) -> None:
        if self._kill_switch_blocked("replace"):
            return
        self._enforce_pilot_boundary(orders)
        for order in orders:
            existing_order_id = None
            if order.metadata:
                existing_order_id = str(order.metadata.get("order_id") or "")
            if not existing_order_id:
                LOGGER.warning("Replace called without existing order id; queuing as place only.")
                self.place([order])
                continue
            self._order_queue.enqueue_replace(
                order_id=existing_order_id,
                new_order=order,
                cancel_fn=self._submit_cancel,
                place_fn=self._submit_replace,
            )

    def status(self) -> dict[str, Any]:
        queue_depth = self._order_queue.depth()
        return {
            "mode": self.mode,
            "queue_depth": queue_depth,
            "orders_recorded": len(self._seen_idempotency),
            "message": "Live broker armed; submissions gated by CLI flags.",
        }

    # Internal helpers ---------------------------------------------------------------------------

    def _queue_audit(self, action: str, metadata: dict[str, Any]) -> None:
        payload = dict(metadata)
        payload.setdefault("queue_action", action)
        self._write_audit("queue_drop", extra=payload)
        if metadata:
            telem_payload = {
                "queue_action": action,
                "idempotency_key": metadata.get("idempotency_key"),
                "order_id": metadata.get("order_id"),
            }
            self._emit_telemetry("reject", telem_payload)

    def _kill_switch_blocked(self, action: str) -> bool:
        if not kill_switch_engaged(self._kill_switch_path):
            return False
        LOGGER.warning(
            "Kill switch engaged at %s; blocking live %s",
            self._kill_switch_path.as_posix(),
            action,
        )
        self._emit_telemetry(
            "reject",
            {"action": action, "reason": "kill_switch_engaged", "path": self._kill_switch_path.as_posix()},
        )
        return True

    def _now(self) -> datetime:
        moment = self._clock()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment

    def _fetch_orderbook(self, market_id: str) -> Orderbook | None:
        if self._orderbook_fetcher is not None:
            return self._orderbook_fetcher(market_id)
        if self._orderbook_client is None:
            self._orderbook_client = KalshiPublicClient()
        return self._orderbook_client.get_orderbook(market_id)

    def _resolve_best_bid_ask(
        self,
        order: BrokerOrder,
        cache: dict[str, tuple[float | None, float | None]],
    ) -> tuple[float | None, float | None]:
        metadata = order.metadata or {}
        snapshot = metadata.get("book_snapshot") or metadata.get("orderbook")
        bid, ask = _extract_best_prices(snapshot)
        if bid is not None or ask is not None:
            return bid, ask
        market_id = order.market_id or str(metadata.get("market_id") or "")
        if not market_id:
            return None, None
        if market_id in cache:
            return cache[market_id]
        try:
            orderbook = self._fetch_orderbook(market_id)
        except Exception as exc:
            LOGGER.warning("Pilot maker-only book fetch failed for %s: %s", market_id, exc)
            cache[market_id] = (None, None)
            return None, None
        bid, ask = _extract_best_prices(orderbook)
        cache[market_id] = (bid, ask)
        return bid, ask

    def _load_outstanding_bins(self) -> tuple[set[tuple[str, float, str]], set[str]]:
        bins: set[tuple[str, float, str]] = set()
        series_seen: set[str] = set()
        outstanding = OutstandingOrdersState.load(self._orders_state_path).outstanding_for("live")
        for record in outstanding.values():
            metadata = record.get("metadata", {}) if isinstance(record, dict) else {}
            series = str(metadata.get("series") or "").strip().upper()
            if not series:
                raise RuntimeError("pilot_outstanding_series_missing")
            series_seen.add(series)
            market_id = str(record.get("market_id") or "").strip()
            side = str(record.get("side") or "").strip().upper()
            strike = record.get("strike")
            if not market_id or not side or strike is None:
                raise RuntimeError("pilot_outstanding_missing_fields")
            bins.add((market_id, float(strike), side))
        return bins, series_seen

    def _enforce_pilot_boundary(self, orders: Sequence[BrokerOrder]) -> None:
        if not self._pilot_mode:
            return
        if not orders:
            return
        config = self._pilot_config
        if config is None:
            raise RuntimeError("pilot_config_missing")

        max_contracts = int(config.max_contracts_per_order)
        max_bins = int(config.max_unique_bins)
        if max_contracts <= 0 or max_bins <= 0:
            raise RuntimeError("pilot_config_invalid_caps")

        series_values: set[str] = set()
        for order in orders:
            metadata = order.metadata or {}
            series = str(metadata.get("series") or "").strip().upper()
            if not series:
                raise RuntimeError("pilot_series_missing")
            series_values.add(series)
            if series not in INDEX_SERIES:
                raise RuntimeError(f"pilot_series_not_allowed:{series}")
            if config.allowed_series and series not in config.allowed_series:
                raise RuntimeError(f"pilot_series_not_allowed:{series}")
            if int(order.contracts) <= 0:
                raise RuntimeError("pilot_contracts_invalid")
            if int(order.contracts) > max_contracts:
                raise RuntimeError("pilot_max_contracts_exceeded")
            if config.enforce_maker_only:
                liquidity = str(metadata.get("liquidity") or "maker").lower()
                if liquidity != "maker":
                    raise RuntimeError("pilot_maker_only_required")

        if len(series_values) > 1:
            raise RuntimeError("pilot_multiple_series")
        series = next(iter(series_values))

        now = self._now()
        window = scheduler_current_window(series, now)
        if window is None:
            raise RuntimeError(f"pilot_window_closed:{series}")
        if window.seconds_to_freeze(now) <= 0:
            raise RuntimeError(f"pilot_window_frozen:{series}")

        new_bins = {
            (order.market_id, float(order.strike), str(order.side).strip().upper()) for order in orders
        }
        if len(new_bins) > max_bins:
            raise RuntimeError("pilot_max_unique_bins_exceeded")

        outstanding_bins, outstanding_series = self._load_outstanding_bins()
        if outstanding_series and (len(outstanding_series) > 1 or series not in outstanding_series):
            raise RuntimeError("pilot_outstanding_series_mismatch")
        if len(outstanding_bins | new_bins) > max_bins:
            raise RuntimeError("pilot_max_unique_bins_exceeded")

        if config.enforce_maker_only:
            cache: dict[str, tuple[float | None, float | None]] = {}
            for order in orders:
                side = str(order.side).strip().upper()
                bid, ask = self._resolve_best_bid_ask(order, cache)
                if bid is not None and ask is not None and bid >= ask:
                    raise RuntimeError("pilot_maker_only_invalid_book")
                if side == "YES":
                    if ask is None:
                        raise RuntimeError("pilot_maker_only_missing_tob")
                    if float(order.price) >= float(ask) - 1e-9:
                        raise RuntimeError("pilot_maker_only_crossing")
                elif side == "NO":
                    if bid is None:
                        raise RuntimeError("pilot_maker_only_missing_tob")
                    if float(order.price) <= float(bid) + 1e-9:
                        raise RuntimeError("pilot_maker_only_crossing")
                else:
                    raise RuntimeError(f"pilot_unknown_side:{order.side}")

    def _submit_cancel(self, order_id: str) -> None:
        if self._kill_switch_blocked("cancel"):
            return
        endpoint = f"/portfolio/orders/{order_id}"
        start_ns = time.perf_counter_ns()
        try:
            response = self._request("DELETE", endpoint, json_body={})
        except Exception as exc:  # pragma: no cover - network errors surfaced in tests
            self._emit_telemetry(
                "reject",
                {
                    "order_id": order_id,
                    "action": "cancel",
                    "error": str(exc),
                    "error_cause": str(exc.__cause__) if exc.__cause__ is not None else None,
                    "latency_ms": self._elapsed_ms(start_ns),
                },
            )
            raise
        self._emit_telemetry(
            "cancel",
            {
                "order_id": order_id,
                "status_code": response.status_code,
                "latency_ms": self._elapsed_ms(start_ns),
            },
        )
        self._write_audit("cancel_intent", extra={"order_id": order_id})

    def _submit_replace(self, order_id: str, order: BrokerOrder) -> None:
        if self._kill_switch_blocked("replace"):
            return
        payload = self._order_payload(order)
        endpoint = f"/portfolio/orders/{order_id}/replace"
        event_payload = self._telemetry_payload(order, payload)
        event_payload["replace_of"] = order_id
        self._emit_telemetry("sent", event_payload)
        start_ns = time.perf_counter_ns()
        try:
            response = self._request(
                "POST",
                endpoint,
                json_body=payload,
                idempotency_key=order.idempotency_key,
            )
        except Exception as exc:  # pragma: no cover
            error_payload = dict(event_payload)
            error_payload["error"] = str(exc)
            if exc.__cause__ is not None:
                error_payload["error_cause"] = str(exc.__cause__)
            error_payload["latency_ms"] = self._elapsed_ms(start_ns)
            self._emit_telemetry("reject", error_payload)
            raise
        ack_payload = dict(event_payload)
        ack_payload["status_code"] = response.status_code
        ack_payload["latency_ms"] = self._elapsed_ms(start_ns)
        self._emit_telemetry("ack", ack_payload)
        self._write_audit(
            "replace_intent",
            extra={"order_id": order_id, "idempotency_key": order.idempotency_key},
        )

    def _order_payload(self, order: BrokerOrder) -> dict[str, Any]:
        metadata = dict(order.metadata or {})
        ticker = metadata.get("market_ticker") or metadata.get("ticker")
        if not ticker:
            raise RuntimeError("Missing market ticker for Kalshi order payload")
        liquidity = str(metadata.get("liquidity") or "maker").lower()
        action = metadata.get("action") or ("sell" if liquidity == "maker" else "buy")
        side = order.side.lower()
        yes_price = float(order.price)
        payload: dict[str, Any] = {
            "ticker": ticker,
            "action": action.lower(),
            "side": side,
            "type": "limit",
            "count": int(order.contracts),
            "yes_price": yes_price,
            "client_order_id": order.idempotency_key,
        }
        # Provide the complementary price to avoid ambiguity on NO orders.
        payload["no_price"] = max(0.0, min(1.0, 1.0 - yes_price))
        return payload

    def _telemetry_payload(self, order: BrokerOrder, payload: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(order.metadata or {})
        book_snapshot = metadata.get("book_snapshot") or metadata.get("orderbook")
        sanitized_snapshot = sanitize_book_snapshot(book_snapshot)
        metadata.pop("book_snapshot", None)
        metadata.pop("orderbook", None)
        event_payload: dict[str, Any] = {
            "idempotency_key": order.idempotency_key,
            "market_id": order.market_id,
            "side": order.side,
            "contracts": order.contracts,
            "price": order.price,
            "size": order.contracts,
            "probability": order.probability,
        }
        if "order_id" in metadata:
            event_payload["order_id"] = metadata.get("order_id")
        if sanitized_snapshot is not None:
            event_payload["book_snapshot"] = sanitized_snapshot
        if metadata:
            event_payload["metadata"] = metadata
        return event_payload

    def _emit_telemetry(self, event_type: str, data: dict[str, Any]) -> None:
        if self._telemetry is None:
            return
        self._telemetry.emit(event_type, source="rest", data=data)

    @staticmethod
    def _elapsed_ms(start_ns: int) -> float:
        return max(0.0, (time.perf_counter_ns() - start_ns) / 1_000_000.0)

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        json_body: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> Response:
        self._rate_limiter.wait()
        try:
            response = self._http.request(
                method,
                endpoint,
                json_body=json_body,
                idempotency_key=idempotency_key,
            )
        except KalshiClockSkewError:
            raise
        except KalshiHttpError as exc:
            raise RuntimeError("Failed to execute Kalshi trading API request") from exc
        return response

    def _write_audit(
        self,
        action: str,
        order: BrokerOrder | None = None,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "mode": self.mode,
            "action": action,
        }
        if order is not None:
            entry.update(
                {
                    "idempotency_key": order.idempotency_key,
                    "market_id": order.market_id,
                    "side": order.side,
                    "price": order.price,
                    "contracts": order.contracts,
                    "probability": order.probability,
                    "strike": order.strike,
                }
            )
        if extra:
            entry.update(extra)

        filename = self._audit_dir / f"live_orders_{datetime.now(tz=UTC).date().isoformat()}.jsonl"
        with filename.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True))
            handle.write("\n")
