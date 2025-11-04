"""Read-only smoke test for Kalshi index ladders.

The script performs signed GET requests against the Kalshi trade API, verifies that the
next-hour INXU/NASDAQ100U events are available, and reports outstanding order state. It
exits with a non-zero status when any checks fail so it can be wired into pre-launch or
operations checklists.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from kalshi_alpha.brokers.kalshi.http_client import KalshiHttpClient, KalshiHttpError
from kalshi_alpha.exec.state.orders import OutstandingOrdersState

ET = ZoneInfo("America/New_York")
U_SERIES_TICKERS = ("INXU", "NASDAQ100U")


@dataclass(slots=True)
class SeriesStatus:
    ticker: str
    series_id: str | None
    target_label: str
    events: list[str]
    has_target: bool


def _now_et() -> datetime:
    return datetime.now(tz=ET)


def _target_hour_label(now_et: datetime) -> str:
    hour = now_et.hour
    if now_et.minute >= 40:
        hour = (hour + 1) % 24
    return f"H{hour:02d}00"


def _series_map(client: KalshiHttpClient) -> dict[str, str]:
    response = client.get("/series")
    payload = response.json()
    series_payload = payload.get("series") if isinstance(payload, dict) else payload
    mapping: dict[str, str] = {}
    for item in series_payload or []:
        ticker = str(item.get("ticker", "")).upper()
        series_id = str(item.get("id") or item.get("series_id") or "").strip()
        if ticker and series_id:
            mapping[ticker] = series_id
    return mapping


def _event_tickers(client: KalshiHttpClient, series_id: str) -> list[str]:
    response = client.get(f"/series/{series_id}/events")
    payload = response.json()
    events_payload = payload.get("events") if isinstance(payload, dict) else payload
    return [str(item.get("ticker") or "") for item in events_payload or []]


def _check_u_series(
    client: KalshiHttpClient,
    now_et: datetime,
) -> list[SeriesStatus]:
    try:
        mapping = _series_map(client)
    except Exception as exc:  # pragma: no cover - network failure guard
        raise RuntimeError(f"Failed to fetch series from Kalshi API: {exc}") from exc

    target_label = _target_hour_label(now_et)
    statuses: list[SeriesStatus] = []
    for ticker in U_SERIES_TICKERS:
        series_id = mapping.get(ticker)
        events: list[str] = []
        has_target = False
        if series_id:
            try:
                events = _event_tickers(client, series_id)
                has_target = any(target_label in event for event in events)
            except KalshiHttpError as exc:  # pragma: no cover - bubble up
                raise RuntimeError(f"Failed to fetch events for {ticker}: {exc}") from exc
            except Exception as exc:  # pragma: no cover - network guard
                raise RuntimeError(f"Failed to fetch events for {ticker}: {exc}") from exc
        statuses.append(
            SeriesStatus(
                ticker=ticker,
                series_id=series_id,
                target_label=target_label,
                events=events,
                has_target=has_target,
            )
        )
    return statuses


def _outstanding_summary() -> dict[str, Any]:
    state = OutstandingOrdersState.load()
    summary = state.summary()
    total = state.total()
    return {"total": total, "breakdown": summary}


def run_smoke(base_url: str | None = None) -> tuple[int, dict[str, Any]]:
    payload: dict[str, Any] = {}
    try:
        client = KalshiHttpClient(base_url=base_url or "https://api.elections.kalshi.com/trade-api/v2")
    except Exception as exc:  # pragma: no cover - auth guard
        now_et = _now_et()
        payload = {
            "timestamp_et": now_et.isoformat(),
            "series": [],
            "outstanding_orders": _outstanding_summary(),
            "reasons": [f"kalshi auth failure: {exc}"],
        }
        return 1, payload
    now_et = _now_et()
    reasons: list[str] = []
    try:
        statuses = _check_u_series(client, now_et)
    except RuntimeError as exc:
        statuses = []
        reasons.append(str(exc))
    outstanding = _outstanding_summary()
    for status in statuses:
        if not status.series_id:
            reasons.append(f"series {status.ticker} not returned by /series")
        elif not status.has_target:
            reasons.append(f"missing {status.target_label} event for {status.ticker}")

    if outstanding["total"]:
        reasons.append(f"{outstanding['total']} outstanding orders pending cancel ACK")

    payload = {
        "timestamp_et": now_et.isoformat(),
        "series": [
            {
                "ticker": status.ticker,
                "series_id": status.series_id,
                "target_label": status.target_label,
                "events_sample": status.events[:5],
                "has_target": status.has_target,
            }
            for status in statuses
        ],
        "outstanding_orders": outstanding,
        "reasons": reasons,
    }
    exit_code = 0 if not reasons else 1
    return exit_code, payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Read-only Kalshi live smoke check.")
    parser.add_argument(
        "--base-url",
        default="https://api.elections.kalshi.com/trade-api/v2",
        help="Kalshi API base URL (default: %(default)s)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human text")
    args = parser.parse_args(argv)

    exit_code, payload = run_smoke(base_url=args.base_url)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        status = "OK" if exit_code == 0 else "FAIL"
        print(f"[live_smoke] {status} {payload['timestamp_et']}")
        for entry in payload["series"]:
            label = "found" if entry["has_target"] else "missing"
            print(
                f"  {entry['ticker']}: target {entry['target_label']} {label}"
            )
        outstanding = payload["outstanding_orders"]
        print(
            f"  outstanding orders: {outstanding['total']} ({outstanding['breakdown']})"
        )
        if payload["reasons"]:
            print("Reasons:")
            for reason in payload["reasons"]:
                print(f"  - {reason}")

    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
